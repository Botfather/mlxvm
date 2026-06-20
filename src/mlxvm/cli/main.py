from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from mlxvm import __version__
from mlxvm.config import AppPaths, load_settings, validate_generation
from mlxvm.diagnostics import Diagnostic, run_diagnostics
from mlxvm.diagnostics.doctor import _memory_bytes
from mlxvm.errors import (
    ConfigurationError,
    MlxvmError,
    ModelNotFoundError,
    SafetyError,
)
from mlxvm.hub import HubClient, RemoteModel
from mlxvm.lifecycle import ModelManager
from mlxvm.logging import configure_logging
from mlxvm.onboarding import (
    detect_shell,
    install_shell_integration,
    is_shell_integrated,
    memory_guidance,
    recommendations_for_memory,
    shell_config_path,
)
from mlxvm.registry import ModelRecord, Registry
from mlxvm.resolver import ModelResolver, Resolution
from mlxvm.runtime import RuntimeRunner
from mlxvm.shell import completion_script, environment_commands, shell_init
from mlxvm.upgrade import check_upgrade, install_upgrade

GLOBAL_FLAGS = {"--json", "--verbose", "--yes", "--no-interactive", "--offline"}


class MlxvmArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ConfigurationError(message)


def _normalize_global_args(argv: Sequence[str]) -> List[str]:
    """Allow global boolean flags before or after the subcommand."""
    before_separator: List[str] = []
    after_separator: List[str] = []
    separator_seen = False
    globals_found: List[str] = []
    for value in argv:
        if value == "--":
            separator_seen = True
            after_separator.append(value)
        elif not separator_seen and value in GLOBAL_FLAGS:
            globals_found.append(value)
        elif separator_seen:
            after_separator.append(value)
        else:
            before_separator.append(value)
    return globals_found + before_separator + after_separator


def _add_model_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", help="override the effective model or alias")
    parser.add_argument("--profile", help="generation profile from config.toml")
    parser.add_argument("--trust-remote-code", action="store_true")


def _add_generation_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--min-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-kv-size", type=int)
    parser.add_argument("--system-prompt")
    parser.add_argument("--seed", type=int)


def _parser() -> argparse.ArgumentParser:
    parser = MlxvmArgumentParser(prog="mlxvm", description="Manage local MLX models")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--json", action="store_true", help="emit stable JSON output")
    parser.add_argument("--verbose", action="store_true", help="show debug logs on stderr")
    parser.add_argument("--yes", action="store_true", help="accept confirmation prompts")
    parser.add_argument("--no-interactive", action="store_true", help="never prompt for input")
    parser.add_argument("--offline", action="store_true", help="use only locally cached data")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("doctor", help="check system and runtime compatibility")

    search = subparsers.add_parser("search", help="search compatible Hugging Face models")
    search.add_argument("query", nargs="?")
    search.add_argument("--limit", type=int, default=20)

    remote = subparsers.add_parser("ls-remote", help="script-friendly remote model search")
    remote.add_argument("query", nargs="?")
    remote.add_argument("--limit", type=int, default=20)

    install = subparsers.add_parser("install", help="download or register a model")
    install.add_argument("spec", help="Hugging Face repo[@revision] or local directory")
    install.add_argument("--alias")
    install.add_argument("--quantize", type=int, choices=(2, 3, 4, 6, 8))
    install.add_argument("--trust-remote-code", action="store_true")
    install.add_argument("--dry-run", action="store_true")

    subparsers.add_parser("ls", help="list installed models")

    current = subparsers.add_parser("current", help="show the effective model selection")
    current.add_argument("--model")
    current.add_argument("--directory", type=Path, default=Path.cwd(), help=argparse.SUPPRESS)

    use = subparsers.add_parser("use", help="select a model for the current shell")
    use.add_argument("target")
    use.add_argument("--profile")
    use.add_argument("--shell", choices=("bash", "zsh", "fish"), default="zsh")

    subparsers.add_parser("deactivate", help="clear current-shell selection")

    env = subparsers.add_parser("env", help="emit shell environment changes")
    env.add_argument("action", choices=("use", "deactivate"))
    env.add_argument("target", nargs="?")
    env.add_argument("--profile")
    env.add_argument("--shell", choices=("bash", "zsh", "fish"), required=True)

    alias = subparsers.add_parser("alias", help="create or replace a model alias")
    alias.add_argument("name")
    alias.add_argument("target")

    unalias = subparsers.add_parser("unalias", help="remove a model alias")
    unalias.add_argument("name")

    uninstall = subparsers.add_parser("uninstall", help="unregister and safely remove a model")
    uninstall.add_argument("target")

    run = subparsers.add_parser("run", help="stream one-shot generation")
    run.add_argument("prompt", nargs="?")
    run.add_argument("--prompt-cache", help="named prompt cache created by 'mlxvm cache create'")
    _add_model_option(run)
    _add_generation_options(run)

    chat = subparsers.add_parser("chat", help="start a stateful chat session")
    _add_model_option(chat)
    _add_generation_options(chat)

    serve = subparsers.add_parser("serve", help="start MLX-LM's development HTTP server")
    _add_model_option(serve)
    _add_generation_options(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)

    execute = subparsers.add_parser("exec", help="run a command with model environment variables")
    execute.add_argument("target")
    execute.add_argument("--profile")
    execute.add_argument("program", nargs=argparse.REMAINDER)

    cache = subparsers.add_parser("cache", help="manage downloads and prompt caches")
    cache_commands = cache.add_subparsers(dest="cache_command", required=True)
    cache_commands.add_parser("prune", help="remove unreferenced revisions and partials")
    cache_commands.add_parser("ls", help="list prompt caches")
    cache_create = cache_commands.add_parser("create", help="create a reusable prompt cache")
    cache_create.add_argument("name")
    cache_create.add_argument("prompt")
    _add_model_option(cache_create)
    cache_create.add_argument("--max-kv-size", type=int)
    cache_remove = cache_commands.add_parser("remove", help="remove a prompt cache")
    cache_remove.add_argument("name")

    init = subparsers.add_parser("shell-init", help="print shell integration")
    init.add_argument("shell", choices=("bash", "zsh", "fish"))
    completions = subparsers.add_parser("completions", help="print shell completions")
    completions.add_argument("shell", choices=("bash", "zsh", "fish"))
    upgrade = subparsers.add_parser("upgrade", help="check for or install a newer mlxvm release")
    upgrade.add_argument("--check", action="store_true")
    return parser


def _payload(
    command: str,
    data: Any,
    *,
    ok: bool = True,
    error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {"schema_version": 1, "command": command, "ok": ok, "data": data, "error": error}


def _emit_json(value: Dict[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _human_size(size: Optional[int]) -> str:
    if size is None:
        return "unknown"
    value = float(size)
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or suffix == "TiB":
            return f"{value:.1f} {suffix}" if suffix != "B" else f"{int(value)} B"
        value /= 1024
    raise AssertionError("unreachable")


def _print_doctor(diagnostics: Iterable[Diagnostic]) -> None:
    symbols = {"pass": "OK", "warn": "WARN", "fail": "FAIL"}
    for diagnostic in diagnostics:
        print(f"{symbols[diagnostic.status]:<4} {diagnostic.name:<20} {diagnostic.message}")


def _print_models(models: List[ModelRecord], active: Resolution) -> None:
    if not models:
        print("No models installed.")
        return
    for model in models:
        marker = "*" if active.model and active.model.id == model.id else " "
        aliases = f" [{', '.join(model.aliases)}]" if model.aliases else ""
        missing = " (missing)" if not model.path.exists() else ""
        print(f"{marker} {model.reference}{aliases}  {_human_size(model.size_bytes)}{missing}")


def _print_current(resolution: Resolution) -> None:
    if not resolution.selected:
        print(resolution.error, file=sys.stderr)
        return
    reference = resolution.model.reference if resolution.model else resolution.requested
    location = f" ({resolution.source_path})" if resolution.source_path else ""
    print(reference)
    print(f"selected by: {resolution.source}{location}")
    if resolution.error:
        print(f"warning: {resolution.error}", file=sys.stderr)


def _print_remote(models: List[RemoteModel]) -> None:
    if not models:
        print("No compatible models found.")
        return
    for index, model in enumerate(models, 1):
        gated = " gated" if model.gated else ""
        print(
            f"{index:>2}. {model.repo_id:<55} "
            f"downloads={model.downloads:<10} likes={model.likes}{gated}"
        )


def _confirm(
    prompt: str,
    args: argparse.Namespace,
    *,
    destructive: bool = False,
    default: bool = False,
) -> bool:
    if args.yes:
        return True
    if args.no_interactive or not sys.stdin.isatty():
        if destructive:
            raise SafetyError(f"confirmation required: {prompt}", hint="pass --yes to proceed")
        return True
    choice = "[Y/n]" if default else "[y/N]"
    response = input(f"{prompt} {choice} ").strip().lower()
    if not response:
        return default
    return response in {"y", "yes"}


def _require_model(
    resolver: ModelResolver, explicit: Optional[str] = None
) -> tuple[ModelRecord, Resolution]:
    resolution = resolver.resolve(explicit=explicit)
    if resolution.model is None:
        raise ModelNotFoundError(
            resolution.error or "no model selected",
            hint="run 'mlxvm ls' or install a model and select it",
        )
    if not resolution.model.path.exists():
        raise ModelNotFoundError(
            f"registered model files are missing: {resolution.model.path}",
            hint="uninstall the stale registry entry and install the model again",
        )
    return resolution.model, resolution


def _generation_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    keys = (
        "temperature",
        "top_p",
        "min_p",
        "top_k",
        "max_tokens",
        "max_kv_size",
        "system_prompt",
        "seed",
    )
    return {key: getattr(args, key) for key in keys if getattr(args, key, None) is not None}


def _install_with_confirmation(
    manager: ModelManager,
    spec: str,
    args: argparse.Namespace,
    *,
    alias: Optional[str] = None,
    friendly: bool = False,
):
    parsed, revision, variant, plan, existing = manager.prepare_install(
        spec, quantize=getattr(args, "quantize", None)
    )
    if existing:
        return manager.install(
            spec,
            alias=alias,
            quantize=getattr(args, "quantize", None),
            trust_remote_code=getattr(args, "trust_remote_code", False),
            dry_run=getattr(args, "dry_run", False),
            capture_runtime=args.json,
        )
    if plan:
        if friendly:
            print(f"\nDownload size: about {_human_size(plan.download_bytes)}.")
            print(
                "The model stays on this Mac in mlxvm's private storage. "
                "You can remove it later with 'mlxvm uninstall default'."
            )
        else:
            print(
                f"Pinned {parsed} to {revision}. Required size: {_human_size(plan.total_bytes)}; "
                f"download: {_human_size(plan.download_bytes)}.",
                file=sys.stderr,
            )
        memory = _memory_bytes()
        if memory and plan.total_bytes > memory * 0.8:
            print(
                f"warning: model size is high relative to {_human_size(memory)} unified memory",
                file=sys.stderr,
            )
        anchor = manager.paths.home
        while not anchor.exists() and anchor != anchor.parent:
            anchor = anchor.parent
        free = shutil.disk_usage(anchor).free
        if plan.download_bytes > free:
            raise SafetyError(
                f"insufficient disk space: need {_human_size(plan.download_bytes)}, "
                f"have {_human_size(free)}"
            )
    if not getattr(args, "dry_run", False) and not _confirm(
        "Install this model?", args, default=friendly
    ):
        raise SafetyError("installation cancelled")
    return manager.install(
        spec,
        alias=alias,
        quantize=getattr(args, "quantize", None),
        trust_remote_code=getattr(args, "trust_remote_code", False),
        dry_run=getattr(args, "dry_run", False),
        capture_runtime=args.json,
    )


def _choose_number(prompt: str, maximum: int, *, default: Optional[int] = None) -> Optional[int]:
    while True:
        response = input(prompt).strip().lower()
        if not response and default is not None:
            return default
        if response in {"q", "quit"} or not response:
            return None
        try:
            choice = int(response)
        except ValueError:
            choice = 0
        if 1 <= choice <= maximum:
            return choice
        print(f"Please enter a number from 1 to {maximum}, or q to quit.")


def _setup_shell_for_beginner(args: argparse.Namespace) -> None:
    shell = detect_shell()
    if shell is None:
        print("\nI couldn't identify your shell. Run 'mlxvm shell-init --help' later for setup.")
        return
    config_path = shell_config_path(shell)
    if is_shell_integrated(shell):
        print(f"\nShell selection is ready in {config_path}.")
        return
    if _confirm(
        f"Set up {shell} so 'mlxvm use' works in future terminal sessions?",
        args,
        default=True,
    ):
        try:
            result = install_shell_integration(shell)
        except (OSError, UnicodeError) as exc:
            print(f"I couldn't update {config_path}: {exc}")
            print(f"You can set it up later with 'mlxvm shell-init {shell}'.")
        else:
            action = "Updated" if result.changed else "Already configured"
            print(f"{action}: {result.path}")
            print("New terminal windows will load mlxvm automatically.")
    else:
        command = (
            "mlxvm shell-init fish | source"
            if shell == "fish"
            else f'eval "$(mlxvm shell-init {shell})"'
        )
        print(f"Skipped shell setup. Later, add: {command}")


def _start_beginner_chat(manager: ModelManager, runtime: RuntimeRunner, model: ModelRecord) -> int:
    settings = load_settings(manager.paths.config)
    print("\nStarting your first chat now.")
    print("Type q and press Return when you want to leave. Type h inside chat for help.\n")
    runtime.chat(
        model,
        settings.generation_for(),
        trust_remote_code=settings.trust_remote_code,
    )
    return 0


def _interactive_workflow(
    args: argparse.Namespace,
    manager: ModelManager,
    hub: HubClient,
    resolver: ModelResolver,
    runtime: RuntimeRunner,
) -> int:
    if args.no_interactive or not sys.stdin.isatty():
        _parser().print_help()
        return 0

    print("\nWelcome to mlxvm — let's get a local model running.\n")
    _setup_shell_for_beginner(args)

    installed = manager.registry.list_models()
    active = resolver.resolve()
    if active.model is not None:
        print(f"\nYour active model is {active.model.repo_id}.")
        return _start_beginner_chat(manager, runtime, active.model)
    if installed:
        print("\nYou already have these models installed:")
        for index, model in enumerate(installed, 1):
            print(f"  {index}. {model.repo_id} ({_human_size(model.size_bytes)})")
        choice = _choose_number("Choose one to make the default [1]: ", len(installed), default=1)
        if choice is None:
            return 0
        model = installed[choice - 1]
        manager.alias("default", model.reference)
        return _start_beginner_chat(manager, runtime, model)

    memory = _memory_bytes()
    print(f"\n{memory_guidance(memory)}\n")
    recommendations = recommendations_for_memory(memory)
    print("Good starting choices for this Mac:")
    for index, model in enumerate(recommendations, 1):
        marker = " — Recommended" if index == 1 else ""
        print(
            f"  {index}. {model.name}{marker}\n"
            f"     {model.description} Download: about {_human_size(model.download_bytes)}."
        )
    search_choice = len(recommendations) + 1
    quit_choice = search_choice + 1
    print(f"  {search_choice}. Search for a different model")
    print(f"  {quit_choice}. Quit")
    choice = _choose_number("\nChoose a model [1]: ", quit_choice, default=1)
    if choice is None or choice == quit_choice:
        return 0

    if choice == search_choice:
        query = input("Search MLX models: ").strip()
        if not query:
            return 0
        models = hub.search(query, limit=10)
        _print_remote(models)
        if not models:
            return 0
        remote_choice = _choose_number("Choose a model (q to quit): ", len(models))
        if remote_choice is None:
            return 0
        repo_id = models[remote_choice - 1].repo_id
    else:
        repo_id = recommendations[choice - 1].repo_id

    result = _install_with_confirmation(
        manager,
        repo_id,
        args,
        alias="default",
        friendly=True,
    )
    if result.model is None:
        raise SafetyError("installation completed without registering a model")
    status = "Already installed" if result.already_installed else "Installed"
    print(f"\n{status}: {result.model.repo_id}")
    return _start_beginner_chat(manager, runtime, result.model)


def _dispatch(
    args: argparse.Namespace,
    paths: AppPaths,
    registry: Registry,
    resolver: ModelResolver,
    manager: ModelManager,
    hub: HubClient,
    runtime: RuntimeRunner,
) -> int:
    command = args.command
    if command == "doctor":
        checks = run_diagnostics(paths)
        failed = any(check.status == "fail" for check in checks)
        if args.json:
            _emit_json(
                _payload("doctor", {"checks": [check.to_dict() for check in checks]}, ok=not failed)
            )
        else:
            _print_doctor(checks)
        return 1 if failed else 0

    if command in {"search", "ls-remote"}:
        if not 1 <= args.limit <= 100:
            raise SafetyError("search limit must be between 1 and 100")
        models = hub.search(args.query, limit=args.limit)
        if args.json:
            _emit_json(_payload(command, {"models": [model.to_dict() for model in models]}))
        else:
            _print_remote(models)
        if (
            command == "search"
            and not args.json
            and not args.no_interactive
            and sys.stdin.isatty()
            and models
        ):
            choice = input("Install model number (blank to stop): ").strip()
            if choice:
                try:
                    selected = models[int(choice) - 1]
                except (ValueError, IndexError) as exc:
                    raise SafetyError("invalid model selection") from exc
                result = _install_with_confirmation(manager, selected.repo_id, args)
                print(f"Installed {result.model.reference}")
        return 0

    if command == "install":
        result = _install_with_confirmation(manager, args.spec, args, alias=args.alias)
        if args.json:
            _emit_json(_payload("install", result.to_dict()))
        elif result.dry_run:
            print("Dry run complete; no files were changed.")
        elif result.already_installed:
            print(f"Already installed: {result.model.reference}")
        else:
            print(f"Installed: {result.model.reference}")
        return 0

    if command == "ls":
        models = registry.list_models()
        active = resolver.resolve()
        if args.json:
            _emit_json(
                _payload(
                    "ls", {"models": [m.to_dict() for m in models], "active": active.to_dict()}
                )
            )
        else:
            _print_models(models, active)
        return 0

    if command == "current":
        active = resolver.resolve(explicit=args.model, start=args.directory)
        ok = active.model is not None and active.error is None
        if args.json:
            error = {"code": "model_not_found", "message": active.error} if active.error else None
            _emit_json(_payload("current", active.to_dict(), ok=ok, error=error))
        else:
            _print_current(active)
        return 0 if ok else 1

    if command in {"use", "deactivate", "env"}:
        action = args.action if command == "env" else command
        shell = getattr(args, "shell", "zsh")
        if action == "deactivate":
            values = {"MLXVM_MODEL": None, "MLXVM_REVISION": None, "MLXVM_PROFILE": None}
            model_data = None
        else:
            target = args.target
            model = registry.resolve(target)
            if model is None:
                raise ModelNotFoundError(f"model or alias '{target}' is not installed")
            settings = load_settings(paths.config)
            settings.generation_for(getattr(args, "profile", None))
            values = {
                "MLXVM_MODEL": model.reference,
                "MLXVM_REVISION": model.revision,
                "MLXVM_PROFILE": getattr(args, "profile", None),
            }
            model_data = model.to_dict()
        if args.json and command != "env":
            _emit_json(_payload(command, {"environment": values, "model": model_data}))
        else:
            print(environment_commands(shell, values))
        return 0

    if command == "alias":
        model = manager.alias(args.name, args.target)
        if args.json:
            _emit_json(_payload("alias", {"name": args.name, "model": model.to_dict()}))
        else:
            print(f"{args.name} -> {model.reference}")
        return 0

    if command == "unalias":
        manager.unalias(args.name)
        if args.json:
            _emit_json(_payload("unalias", {"name": args.name}))
        else:
            print(f"Removed alias: {args.name}")
        return 0

    if command == "uninstall":
        model = registry.resolve(args.target)
        if model is None:
            raise ModelNotFoundError(f"model or alias '{args.target}' is not installed")
        if not _confirm(f"Uninstall {model.reference}?", args, destructive=True):
            raise SafetyError("uninstall cancelled")
        removed = manager.uninstall(args.target)
        if args.json:
            _emit_json(_payload("uninstall", {"model": removed.to_dict()}))
        else:
            print(f"Uninstalled: {removed.reference}")
        return 0

    if command in {"run", "chat", "serve"}:
        model, resolution = _require_model(resolver, args.model)
        settings = load_settings(paths.config)
        profile = args.profile or os.environ.get("MLXVM_PROFILE")
        generation = settings.generation_for(profile, resolution.generation)
        generation.update(_generation_overrides(args))
        generation = validate_generation(generation, "effective generation settings")
        trust = bool(args.trust_remote_code or settings.trust_remote_code)
        if command == "run":
            prompt = args.prompt
            if prompt is None:
                prompt = "-" if not sys.stdin.isatty() else input("Prompt: ")
            prompt_cache = None
            if args.prompt_cache:
                prompt_cache = runtime.prompt_cache_path(args.prompt_cache)
                if not prompt_cache.is_file():
                    raise SafetyError(f"prompt cache '{args.prompt_cache}' does not exist")
            text = runtime.generate(
                model,
                prompt,
                generation,
                trust_remote_code=trust,
                capture=args.json,
                prompt_cache=prompt_cache,
            )
            if args.json:
                _emit_json(_payload("run", {"model": model.to_dict(), "text": text}))
            return 0
        if args.json:
            raise SafetyError(f"--json is not supported for interactive '{command}'")
        if command == "chat":
            runtime.chat(model, generation, trust_remote_code=trust)
            return 0
        if args.host not in {"127.0.0.1", "localhost", "::1"}:
            if not _confirm(
                f"Expose MLX-LM's development server on non-local address {args.host}?",
                args,
                destructive=True,
            ):
                raise SafetyError("serve cancelled")
        print(
            "warning: MLX-LM's HTTP server is development-oriented, not production hardened.",
            file=sys.stderr,
        )
        runtime.serve(
            model,
            generation,
            host=args.host,
            port=args.port,
            trust_remote_code=trust,
        )
        return 0

    if command == "exec":
        model = registry.resolve(args.target)
        if model is None:
            raise ModelNotFoundError(f"model or alias '{args.target}' is not installed")
        program = args.program[1:] if args.program and args.program[0] == "--" else args.program
        return runtime.execute(model, program, profile=args.profile)

    if command == "cache":
        if args.cache_command == "prune":
            result = manager.prune()
            if args.json:
                _emit_json(_payload("cache prune", result))
            else:
                total = result["bytes"] + result["converted_bytes"]
                print(f"Freed {_human_size(total)}; removed {result['revisions']} revisions.")
            return 0
        if args.cache_command == "ls":
            caches = runtime.list_prompt_caches()
            if args.json:
                _emit_json(_payload("cache ls", {"prompt_caches": caches}))
            elif not caches:
                print("No prompt caches.")
            else:
                for cache in caches:
                    print(f"{cache['name']:<30} {_human_size(cache['size_bytes'])}")
            return 0
        if args.cache_command == "create":
            model, _ = _require_model(resolver, args.model)
            settings = load_settings(paths.config)
            trust = bool(args.trust_remote_code or settings.trust_remote_code)
            path = runtime.create_prompt_cache(
                model,
                args.name,
                args.prompt,
                max_kv_size=args.max_kv_size,
                trust_remote_code=trust,
                capture=args.json,
            )
            if args.json:
                _emit_json(_payload("cache create", {"name": args.name, "path": str(path)}))
            else:
                print(path)
            return 0
        path = runtime.remove_prompt_cache(args.name)
        if args.json:
            _emit_json(_payload("cache remove", {"name": args.name, "path": str(path)}))
        else:
            print(f"Removed prompt cache: {args.name}")
        return 0

    if command == "shell-init":
        print(shell_init(args.shell), end="")
        return 0

    if command == "completions":
        print(completion_script(args.shell))
        return 0

    if command == "upgrade":
        if args.offline:
            raise SafetyError("upgrade is unavailable in offline mode")
        info = check_upgrade()
        if args.check or not info.update_available:
            if args.json:
                _emit_json(_payload("upgrade", info.to_dict()))
            elif info.update_available:
                print(f"Update available: {info.current} -> {info.latest}")
            else:
                print(f"mlxvm {info.current} is up to date.")
            return 0
        if not _confirm(f"Upgrade mlxvm {info.current} to {info.latest}?", args, destructive=True):
            raise SafetyError("upgrade cancelled")
        if args.json:
            raise SafetyError("--json requires --check for upgrade")
        install_upgrade()
        print(f"Upgraded mlxvm to {info.latest}.")
        return 0

    raise ConfigurationError(f"unknown command: {command}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = _parser()
    try:
        args = parser.parse_args(_normalize_global_args(raw_argv))
    except ConfigurationError as exc:
        paths = AppPaths.discover()
        try:
            parse_logger = configure_logging(paths.logs, verbose="--verbose" in raw_argv)
        except OSError:
            parse_logger = None
        if parse_logger:
            parse_logger.warning("configuration_error: %s", exc.message)
        if "--json" in raw_argv:
            command = next((value for value in raw_argv if not value.startswith("-")), "unknown")
            _emit_json(
                _payload(
                    command,
                    None,
                    ok=False,
                    error={"code": exc.code, "message": exc.message, "hint": exc.hint},
                )
            )
        else:
            print(f"error: {exc.message}", file=sys.stderr)
            print(f"hint: run '{parser.prog} --help' for usage", file=sys.stderr)
        return exc.exit_code

    paths = AppPaths.discover()
    try:
        logger = configure_logging(paths.logs, verbose=args.verbose)
    except OSError as exc:
        message = f"cannot initialize logs under {paths.logs}: {exc}"
        if "--json" in raw_argv:
            _emit_json(
                _payload(
                    "startup",
                    None,
                    ok=False,
                    error={"code": "storage_error", "message": message},
                )
            )
        else:
            print(f"error: {message}", file=sys.stderr)
        return 1
    try:
        settings = load_settings(paths.config)
        offline = bool(args.offline or settings.offline)
        registry = Registry(paths.registry)
        resolver = ModelResolver(registry)
        hub = HubClient(paths.hub_cache, offline=offline)
        manager = ModelManager(paths, registry, hub)
        runtime = RuntimeRunner(paths)
        if not args.command:
            if args.json:
                active = resolver.resolve()
                _emit_json(
                    _payload(
                        "interactive",
                        {
                            "models": [model.to_dict() for model in registry.list_models()],
                            "active": active.to_dict(),
                        },
                    )
                )
                return 0
            return _interactive_workflow(args, manager, hub, resolver, runtime)
        logger.debug("command=%s json=%s offline=%s", args.command, args.json, offline)
        return _dispatch(args, paths, registry, resolver, manager, hub, runtime)
    except MlxvmError as exc:
        logger.warning("%s: %s", exc.code, exc.message)
        if args.verbose:
            logger.debug("error details: %s", exc.details)
        if args.json:
            _emit_json(
                _payload(
                    args.command or "interactive",
                    None,
                    ok=False,
                    error={
                        "code": exc.code,
                        "message": exc.message,
                        "hint": exc.hint,
                        "details": exc.details,
                    },
                )
            )
        else:
            print(f"error: {exc.message}", file=sys.stderr)
            if exc.hint:
                print(f"hint: {exc.hint}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        logger.info("operation interrupted by user")
        if args.json:
            _emit_json(
                _payload(
                    args.command or "interactive",
                    None,
                    ok=False,
                    error={"code": "interrupted", "message": "operation interrupted"},
                )
            )
        else:
            print("\nInterrupted.", file=sys.stderr)
        return 130
    except (sqlite3.Error, OSError) as exc:
        logger.exception("storage operation failed")
        error = MlxvmError(
            f"storage operation failed: {exc}",
            code="storage_error",
            hint=f"inspect {paths.logs / 'mlxvm.log'}",
        )
        if args.json:
            _emit_json(
                _payload(
                    args.command or "interactive",
                    None,
                    ok=False,
                    error={"code": error.code, "message": error.message, "hint": error.hint},
                )
            )
        else:
            print(f"error: {error.message}\nhint: {error.hint}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("unexpected failure")
        if args.json:
            _emit_json(
                _payload(
                    args.command or "interactive",
                    None,
                    ok=False,
                    error={
                        "code": "internal_error",
                        "message": "unexpected internal error",
                        "hint": f"inspect {paths.logs / 'mlxvm.log'}",
                    },
                )
            )
        else:
            print(
                f"error: unexpected internal error: {exc}\n"
                f"hint: inspect {paths.logs / 'mlxvm.log'}",
                file=sys.stderr,
            )
        return 70
