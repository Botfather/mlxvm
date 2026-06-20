from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from mlxvm.config import AppPaths
from mlxvm.errors import DependencyError, RuntimeFailure, SafetyError
from mlxvm.locks import FileLock
from mlxvm.logging import get_logger
from mlxvm.registry import ModelRecord


class RuntimeRunner:
    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths
        self.logger = get_logger()

    @staticmethod
    def ensure_available() -> None:
        if importlib.util.find_spec("mlx_lm") is None:
            raise DependencyError(
                "mlx-lm is not installed",
                hint="reinstall mlxvm with its required dependencies",
            )

    def generate(
        self,
        model: ModelRecord,
        prompt: str,
        settings: Dict[str, Any],
        *,
        trust_remote_code: bool = False,
        capture: bool = False,
        prompt_cache: Optional[Path] = None,
    ) -> Optional[str]:
        self.ensure_available()
        command = [
            sys.executable,
            "-m",
            "mlxvm.runtime.worker",
            "--model",
            str(model.path),
            "--prompt",
            prompt,
            "--settings",
            json.dumps(settings, separators=(",", ":")),
        ]
        if trust_remote_code:
            command.append("--trust-remote-code")
        if prompt_cache:
            command.extend(["--prompt-cache", str(prompt_cache)])
        return self._run(command, capture=capture)

    def chat(
        self,
        model: ModelRecord,
        settings: Dict[str, Any],
        *,
        trust_remote_code: bool = False,
    ) -> None:
        self.ensure_available()
        command = [sys.executable, "-m", "mlx_lm.chat", "--model", str(model.path)]
        self._add_generation_options(command, settings, chat=True)
        if trust_remote_code:
            command.append("--trust-remote-code")
        self._run(command)

    def serve(
        self,
        model: ModelRecord,
        settings: Dict[str, Any],
        *,
        host: str,
        port: int,
        trust_remote_code: bool = False,
    ) -> None:
        self.ensure_available()
        if not 1 <= port <= 65535:
            raise SafetyError("port must be between 1 and 65535")
        command = [
            sys.executable,
            "-m",
            "mlx_lm.server",
            "--model",
            str(model.path),
            "--host",
            host,
            "--port",
            str(port),
        ]
        self._add_generation_options(command, settings, server=True)
        if trust_remote_code:
            command.append("--trust-remote-code")
        self._run(command)

    def create_prompt_cache(
        self,
        model: ModelRecord,
        name: str,
        prompt: str,
        *,
        max_kv_size: Optional[int] = None,
        trust_remote_code: bool = False,
        capture: bool = False,
    ) -> Path:
        self.ensure_available()
        self._validate_cache_name(name)
        self.paths.prompt_cache.mkdir(parents=True, exist_ok=True)
        destination = self.paths.prompt_cache / f"{name}.safetensors"
        temp = self.paths.prompt_cache / f"{name}.tmp"
        with FileLock(self.paths.locks / f"prompt-{name}.lock"):
            command = [
                sys.executable,
                "-m",
                "mlx_lm.cache_prompt",
                "--model",
                str(model.path),
                "--prompt",
                prompt,
                "--prompt-cache-file",
                str(temp),
            ]
            if max_kv_size is not None:
                command.extend(["--max-kv-size", str(max_kv_size)])
            if trust_remote_code:
                command.append("--trust-remote-code")
            try:
                self._run(command, capture=capture)
                os.replace(temp, destination)
            finally:
                temp.unlink(missing_ok=True)
        return destination

    def list_prompt_caches(self) -> list[Dict[str, Any]]:
        if not self.paths.prompt_cache.exists():
            return []
        return [
            {"name": path.stem, "path": str(path), "size_bytes": path.stat().st_size}
            for path in sorted(self.paths.prompt_cache.glob("*.safetensors"))
        ]

    def remove_prompt_cache(self, name: str) -> Path:
        path = self.prompt_cache_path(name)
        if not path.is_file():
            raise SafetyError(f"prompt cache '{name}' does not exist")
        with FileLock(self.paths.locks / f"prompt-{name}.lock"):
            path.unlink()
        return path

    def prompt_cache_path(self, name: str) -> Path:
        self._validate_cache_name(name)
        return self.paths.prompt_cache / f"{name}.safetensors"

    @staticmethod
    def _validate_cache_name(name: str) -> None:
        if not name or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for character in name
        ):
            raise SafetyError(
                "prompt cache name may contain only letters, numbers, '.', '_', and '-'"
            )

    @staticmethod
    def command_environment(model: ModelRecord, profile: Optional[str]) -> Dict[str, str]:
        return {
            "MLXVM_MODEL": model.reference,
            "MLXVM_REVISION": model.revision,
            "MLXVM_PROFILE": profile or "",
            "MLXVM_MODEL_PATH": str(model.path),
        }

    def execute(self, model: ModelRecord, command: Sequence[str], *, profile: Optional[str]) -> int:
        if not command:
            raise SafetyError("exec requires a command after '--'")
        environment = os.environ.copy()
        environment.update(self.command_environment(model, profile))
        try:
            return subprocess.run(list(command), env=environment, check=False).returncode
        except FileNotFoundError as exc:
            raise RuntimeFailure(f"command not found: {command[0]}") from exc

    def _run(self, command: list[str], *, capture: bool = False) -> Optional[str]:
        self.logger.debug("starting runtime subprocess: %s", command[:4])
        try:
            completed = subprocess.run(
                command,
                check=False,
                text=capture,
                stdout=subprocess.PIPE if capture else None,
                stderr=subprocess.PIPE if capture else None,
            )
        except KeyboardInterrupt:
            self.logger.info("runtime operation interrupted")
            raise
        if completed.returncode != 0:
            stderr = completed.stderr.strip()[-4000:] if capture and completed.stderr else None
            raise RuntimeFailure(
                f"MLX-LM exited with status {completed.returncode}",
                details={"stderr": stderr} if stderr else {},
                hint="run with --verbose and inspect mlxvm.log for context",
            )
        return completed.stdout if capture else None

    @staticmethod
    def _add_generation_options(
        command: list[str], settings: Dict[str, Any], *, chat: bool = False, server: bool = False
    ) -> None:
        mapping = {
            "temperature": "--temp",
            "top_p": "--top-p",
            "max_tokens": "--max-tokens",
            "system_prompt": "--system-prompt",
        }
        if not chat:
            mapping.update({"min_p": "--min-p", "top_k": "--top-k"})
        if not server:
            mapping.update({"max_kv_size": "--max-kv-size", "seed": "--seed"})
        for key, option in mapping.items():
            if settings.get(key) is not None:
                command.extend([option, str(settings[key])])
