# mlxvm

[![CI](https://github.com/Botfather/mlxvm/actions/workflows/ci.yml/badge.svg)](https://github.com/Botfather/mlxvm/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/mlxvm.svg)](https://pypi.org/project/mlxvm/)
[![Python](https://img.shields.io/pypi/pyversions/mlxvm.svg)](https://pypi.org/project/mlxvm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

`mlxvm` is an [nvm](https://github.com/nvm-sh/nvm)-style manager for local
text-generation models powered by [MLX-LM](https://github.com/ml-explore/mlx-lm).
It pins Hugging Face revisions, keeps a private cache, supports shell / project /
global model selection, and delegates inference to MLX-LM without hiding its
output.

If you have used `nvm` to switch Node versions per project, `mlxvm` will feel
familiar: install a model, alias it, `use` it in a shell, or pin it to a
directory with a `.mlxvmrc` file.

```sh
mlxvm install mlx-community/Qwen3-1.7B-4bit --alias default
mlxvm use default
mlxvm chat
```

## Contents

- [Why mlxvm](#why-mlxvm)
- [Requirements](#requirements)
- [Install](#install)
- [Quick start](#quick-start)
- [Commands](#commands)
- [Model resolution](#model-resolution)
- [Configuration](#configuration)
- [Prompt caches and serving](#prompt-caches-and-serving)
- [Automation and JSON output](#automation-and-json-output)
- [Storage layout](#storage-layout)
- [How it works](#how-it-works)
- [Development](#development)
- [Contributing](#contributing)
- [Security](#security)
- [License](#license)

## Why mlxvm

- **Reproducible.** Branches and tags are resolved to immutable Hugging Face
  commit hashes at install time, so a project pinned to a revision keeps working
  even if the repository's `main` branch changes.
- **Private and safe.** Models live in `mlxvm`'s own cache, so uninstalling
  never deletes files owned by other Hugging Face applications, and a registered
  local model directory is never removed.
- **Per-shell, per-project, and global selection**, mirroring nvm's ergonomics.
- **Scriptable.** Every non-interactive command supports `--json` with a stable
  schema and stable error codes, plus `--no-interactive`, `--yes`, and
  `--offline` for automation.
- **Thin by design.** Downloads use Hugging Face's resumable cache; inference,
  chat, quantization, and serving are delegated to MLX-LM rather than
  reimplemented.

## Requirements

- Apple Silicon Mac (M-series)
- macOS 15 or newer recommended
- Python 3.9 or newer
- Sufficient unified memory and disk space for the selected model

MLX is Apple-Silicon-only; `mlxvm` validates this with `mlxvm doctor`.

## Install

Use an isolated application environment:

```sh
pipx install mlxvm
# or
uv tool install mlxvm
```

Enable current-shell model selection by evaluating the integration snippet and
adding the same line to your shell startup file:

```sh
eval "$(mlxvm shell-init zsh)"      # bash and fish are also supported
```

The shell function intercepts `mlxvm use` / `mlxvm deactivate` so they can
export variables into the current shell (a child process cannot mutate its
parent's environment); every other command is delegated unchanged. Tab
completions are available with `mlxvm completions zsh` (or `bash` / `fish`).

## Quick start

For a guided first run, just launch `mlxvm` with no arguments. It explains
unified memory, recommends a model that fits your Mac, offers to configure your
shell, installs the selection, and starts your first chat automatically.

```sh
mlxvm
```

The same workflow, one command at a time:

```sh
mlxvm doctor                                                   # check compatibility
mlxvm search Qwen                                              # browse compatible models
mlxvm install mlx-community/Qwen3-0.6B-4bit --alias default    # download + register
mlxvm use default                                              # select for this shell
mlxvm run "Explain unified memory in one paragraph."          # one-shot generation
mlxvm chat                                                     # interactive chat
```

Pin a revision, register a local model, or produce a private quantized
conversion:

```sh
mlxvm install org/model@40-character-commit-sha
mlxvm install /absolute/path/to/model --alias local
mlxvm install org/model --quantize 4 --alias model-q4
```

## Commands

| Command | Purpose |
| --- | --- |
| `mlxvm doctor` | Validate architecture, macOS, Python, MLX, memory, disk, and Hugging Face auth. |
| `mlxvm search [query]` | Search compatible Hugging Face models interactively. |
| `mlxvm ls-remote [query]` | Script-friendly remote search. |
| `mlxvm install <spec>` | Download and register a model (`repo[@revision]` or a local path). |
| `mlxvm install <spec> --quantize 4` | Convert and quantize through MLX-LM (bits: 2, 3, 4, 6, 8). |
| `mlxvm ls` | List installed models, sizes, aliases, and the active selection. |
| `mlxvm current` | Show the effective model and why it was selected. |
| `mlxvm use <model-or-alias>` | Select a model for the current shell. |
| `mlxvm deactivate` | Clear the current-shell selection. |
| `mlxvm alias <name> <target>` | Create or replace an alias such as `default`, `coding`, or `small`. |
| `mlxvm unalias <name>` | Remove an alias. |
| `mlxvm uninstall <model>` | Unregister and safely remove a managed model. |
| `mlxvm run [prompt]` | Stream one-shot generation (reads stdin when the prompt is `-`). |
| `mlxvm chat` | Start a stateful chat session. |
| `mlxvm serve` | Start MLX-LM's development HTTP server. |
| `mlxvm exec <model> -- <command>` | Run a command with the model exposed through environment variables. |
| `mlxvm cache prune` | Remove unreferenced revisions, orphaned conversions, and partials. |
| `mlxvm cache create/ls/remove` | Manage reusable prompt caches. |
| `mlxvm shell-init <shell>` | Print shell integration for `bash`, `zsh`, or `fish`. |
| `mlxvm completions <shell>` | Print shell completions. |
| `mlxvm upgrade [--check]` | Check for or install a newer `mlxvm` release. |

`mlxvm exec MODEL -- COMMAND` exposes `MLXVM_MODEL`, `MLXVM_REVISION`,
`MLXVM_PROFILE`, and `MLXVM_MODEL_PATH` to the child process.

Run `mlxvm <command> --help` for the full set of options on any command.

## Model resolution

Commands resolve a model using a deterministic precedence order:

1. Explicit `--model`
2. The current shell's `MLXVM_MODEL`
3. The nearest `.mlxvmrc`, found by walking toward the filesystem root
4. The global `default` alias
5. Otherwise, a friendly error suggesting `mlxvm install` or `mlxvm use`

Example `.mlxvmrc`:

```toml
model = "mlx-community/Qwen3-0.6B-4bit"
revision = "immutable-commit-hash"

[generation]
temperature = 0.7
max_tokens = 1024
max_kv_size = 4096
```

## Configuration

Global settings live at `$MLXVM_HOME/config.toml` (the platform data directory
is used when `MLXVM_HOME` is unset):

```toml
offline = false
trust_remote_code = false

[generation]
temperature = 0.2
max_tokens = 512

[profiles.creative]
temperature = 0.9
top_p = 0.95
```

Select a profile with `mlxvm use default --profile creative`, or pass
`--profile creative` to any runtime command. Generation settings layer in this
order, each overriding the previous: global `[generation]` → selected profile →
project `.mlxvmrc` → explicit command-line flags (`--temperature`,
`--max-tokens`, …).

Supported generation keys: `temperature`, `top_p`, `min_p`, `top_k`,
`max_tokens`, `max_kv_size`, `system_prompt`, and `seed`.

## Prompt caches and serving

```sh
mlxvm cache create context "A long reusable context"
mlxvm run --prompt-cache context "Summarize the context"
mlxvm cache ls
mlxvm cache remove context
mlxvm serve --host 127.0.0.1 --port 8080
```

MLX-LM's HTTP server is development-oriented and is deliberately bound to
localhost by default. `mlxvm` requires explicit confirmation before binding it
to a non-local address.

## Automation and JSON output

All finite commands support `--json`; envelopes use `schema_version: 1` and a
stable set of error codes:

```json
{"schema_version":1,"command":"ls","ok":true,"data":{"models":[]},"error":null}
```

Use `--no-interactive` to prohibit prompts, `--yes` to accept a destructive
confirmation up front, and `--offline` to prohibit all Hub access. Error codes
include `configuration_error`, `model_not_found`, `dependency_error`,
`network_error`, `lock_timeout`, `safety_error`, and `runtime_failure`, each
with a stable process exit status.

Downloads use Hugging Face's resumable cache and a model is registered only
after a complete snapshot or conversion. Mutations are guarded by process
locks, converted output is staged in a temporary directory before an atomic
rename, and uninstall never deletes a registered local directory. Rotating logs
are kept under `$MLXVM_HOME/logs/mlxvm.log`; `--verbose` mirrors debug context
to stderr.

## Storage layout

`mlxvm` keeps everything under a single private home directory, chosen via
[platformdirs](https://github.com/tox-dev/platformdirs) and overridable with the
`MLXVM_HOME` environment variable:

```text
$MLXVM_HOME/
├── config.toml          # global settings and profiles
├── registry.sqlite      # installed-model metadata and aliases (WAL mode)
├── cache/huggingface/   # private Hugging Face snapshot cache
├── models/converted/    # quantized / converted models
├── prompt-cache/        # reusable prompt caches
├── locks/               # advisory process locks
└── logs/                # rotating logs
```

## How it works

The package is organized into small, independently testable modules; core
services stay free of terminal prompts so they remain scriptable.

```text
src/mlxvm/
├── cli/           # argument parsing, dispatch, and interactive workflow
├── config/        # paths, global settings, and .mlxvmrc parsing
├── resolver/      # model / alias / revision resolution precedence
├── registry/      # SQLite-backed installed-model metadata and aliases
├── hub/           # Hugging Face search, sizing, download, and pruning
├── lifecycle/     # install, convert, alias, uninstall, and prune orchestration
├── runtime/       # MLX-LM run, chat, serve, and prompt caching (isolated worker)
├── shell/         # bash / zsh / fish integration and completions
├── diagnostics/   # hardware, memory, disk, and compatibility checks
└── locks/         # advisory file locks for concurrent operations
```

MLX initializes Metal on import, so generation runs in an isolated worker
subprocess (`mlxvm.runtime.worker`); this keeps the manager and its registry
recoverable even when Metal initialization fails.

## Development

```sh
git clone https://github.com/Botfather/mlxvm
cd mlxvm
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'

.venv/bin/ruff check src tests        # lint
.venv/bin/ruff format src tests       # format
.venv/bin/pytest                      # tests (mocked runtime; no model download)
```

The unit suite mocks the MLX/Hugging Face runtime and needs no network access or
model downloads. An optional Apple Silicon smoke test that performs a real
install-and-generate cycle lives in `.github/workflows/smoke.yml` and runs on
demand.

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for the
workflow, coding conventions, and how to run the checks. Please open an issue to
discuss substantial changes before sending a pull request.

## Security

`mlxvm` defaults to safe behavior: `trust_remote_code` is off unless explicitly
enabled, Hugging Face tokens are never stored in project configuration, and the
development server binds to localhost unless you confirm otherwise. If you
discover a security issue, please report it privately to the maintainer rather
than opening a public issue.

## License

[MIT](LICENSE) © Tushar Mohan

See [CHANGELOG.md](CHANGELOG.md) for release notes.
