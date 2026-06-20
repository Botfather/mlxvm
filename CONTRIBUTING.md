# Contributing to mlxvm

Thanks for your interest in improving `mlxvm`! This document covers how to set
up a development environment, the conventions the project follows, and how to
get a change merged.

## Getting started

`mlxvm` targets Apple Silicon Macs and Python 3.9+.

```sh
git clone https://github.com/Botfather/mlxvm
cd mlxvm
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

## Running the checks

Before opening a pull request, run the same checks CI does:

```sh
.venv/bin/ruff check src tests        # lint
.venv/bin/ruff format --check src tests  # formatting
.venv/bin/pytest                      # tests
```

To auto-fix formatting and lint issues:

```sh
.venv/bin/ruff format src tests
.venv/bin/ruff check --fix src tests
```

The unit suite mocks the MLX and Hugging Face runtime, so it requires no network
access and downloads no models. If you have an Apple Silicon machine and want to
exercise a real install-and-generate cycle, the workflow in
`.github/workflows/smoke.yml` shows the end-to-end commands.

## Conventions

- **Keep core services free of terminal prompts.** Interactive behavior belongs
  in the `cli` layer; everything in `config`, `resolver`, `registry`, `hub`,
  `lifecycle`, `runtime`, `shell`, `diagnostics`, and `locks` should remain
  importable and testable without a TTY.
- **Don't reinvent the wheel.** Download/caching is delegated to
  `huggingface_hub`; inference, chat, quantization, and serving are delegated to
  `mlx-lm`. Prefer extending those integrations over reimplementing them.
- **Fail safely.** Stage writes in temporary paths and use atomic renames, guard
  mutations with the file lock, and never delete user-owned files. Raise the
  typed errors in `mlxvm.errors` so failures map to stable codes and exit
  statuses.
- **Preserve the machine-readable contract.** Output behind `--json` uses
  `schema_version: 1`; if you change a payload shape or add an error code, update
  the schema version and the docs together.
- **Match the surrounding style.** Code is formatted and linted with
  [ruff](https://docs.astral.sh/ruff/) (line length 100). Add tests for new
  behavior and keep them runtime-mocked where possible.

## Pull requests

1. Open an issue first for anything substantial so we can agree on the approach.
2. Keep changes focused; unrelated cleanups are easier to review separately.
3. Make sure lint, format, and tests pass.
4. Add a short note to [CHANGELOG.md](CHANGELOG.md) under an *Unreleased*
   heading describing the user-visible change.

## Reporting bugs and security issues

For ordinary bugs, open a GitHub issue with the command you ran, what you
expected, and what happened (including relevant lines from
`$MLXVM_HOME/logs/mlxvm.log`).

For security-sensitive reports, please contact the maintainer privately rather
than filing a public issue.

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE).
