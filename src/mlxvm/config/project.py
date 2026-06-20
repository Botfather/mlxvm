from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from mlxvm.errors import ConfigurationError


class ConfigError(ConfigurationError, ValueError):
    pass


@dataclass(frozen=True)
class ProjectConfig:
    path: Path
    model: str
    revision: Optional[str] = None
    generation: Dict[str, Any] = field(default_factory=dict)


def find_project_config(start: Optional[Path] = None) -> Optional[Path]:
    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for directory in (current, *current.parents):
        candidate = directory / ".mlxvmrc"
        if candidate.is_file():
            return candidate
    return None


def _toml_load(path: Path) -> Dict[str, Any]:
    try:
        import tomllib  # type: ignore[import-not-found]
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef,import-not-found]
        except ImportError as exc:
            raise ConfigError("TOML support is unavailable; install mlxvm normally") from exc
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc


def load_project_config(path: Path) -> ProjectConfig:
    data = _toml_load(path)
    model = data.get("model")
    revision = data.get("revision")
    generation = data.get("generation", {})
    if not isinstance(model, str) or not model.strip():
        raise ConfigError(f"{path} must define a non-empty string 'model'")
    if revision is not None and not isinstance(revision, str):
        raise ConfigError(f"{path} 'revision' must be a string")
    if not isinstance(generation, dict):
        raise ConfigError(f"{path} 'generation' must be a table")
    try:
        json.dumps(generation)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{path} 'generation' values must be JSON-compatible") from exc
    return ProjectConfig(
        path=path,
        model=model.strip(),
        revision=revision.strip() if revision else None,
        generation=generation,
    )
