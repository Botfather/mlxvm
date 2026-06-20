from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _default_home() -> Path:
    try:
        from platformdirs import user_data_path

        return user_data_path("mlxvm", appauthor=False)
    except ImportError:
        if os.name == "posix" and os.uname().sysname == "Darwin":
            return Path.home() / "Library" / "Application Support" / "mlxvm"
        return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "mlxvm"


@dataclass(frozen=True)
class AppPaths:
    home: Path

    @classmethod
    def discover(cls) -> "AppPaths":
        override = os.environ.get("MLXVM_HOME")
        return cls(Path(override).expanduser() if override else _default_home())

    @property
    def config(self) -> Path:
        return self.home / "config.toml"

    @property
    def registry(self) -> Path:
        return self.home / "registry.sqlite"

    @property
    def hub_cache(self) -> Path:
        return self.home / "cache" / "huggingface"

    @property
    def converted_models(self) -> Path:
        return self.home / "models" / "converted"

    @property
    def prompt_cache(self) -> Path:
        return self.home / "prompt-cache"

    @property
    def locks(self) -> Path:
        return self.home / "locks"

    @property
    def logs(self) -> Path:
        return self.home / "logs"

    def ensure(self) -> None:
        for path in (
            self.home,
            self.hub_cache,
            self.converted_models,
            self.prompt_cache,
            self.locks,
            self.logs,
        ):
            path.mkdir(parents=True, exist_ok=True)
