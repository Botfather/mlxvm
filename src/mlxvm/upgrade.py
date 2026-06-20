from __future__ import annotations

import json
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict

from packaging.version import Version

from mlxvm import __version__
from mlxvm.errors import NetworkError, RuntimeFailure


@dataclass(frozen=True)
class UpgradeInfo:
    current: str
    latest: str
    update_available: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current": self.current,
            "latest": self.latest,
            "update_available": self.update_available,
        }


def check_upgrade() -> UpgradeInfo:
    try:
        with urllib.request.urlopen("https://pypi.org/pypi/mlxvm/json", timeout=10) as response:
            payload = json.load(response)
        latest = str(payload["info"]["version"])
    except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
        raise NetworkError(f"cannot check PyPI for updates: {exc}") from exc
    return UpgradeInfo(__version__, latest, Version(latest) > Version(__version__))


def install_upgrade() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "mlxvm"], check=False
    )
    if completed.returncode != 0:
        raise RuntimeFailure(
            f"package upgrade failed with status {completed.returncode}",
            hint="if installed with pipx or uv, use that tool's upgrade command",
        )
