from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from packaging.version import InvalidVersion, Version

from mlxvm.config import AppPaths


@dataclass(frozen=True)
class Diagnostic:
    name: str
    status: str
    message: str
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "details": self.details,
        }


def _package_version(distribution: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _version_is_supported(value: Optional[str], minimum: str) -> bool:
    if value is None:
        return False
    try:
        return Version(value) >= Version(minimum)
    except InvalidVersion:
        return False


def _mlx_runtime_check() -> Diagnostic:
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import mlx.core as mx; print(mx.default_device())",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Diagnostic("mlx_runtime", "fail", f"MLX probe failed: {exc}", {})
    if completed.returncode != 0:
        return Diagnostic(
            "mlx_runtime",
            "fail",
            f"Metal initialization failed (status {completed.returncode})",
            {"returncode": completed.returncode},
        )
    return Diagnostic(
        "mlx_runtime",
        "pass",
        completed.stdout.strip() or "Metal initialized",
        {},
    )


def _memory_bytes() -> Optional[int]:
    if platform.system() == "Darwin":
        try:
            output = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True, stderr=subprocess.DEVNULL
            )
            return int(output.strip())
        except (OSError, ValueError, subprocess.CalledProcessError):
            pass
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return int(page_size * pages)
    except (AttributeError, OSError, ValueError):
        return None


def _hf_auth_status() -> Diagnostic:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return Diagnostic("huggingface_auth", "pass", "token found in environment", {})
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    token_path = hf_home / "token"
    if token_path.is_file():
        return Diagnostic("huggingface_auth", "pass", "cached token found", {})
    return Diagnostic(
        "huggingface_auth",
        "warn",
        "not authenticated; public models remain available",
        {},
    )


def run_diagnostics(paths: AppPaths) -> List[Diagnostic]:
    system = platform.system()
    machine = platform.machine().lower()
    diagnostics = [
        Diagnostic(
            "platform",
            "pass" if system == "Darwin" else "fail",
            f"{system} {platform.release()}",
            {"system": system, "release": platform.release()},
        ),
        Diagnostic(
            "architecture",
            "pass" if machine in {"arm64", "aarch64"} else "fail",
            machine,
            {"machine": machine},
        ),
    ]

    python_ok = sys.version_info >= (3, 9)
    diagnostics.append(
        Diagnostic(
            "python",
            "pass" if python_ok else "fail",
            platform.python_version(),
            {"executable": sys.executable, "minimum": "3.9"},
        )
    )

    package_results: Dict[str, Optional[str]] = {}
    for name, distribution, minimum in (
        ("mlx", "mlx", "0.29"),
        ("mlx_lm", "mlx-lm", "0.29"),
    ):
        package_version = _package_version(distribution)
        package_results[name] = package_version
        supported = _version_is_supported(package_version, minimum)
        diagnostics.append(
            Diagnostic(
                name,
                "pass" if supported else "fail",
                package_version or "not installed",
                {"version": package_version, "minimum": minimum},
            )
        )

    if package_results["mlx"] and system == "Darwin" and machine in {"arm64", "aarch64"}:
        diagnostics.append(_mlx_runtime_check())

    memory = _memory_bytes()
    diagnostics.append(
        Diagnostic(
            "memory",
            "pass" if memory else "warn",
            f"{memory / 1024**3:.1f} GiB unified memory" if memory else "could not detect memory",
            {"bytes": memory},
        )
    )

    disk_anchor = paths.home
    while not disk_anchor.exists() and disk_anchor != disk_anchor.parent:
        disk_anchor = disk_anchor.parent
    try:
        disk = shutil.disk_usage(disk_anchor)
        diagnostics.append(
            Diagnostic(
                "disk",
                "pass" if disk.free >= 10 * 1024**3 else "warn",
                f"{disk.free / 1024**3:.1f} GiB available",
                {"path": str(paths.home), "free_bytes": disk.free, "total_bytes": disk.total},
            )
        )
    except OSError as exc:
        diagnostics.append(Diagnostic("disk", "warn", str(exc), {"path": str(paths.home)}))

    diagnostics.append(_hf_auth_status())
    return diagnostics
