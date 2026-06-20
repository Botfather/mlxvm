from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

GIB = 1024**3
START_MARKER = "# >>> mlxvm initialize >>>"
END_MARKER = "# <<< mlxvm initialize <<<"


@dataclass(frozen=True)
class RecommendedModel:
    repo_id: str
    name: str
    description: str
    download_bytes: int


@dataclass(frozen=True)
class ShellSetupResult:
    shell: str
    path: Path
    changed: bool


SMALL = RecommendedModel(
    "mlx-community/Qwen3-0.6B-4bit",
    "Qwen3 0.6B",
    "Very fast and compact; best for trying mlxvm and simple questions.",
    351_386_061,
)
BALANCED = RecommendedModel(
    "mlx-community/Qwen3-1.7B-4bit",
    "Qwen3 1.7B",
    "A comfortable everyday model with a good balance of speed and quality.",
    984_015_687,
)
CAPABLE = RecommendedModel(
    "mlx-community/Qwen3-4B-4bit",
    "Qwen3 4B",
    "Better for writing, coding, and reasoning; somewhat slower.",
    2_278_972_183,
)
LARGE = RecommendedModel(
    "Qwen/Qwen3-8B-MLX-4bit",
    "Qwen3 8B",
    "The strongest starter option here; higher quality with slower responses.",
    4_367_857_874,
)


def recommendations_for_memory(memory_bytes: Optional[int]) -> list[RecommendedModel]:
    """Return the recommended model first, followed by two useful alternatives."""
    if memory_bytes is None or memory_bytes < 12 * GIB:
        return [BALANCED, SMALL, CAPABLE]
    if memory_bytes < 24 * GIB:
        return [CAPABLE, BALANCED, LARGE]
    return [LARGE, CAPABLE, BALANCED]


def memory_guidance(memory_bytes: Optional[int]) -> str:
    if memory_bytes is None:
        return (
            "I couldn't measure your Mac's memory, so I've chosen a small, safe default. "
            "You can switch models later without losing anything."
        )
    gib = memory_bytes / GIB
    if memory_bytes < 12 * GIB:
        advice = "A compact model will stay responsive while leaving room for macOS and your apps."
    elif memory_bytes < 24 * GIB:
        advice = "A mid-sized model should run comfortably while leaving useful working room."
    else:
        advice = "You have room for a larger local model, though larger models respond more slowly."
    return (
        f"Your Mac has about {gib:.0f} GB of unified memory. This memory is shared by macOS, "
        f"your apps, the model, and its conversation history. {advice}"
    )


def detect_shell(environment: Optional[Mapping[str, str]] = None) -> Optional[str]:
    shell_path = (environment or os.environ).get("SHELL", "")
    shell = Path(shell_path).name
    return shell if shell in {"bash", "zsh", "fish"} else None


def shell_config_path(shell: str, home: Optional[Path] = None) -> Path:
    home = home or Path.home()
    if shell == "zsh":
        return home / ".zshrc"
    if shell == "bash":
        return home / ".bash_profile"
    if shell == "fish":
        return home / ".config" / "fish" / "config.fish"
    raise ValueError(f"unsupported shell: {shell}")


def _shell_block(shell: str) -> str:
    command = (
        "mlxvm shell-init fish | source"
        if shell == "fish"
        else f'eval "$(mlxvm shell-init {shell})"'
    )
    return f"{START_MARKER}\n{command}\n{END_MARKER}"


def is_shell_integrated(shell: str, home: Optional[Path] = None) -> bool:
    path = shell_config_path(shell, home)
    if not path.is_file():
        return False
    try:
        return START_MARKER in path.read_text(encoding="utf-8", errors="surrogateescape")
    except OSError:
        return False


def install_shell_integration(shell: str, home: Optional[Path] = None) -> ShellSetupResult:
    configured_path = shell_config_path(shell, home)
    path = configured_path.resolve() if configured_path.is_symlink() else configured_path
    path.parent.mkdir(parents=True, exist_ok=True)
    original = path.read_text(encoding="utf-8", errors="surrogateescape") if path.exists() else ""
    block = _shell_block(shell)
    pattern = re.compile(rf"{re.escape(START_MARKER)}.*?{re.escape(END_MARKER)}", re.DOTALL)
    if pattern.search(original):
        updated = pattern.sub(block, original)
    else:
        separator = "" if not original else ("" if original.endswith("\n") else "\n")
        updated = f"{original}{separator}{block}\n"
    if updated == original:
        return ShellSetupResult(shell, configured_path, False)

    temporary = path.with_name(f".{path.name}.mlxvm-{os.getpid()}.tmp")
    try:
        temporary.write_text(updated, encoding="utf-8", errors="surrogateescape")
        if path.exists():
            temporary.chmod(path.stat().st_mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return ShellSetupResult(shell, configured_path, True)
