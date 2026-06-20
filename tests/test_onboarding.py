from pathlib import Path

from mlxvm.onboarding import (
    GIB,
    detect_shell,
    install_shell_integration,
    is_shell_integrated,
    memory_guidance,
    recommendations_for_memory,
    shell_config_path,
)


def test_recommendations_scale_conservatively_with_memory() -> None:
    assert recommendations_for_memory(8 * GIB)[0].repo_id.endswith("Qwen3-1.7B-4bit")
    assert recommendations_for_memory(16 * GIB)[0].repo_id.endswith("Qwen3-4B-4bit")
    assert recommendations_for_memory(32 * GIB)[0].repo_id == "Qwen/Qwen3-8B-MLX-4bit"


def test_memory_guidance_uses_plain_language() -> None:
    guidance = memory_guidance(8 * GIB)
    assert "8 GB" in guidance
    assert "shared by macOS" in guidance
    assert "apps" in guidance


def test_shell_detection_and_paths(tmp_path: Path) -> None:
    assert detect_shell({"SHELL": "/bin/zsh"}) == "zsh"
    assert detect_shell({"SHELL": "/bin/unknown"}) is None
    assert shell_config_path("zsh", tmp_path) == tmp_path / ".zshrc"
    assert shell_config_path("bash", tmp_path) == tmp_path / ".bash_profile"


def test_shell_setup_is_atomic_and_idempotent(tmp_path: Path) -> None:
    config = tmp_path / ".zshrc"
    config.write_text("export EXISTING=value\n")
    first = install_shell_integration("zsh", tmp_path)
    assert first.changed is True
    assert "export EXISTING=value" in config.read_text()
    assert 'eval "$(mlxvm shell-init zsh)"' in config.read_text()
    assert is_shell_integrated("zsh", tmp_path)

    second = install_shell_integration("zsh", tmp_path)
    assert second.changed is False
    assert config.read_text().count("mlxvm initialize") == 2


def test_fish_setup_uses_source(tmp_path: Path) -> None:
    install_shell_integration("fish", tmp_path)
    config = tmp_path / ".config" / "fish" / "config.fish"
    assert "mlxvm shell-init fish | source" in config.read_text()
