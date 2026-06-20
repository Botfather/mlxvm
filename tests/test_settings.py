from pathlib import Path

import pytest

from mlxvm.config.project import ConfigError
from mlxvm.config.settings import load_settings


def test_settings_merge_profile_and_project(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        "offline = true\n\n"
        "[generation]\nmax_tokens = 128\ntemperature = 0.1\n\n"
        "[profiles.creative]\ntemperature = 0.8\ntop_p = 0.9\n"
    )
    settings = load_settings(config)
    result = settings.generation_for("creative", {"max_kv_size": 2048})
    assert settings.offline is True
    assert result == {
        "max_tokens": 128,
        "temperature": 0.8,
        "top_p": 0.9,
        "max_kv_size": 2048,
    }


def test_settings_reject_unknown_generation_key(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[generation]\nmagic = 1\n")
    with pytest.raises(ConfigError, match="unsupported keys"):
        load_settings(config)
