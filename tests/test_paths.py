from pathlib import Path

from mlxvm.config import AppPaths


def test_home_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MLXVM_HOME", str(tmp_path))
    paths = AppPaths.discover()
    assert paths.home == tmp_path
    assert paths.registry == tmp_path / "registry.sqlite"


def test_ensure_creates_private_layout(tmp_path: Path) -> None:
    paths = AppPaths(tmp_path)
    paths.ensure()
    assert paths.hub_cache.is_dir()
    assert paths.converted_models.is_dir()
    assert paths.locks.is_dir()
