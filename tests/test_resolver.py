from pathlib import Path

from mlxvm.registry import Registry
from mlxvm.resolver import ModelResolver


def _registry(tmp_path: Path) -> Registry:
    registry = Registry(tmp_path / "registry.sqlite")
    model = registry.add_model("org/model", "commit", tmp_path / "model")
    registry.set_alias("default", model.id)
    return registry


def test_precedence_explicit_then_shell_then_default(monkeypatch, tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    resolver = ModelResolver(registry)
    monkeypatch.setenv("MLXVM_MODEL", "default")

    explicit = resolver.resolve(explicit="org/model@commit", start=tmp_path)
    assert explicit.source == "argument"
    assert explicit.model is not None

    shell = resolver.resolve(start=tmp_path)
    assert shell.source == "shell"

    monkeypatch.delenv("MLXVM_MODEL")
    fallback = resolver.resolve(start=tmp_path)
    assert fallback.source == "default alias"


def test_uninstalled_selection_is_preserved(monkeypatch, tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    monkeypatch.setenv("MLXVM_MODEL", "org/missing")
    result = ModelResolver(registry).resolve(start=tmp_path)
    assert result.selected
    assert result.model is None
    assert "not installed" in result.error


def test_nearest_project_config_wins(monkeypatch, tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    nested = tmp_path / "project" / "packages" / "app"
    nested.mkdir(parents=True)
    (tmp_path / "project" / ".mlxvmrc").write_text(
        'model = "org/model"\nrevision = "commit"\n\n'
        "[generation]\ntemperature = 0.2\nmax_tokens = 128\n"
    )
    monkeypatch.delenv("MLXVM_MODEL", raising=False)

    result = ModelResolver(registry).resolve(start=nested)
    assert result.source == "project"
    assert result.source_path == tmp_path / "project" / ".mlxvmrc"
    assert result.model is not None
    assert result.generation == {"temperature": 0.2, "max_tokens": 128}


def test_invalid_project_config_is_reported(monkeypatch, tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    (project / ".mlxvmrc").write_text("revision = 123\n")
    monkeypatch.delenv("MLXVM_MODEL", raising=False)

    result = ModelResolver(registry).resolve(start=project)
    assert result.source == "project"
    assert result.model is None
    assert "must define" in result.error
