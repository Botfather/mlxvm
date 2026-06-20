import sqlite3
from pathlib import Path

from mlxvm.registry import Registry


def test_register_list_resolve_and_alias(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    model = registry.add_model(
        "mlx-community/example",
        "abc123",
        tmp_path / "model",
        size_bytes=1024,
        metadata={"format": "safetensors"},
    )
    registry.set_alias("default", model.id)

    models = registry.list_models()
    assert len(models) == 1
    assert models[0].aliases == ("default",)
    assert registry.resolve("default").revision == "abc123"
    assert registry.resolve("mlx-community/example@abc123").id == model.id


def test_revision_is_part_of_identity(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    first = registry.add_model("org/model", "one", tmp_path / "one")
    second = registry.add_model("org/model", "two", tmp_path / "two")
    assert first.id != second.id
    assert registry.resolve("org/model").id == second.id


def test_v1_registry_migrates_without_losing_aliases(tmp_path: Path) -> None:
    path = tmp_path / "registry.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE models (
            id INTEGER PRIMARY KEY, repo_id TEXT NOT NULL, revision TEXT NOT NULL,
            path TEXT NOT NULL UNIQUE, source TEXT NOT NULL, size_bytes INTEGER NOT NULL,
            installed_at TEXT NOT NULL, metadata_json TEXT NOT NULL,
            UNIQUE(repo_id, revision)
        );
        CREATE TABLE aliases (
            name TEXT PRIMARY KEY,
            model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE
        );
        INSERT INTO models VALUES (1, 'org/model', 'abc', '/tmp/model', 'hub', 1, 'now', '{}');
        INSERT INTO aliases VALUES ('default', 1);
        PRAGMA user_version = 1;
        """
    )
    connection.close()

    registry = Registry(path)
    model = registry.resolve("default")
    assert model is not None
    assert model.variant == "default"
    assert model.reference == "org/model@abc"
