from pathlib import Path
from types import SimpleNamespace

import pytest

from mlxvm.config import AppPaths
from mlxvm.hub import DownloadPlan, HubClient
from mlxvm.lifecycle import ModelManager
from mlxvm.registry import Registry


def _model_dir(path: Path) -> Path:
    path.mkdir()
    (path / "config.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"weights")
    return path


def test_local_model_lifecycle_does_not_delete_user_files(tmp_path: Path) -> None:
    paths = AppPaths(tmp_path / "home")
    registry = Registry(paths.registry)
    manager = ModelManager(paths, registry, HubClient(paths.hub_cache))
    local = _model_dir(tmp_path / "my-model")

    result = manager.install(str(local), alias="default")
    assert result.model.source == "local"
    assert registry.resolve("default").id == result.model.id

    removed = manager.uninstall("default")
    assert removed.id == result.model.id
    assert local.is_dir()
    assert registry.list_models() == []


def test_repeated_local_install_is_idempotent(tmp_path: Path) -> None:
    paths = AppPaths(tmp_path / "home")
    registry = Registry(paths.registry)
    manager = ModelManager(paths, registry, HubClient(paths.hub_cache))
    local = _model_dir(tmp_path / "my-model")
    first = manager.install(str(local)).model
    second = manager.install(str(local)).model
    assert first.id == second.id
    assert len(registry.list_models()) == 1


def test_prune_removes_only_unregistered_converted_directories(tmp_path: Path) -> None:
    paths = AppPaths(tmp_path / "home")
    registry = Registry(paths.registry)
    manager = ModelManager(paths, registry, HubClient(paths.hub_cache))
    paths.converted_models.mkdir(parents=True)
    orphan = paths.converted_models / "orphan"
    _model_dir(orphan)
    result = manager.prune()
    assert result["converted_models"] == 1
    assert not orphan.exists()


def test_installed_model_short_circuits_remote_lookup(tmp_path: Path) -> None:
    paths = AppPaths(tmp_path / "home")
    registry = Registry(paths.registry)
    local_snapshot = _model_dir(tmp_path / "snapshot")
    installed = registry.add_model(
        "org/model", "a" * 40, local_snapshot, source="hub", size_bytes=7
    )

    class OfflineGuard:
        def resolve_revision(self, *args, **kwargs):
            raise AssertionError("remote lookup should not occur")

    manager = ModelManager(paths, registry, OfflineGuard())
    result = manager.install("org/model")
    assert result.already_installed is True
    assert result.model.id == installed.id
    assert result.plan.download_bytes == 0


def test_interrupted_download_is_not_registered_and_releases_lock(tmp_path: Path) -> None:
    paths = AppPaths(tmp_path / "home")
    registry = Registry(paths.registry)

    class InterruptedHub:
        def resolve_revision(self, repo_id, revision):
            return "a" * 40, 10

        def plan(self, repo_id, revision, known_size=0):
            return DownloadPlan(repo_id, revision, known_size, known_size, 1)

        def download(self, repo_id, revision):
            raise KeyboardInterrupt

    manager = ModelManager(paths, registry, InterruptedHub())
    with pytest.raises(KeyboardInterrupt):
        manager.install("org/model")
    assert registry.list_models() == []

    # A failed operation must not leave a lock that blocks recovery.
    from mlxvm.locks import FileLock

    with FileLock(paths.locks / "models.lock", timeout=0.1):
        pass


def test_uninstall_keeps_other_hub_revision_referenced(tmp_path: Path) -> None:
    paths = AppPaths(tmp_path / "home")
    registry = Registry(paths.registry)
    first_path = _model_dir(tmp_path / "first")
    second_path = _model_dir(tmp_path / "second")
    first = registry.add_model("org/model", "a" * 40, first_path, source="hub")
    registry.add_model("org/model", "b" * 40, second_path, source="hub")

    class RecordingHub:
        referenced = None

        def prune(self, referenced):
            self.referenced = referenced
            return {}

    hub = RecordingHub()
    manager = ModelManager(paths, registry, hub)
    manager.uninstall(first.reference)
    assert hub.referenced == {"b" * 40}


def test_quantized_conversion_is_staged_and_registered(monkeypatch, tmp_path: Path) -> None:
    paths = AppPaths(tmp_path / "home")
    registry = Registry(paths.registry)
    source = _model_dir(tmp_path / "source")
    manager = ModelManager(paths, registry, HubClient(paths.hub_cache))

    def fake_run(command, **kwargs):
        output = Path(command[command.index("--mlx-path") + 1])
        _model_dir(output)
        return SimpleNamespace(returncode=0, stdout="converted", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    model = manager._convert(
        "org/model",
        "a" * 40,
        source,
        4,
        trust_remote_code=False,
        capture=True,
    )
    assert model.variant == "q4"
    assert model.source == "converted"
    assert model.path.parent == paths.converted_models
    assert model.path.is_dir()
