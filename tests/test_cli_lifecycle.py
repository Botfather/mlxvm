import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from mlxvm.cli import main
from mlxvm.cli.main import _interactive_workflow
from mlxvm.config import AppPaths
from mlxvm.lifecycle import InstallResult
from mlxvm.registry import ModelRecord
from mlxvm.resolver import Resolution


def _model_dir(path: Path) -> Path:
    path.mkdir()
    (path / "config.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"weights")
    return path


def test_cli_local_install_select_and_uninstall(monkeypatch, tmp_path: Path, capsys) -> None:
    home = tmp_path / "home"
    model = _model_dir(tmp_path / "model")
    monkeypatch.setenv("MLXVM_HOME", str(home))
    monkeypatch.delenv("MLXVM_MODEL", raising=False)

    assert main(["install", str(model), "--alias", "default", "--json"]) == 0
    installed = json.loads(capsys.readouterr().out)
    assert installed["data"]["model"]["source"] == "local"

    assert main(["current", "--directory", str(tmp_path), "--json"]) == 0
    current = json.loads(capsys.readouterr().out)
    assert current["data"]["model"]["aliases"] == ["default"]

    assert main(["use", "default", "--json"]) == 0
    selected = json.loads(capsys.readouterr().out)
    assert selected["data"]["environment"]["MLXVM_MODEL"].startswith("local/model-")

    assert main(["uninstall", "default", "--yes", "--json"]) == 0
    json.loads(capsys.readouterr().out)
    assert model.exists()


def test_cli_errors_have_stable_codes(monkeypatch, tmp_path: Path, capsys) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("MLXVM_HOME", str(home))
    assert main(["alias", "coding", "missing", "--json"]) == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "model_not_found"
    assert "model_not_found" in (home / "logs" / "mlxvm.log").read_text()


def test_beginner_flow_installs_recommendation_and_starts_chat(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    paths = AppPaths(tmp_path / "home")
    selected = ModelRecord(
        id=1,
        repo_id="mlx-community/Qwen3-1.7B-4bit",
        revision="abc",
        path=tmp_path / "model",
        source="hub",
        size_bytes=10,
        installed_at="now",
        metadata={},
        aliases=("default",),
    )

    class Registry:
        def list_models(self):
            return []

    class Manager:
        def __init__(self, app_paths):
            self.registry = Registry()
            self.paths = app_paths

    class Resolver:
        def resolve(self):
            return Resolution(None, None, None, None, None, {}, "none")

    class Runtime:
        chatted_with = None

        def chat(self, model, settings, trust_remote_code=False):
            self.chatted_with = model

    runtime = Runtime()
    cli_module = importlib.import_module("mlxvm.cli.main")
    args = SimpleNamespace(no_interactive=False, yes=False, json=False)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(cli_module, "detect_shell", lambda: None)
    monkeypatch.setattr(cli_module, "_memory_bytes", lambda: 8 * 1024**3)
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    monkeypatch.setattr(
        cli_module,
        "_install_with_confirmation",
        lambda *args, **kwargs: InstallResult(selected, None, False, False),
    )

    assert _interactive_workflow(args, Manager(paths), object(), Resolver(), runtime) == 0
    assert runtime.chatted_with is selected
    output = capsys.readouterr().out
    assert "Recommended" in output
    assert "Starting your first chat" in output
