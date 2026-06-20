import json
from pathlib import Path

from mlxvm.cli import main


def test_empty_list_json(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv("MLXVM_HOME", str(tmp_path))
    assert main(["ls", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1
    assert payload["command"] == "ls"
    assert payload["data"]["models"] == []


def test_current_without_selection_is_failure(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv("MLXVM_HOME", str(tmp_path))
    monkeypatch.delenv("MLXVM_MODEL", raising=False)
    assert main(["current", "--directory", str(tmp_path), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["data"]["selected"] is False


def test_argument_errors_are_structured(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv("MLXVM_HOME", str(tmp_path))
    assert main(["install", "--json"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "configuration_error"
