from pathlib import Path
from types import SimpleNamespace

import pytest

from mlxvm.config import AppPaths
from mlxvm.errors import SafetyError
from mlxvm.registry import ModelRecord
from mlxvm.runtime import RuntimeRunner


def _model(tmp_path: Path) -> ModelRecord:
    return ModelRecord(
        id=1,
        repo_id="org/model",
        revision="abc",
        path=tmp_path / "model",
        source="local",
        size_bytes=1,
        installed_at="now",
        metadata={},
    )


def test_generation_runs_in_isolated_worker(monkeypatch, tmp_path: Path) -> None:
    runner = RuntimeRunner(AppPaths(tmp_path / "home"))
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        return SimpleNamespace(returncode=0, stdout="answer\n", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    output = runner.generate(_model(tmp_path), "hello", {"max_tokens": 12}, capture=True)
    assert output == "answer\n"
    assert "mlxvm.runtime.worker" in seen["command"]
    assert seen["command"][seen["command"].index("--prompt") + 1] == "hello"


def test_prompt_cache_names_cannot_escape_private_directory(tmp_path: Path) -> None:
    runner = RuntimeRunner(AppPaths(tmp_path / "home"))
    with pytest.raises(SafetyError):
        runner.prompt_cache_path("../outside")
