from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from mlxvm.config import AppPaths
from mlxvm.errors import ModelNotFoundError, RuntimeFailure, SafetyError
from mlxvm.hub import DownloadPlan, HubClient, parse_model_spec
from mlxvm.hub.client import directory_size
from mlxvm.locks import FileLock
from mlxvm.logging import get_logger
from mlxvm.registry import ModelRecord, Registry


@dataclass(frozen=True)
class InstallResult:
    model: Optional[ModelRecord]
    plan: Optional[DownloadPlan]
    already_installed: bool
    dry_run: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.to_dict() if self.model else None,
            "plan": self.plan.to_dict() if self.plan else None,
            "already_installed": self.already_installed,
            "dry_run": self.dry_run,
        }


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")[:100] or "model"


def _validate_model_dir(path: Path) -> None:
    if not path.is_dir():
        raise ModelNotFoundError(f"model directory does not exist: {path}")
    if not (path / "config.json").is_file():
        raise SafetyError(f"{path} is not a model directory: config.json is missing")
    weight_patterns = ("*.safetensors", "*.npz", "*.bin")
    if not any(next(path.glob(pattern), None) for pattern in weight_patterns):
        raise SafetyError(f"{path} does not contain recognizable model weights")


class ModelManager:
    def __init__(self, paths: AppPaths, registry: Registry, hub: HubClient) -> None:
        self.paths = paths
        self.registry = registry
        self.hub = hub
        self.logger = get_logger()

    def prepare_install(
        self, spec: str, *, quantize: Optional[int] = None
    ) -> tuple[str, Optional[str], str, Optional[DownloadPlan], Optional[ModelRecord]]:
        parsed, requested_revision = parse_model_spec(spec)
        local = Path(parsed)
        if local.exists():
            return parsed, None, "local", None, self.registry.get_by_path(local)
        variant = f"q{quantize}" if quantize else "default"
        installed = self.registry.resolve(parsed, requested_revision, variant)
        if installed and (requested_revision is None or requested_revision == installed.revision):
            cached_plan = DownloadPlan(
                parsed,
                installed.revision,
                installed.size_bytes,
                0,
                0,
            )
            return parsed, installed.revision, variant, cached_plan, installed
        revision, size = self.hub.resolve_revision(parsed, requested_revision)
        existing = self.registry.resolve(parsed, revision, variant)
        plan = self.hub.plan(parsed, revision, known_size=size)
        return parsed, revision, variant, plan, existing

    def install(
        self,
        spec: str,
        *,
        alias: Optional[str] = None,
        quantize: Optional[int] = None,
        trust_remote_code: bool = False,
        dry_run: bool = False,
        capture_runtime: bool = False,
    ) -> InstallResult:
        if alias:
            self._validate_alias(alias)
        parsed, revision, variant, plan, existing = self.prepare_install(spec, quantize=quantize)
        if existing:
            if alias:
                self.registry.set_alias(alias, existing.id)
            return InstallResult(existing, plan, True, dry_run)
        if dry_run:
            return InstallResult(None, plan, False, True)

        with FileLock(self.paths.locks / "models.lock", timeout=300):
            if revision:
                existing = self.registry.resolve(parsed, revision, variant)
                if existing:
                    if alias:
                        self.registry.set_alias(alias, existing.id)
                    return InstallResult(existing, plan, True, False)
                snapshot = self.hub.download(parsed, revision)
                _validate_model_dir(snapshot)
                if quantize:
                    model = self._convert(
                        parsed,
                        revision,
                        snapshot,
                        quantize,
                        trust_remote_code=trust_remote_code,
                        capture=capture_runtime,
                    )
                else:
                    model = self.registry.add_model(
                        parsed,
                        revision,
                        snapshot,
                        source="hub",
                        size_bytes=directory_size(snapshot),
                        metadata={"trust_remote_code": trust_remote_code},
                    )
            else:
                model = self._register_local(Path(parsed))
            if alias:
                self.registry.set_alias(alias, model.id)
                refreshed = self.registry.get_by_id(model.id)
                if refreshed:
                    model = refreshed
            self.logger.info("installed model %s at %s", model.reference, model.path)
            return InstallResult(model, plan, False, False)

    def _register_local(self, path: Path) -> ModelRecord:
        path = path.resolve()
        _validate_model_dir(path)
        for model in self.registry.list_models():
            if model.path.resolve() == path:
                return model
        path_hash = hashlib.sha256(str(path).encode()).hexdigest()[:8]
        repo_id = f"local/{_slug(path.name)}-{path_hash}"
        return self.registry.add_model(
            repo_id,
            "local",
            path,
            source="local",
            size_bytes=directory_size(path),
            metadata={"registered_path": str(path)},
        )

    def _convert(
        self,
        repo_id: str,
        revision: str,
        source: Path,
        bits: int,
        *,
        trust_remote_code: bool,
        capture: bool,
    ) -> ModelRecord:
        if bits not in (2, 3, 4, 6, 8):
            raise SafetyError("quantization bits must be one of 2, 3, 4, 6, or 8")
        self.paths.converted_models.mkdir(parents=True, exist_ok=True)
        name = f"{_slug(repo_id)}-{revision[:12]}-q{bits}"
        destination = self.paths.converted_models / name
        temp = self.paths.converted_models / f".{name}.tmp-{uuid.uuid4().hex}"
        if destination.exists():
            raise SafetyError(f"conversion destination already exists: {destination}")
        command = [
            sys.executable,
            "-m",
            "mlx_lm.convert",
            "--hf-path",
            str(source),
            "--mlx-path",
            str(temp),
            "--quantize",
            "--q-bits",
            str(bits),
        ]
        if trust_remote_code:
            command.append("--trust-remote-code")
        self.logger.info("starting conversion for %s@%s to q%s", repo_id, revision, bits)
        try:
            completed = subprocess.run(
                command,
                check=False,
                text=capture,
                stdout=subprocess.PIPE if capture else None,
                stderr=subprocess.PIPE if capture else None,
            )
            if completed.returncode != 0:
                raise RuntimeFailure(
                    f"MLX-LM conversion failed with exit code {completed.returncode}",
                    hint="inspect the MLX-LM output above and mlxvm.log",
                    details={"stderr": completed.stderr.strip()[-4000:]}
                    if capture and completed.stderr
                    else {},
                )
            if capture and completed.stdout:
                self.logger.debug("conversion output: %s", completed.stdout.strip())
            _validate_model_dir(temp)
            os.replace(temp, destination)
        except KeyboardInterrupt:
            self.logger.warning("conversion interrupted for %s", repo_id)
            raise
        finally:
            if temp.exists():
                shutil.rmtree(temp, ignore_errors=True)
        return self.registry.add_model(
            repo_id,
            revision,
            destination,
            source="converted",
            size_bytes=directory_size(destination),
            metadata={"quantization_bits": bits, "source_path": str(source)},
            variant=f"q{bits}",
        )

    def alias(self, name: str, target: str) -> ModelRecord:
        self._validate_alias(name)
        model = self.registry.resolve(target)
        if model is None:
            raise ModelNotFoundError(
                f"model or alias '{target}' is not installed",
                hint="run 'mlxvm ls' to see installed models",
            )
        self.registry.set_alias(name, model.id)
        return self.registry.get_by_id(model.id) or model

    @staticmethod
    def _validate_alias(name: str) -> None:
        if not name or not all(character.isalnum() or character in "._-" for character in name):
            raise SafetyError("alias may contain only letters, numbers, '.', '_', and '-'")

    def unalias(self, name: str) -> None:
        if not self.registry.remove_alias(name):
            raise ModelNotFoundError(f"alias '{name}' does not exist")

    def uninstall(self, target: str) -> ModelRecord:
        model = self.registry.resolve(target)
        if model is None:
            raise ModelNotFoundError(f"model or alias '{target}' is not installed")
        with FileLock(self.paths.locks / "models.lock"):
            tombstone: Optional[Path] = None
            if model.source == "converted":
                if not _is_relative_to(model.path, self.paths.converted_models):
                    raise SafetyError(
                        f"refusing to remove converted model outside MLXVM_HOME: {model.path}"
                    )
                tombstone = model.path.with_name(f".{model.path.name}.deleting-{uuid.uuid4().hex}")
                os.replace(model.path, tombstone)
            try:
                removed = self.registry.remove_model(model.id)
                if removed is None:
                    raise RuntimeFailure(f"registry entry disappeared during uninstall: {model.id}")
            except Exception:
                if tombstone and tombstone.exists():
                    os.replace(tombstone, model.path)
                raise
            if tombstone:
                shutil.rmtree(tombstone, ignore_errors=True)
            try:
                self.hub.prune(self.registry.referenced_revisions())
            except Exception as exc:
                self.logger.warning("model removed but cache cleanup failed: %s", exc)
            self.logger.info("uninstalled model %s", model.reference)
            return removed

    def prune(self) -> Dict[str, int]:
        with FileLock(self.paths.locks / "models.lock"):
            result = self.hub.prune(self.registry.referenced_revisions())
            registered = {model.path.resolve() for model in self.registry.list_models()}
            converted = 0
            converted_bytes = 0
            if self.paths.converted_models.exists():
                for path in self.paths.converted_models.iterdir():
                    if not path.is_dir() or path.resolve() in registered:
                        continue
                    if not _is_relative_to(path, self.paths.converted_models):
                        continue
                    converted_bytes += directory_size(path)
                    shutil.rmtree(path)
                    converted += 1
            prompt_partials = 0
            if self.paths.prompt_cache.exists():
                for partial in self.paths.prompt_cache.glob("*.tmp"):
                    partial.unlink(missing_ok=True)
                    prompt_partials += 1
            result.update(
                {
                    "converted_models": converted,
                    "converted_bytes": converted_bytes,
                    "prompt_partials": prompt_partials,
                }
            )
            self.logger.info("pruned cache: %s", result)
            return result
