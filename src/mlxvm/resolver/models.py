from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from mlxvm.config.project import ConfigError, find_project_config, load_project_config
from mlxvm.registry import ModelRecord, Registry


@dataclass(frozen=True)
class Resolution:
    requested: Optional[str]
    revision: Optional[str]
    source: Optional[str]
    source_path: Optional[Path]
    model: Optional[ModelRecord]
    generation: Dict[str, Any]
    error: Optional[str] = None

    @property
    def selected(self) -> bool:
        return self.requested is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected": self.selected,
            "requested": self.requested,
            "revision": self.revision,
            "source": self.source,
            "source_path": str(self.source_path) if self.source_path else None,
            "installed": self.model is not None,
            "model": self.model.to_dict() if self.model else None,
            "generation": self.generation,
            "error": self.error,
        }


class ModelResolver:
    def __init__(self, registry: Registry):
        self.registry = registry

    def resolve(
        self, *, explicit: Optional[str] = None, start: Optional[Path] = None
    ) -> Resolution:
        if explicit:
            return self._finish(explicit, None, "argument")

        shell_model = os.environ.get("MLXVM_MODEL")
        if shell_model:
            return self._finish(
                shell_model,
                os.environ.get("MLXVM_REVISION"),
                "shell",
            )

        project_path = find_project_config(start)
        if project_path:
            try:
                project = load_project_config(project_path)
            except ConfigError as exc:
                return Resolution(None, None, "project", project_path, None, {}, str(exc))
            return self._finish(
                project.model,
                project.revision,
                "project",
                source_path=project.path,
                generation=project.generation,
            )

        default = self.registry.resolve("default")
        if default:
            return Resolution("default", default.revision, "default alias", None, default, {})

        return Resolution(
            None,
            None,
            None,
            None,
            None,
            {},
            "no model selected; run 'mlxvm install <repo>' or 'mlxvm use <model>'",
        )

    def _finish(
        self,
        requested: str,
        revision: Optional[str],
        source: str,
        *,
        source_path: Optional[Path] = None,
        generation: Optional[Dict[str, Any]] = None,
    ) -> Resolution:
        model = self.registry.resolve(requested, revision)
        embedded_revision: Optional[str] = None
        revision_source = requested.rpartition("#")[0] or requested
        repo_id, separator, candidate_revision = revision_source.rpartition("@")
        if separator and repo_id:
            embedded_revision = candidate_revision
        return Resolution(
            requested=requested,
            revision=model.revision if model else revision or embedded_revision,
            source=source,
            source_path=source_path,
            model=model,
            generation=generation or {},
            error=None if model else f"selected model '{requested}' is not installed",
        )
