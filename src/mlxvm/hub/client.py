from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mlxvm.errors import DependencyError, NetworkError

IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40,64}$", re.IGNORECASE)


@dataclass(frozen=True)
class RemoteModel:
    repo_id: str
    revision: Optional[str]
    downloads: int
    likes: int
    pipeline_tag: Optional[str]
    updated_at: Optional[str]
    size_bytes: Optional[int]
    gated: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "downloads": self.downloads,
            "likes": self.likes,
            "pipeline_tag": self.pipeline_tag,
            "updated_at": self.updated_at,
            "size_bytes": self.size_bytes,
            "gated": self.gated,
        }


@dataclass(frozen=True)
class DownloadPlan:
    repo_id: str
    revision: str
    total_bytes: int
    download_bytes: int
    files: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "total_bytes": self.total_bytes,
            "download_bytes": self.download_bytes,
            "files": self.files,
        }


def parse_model_spec(spec: str) -> Tuple[str, Optional[str]]:
    candidate = Path(spec).expanduser()
    if candidate.exists():
        return str(candidate.resolve()), None
    repo, separator, revision = spec.rpartition("@")
    if separator and "/" in repo and revision:
        return repo, revision
    return spec, None


def directory_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except FileNotFoundError:
                continue
    return total


class HubClient:
    def __init__(self, cache_dir: Path, *, offline: bool = False) -> None:
        self.cache_dir = cache_dir
        self.offline = offline

    @staticmethod
    def _imports():
        try:
            from huggingface_hub import HfApi, scan_cache_dir, snapshot_download
            from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
        except ImportError as exc:
            raise DependencyError(
                "huggingface-hub is not installed",
                hint="reinstall mlxvm with its required dependencies",
            ) from exc
        return HfApi, snapshot_download, scan_cache_dir, HfHubHTTPError, LocalEntryNotFoundError

    def search(self, query: Optional[str], *, limit: int = 20) -> List[RemoteModel]:
        if self.offline:
            raise NetworkError("remote search is unavailable in offline mode")
        HfApi, _, _, hub_error, _ = self._imports()
        try:
            kwargs: Dict[str, Any] = {
                "author": "mlx-community",
                "search": query or None,
                "sort": "downloads",
                "limit": min(limit * 3, 100),
                "full": True,
            }
            if "direction" in inspect.signature(HfApi.list_models).parameters:
                kwargs["direction"] = -1
            models = HfApi().list_models(**kwargs)
            compatible = [
                self._remote_model(model)
                for model in models
                if getattr(model, "pipeline_tag", None) in {None, "text-generation"}
            ]
            return compatible[:limit]
        except (hub_error, OSError) as exc:
            raise NetworkError(f"Hugging Face search failed: {exc}") from exc

    def resolve_revision(self, repo_id: str, revision: Optional[str]) -> Tuple[str, int]:
        if self.offline:
            if revision and IMMUTABLE_REVISION.fullmatch(revision):
                return revision, 0
            raise NetworkError(
                "offline installation requires an immutable commit revision",
                hint=f"use {repo_id}@<commit-sha> or install it once while online",
            )
        HfApi, _, _, hub_error, _ = self._imports()
        try:
            info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
        except (hub_error, OSError) as exc:
            raise NetworkError(f"cannot resolve {repo_id}: {exc}") from exc
        sha = getattr(info, "sha", None)
        if not sha:
            raise NetworkError(f"Hugging Face did not return an immutable revision for {repo_id}")
        size = sum((getattr(file, "size", None) or 0) for file in (info.siblings or []))
        return str(sha), int(size)

    def plan(self, repo_id: str, revision: str, *, known_size: int = 0) -> DownloadPlan:
        _, snapshot_download, scan_cache_dir, hub_error, local_error = self._imports()
        if "dry_run" not in inspect.signature(snapshot_download).parameters:
            if self.cache_dir.exists():
                cache = scan_cache_dir(self.cache_dir)
                for repo in cache.repos:
                    if repo.repo_id != repo_id:
                        continue
                    for cached_revision in repo.revisions:
                        if cached_revision.commit_hash == revision:
                            return DownloadPlan(
                                repo_id,
                                revision,
                                int(cached_revision.size_on_disk),
                                0,
                                int(cached_revision.nb_files),
                            )
            return DownloadPlan(repo_id, revision, known_size, known_size, 0)
        try:
            files = snapshot_download(
                repo_id,
                revision=revision,
                cache_dir=self.cache_dir,
                local_files_only=self.offline,
                dry_run=True,
            )
        except (hub_error, local_error, OSError) as exc:
            raise NetworkError(f"cannot calculate download size for {repo_id}: {exc}") from exc
        total = sum(int(getattr(file, "file_size", 0) or 0) for file in files)
        download = sum(
            int(getattr(file, "file_size", 0) or 0)
            for file in files
            if getattr(file, "will_download", True)
        )
        return DownloadPlan(repo_id, revision, total, download, len(files))

    def download(self, repo_id: str, revision: str) -> Path:
        _, snapshot_download, _, hub_error, local_error = self._imports()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = snapshot_download(
                repo_id,
                revision=revision,
                cache_dir=self.cache_dir,
                local_files_only=self.offline,
            )
        except (hub_error, local_error, OSError) as exc:
            mode = "offline cache lookup" if self.offline else "download"
            raise NetworkError(f"{mode} failed for {repo_id}@{revision}: {exc}") from exc
        return Path(result).resolve()

    def prune(self, referenced_revisions: set[str]) -> Dict[str, int]:
        _, _, scan_cache_dir, _, _ = self._imports()
        if not self.cache_dir.exists():
            return {"revisions": 0, "bytes": 0, "partials": 0}
        cache = scan_cache_dir(self.cache_dir)
        stale = [
            revision.commit_hash
            for repo in cache.repos
            for revision in repo.revisions
            if revision.commit_hash not in referenced_revisions
        ]
        freed = 0
        if stale:
            strategy = cache.delete_revisions(*stale)
            freed = int(strategy.expected_freed_size)
            strategy.execute()
        partials = 0
        for partial in self.cache_dir.rglob("*.incomplete"):
            try:
                partial.unlink()
                partials += 1
            except FileNotFoundError:
                pass
        return {"revisions": len(stale), "bytes": freed, "partials": partials}

    @staticmethod
    def _remote_model(model: Any) -> RemoteModel:
        size = getattr(model, "used_storage", None)
        updated = getattr(model, "last_modified", None)
        return RemoteModel(
            repo_id=str(model.id),
            revision=getattr(model, "sha", None),
            downloads=int(getattr(model, "downloads", 0) or 0),
            likes=int(getattr(model, "likes", 0) or 0),
            pipeline_tag=getattr(model, "pipeline_tag", None),
            updated_at=updated.isoformat() if hasattr(updated, "isoformat") else None,
            size_bytes=int(size) if isinstance(size, (int, float)) else None,
            gated=bool(getattr(model, "gated", False)),
        )
