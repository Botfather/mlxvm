from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

SCHEMA_VERSION = 2


@dataclass(frozen=True)
class ModelRecord:
    id: int
    repo_id: str
    revision: str
    path: Path
    source: str
    size_bytes: int
    installed_at: str
    metadata: Dict[str, Any]
    aliases: tuple[str, ...] = ()
    variant: str = "default"

    @property
    def reference(self) -> str:
        suffix = f"#{self.variant}" if self.variant != "default" else ""
        return f"{self.repo_id}@{self.revision}{suffix}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "variant": self.variant,
            "reference": self.reference,
            "path": str(self.path),
            "source": self.source,
            "size_bytes": self.size_bytes,
            "installed_at": self.installed_at,
            "metadata": self.metadata,
            "aliases": list(self.aliases),
        }


class Registry:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.path), timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            self._migrate(connection)
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _migrate(self, connection: sqlite3.Connection) -> None:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise RuntimeError(
                f"registry schema {version} is newer than supported schema {SCHEMA_VERSION}"
            )
        if version == 0:
            connection.executescript(
                """
                CREATE TABLE models (
                    id INTEGER PRIMARY KEY,
                    repo_id TEXT NOT NULL,
                    revision TEXT NOT NULL,
                    path TEXT NOT NULL UNIQUE,
                    source TEXT NOT NULL CHECK(source IN ('hub', 'local', 'converted')),
                    size_bytes INTEGER NOT NULL DEFAULT 0 CHECK(size_bytes >= 0),
                    installed_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    variant TEXT NOT NULL DEFAULT 'default',
                    UNIQUE(repo_id, revision, variant)
                );
                CREATE TABLE aliases (
                    name TEXT PRIMARY KEY,
                    model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE
                );
                CREATE INDEX aliases_model_id ON aliases(model_id);
                PRAGMA user_version = 2;
                """
            )
        elif version == 1:
            connection.executescript(
                """
                PRAGMA foreign_keys = OFF;
                ALTER TABLE models RENAME TO models_v1;
                ALTER TABLE aliases RENAME TO aliases_v1;
                CREATE TABLE models (
                    id INTEGER PRIMARY KEY,
                    repo_id TEXT NOT NULL,
                    revision TEXT NOT NULL,
                    path TEXT NOT NULL UNIQUE,
                    source TEXT NOT NULL CHECK(source IN ('hub', 'local', 'converted')),
                    size_bytes INTEGER NOT NULL DEFAULT 0 CHECK(size_bytes >= 0),
                    installed_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    variant TEXT NOT NULL DEFAULT 'default',
                    UNIQUE(repo_id, revision, variant)
                );
                CREATE TABLE aliases (
                    name TEXT PRIMARY KEY,
                    model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE
                );
                INSERT INTO models
                    (id, repo_id, revision, path, source, size_bytes, installed_at, metadata_json)
                SELECT id, repo_id, revision, path, source, size_bytes, installed_at, metadata_json
                FROM models_v1;
                INSERT INTO aliases(name, model_id) SELECT name, model_id FROM aliases_v1;
                DROP TABLE aliases_v1;
                DROP TABLE models_v1;
                CREATE INDEX aliases_model_id ON aliases(model_id);
                PRAGMA user_version = 2;
                PRAGMA foreign_keys = ON;
                """
            )

    def add_model(
        self,
        repo_id: str,
        revision: str,
        path: Path,
        *,
        source: str = "hub",
        size_bytes: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        variant: str = "default",
    ) -> ModelRecord:
        installed_at = datetime.now(timezone.utc).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO models
                    (repo_id, revision, path, source, size_bytes, installed_at,
                     metadata_json, variant)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    repo_id,
                    revision,
                    str(path.resolve()),
                    source,
                    size_bytes,
                    installed_at,
                    json.dumps(metadata or {}, sort_keys=True),
                    variant,
                ),
            )
            model_id = int(cursor.lastrowid)
        record = self.get_by_id(model_id)
        if record is None:
            raise RuntimeError(f"registry failed to read newly inserted model {model_id}")
        return record

    def list_models(self) -> List[ModelRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT m.*, GROUP_CONCAT(a.name, char(31)) AS alias_names
                FROM models m
                LEFT JOIN aliases a ON a.model_id = m.id
                GROUP BY m.id
                ORDER BY m.repo_id COLLATE NOCASE, m.installed_at DESC
                """
            ).fetchall()
        return [self._record(row) for row in rows]

    def get_by_id(self, model_id: int) -> Optional[ModelRecord]:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT m.*, GROUP_CONCAT(a.name, char(31)) AS alias_names
                FROM models m LEFT JOIN aliases a ON a.model_id = m.id
                WHERE m.id = ? GROUP BY m.id
                """,
                (model_id,),
            ).fetchone()
        return self._record(row) if row else None

    def get_by_path(self, path: Path) -> Optional[ModelRecord]:
        resolved = str(path.resolve())
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT m.*, GROUP_CONCAT(a.name, char(31)) AS alias_names
                FROM models m LEFT JOIN aliases a ON a.model_id = m.id
                WHERE m.path = ? GROUP BY m.id
                """,
                (resolved,),
            ).fetchone()
        return self._record(row) if row else None

    def resolve(
        self, value: str, revision: Optional[str] = None, variant: Optional[str] = None
    ) -> Optional[ModelRecord]:
        value, variant_separator, embedded_variant = value.rpartition("#")
        if not variant_separator:
            value = embedded_variant
        else:
            variant = embedded_variant
        repo_id, separator, embedded_revision = value.rpartition("@")
        if separator and repo_id:
            value, revision = repo_id, embedded_revision
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT m.*, GROUP_CONCAT(a.name, char(31)) AS alias_names
                FROM models m LEFT JOIN aliases a ON a.model_id = m.id
                WHERE a.name = ? OR (
                    m.repo_id = ? AND (? IS NULL OR m.revision = ?)
                    AND (? IS NULL OR m.variant = ?)
                )
                GROUP BY m.id
                ORDER BY CASE WHEN a.name = ? THEN 0 ELSE 1 END,
                    CASE WHEN m.variant = 'default' THEN 0 ELSE 1 END,
                    m.installed_at DESC
                LIMIT 1
                """,
                (value, value, revision, revision, variant, variant, value),
            ).fetchone()
        return self._record(row) if row else None

    def set_alias(self, name: str, model_id: int) -> None:
        if not name or not all(character.isalnum() or character in "._-" for character in name):
            raise ValueError("alias may contain only letters, numbers, '.', '_', and '-'")
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO aliases(name, model_id) VALUES (?, ?)
                ON CONFLICT(name) DO UPDATE SET model_id = excluded.model_id
                """,
                (name, model_id),
            )

    def remove_alias(self, name: str) -> bool:
        with self.connect() as connection:
            cursor = connection.execute("DELETE FROM aliases WHERE name = ?", (name,))
            return cursor.rowcount > 0

    def remove_model(self, model_id: int) -> Optional[ModelRecord]:
        record = self.get_by_id(model_id)
        if record is None:
            return None
        with self.connect() as connection:
            connection.execute("DELETE FROM models WHERE id = ?", (model_id,))
        return record

    def referenced_revisions(self) -> set[str]:
        with self.connect() as connection:
            rows = connection.execute("SELECT revision FROM models WHERE source = 'hub'").fetchall()
        return {str(row["revision"]) for row in rows}

    @staticmethod
    def _record(row: sqlite3.Row) -> ModelRecord:
        aliases = tuple(sorted(filter(None, (row["alias_names"] or "").split(chr(31)))))
        return ModelRecord(
            id=row["id"],
            repo_id=row["repo_id"],
            revision=row["revision"],
            path=Path(row["path"]),
            source=row["source"],
            size_bytes=row["size_bytes"],
            installed_at=row["installed_at"],
            metadata=json.loads(row["metadata_json"]),
            aliases=aliases,
            variant=row["variant"],
        )
