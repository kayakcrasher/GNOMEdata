"""Persistent GNOMEdata workspace model."""

from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3


@dataclass(frozen=True)
class Workspace:
    workspace_id: str
    name: str
    description: str
    created_at: str


def normalize_workspace_id(value: str) -> str:
    """Convert a human workspace name into a stable identifier."""

    value = value.strip().casefold()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


class WorkspaceStore:
    """Persistent registry of GNOMEdata workspaces."""

    def __init__(self, database: str | Path) -> None:
        self.database = str(database)

        self.connection = sqlite3.connect(
            self.database,
            check_same_thread=False,
        )

        self.connection.row_factory = sqlite3.Row

        with self.connection:
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS workspaces (
                    workspace_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def create(
        self,
        name: str,
        description: str = "",
        workspace_id: str | None = None,
    ) -> Workspace:
        name = name.strip()

        if not name:
            raise ValueError("workspace name cannot be empty.")

        if workspace_id is None:
            workspace_id = normalize_workspace_id(name)
        else:
            workspace_id = workspace_id.strip()

        if not workspace_id:
            raise ValueError("workspace_id cannot be empty.")

        with self.connection:
            self.connection.execute("""
                INSERT INTO workspaces (
                    workspace_id,
                    name,
                    description
                )
                VALUES (?, ?, ?)
            """, (
                workspace_id,
                name,
                description.strip(),
            ))

        workspace = self.get(workspace_id)

        if workspace is None:
            raise RuntimeError("workspace creation failed.")

        return workspace

    def get(
        self,
        workspace_id: str,
    ) -> Workspace | None:
        row = self.connection.execute("""
            SELECT
                workspace_id,
                name,
                description,
                created_at
            FROM workspaces
            WHERE workspace_id = ?
        """, (
            workspace_id,
        )).fetchone()

        if row is None:
            return None

        return Workspace(
            workspace_id=row["workspace_id"],
            name=row["name"],
            description=row["description"],
            created_at=row["created_at"],
        )

    def list(self) -> list[Workspace]:
        rows = self.connection.execute("""
            SELECT
                workspace_id,
                name,
                description,
                created_at
            FROM workspaces
            ORDER BY created_at, workspace_id
        """).fetchall()

        return [
            Workspace(
                workspace_id=row["workspace_id"],
                name=row["name"],
                description=row["description"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "WorkspaceStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
