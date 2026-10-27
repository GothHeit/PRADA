"""PRADA's shared SQLite memory.

One database file for the whole node. Each module owns its own tables and
registers them with a list of migrations; the store tracks how many of each
module's migrations have run, so schemas can grow without breaking old files.

The file lives in $PRADA_DATA_DIR, else $XDG_DATA_HOME/prada, else
~/.local/share/prada, as prada.db.
"""

import os
import sqlite3
from pathlib import Path


def data_dir() -> Path:
    if env := os.environ.get("PRADA_DATA_DIR"):
        return Path(env)
    base = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(base) / "prada"


def db_path() -> Path:
    return data_dir() / "prada.db"


def open_store(module: str, migrations: list[str], path: Path | str | None = None):
    """Open the shared database and bring `module`'s tables up to date.

    `migrations` is append-only: entry N is a SQL script run once when the
    module's stored version is below N + 1. Pass path=":memory:" in tests.
    """
    path = db_path() if path is None else path
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    # The node's server threads share one connection per module, serialized by the node.
    db = sqlite3.connect(path, check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.execute(
        "CREATE TABLE IF NOT EXISTS schema_versions (module TEXT PRIMARY KEY, version INTEGER NOT NULL)"
    )
    row = db.execute("SELECT version FROM schema_versions WHERE module = ?", (module,)).fetchone()
    version = row["version"] if row else 0
    if version > len(migrations):
        db.close()
        raise RuntimeError(f"{path} was written by a newer PRADA ({module} schema v{version})")
    for i in range(version, len(migrations)):
        with db:
            db.executescript(f"BEGIN;{migrations[i]}")
            db.execute(
                "INSERT INTO schema_versions (module, version) VALUES (?, ?) "
                "ON CONFLICT(module) DO UPDATE SET version = excluded.version",
                (module, i + 1),
            )
    return db
