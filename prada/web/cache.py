"""SQLite cache for search results and page reads (table `web_cache`, in the web module's
store). Saves rate limits (Reddit allows about 10 reads a minute) and makes a follow-up on
the same search instant. Entries are kept for their source's TTL; anything older than
`KEEP` is deleted when the cache opens."""

import json
import sqlite3
import time
from collections.abc import Callable

MIGRATIONS = [
    """
    CREATE TABLE web_cache (
        key TEXT PRIMARY KEY,     -- "<source>:search:<intent>:<query>" or "<source>:read:<url>"
        value TEXT NOT NULL,      -- JSON
        fetched_at REAL NOT NULL  -- unix time
    );
    """
]
KEEP = 7 * 24 * 3600


def key(*parts: str) -> str:
    return ":".join(" ".join(p.lower().split()) for p in parts)


class Cache:
    """Read-through cache; with no database (store unavailable) it caches nothing."""

    def __init__(self, db: sqlite3.Connection | None, clock: Callable[[], float] = time.time):
        self._db = db
        self._clock = clock
        if db is not None:
            try:
                with db:
                    db.execute("DELETE FROM web_cache WHERE fetched_at < ?", (clock() - KEEP,))
            except sqlite3.Error:
                self._db = None

    def get(self, k: str, ttl: float):
        if self._db is None:
            return None
        try:
            row = self._db.execute(
                "SELECT value FROM web_cache WHERE key = ? AND fetched_at >= ?",
                (k, self._clock() - ttl),
            ).fetchone()
        except sqlite3.Error:
            return None
        return json.loads(row["value"]) if row else None

    def put(self, k: str, value) -> None:
        if self._db is None:
            return
        try:
            with self._db:
                self._db.execute(
                    "INSERT OR REPLACE INTO web_cache (key, value, fetched_at) VALUES (?, ?, ?)",
                    (k, json.dumps(value, ensure_ascii=False), self._clock()),
                )
        except sqlite3.Error:
            pass  # a cache that can't write is only slower
