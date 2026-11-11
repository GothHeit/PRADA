"""PRADA's copy of Corpus' music catalog, and finding music in it by name or genre.

Corpus exports its catalog (`--export-music`); PRADA keeps it in `music_track` and an
in-memory index, so "play Space Dementia" is matched in code, in milliseconds, with no
model call. Names are compared without case, accents or punctuation, word order is free
("Muse Space Dementia" = "Space Dementia Muse") and small mishearings still match.
"""

import re
import threading
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher

STORE = "music_catalog"  # its own schema version, apart from the play log's
MIGRATIONS = [
    """
    CREATE TABLE music_track (
        path TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        artist TEXT NOT NULL DEFAULT '',
        album TEXT NOT NULL DEFAULT '',
        genre TEXT NOT NULL DEFAULT '',
        duration REAL
    );
    CREATE TABLE music_catalog (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        imported_at REAL NOT NULL,
        songs INTEGER NOT NULL
    );
    """,
]
CLEAR = 0.85  # a name this close is what was asked for
TIE = 0.03  # two different kinds of hit this close: ask which one


def norm(text: str) -> str:
    """Lower case, no accents or punctuation: "Próxima, please!" -> "proxima please"."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"['’]", "", text)
    return re.sub(r"[^\w]+", " ", text).strip()


def key(text: str) -> str:
    """`norm` without a leading "the": "The Beatles" -> "beatles"."""
    return re.sub(r"^the ", "", norm(text))


def similar(a: str, b: str) -> float:
    """1.0 for the same words (in any order), else how alike the two keys are."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if sorted(a.split()) == sorted(b.split()):
        return 0.98
    matcher = SequenceMatcher(None, a, b)
    if matcher.real_quick_ratio() < CLEAR - 0.1:
        return 0.0
    return matcher.ratio()


@dataclass(frozen=True)
class Track:
    path: str  # the file as Corpus knows it (a Windows path)
    title: str
    artist: str = ""
    album: str = ""
    genre: str = ""
    duration: float | None = None

    def describe(self) -> str:
        return f"{self.title} by {self.artist}" if self.artist else self.title


@dataclass
class Hit:
    kind: str  # "song", "album" or "artist"
    name: str  # the title, album or artist as tagged
    artist: str
    tracks: list[Track] = field(default_factory=list)
    score: float = 0.0

    def describe(self) -> str:
        if self.kind == "artist" or not self.artist:
            return self.name
        return f"{self.name} by {self.artist}"


def _stem(path: str) -> str:
    name = re.split(r"[\\/]", path)[-1]
    return name.rsplit(".", 1)[0] if "." in name else name


def parse_export(data: list) -> list[Track]:
    """Corpus' export (a list of song dicts) as tracks; junk entries are skipped."""
    tracks: dict[str, Track] = {}
    for song in data:
        if not isinstance(song, dict) or not isinstance(song.get("path"), str):
            continue
        genre = song.get("genre") or ""
        if isinstance(genre, list):
            genre = "; ".join(str(g) for g in genre)
        try:
            duration = float(song["duration"]) if song.get("duration") else None
        except (TypeError, ValueError):
            duration = None
        path = song["path"]
        tracks[path] = Track(
            path=path,
            title=str(song.get("title") or "").strip() or _stem(path),
            artist=str(song.get("artist") or "").strip(),
            album=str(song.get("album") or "").strip(),
            genre=str(genre).strip(),
            duration=duration,
        )
    return list(tracks.values())


def _folder(path: str) -> str:
    return re.split(r"[\\/](?=[^\\/]*$)", path)[0]


def _most(names: list[str]) -> str:
    return Counter(names).most_common(1)[0][0] if names else ""


class Catalog:
    """The catalog in `music_track`, indexed in memory. Safe to share between threads."""

    def __init__(self, db):
        self._db = db
        self._lock = threading.Lock()
        self._songs: dict[tuple[str, str], list[Track]] = {}
        self._albums: dict[tuple[str, str], list[Track]] = {}
        self._artists: dict[str, list[Track]] = {}
        self._tracks: list[Track] = []
        self.imported_at: float | None = None
        self._load()

    def __len__(self) -> int:
        return len(self._tracks)

    def replace(self, tracks: list[Track], now: float) -> None:
        """The whole catalog becomes `tracks` (a fresh export)."""
        with self._lock, self._db:
            self._db.execute("DELETE FROM music_track")
            self._db.executemany(
                "INSERT INTO music_track (path, title, artist, album, genre, duration)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                [(t.path, t.title, t.artist, t.album, t.genre, t.duration) for t in tracks],
            )
            self._db.execute(
                "INSERT INTO music_catalog (id, imported_at, songs) VALUES (1, ?, ?)"
                " ON CONFLICT(id) DO UPDATE SET imported_at = excluded.imported_at,"
                " songs = excluded.songs",
                (now, len(tracks)),
            )
        self._load()

    def sync(self) -> None:
        """Reload if another process (`prada music import`) wrote a newer catalog."""
        with self._lock:
            row = self._db.execute("SELECT imported_at FROM music_catalog").fetchone()
        if row and row["imported_at"] != self.imported_at:
            self._load()

    def _load(self) -> None:
        with self._lock:
            rows = self._db.execute("SELECT * FROM music_track ORDER BY path").fetchall()
            row = self._db.execute("SELECT imported_at FROM music_catalog").fetchone()
            tracks = [Track(**dict(r)) for r in rows]
            songs, albums, artists = defaultdict(list), defaultdict(list), defaultdict(list)
            for t in tracks:
                songs[(key(t.title), key(t.artist))].append(t)
                if t.album:
                    albums[(key(t.album), _folder(t.path))].append(t)
                if t.artist:
                    artists[key(t.artist)].append(t)
            self._tracks, self._songs = tracks, dict(songs)
            self._albums, self._artists = dict(albums), dict(artists)
            self.imported_at = row["imported_at"] if row else None

    def find(
        self, query: str, by: list[tuple[str, str]] = (), kind: str | None = None
    ) -> list[Hit]:
        """Hits for `query` (a name, maybe with the artist in it), best first. `by` = other
        readings as (name, artist), e.g. ("space dementia", "muse") for "... by Muse";
        `kind` keeps only songs, albums or artists."""
        q = key(query)
        readings = [(key(name), key(artist)) for name, artist in by]
        hits: list[Hit] = []

        def score(name: str, artist: str) -> float:
            best = max(
                similar(q, name), similar(q, f"{artist} {name}"), similar(q, f"{name} {artist}")
            )
            for n, a in readings:
                best = max(best, min(similar(n, name), similar(a, artist)))
            return best

        with self._lock:
            if kind in (None, "song"):
                for (title, artist), tracks in self._songs.items():
                    if (s := score(title, artist)) >= CLEAR:
                        first = tracks[0]
                        hits.append(Hit("song", first.title, first.artist, tracks[:1], s))
            if kind in (None, "album"):
                for (album, _), tracks in self._albums.items():
                    artist = _most([t.artist for t in tracks])
                    if (s := score(album, key(artist))) >= CLEAR:
                        hits.append(Hit("album", tracks[0].album, artist, list(tracks), s))
            if kind in (None, "artist"):
                for name, tracks in self._artists.items():
                    s = similar(q, name)
                    if s >= CLEAR:
                        hits.append(
                            Hit("artist", tracks[0].artist, tracks[0].artist, list(tracks), s)
                        )
            # Same score: the artist with more songs in the library (kafka likes them more).
            plenty = {name: len(t) for name, t in self._artists.items()}
            hits.sort(key=lambda h: (-h.score, -plenty.get(key(h.artist), 0)))
        return hits

    def tracks(self) -> list[Track]:
        with self._lock:
            return list(self._tracks)

    def artists(self) -> list[str]:
        """Artists as tagged, the ones with the most songs first."""
        with self._lock:
            ranked = sorted(self._artists.values(), key=len, reverse=True)
            return [tracks[0].artist for tracks in ranked]

    def by_genre(self, words: list[str]) -> list[Track]:
        """Tracks whose genre holds any of `words` as whole words ("rock" in "J-Rock")."""
        wanted = [f" {key(w)} " for w in words if key(w)]
        with self._lock:
            return [t for t in self._tracks if any(w in f" {key(t.genre)} " for w in wanted)]
