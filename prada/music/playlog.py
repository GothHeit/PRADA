"""The play log: every track Spotify or Corpus played, whoever started it.

While the node runs, a watcher thread reads the players every `POLL` seconds and turns what
changed into rows of `music_play`: one row per track played, with the seconds actually
listened (pauses not counted), how it ended (finished, skipped early, stopped, or still
playing) and whether PRADA started it. This is what taste learning will read later; it
stays on this computer like the rest of her memory.
"""

import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from prada.music.bridge import BridgeError, Player, Status

MIGRATIONS = [
    """
    CREATE TABLE music_play (
        id INTEGER PRIMARY KEY,
        app TEXT NOT NULL,
        title TEXT NOT NULL,
        artist TEXT NOT NULL DEFAULT '',
        album TEXT NOT NULL DEFAULT '',
        started_at TEXT NOT NULL,
        ended_at TEXT,
        seconds REAL NOT NULL DEFAULT 0,
        duration REAL,
        ended TEXT NOT NULL DEFAULT 'playing',
        by_her INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX music_play_started ON music_play (started_at);
    """
]
POLL = 2.0  # seconds between looks at the players
MAX_STEP = 3 * POLL  # longest gap counted as listening (the node or the bridge stalled)
FLUSH = 30.0  # seconds between writes of a playing track's listened time
END_SLACK = 10.0  # this close to the end counts as finished
FINISHED_SHARE = 0.9  # or this much of the track was heard
HER_WINDOW = 10.0  # a track that starts this soon after she pressed a button is hers
RETRY = 30.0  # seconds to wait after the players couldn't be read


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class _Open:
    row: int
    track: tuple[str, str, str]
    seen: float  # when it was last seen
    playing: bool
    seconds: float
    position: float
    duration: float
    flushed: float


class PlayLog:
    """Turns successive `Status` readings into `music_play` rows. Only Spotify and Corpus
    are logged (`Player.known`); a browser video is not music."""

    def __init__(self, db):
        self._db = db
        self._open: dict[str, _Open] = {}
        self._hers: dict[str, float] = {}
        self._lock = threading.Lock()
        with db:  # a row left playing by an earlier run ended when that run did
            db.execute("UPDATE music_play SET ended = 'stopped' WHERE ended = 'playing'")

    def mark(self, app: str, now: float) -> None:
        """She pressed a button on `app`; the next new track there was started by her."""
        with self._lock:
            self._hers[app] = now

    def see(self, status: Status, now: float) -> None:
        players = {p.app: p for p in status.players if p.known and p.title}
        for app in list(self._open):
            if app not in players:
                self._close(app, now)  # the player was closed or shows no track
        for player in players.values():
            self._see(player, now)

    def close_all(self, now: float) -> None:
        for app in list(self._open):
            self._close(app, now)

    def _see(self, player: Player, now: float) -> None:
        o = self._open.get(player.app)
        if o is not None and o.playing:
            o.seconds += min(max(now - o.seen, 0.0), MAX_STEP)
        if o is not None and o.track != player.track:
            self._close(player.app, now)
            o = None
        if o is None:
            if player.playing:  # a track sitting paused since before is not a play
                self._start(player, now)
            return
        changed = o.playing != player.playing
        o.seen, o.playing = now, player.playing
        o.position = player.position or o.position
        o.duration = player.duration or o.duration
        if changed or now - o.flushed >= FLUSH:
            self._write(o, now)

    def _start(self, player: Player, now: float) -> None:
        with self._lock:
            her = now - self._hers.pop(player.app, -HER_WINDOW - 1) <= HER_WINDOW
        with self._db:
            cur = self._db.execute(
                "INSERT INTO music_play (app, title, artist, album, started_at, duration, by_her)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    player.name,
                    player.title,
                    player.artist,
                    player.album,
                    _iso(now),
                    player.duration or None,
                    int(her),
                ),
            )
        self._open[player.app] = _Open(
            cur.lastrowid,
            player.track,
            now,
            True,
            0.0,
            player.position,
            player.duration,
            now,
        )

    def _write(self, o: _Open, now: float) -> None:
        o.flushed = now
        with self._db:
            self._db.execute(
                "UPDATE music_play SET seconds = ?, duration = ? WHERE id = ?",
                (round(o.seconds, 1), o.duration or None, o.row),
            )

    def _close(self, app: str, now: float) -> None:
        o = self._open.pop(app)
        if o.duration and (
            o.position >= o.duration - END_SLACK or o.seconds >= FINISHED_SHARE * o.duration
        ):
            ended = "finished"
        elif o.playing:
            ended = "skipped"  # moved on (or closed) while it was still playing
        else:
            ended = "stopped"  # left paused, then changed or closed
        with self._db:
            self._db.execute(
                "UPDATE music_play SET seconds = ?, duration = ?, ended = ?, ended_at = ?"
                " WHERE id = ?",
                (round(o.seconds, 1), o.duration or None, ended, _iso(now), o.row),
            )


class Watcher:
    """Reads the players every `POLL` seconds in a daemon thread and feeds a `PlayLog`."""

    def __init__(self, status: Callable[[], Status], log: PlayLog, clock=time.time):
        self._status = status
        self._log = log
        self._clock = clock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error = ""  # the last failure printed, so a closed Windows isn't said every time

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="music-watch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def tick(self) -> float:
        """One look at the players; returns how long to wait before the next."""
        try:
            status = self._status()
        except BridgeError as e:
            if str(e) != self._error:
                print(f"music: can't read the players, retrying ({e})", file=sys.stderr)
            self._error = str(e)
            return RETRY
        self._error = ""
        self._log.see(status, self._clock())
        return POLL

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                wait = self.tick()
            except Exception as e:  # a bad row must not end the log for good
                print(f"music: play log error ({e})", file=sys.stderr)
                wait = RETRY
            self._stop.wait(wait)
