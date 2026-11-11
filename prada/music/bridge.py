"""The way to the music players: Windows' media controls (SMTC), seen from WSL.

Spotify, Corpus and most other players publish what they play to Windows and obey its
play/pause/next/previous buttons. `WindowsBridge` keeps one `powershell.exe` running (WSL
interop, nothing to install) with `bridge.ps1`, and talks to it in JSON lines: `status()`
reads every player, `control(app, verb)` presses a button on one. `FakeBridge` plays a
canned list in memory, for tests and PRADA_FAKE_LLM.
"""

import base64
import json
import os
import queue
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from importlib import resources

TIMEOUT = 6.0  # seconds to wait for one answer (the first includes PowerShell's start)
POWERSHELL = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
VERBS = ("pause", "play", "next", "previous")
KNOWN = {"spotify": "Spotify", "corpus": "Corpus"}  # app id contains -> name; logged plays


class BridgeError(Exception):
    """The players can't be reached (no Windows, PowerShell failed, no answer)."""


@dataclass
class Player:
    app: str  # Windows' app id, e.g. "Spotify.exe" or "Corpus.Corpus"
    title: str = ""
    artist: str = ""
    album: str = ""
    state: str = "stopped"  # playing, paused, stopped, changing, opened, closed
    position: float = 0.0  # seconds into the track, 0 if the player doesn't say
    duration: float = 0.0  # 0 if unknown

    @property
    def name(self) -> str:
        """ "Spotify", "Corpus", or the app id for any other player."""
        low = self.app.lower()
        return next((name for key, name in KNOWN.items() if key in low), self.app)

    @property
    def known(self) -> bool:
        return self.name in KNOWN.values()

    @property
    def playing(self) -> bool:
        return self.state == "playing"

    @property
    def track(self) -> tuple[str, str, str]:
        return (self.title, self.artist, self.album)

    def describe(self) -> str:
        """ "Space Dementia by Muse"; just the title when there's no artist."""
        if self.title and self.artist:
            return f"{self.title} by {self.artist}"
        return self.title or "something untitled"


@dataclass
class Status:
    players: list[Player] = field(default_factory=list)
    current: str | None = None  # the app Windows shows in its own media flyout

    def get(self, app: str) -> Player | None:
        return next((p for p in self.players if p.app == app), None)


def parse_status(data: dict) -> Status:
    players = []
    for p in data.get("players") or []:
        if not isinstance(p, dict) or not p.get("app"):
            continue
        players.append(
            Player(
                app=str(p["app"]),
                title=str(p.get("title") or ""),
                artist=str(p.get("artist") or ""),
                album=str(p.get("album") or ""),
                state=str(p.get("state") or "stopped"),
                position=float(p.get("position") or 0),
                duration=float(p.get("duration") or 0),
            )
        )
    return Status(players=players, current=data.get("current") or None)


def find_powershell() -> str | None:
    """Windows PowerShell through WSL interop (or on Windows itself), None elsewhere."""
    found = shutil.which("powershell.exe")
    if found:
        return found
    return POWERSHELL if os.path.exists(POWERSHELL) else None


def script() -> str:
    return resources.files("prada.music").joinpath("bridge.ps1").read_text(encoding="utf-8")


class WindowsBridge:
    """One long-running PowerShell, started on first use and again after it dies. Safe to
    call from several threads (one request at a time)."""

    def __init__(self, powershell: str | None = None, timeout: float = TIMEOUT):
        self._powershell = powershell
        self._timeout = timeout
        self._proc: subprocess.Popen | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.Lock()
        self._next = 0

    def status(self) -> Status:
        return parse_status(self._request({"cmd": "status"}))

    def control(self, app: str, verb: str) -> bool:
        if verb not in VERBS:
            raise ValueError(f"unknown verb {verb!r}")
        return bool(self._request({"cmd": "control", "app": app, "verb": verb}).get("ok"))

    def close(self) -> None:
        with self._lock:
            self._stop()

    def _request(self, body: dict) -> dict:
        with self._lock:
            proc = self._start()
            self._next += 1
            body = {"id": self._next, **body}
            try:
                proc.stdin.write(json.dumps(body) + "\n")
                proc.stdin.flush()
            except OSError as e:
                self._stop()
                raise BridgeError(f"the media bridge stopped ({e})") from e
            while True:
                reply = self._read()
                if reply.get("id") == body["id"]:
                    break  # an older answer that came too late is skipped
            if body["cmd"] == "status" and not reply.get("ok"):
                raise BridgeError(reply.get("error") or "Windows didn't list the players")
            return reply

    def _read(self) -> dict:
        try:
            line = self._lines.get(timeout=self._timeout)
        except queue.Empty:
            self._stop()
            raise BridgeError("Windows' media controls didn't answer") from None
        if line is None:
            self._stop()
            raise BridgeError("the media bridge stopped")
        try:
            data = json.loads(line.lstrip("﻿"))
        except ValueError:
            return {}
        if "fatal" in data:
            self._stop()
            raise BridgeError(f"Windows' media controls failed: {data['fatal']}")
        return data if isinstance(data, dict) else {}

    def _start(self) -> subprocess.Popen:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        exe = self._powershell or find_powershell()
        if exe is None:
            raise BridgeError("I can only reach the music players on Windows or WSL")
        encoded = base64.b64encode(script().encode("utf-16-le")).decode("ascii")
        try:
            self._proc = subprocess.Popen(
                [exe, "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError as e:
            raise BridgeError(f"PowerShell didn't start ({e})") from e
        self._lines = queue.Queue()
        threading.Thread(target=_pump, args=(self._proc.stdout, self._lines), daemon=True).start()
        while not self._read().get("ready"):
            pass
        return self._proc

    def _stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.kill()
            proc.wait(timeout=2)  # reap it, or a 24/7 node piles up zombies
        except (OSError, subprocess.TimeoutExpired):
            pass
        for pipe in (proc.stdin, proc.stdout):
            try:
                pipe.close()
            except (OSError, ValueError):
                pass


def _pump(stream, lines: queue.Queue) -> None:
    for line in stream:
        if line.strip():
            lines.put(line)
    lines.put(None)


# A small canned library for FakeBridge: (title, artist, album, seconds).
FAKE_TRACKS = [
    ("Space Dementia", "Muse", "Origin of Symmetry", 381.0),
    ("Unravel", "TK from Ling tosite sigure", "Fantastic Magic", 238.0),
    ("Sugar Song and Bitter Step", "UNISON SQUARE GARDEN", "Catcher In The Spy", 246.0),
]


class FakeBridge:
    """Players in memory: Spotify playing the first fake track, Corpus paused on the second.
    `stuck` = buttons answer ok but nothing changes (to test the checks)."""

    def __init__(self, players: list[Player] | None = None, stuck: bool = False):
        if players is None:
            players = [
                _fake_player("Spotify.exe", 0, "playing"),
                _fake_player("Corpus.Corpus", 1, "paused"),
            ]
        self.players = players
        self.current = players[0].app if players else None
        self.stuck = stuck
        self.pressed: list[tuple[str, str]] = []

    def status(self) -> Status:
        copy = [Player(**vars(p)) for p in self.players]
        return Status(players=copy, current=self.current)

    def control(self, app: str, verb: str) -> bool:
        if verb not in VERBS:
            raise ValueError(f"unknown verb {verb!r}")
        player = next((p for p in self.players if p.app == app), None)
        if player is None:
            return False
        self.pressed.append((app, verb))
        self.current = app
        if self.stuck:
            return True
        if verb == "pause":
            player.state = "paused"
        elif verb == "play":
            player.state = "playing"
        else:
            i = next((i for i, t in enumerate(FAKE_TRACKS) if t[0] == player.title), 0)
            step = 1 if verb == "next" else -1
            moved = _fake_player(app, (i + step) % len(FAKE_TRACKS), "playing")
            vars(player).update(vars(moved))
        return True


def _fake_player(app: str, i: int, state: str) -> Player:
    title, artist, album, seconds = FAKE_TRACKS[i]
    return Player(app, title, artist, album, state, 0.0, seconds)
