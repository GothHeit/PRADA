"""The way to Spotify: its official Web API (Premium), with the standard library only.

kafka registers a Spotify app once (its client id goes under `[spotify]` in the config) and
logs in once with `prada music login` (OAuth Authorization Code with PKCE: no client
secret). The refresh token is kept next to the config file (`spotify-token.json`, only
readable by kafka), never in the repo. From then on `SpotifyApi` can:

- `search(q)`: tracks, artists, albums and playlists;
- `devices()`: the Spotify apps and speakers of kafka's account (Spotify Connect);
- `play(hit, device)`: start a track or an album/artist/playlist on one of them;
- `now()`: what plays right now, to confirm it before she says so.

This is a deliberate cloud call (approved 2026-11-11): Spotify's music lives on Spotify's
side anyway. `FakeSpotify` answers the same calls from a small canned catalog (tests,
PRADA_FAKE_LLM).
"""

import base64
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from prada import config

API = "https://api.spotify.com/v1"
ACCOUNTS = "https://accounts.spotify.com"
REDIRECT_PORT = 7725  # the redirect URI registered in kafka's Spotify app
REDIRECT_URI = f"http://127.0.0.1:{REDIRECT_PORT}/callback"
SCOPES = "user-read-playback-state user-modify-playback-state user-read-currently-playing"
TIMEOUT = 6.0  # seconds for one HTTP call
LOGIN_WAIT = 300  # seconds `prada music login` waits for the browser
CMD = "/mnt/c/Windows/System32/cmd.exe"


class SpotifyError(Exception):
    """Spotify can't be reached, isn't logged in, or said no."""


@dataclass
class Hit:
    """One search result."""

    kind: str  # track, artist, album, playlist
    uri: str
    name: str
    by: str = ""  # the artists, or a playlist's owner
    album: str = ""
    album_uri: str = ""  # a track's album: tracks are started inside it (see `play`)

    def describe(self) -> str:
        """ "Space Dementia by Muse", "the album Origin of Symmetry", "Muse"."""
        if self.kind == "track":
            return f"{self.name} by {self.by}" if self.by else self.name
        if self.kind == "album":
            return f"the album {self.name}" + (f" by {self.by}" if self.by else "")
        if self.kind == "playlist":
            return f"the playlist {self.name}"
        return self.name


@dataclass
class Results:
    hits: list[Hit] = field(default_factory=list)
    top: str | None = None  # the uri of Spotify's own top result, if it gave one


@dataclass
class Now:
    """What Spotify plays right now."""

    uri: str = ""
    title: str = ""
    artists: str = ""
    album: str = ""
    context: str = ""  # the album, artist or playlist it plays from
    playing: bool = False
    device: str = ""  # the device's name


@dataclass
class Device:
    id: str
    name: str
    kind: str = ""  # Computer, Smartphone, Speaker, ...
    active: bool = False


def _names(items) -> str:
    return ", ".join(
        str(a.get("name")) for a in items or [] if isinstance(a, dict) and a.get("name")
    )


def parse_results(data: dict) -> Results:
    """The Web API's search answer: {"tracks": {"items": [...]}, "artists": ...}."""
    hits = []
    for kind in ("track", "artist", "album", "playlist"):
        block = data.get(kind + "s")
        for item in (block or {}).get("items") or [] if isinstance(block, dict) else []:
            if not isinstance(item, dict) or not item.get("uri") or not item.get("name"):
                continue
            if kind == "playlist":
                by = str((item.get("owner") or {}).get("display_name") or "")
            else:
                by = _names(item.get("artists"))
            album = item.get("album") if kind == "track" else None
            album = album if isinstance(album, dict) else {}
            hits.append(
                Hit(
                    kind,
                    str(item["uri"]),
                    str(item["name"]),
                    by,
                    str(album.get("name") or ""),
                    str(album.get("uri") or ""),
                )
            )
    return Results(hits=hits)


def parse_now(data: dict | None) -> Now:
    """The Web API's player state (GET /me/player); empty when nothing plays."""
    data = data if isinstance(data, dict) else {}
    item = data.get("item") if isinstance(data.get("item"), dict) else {}
    return Now(
        uri=str(item.get("uri") or ""),
        title=str(item.get("name") or ""),
        artists=_names(item.get("artists")),
        album=str((item.get("album") or {}).get("name") or ""),
        context=str((data.get("context") or {}).get("uri") or ""),
        playing=bool(data.get("is_playing")),
        device=str((data.get("device") or {}).get("name") or ""),
    )


def parse_devices(data: dict) -> list[Device]:
    return [
        Device(
            str(d["id"]),
            str(d.get("name") or ""),
            str(d.get("type") or ""),
            bool(d.get("is_active")),
        )
        for d in data.get("devices") or []
        if isinstance(d, dict) and d.get("id") and not d.get("is_restricted")
    ]


def choose(devices: list[Device], anywhere: bool = False) -> Device | None:
    """The one playing already, else this PC's Spotify app; with `anywhere`, else any
    (a phone that happens to be online is a last resort, after opening the PC's app)."""
    wants = [lambda d: d.active, lambda d: d.kind == "Computer"]
    if anywhere:
        wants.append(lambda d: True)
    for want in wants:
        if found := next((d for d in devices if want(d)), None):
            return found
    return None


@dataclass
class Settings:
    client_id: str = ""
    token_path: Path = field(default_factory=Path)


def settings() -> Settings:
    """`[spotify] client_id` from the config (or PRADA_SPOTIFY_CLIENT_ID); the token file
    lives next to the config file."""
    file = config.load_config().get("spotify", {})
    if not isinstance(file, dict):
        raise ValueError(f"[spotify] in {config.config_path()} must be a table")
    client_id = os.environ.get("PRADA_SPOTIFY_CLIENT_ID") or str(file.get("client_id") or "")
    return Settings(client_id.strip(), config.config_path().parent / "spotify-token.json")


def _http(method: str, url: str, headers: dict, body: bytes | None) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except (urllib.error.URLError, OSError) as e:
        raise SpotifyError(f"Spotify can't be reached ({getattr(e, 'reason', e)})") from e


def _json(raw: bytes) -> dict:
    try:
        data = json.loads(raw or b"{}")
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _why(status: int, raw: bytes) -> str:
    error = _json(raw).get("error")
    if isinstance(error, dict):
        reason, message = error.get("reason"), error.get("message")
        if reason == "NO_ACTIVE_DEVICE":
            return "no Spotify device is open"
        if reason == "PREMIUM_REQUIRED":
            return "Spotify needs Premium for that"
        if message:
            return f"Spotify said: {message}"
    if status == 429:
        return "Spotify asked me to slow down"
    return f"Spotify answered {status}"


class SpotifyApi:
    """The Web API as kafka's account. `http(method, url, headers, body) -> (status, body)`
    replaces the network (tests). Safe to call from several threads."""

    def __init__(self, client_id: str = "", token_path: Path | None = None, http=None):
        if token_path is None:
            s = settings()
            client_id, token_path = client_id or s.client_id, s.token_path
        self.client_id = client_id
        self.token_path = token_path
        self._http = http or _http
        self._lock = threading.Lock()
        self._access = ""
        self._expires = 0.0

    def ready(self) -> bool:
        """True once a client id is set and kafka has logged in."""
        return bool(self.client_id) and self.token_path.is_file()

    def search(self, q: str, limit: int = 8) -> Results:
        query = urllib.parse.urlencode(
            {"q": q, "type": "track,artist,album,playlist", "limit": limit}
        )
        return parse_results(self._call("GET", f"/search?{query}"))

    def devices(self) -> list[Device]:
        return parse_devices(self._call("GET", "/me/player/devices"))

    def play(self, hit: Hit, device: Device) -> None:
        """A track starts inside its album (`offset`): the Store app answers 204 to a bare
        `uris` play but never starts it. `uris` only when the album is unknown."""
        if hit.kind != "track":
            body = {"context_uri": hit.uri}
        elif hit.album_uri:
            body = {"context_uri": hit.album_uri, "offset": {"uri": hit.uri}}
        else:
            body = {"uris": [hit.uri]}
        query = urllib.parse.urlencode({"device_id": device.id})
        self._call("PUT", f"/me/player/play?{query}", body)

    def now(self) -> Now:
        return parse_now(self._call("GET", "/me/player"))

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        for attempt in (1, 2):
            headers = {"Authorization": f"Bearer {self._token()}"}
            if data is not None:
                headers["Content-Type"] = "application/json"
            status, raw = self._http(method, API + path, headers, data)
            if status == 401 and attempt == 1:
                self._expires = 0  # refresh and try once more
                continue
            if status >= 400:
                raise SpotifyError(_why(status, raw))
            return _json(raw)
        raise SpotifyError("Spotify didn't accept my login; run `prada music login` again")

    def _token(self) -> str:
        with self._lock:
            if self._access and time.time() < self._expires - 30:
                return self._access
            if not self.client_id:
                raise SpotifyError("Spotify isn't set up: add [spotify] client_id to my config")
            try:
                saved = json.loads(self.token_path.read_text(encoding="utf-8"))
                refresh = str(saved["refresh_token"])
            except (OSError, ValueError, KeyError, TypeError):
                raise SpotifyError(
                    "I'm not logged in to Spotify; run `prada music login`"
                ) from None
            form = {
                "grant_type": "refresh_token",
                "refresh_token": refresh,
                "client_id": self.client_id,
            }
            self._store(self._exchange(form), refresh)
            return self._access

    def _exchange(self, form: dict) -> dict:
        body = urllib.parse.urlencode(form).encode()
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        status, raw = self._http("POST", f"{ACCOUNTS}/api/token", headers, body)
        data = _json(raw)
        if status >= 400 or not data.get("access_token"):
            why = data.get("error_description") or data.get("error") or f"answer {status}"
            raise SpotifyError(f"Spotify refused my login ({why}); run `prada music login` again")
        return data

    def _store(self, data: dict, old_refresh: str = "") -> None:
        self._access = str(data["access_token"])
        self._expires = time.time() + float(data.get("expires_in") or 3600)
        refresh = str(data.get("refresh_token") or old_refresh)
        if refresh != old_refresh:  # Spotify may hand out a new one; keep it
            save_token(self.token_path, refresh)

    # The one-time login.

    def login_url(self, verifier: str, state: str) -> str:
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        query = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": REDIRECT_URI,
            "code_challenge_method": "S256",
            "code_challenge": challenge.decode().rstrip("="),
            "scope": SCOPES,
            "state": state,
        }
        return f"{ACCOUNTS}/authorize?{urllib.parse.urlencode(query)}"

    def finish_login(self, code: str, verifier: str) -> None:
        form = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT_URI,
            "client_id": self.client_id,
            "code_verifier": verifier,
        }
        data = self._exchange(form)
        if not data.get("refresh_token"):
            raise SpotifyError("Spotify gave no refresh token")
        with self._lock:
            self._store(data)


def save_token(path: Path, refresh: str) -> None:
    """Write the refresh token, readable by kafka only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    if hasattr(os, "fchmod"):
        os.fchmod(fd, 0o600)  # a leftover .tmp keeps its old permissions otherwise
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"refresh_token": refresh}, f)
    os.replace(tmp, path)


def login(api: SpotifyApi, wait: float = LOGIN_WAIT, port: int = REDIRECT_PORT) -> None:
    """`prada music login`: open Spotify's consent page, catch the answer on
    127.0.0.1:7725/callback, keep the refresh token. SpotifyError when it fails."""
    if not api.client_id:
        raise SpotifyError(
            f"add [spotify] client_id to {config.config_path()} first (see docs/setup.md)"
        )
    verifier = secrets.token_urlsafe(64)
    state = secrets.token_urlsafe(16)
    got: dict = {}

    class Callback(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urllib.parse.urlsplit(self.path)
            if url.path != "/callback":
                self.send_error(404)
                return
            got.update({k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()})
            ok = "code" in got and got.get("state") == state
            text = (
                "PRADA is logged in to Spotify. You can close this tab."
                if ok
                else ("Spotify login failed. Look at the terminal.")
            )
            body = f"<!doctype html><meta charset=utf-8><p>{text}</p>".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    with HTTPServer(("127.0.0.1", port), Callback) as server:
        server.timeout = 1
        url = api.login_url(verifier, state)
        print("Open this address in your browser and allow PRADA:\n\n" + url + "\n")
        open_url(url)
        deadline = time.monotonic() + wait
        while not got and time.monotonic() < deadline:
            server.handle_request()
    if not got:
        raise SpotifyError("no answer from the browser in time")
    if got.get("state") != state:
        raise SpotifyError("the answer from the browser didn't match this login")
    if "code" not in got:
        raise SpotifyError(f"Spotify said no ({got.get('error', 'no code')})")
    api.finish_login(got["code"], verifier)


def _windows(target: str) -> bool:
    """Open a URL or app link with Windows (from WSL through interop); False if no Windows."""
    if sys.platform == "win32":
        os.startfile(target)  # type: ignore[attr-defined]
        return True
    exe = shutil.which("cmd.exe") or (CMD if os.path.exists(CMD) else None)
    if exe is None:
        return False
    try:
        subprocess.run(
            [exe, "/c", "start", "", target.replace("&", "^&")],
            cwd="/mnt/c" if os.path.isdir("/mnt/c") else None,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return True


def open_url(url: str) -> None:
    _windows(url)  # best effort; the address is printed anyway


def open_app() -> None:
    """Start the Spotify desktop app (Store or classic: both answer `spotify:`)."""
    if not _windows("spotify:"):
        raise SpotifyError("no Spotify device is open, and I can only open Spotify on Windows")


# A small canned catalog for FakeSpotify: (kind, name, by, album).
FAKE_CATALOG = [
    ("track", "Space Dementia", "Muse", "Origin of Symmetry"),
    ("track", "Hysteria", "Muse", "Absolution"),
    ("track", "Unravel", "TK from Ling tosite sigure", "Fantastic Magic"),
    ("track", "Sugar Song and Bitter Step", "UNISON SQUARE GARDEN", "Catcher In The Spy"),
    ("track", "Evidências", "Chitãozinho & Xororó", "Cowboy do Asfalto"),
    ("artist", "Muse", "", ""),
    ("artist", "TK from Ling tosite sigure", "", ""),
    ("album", "Origin of Symmetry", "Muse", ""),
    ("playlist", "Calm Vibes", "Spotify", ""),
]


def _fake_uri(kind: str, name: str) -> str:
    return f"spotify:{kind}:" + "".join(c for c in name.lower() if c.isalnum())


class FakeSpotify:
    """Spotify in memory. `open=False` = no device until `open_app` is called;
    `stuck` = play answers ok but nothing changes (to test the check)."""

    PC = Device("pc", "KAFKA-PC", "Computer", False)

    def __init__(
        self,
        open: bool = True,
        stuck: bool = False,
        logged_in: bool = True,
        others: list[Device] | None = None,
    ):
        """`others` = more devices online besides this PC's app (a phone)."""
        self._open = open
        self.others = others or []
        self.stuck = stuck
        self.logged_in = logged_in
        self.opened = 0
        self.searched: list[str] = []
        self.played: list[tuple[str, str]] = []  # (uri, device id)
        self.playing = Now(
            _fake_uri("track", "Unravel"),
            "Unravel",
            "TK from Ling tosite sigure",
            "Fantastic Magic",
        )

    def ready(self) -> bool:
        return self.logged_in

    def devices(self) -> list[Device]:
        return ([self.PC] if self._open else []) + self.others

    def open_app(self) -> None:
        self.opened += 1
        self._open = True

    def search(self, q: str, limit: int = 8) -> Results:
        from prada.music.play import tokens

        self.searched.append(q)
        wanted = tokens(q)
        hits = [
            Hit(
                kind,
                _fake_uri(kind, name),
                name,
                by,
                album,
                _fake_uri("album", album) if album else "",
            )
            for kind, name, by, album in FAKE_CATALOG
            if wanted & tokens(f"{name} {by}")
        ]
        return Results(hits=hits[:limit])

    def play(self, hit: Hit, device: Device) -> None:
        if device not in self.devices():
            raise SpotifyError("no Spotify device is open")
        self.played.append((hit.uri, device.id))
        if self.stuck:
            return
        if hit.kind == "track":
            context = hit.album_uri
            self.playing = Now(hit.uri, hit.name, hit.by, hit.album, context, True, device.name)
        else:  # a context: its first track
            _, name, by, album = next(t for t in FAKE_CATALOG if t[0] == "track")
            self.playing = Now(
                _fake_uri("track", name), name, by, album, hit.uri, True, device.name
            )

    def now(self) -> Now:
        return self.playing
