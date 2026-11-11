"""Corpus, kafka's own player, driven through its documented command line.

`Corpus.exe --export-music <file.json>` writes the music catalog (read-only, no window);
`--play [--shuffle] <files>`, `--queue <files>` and `--next <files>` play in Corpus' strip
without opening the viewer or taking focus. Exit codes: 0 ok, 1 failed, 2 bad command line.
PRADA uses nothing else: no Corpus database, no window messages.

From WSL the exe runs through interop: Windows paths in the catalog are passed as they are.
The export is written to Windows' own temp folder (`windows_temp`), read back through
/mnt/<drive> and deleted: Corpus can't write to a WSL path (a wsl.localhost network path).
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from prada import config

EXPORT_TIMEOUT = 120.0  # a catalog without stored genres reads every file (~15 s / 1000 songs)
PLAY_TIMEOUT = 10.0  # a command hands its files to the running Corpus and exits
MAX_ARGS = 24_000  # characters of paths on one command line (Windows allows 32 767)
CMD = "/mnt/c/Windows/System32/cmd.exe"


class CorpusError(Exception):
    """Corpus can't be reached or refused the command."""


def to_linux(path: str) -> str:
    """ "C:\\Apps\\Corpus.exe" -> "/mnt/c/Apps/Corpus.exe" (WSL); other paths unchanged."""
    if os.name != "nt" and (m := re.fullmatch(r"([A-Za-z]):[\\/](.*)", path)):
        return f"/mnt/{m[1].lower()}/{m[2].replace(chr(92), '/')}"
    return path


def to_windows(path: Path) -> str:
    """A path the Windows exe can open: as is on Windows, /mnt/<drive>/... by hand under WSL,
    else `wslpath -w` (a \\\\wsl.localhost path, which Corpus can't write to)."""
    if os.name == "nt":
        return str(path)
    if m := re.fullmatch(r"/mnt/([a-z])/(.*)", str(path)):
        return f"{m[1].upper()}:\\{m[2].replace('/', chr(92))}"
    try:
        out = subprocess.run(
            ["wslpath", "-w", str(path)], capture_output=True, text=True, timeout=5, check=True
        )
    except (OSError, subprocess.SubprocessError) as e:
        raise CorpusError(f"I can only reach Corpus from Windows or WSL ({e})") from e
    return out.stdout.strip()


def temp_candidates(echoed: str) -> list[str]:
    """Windows temp folders from `echo %TEMP%;%LOCALAPPDATA%`: TEMP, then the Temp folder in LOCALAPPDATA
    (TEMP can be a short 8.3 name that /mnt/c doesn't show)."""
    parts = [p.strip() for p in echoed.strip().split(";")]
    found = []
    if parts and parts[0] and "%" not in parts[0]:
        found.append(parts[0])
    if len(parts) > 1 and parts[1] and "%" not in parts[1]:
        found.append(parts[1].rstrip("\\") + "\\Temp")
    return found


def windows_temp() -> Path:
    """Windows' temp folder, as a path this process can write: Python's own on Windows,
    else %TEMP% asked from cmd.exe and seen through /mnt/<drive>."""
    if os.name == "nt":
        return Path(tempfile.gettempdir())
    cmd = shutil.which("cmd.exe") or (CMD if os.path.exists(CMD) else None)
    if cmd is None:
        raise CorpusError("I can only reach Corpus from Windows or WSL (no cmd.exe)")
    try:
        out = subprocess.run(
            [cmd, "/d", "/c", "echo %TEMP%;%LOCALAPPDATA%"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd="/mnt/c" if os.path.isdir("/mnt/c") else None,  # cmd.exe warns on a WSL cwd
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as e:
        raise CorpusError(f"Windows didn't say where its temp folder is ({e})") from e
    for folder in temp_candidates(out.stdout):
        path = Path(to_linux(folder))
        if path.is_dir():
            return path
    raise CorpusError(f"Windows' temp folder isn't reachable from here ({out.stdout.strip()})")


def find_corpus() -> str | None:
    """`[music] corpus` from the config file, else Corpus.exe on PATH."""
    if path := config.music_settings().corpus:
        return to_linux(path)
    return shutil.which("Corpus.exe")


def fit(paths: list[str]) -> list[str]:
    """As many paths as fit on one command line, in order."""
    kept, size = [], 0
    for p in paths:
        size += len(p) + 3
        if size > MAX_ARGS:
            break
        kept.append(p)
    return kept


class CorpusCli:
    """Runs Corpus.exe for one command at a time."""

    def __init__(self, exe: str | None = None):
        self._exe = exe

    def available(self) -> bool:
        """Corpus is set up (`[music] corpus` or on PATH); a bad setting counts as no."""
        try:
            return bool(self._exe or find_corpus())
        except ValueError:
            return False

    def export(self) -> list[dict]:
        """The catalog as Corpus exports it (a list of song dicts)."""
        self._find()  # before anything else: "where is Corpus?" is the useful error
        with tempfile.TemporaryDirectory(prefix="prada-corpus-", dir=windows_temp()) as tmp:
            dest = Path(tmp) / "music.json"
            self._run(["--export-music", to_windows(dest)], EXPORT_TIMEOUT, wait=True)
            try:
                data = json.loads(dest.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError) as e:
                raise CorpusError(f"Corpus' music export couldn't be read ({e})") from e
        if not isinstance(data, list):
            raise CorpusError("Corpus' music export isn't a list of songs")
        return data

    def play(self, paths: list[str], mode: str = "play", shuffle: bool = False) -> None:
        """mode "play" replaces the play list, "queue" adds at the end, "next" after this one."""
        args = {"play": ["--play"] + (["--shuffle"] if shuffle else []), "queue": ["--queue"]}
        self._run([*args.get(mode, ["--next"]), *fit(paths)], PLAY_TIMEOUT, wait=False)

    def _find(self) -> str:
        try:
            exe = self._exe or find_corpus()
        except ValueError as e:
            raise CorpusError(str(e)) from e
        if not exe:
            raise CorpusError("I don't know where Corpus is (set corpus under [music] in config)")
        return exe

    def _run(self, args: list[str], timeout: float, wait: bool) -> None:
        exe = self._find()
        try:
            proc = subprocess.Popen(
                [exe, *args],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as e:
            raise CorpusError(f"Corpus didn't start ({e})") from e
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if not wait:
                return  # Corpus wasn't running, so this process became it: leave it playing
            proc.kill()
            proc.wait()
            raise CorpusError("Corpus took too long to answer") from None
        if code == 2:
            raise CorpusError("Corpus didn't understand the command")
        if code != 0:
            raise CorpusError(f"Corpus couldn't do it (exit code {code})")


class FakeCorpus:
    """Corpus in memory, for tests and PRADA_FAKE_LLM. Given a `FakeBridge`, a play shows up
    there as Corpus playing the first file, like the real one in Windows' media controls.
    `fail` = every command fails; `silent` = commands succeed but nothing plays."""

    def __init__(self, songs: list[dict] | None = None, bridge=None, fail=False, silent=False):
        self.songs = FAKE_SONGS if songs is None else songs
        self.bridge = bridge
        self.fail = fail
        self.silent = silent
        self.calls: list[tuple] = []

    def available(self) -> bool:
        return True

    def export(self) -> list[dict]:
        self.calls.append(("export",))
        if self.fail:
            raise CorpusError("Corpus couldn't do it (exit code 1)")
        return [dict(s) for s in self.songs]

    def play(self, paths: list[str], mode: str = "play", shuffle: bool = False) -> None:
        self.calls.append((mode, list(paths), shuffle))
        if self.fail:
            raise CorpusError("Corpus couldn't do it (exit code 1)")
        if self.bridge is None or self.silent:
            return
        from prada.music.bridge import Player

        corpus = next((p for p in self.bridge.players if p.name == "Corpus"), None)
        if corpus is None:
            corpus = Player("Corpus.Corpus")
            self.bridge.players.append(corpus)
        if mode != "play" and corpus.playing:
            return
        song = next((s for s in self.songs if s["path"] == paths[0]), {})
        corpus.title = song.get("title", "")
        corpus.artist = song.get("artist", "")
        corpus.album = song.get("album", "")
        corpus.duration = song.get("duration", 0.0)
        corpus.state, corpus.position = "playing", 0.0
        self.bridge.current = corpus.app


def _song(path, title, artist, album, genre, duration):
    return {
        "path": "E:\\Music\\" + path,
        "title": title,
        "artist": artist,
        "album": album,
        "genre": genre,
        "tags": [],
        "duration": duration,
    }


# fmt: off
FAKE_SONGS = [
    _song("Muse\\Origin\\01 New Born.flac", "New Born", "Muse", "Origin of Symmetry", "Alternative Rock", 363.0),
    _song("Muse\\Origin\\03 Space Dementia.flac", "Space Dementia", "Muse", "Origin of Symmetry", "Alternative Rock", 381.0),
    _song("Muse\\Black Holes\\01 Take a Bow.flac", "Take a Bow", "Muse", "Black Holes and Revelations", "Alternative Rock", 275.0),
    _song("TK\\Fantastic Magic\\01 Unravel.flac", "Unravel", "TK from Ling tosite sigure", "Fantastic Magic", "J-Rock", 238.0),
    _song("Unison\\Catcher\\02 Sugar Song.mp3", "Sugar Song and Bitter Step", "UNISON SQUARE GARDEN", "Catcher In The Spy", "J-Rock", 246.0),
    _song("Bach\\Suites\\01 Prelude.flac", "Cello Suite No. 1: Prelude", "Yo-Yo Ma", "Bach: Cello Suites", "Classical", 150.0),
    _song("Jobim\\Wave\\01 Wave.mp3", "Wave", "Antonio Carlos Jobim", "Wave", "Bossa Nova", 175.0),
]
# fmt: on
