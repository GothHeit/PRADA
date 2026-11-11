import sqlite3
import sys
import textwrap

import pytest

from prada import honesty
from prada.eval import eval_modules
from prada.module import Request
from prada.music.bridge import (
    BridgeError,
    FakeBridge,
    Player,
    Status,
    WindowsBridge,
    parse_status,
)
from prada.music.module import MusicModule, command, now_playing, targets
from prada.music.playlog import MIGRATIONS, POLL, PlayLog, Watcher
from prada.node import Node
from prada.store import open_store
from tutor.module import TutorModule


@pytest.mark.parametrize(
    "text, verb, app",
    [
        ("pause", "pause", None),
        ("Pause the music, please.", "pause", None),
        ("stop the music", "pause", None),
        ("Prada, pause Spotify", "pause", "spotify"),
        ("pausa a música no Corpus", "pause", "corpus"),
        ("play", "resume", None),
        ("resume the song", "resume", None),
        ("volta a tocar", "resume", None),
        ("continua a música", "resume", None),
        ("next", "next", None),
        ("Hey Prada, next song please", "next", None),
        ("skip this one", "next", None),
        ("próxima", "next", None),
        ("pula essa música", "next", None),
        ("previous", "previous", None),
        ("go back a song", "previous", None),
        ("música anterior", "previous", None),
        ("What's playing?", "now", None),
        ("what song is this", "now", None),
        ("what am I listening to right now?", "now", None),
        ("who sings this?", "now", None),
        ("who is singing?", "now", None),
        ("who sings", "now", None),
        ("what's on Spotify?", "now", "spotify"),
        ("quem tá cantando?", "now", None),
        ("quem canta essa música", "now", None),
        ("que música é essa?", "now", None),
        ("qual é a música que está tocando", "now", None),
        ("o que tá tocando no spotify?", "now", "spotify"),
    ],
)
def test_commands(text, verb, app):
    assert command(text) == (verb, app)


@pytest.mark.parametrize(
    "text",
    [
        "stop",  # also "stop talking" or a no to her question
        "continue",
        "go back",
        "play Muse Space Dementia",  # Phase 3
        "next week I travel to Lisbon",
        "what is this?",
        "pause for a second, I need to think",
        "can you note that the next song is great",
    ],
)
def test_not_commands(text):
    assert command(text) is None


def spotify(state="playing", title="Space Dementia", position=0.0):
    return Player("Spotify.exe", title, "Muse", "Origin of Symmetry", state, position, 381.0)


def corpus(state="paused"):
    return Player("Corpus.Corpus", "Unravel", "TK", "Fantastic Magic", state, 0.0, 238.0)


def test_targets_prefer_music_players_and_pause_everything_playing():
    video = Player("Chrome", "A video", "", "", "playing")
    status = Status([video, spotify(), corpus("playing")], current="Chrome")
    assert [p.name for p in targets("pause", status, None)] == ["Spotify", "Corpus", "Chrome"]
    assert [p.name for p in targets("next", status, None)] == ["Spotify"]
    assert targets("resume", status, None) == "It's already playing: Space Dementia by Muse."


def test_targets_without_anything_playing():
    status = Status([spotify("paused"), corpus("paused")], current="Corpus.Corpus")
    assert targets("pause", status, None) == "Nothing's playing right now."
    assert [p.name for p in targets("resume", status, None)] == ["Corpus"]
    assert [p.name for p in targets("next", status, None)] == ["Corpus"]
    assert targets("pause", status, "spotify") == "Spotify isn't playing right now."
    assert targets("pause", Status([corpus()]), "spotify") == "Spotify isn't open."
    assert targets("next", Status([]), None) == "No music player is open right now."


def test_now_playing_comes_from_the_players_only():
    assert now_playing(Status([spotify()]), None) == "That's Space Dementia by Muse, on Spotify."
    paused = Status([spotify("paused")], current="Spotify.exe")
    assert "paused on Space Dementia" in now_playing(paused, None)
    assert now_playing(Status([]), None) == "Nothing's playing right now."


@pytest.fixture
def music_node(tmp_path):
    bridge = FakeBridge()
    node = Node(
        modules=[TutorModule(), MusicModule(bridge=bridge, verify=0.2)],
        db_path=tmp_path / "prada.db",
        background=False,
    )
    return node, bridge


def test_pause_is_pressed_checked_and_logged(music_node, tmp_path):
    node, bridge = music_node
    reply = node.handle(Request(text="pause", session="s"))
    assert reply.module == "music"
    assert reply.text == "Paused Spotify."
    assert bridge.pressed == [("Spotify.exe", "pause")]
    assert [o.ok for o in reply.outcomes] == [True]
    db = sqlite3.connect(tmp_path / "prada.db")
    db.row_factory = sqlite3.Row
    row = db.execute("SELECT kind, status FROM actionlog_entry").fetchone()
    assert (row["kind"], row["status"]) == ("music.pause", "done")


def test_skip_says_the_new_track(music_node):
    node, _ = music_node
    reply = node.handle(Request(text="next song", session="s"))
    assert reply.text == "Skipped. Now it's Unravel by TK from Ling tosite sigure."
    reply = node.handle(Request(text="what's playing?", session="s"))
    assert reply.text == "That's Unravel by TK from Ling tosite sigure, on Spotify."


def test_a_press_that_changes_nothing_is_never_claimed(tmp_path):
    node = Node(
        modules=[MusicModule(bridge=FakeBridge(stuck=True), verify=0.2)],
        db_path=tmp_path / "prada.db",
        background=False,
    )
    reply = node.handle(Request(text="pause", session="s"))
    assert not reply.outcomes[0].ok
    assert reply.text.startswith("I tried to pause Spotify, but it didn't work")
    assert not honesty.claims_action(reply.text)


class Skipping(FakeBridge):
    """After a press, the first look shows the player mid-change: no title, "changing"."""

    def control(self, app, verb):
        self.mid_change = True
        return super().control(app, verb)

    def status(self):
        status = super().status()
        if getattr(self, "mid_change", False):
            self.mid_change = False
            for p in status.players:
                p.title, p.artist, p.album, p.state = "", "", "", "changing"
        return status


@pytest.mark.parametrize("text", ["next", "previous"])
def test_a_skip_waits_past_the_untitled_moment(text):
    bridge = Skipping([spotify(position=95.0)])
    node = Node(
        modules=[MusicModule(bridge=bridge, verify=1.0)], background=False, db_path=":memory:"
    )
    reply = node.handle(Request(text=text, session="s"))
    assert reply.outcomes[0].ok
    assert "untitled" not in reply.text
    assert reply.text.endswith("by TK from Ling tosite sigure.") or "Sugar Song" in reply.text


def test_a_changing_player_is_not_yet_paused():
    from prada.music.module import PauseExecutor

    assert not PauseExecutor(MusicModule()).changed(spotify(), spotify("changing"))
    assert PauseExecutor(MusicModule()).changed(spotify(), spotify("paused"))


def test_previous_that_restarts_the_track_counts():
    bridge = FakeBridge([spotify(position=95.0)])

    def restart(app, verb):
        bridge.players[0].position = 0.5
        return True

    bridge.control = restart
    module = MusicModule(bridge=bridge, verify=0.2)
    node = Node(modules=[module], background=False, db_path=":memory:")
    reply = node.handle(Request(text="previous", session="s"))
    assert reply.text == "Back to the start of Space Dementia by Muse."


def test_unreachable_players_are_said_plainly(tmp_path):
    class Down(FakeBridge):
        def status(self):
            raise BridgeError("Windows' media controls didn't answer")

    node = Node(modules=[MusicModule(bridge=Down())], background=False, db_path=":memory:")
    reply = node.handle(Request(text="pause", session="s"))
    assert reply.text == "I can't reach your music players: Windows' media controls didn't answer."
    assert reply.error is None


def test_explicit_command_wins_mid_lesson(music_node):
    node, bridge = music_node
    node.handle(Request(text="/tutor Je suis fatigué", session="s"))
    reply = node.handle(Request(text="pause", session="s"))  # also a French word
    assert reply.module == "music"
    assert bridge.pressed == [("Spotify.exe", "pause")]


def test_eval_never_reaches_real_players():
    music = next(m for m in eval_modules() if m.name == "music")
    assert isinstance(music.bridge(), FakeBridge)


def test_start_skips_the_watcher_for_fakes(monkeypatch, tmp_path):
    monkeypatch.setenv("PRADA_FAKE_LLM", "1")
    module = MusicModule()
    module.start(tmp_path / "prada.db")
    assert module._log is None


# The play log


@pytest.fixture
def log():
    db = open_store("music", MIGRATIONS, ":memory:")
    return PlayLog(db), db


def rows(db):
    return [dict(r) for r in db.execute("SELECT * FROM music_play ORDER BY id")]


def test_play_log_counts_listening_and_endings(log):
    playlog, db = log
    playlog.see(Status([spotify()]), 0)
    playlog.see(Status([spotify(position=2)]), 2)
    playlog.see(Status([spotify("paused", position=2)]), 4)
    playlog.see(Status([spotify("paused", position=2)]), 100)  # paused time isn't listening
    playlog.see(Status([spotify(position=2)]), 102)
    playlog.see(Status([spotify(title="Hysteria", position=0)]), 104)  # skipped early
    (first, second) = rows(db)
    assert (first["title"], first["app"], first["ended"]) == (
        "Space Dementia",
        "Spotify",
        "skipped",
    )
    assert first["seconds"] == 6.0
    assert first["ended_at"] == "1970-01-01T00:01:44Z"
    assert (second["title"], second["ended"], second["by_her"]) == ("Hysteria", "playing", 0)


def test_play_log_finished_and_closed(log):
    playlog, db = log
    playlog.see(Status([spotify(position=370)]), 0)
    playlog.see(Status([spotify(position=376)]), 2)
    playlog.see(Status([]), 4)  # Spotify closed at the end of the track
    assert rows(db)[0]["ended"] == "finished"


def test_play_log_ignores_paused_tracks_and_other_apps(log):
    playlog, db = log
    playlog.see(Status([spotify("paused"), Player("Chrome", "A video", state="playing")]), 0)
    assert rows(db) == []


def test_play_log_marks_tracks_she_started(log):
    playlog, db = log
    playlog.see(Status([spotify()]), 0)
    playlog.mark("Spotify.exe", 3)
    playlog.see(Status([spotify(title="Hysteria")]), 4)
    assert [r["by_her"] for r in rows(db)] == [0, 1]


def test_play_log_closes_rows_left_by_an_earlier_run(tmp_path):
    db = open_store("music", MIGRATIONS, tmp_path / "prada.db")
    PlayLog(db).see(Status([spotify()]), 0)
    PlayLog(db)
    assert rows(db)[0]["ended"] == "stopped"


def test_watcher_backs_off_when_the_players_are_unreachable(log, capsys):
    playlog, db = log

    def down():
        raise BridgeError("no Windows here")

    watcher = Watcher(down, playlog, clock=lambda: 0)
    assert watcher.tick() > POLL
    watcher.tick()
    assert capsys.readouterr().err.count("no Windows here") == 1  # said once, not every time
    assert Watcher(FakeBridge().status, playlog, clock=lambda: 0).tick() == POLL
    assert rows(db)[0]["title"] == "Space Dementia"


# The PowerShell side, played by a small Python script speaking the same JSON lines.

FAKE_PS = textwrap.dedent(
    """\
    import json, sys
    mode = {mode!r}
    if mode == "fatal":
        print(json.dumps({{"fatal": "no SMTC"}}), flush=True)
        sys.exit(1)
    print("\\ufeff" + json.dumps({{"ready": True}}), flush=True)
    for line in sys.stdin:
        req = json.loads(line)
        if mode == "silent":
            continue
        if req["cmd"] == "status":
            out = {{"ok": True, "current": "Spotify.exe", "players": [
                {{"app": "Spotify.exe", "title": "Señorita", "artist": "Muse",
                  "state": "playing", "position": 12.5, "duration": 200}}]}}
        else:
            out = {{"ok": req["app"] == "Spotify.exe"}}
        out["id"] = req["id"]
        print(json.dumps(out, ensure_ascii=False), flush=True)
    """
)


def fake_powershell(tmp_path, mode="ok"):
    script = tmp_path / f"ps-{mode}.py"
    script.write_text(FAKE_PS.format(mode=mode), encoding="utf-8")
    exe = tmp_path / f"powershell-{mode}"
    exe.write_text(f"#!/bin/sh\nexec {sys.executable} {script}\n")
    exe.chmod(0o755)
    return str(exe)


@pytest.mark.skipif(sys.platform == "win32", reason="the stand-in is a shell script")
def test_windows_bridge_speaks_json_lines(tmp_path):
    bridge = WindowsBridge(fake_powershell(tmp_path))
    try:
        status = bridge.status()
        assert status.current == "Spotify.exe"
        assert status.players[0].title == "Señorita"
        assert status.players[0].position == 12.5
        assert bridge.control("Spotify.exe", "pause") is True
        assert bridge.control("Corpus.Corpus", "next") is False
        with pytest.raises(ValueError):
            bridge.control("Spotify.exe", "delete")
    finally:
        bridge.close()


@pytest.mark.skipif(sys.platform == "win32", reason="the stand-in is a shell script")
@pytest.mark.parametrize("mode, message", [("fatal", "no SMTC"), ("silent", "didn't answer")])
def test_windows_bridge_failures(tmp_path, mode, message):
    bridge = WindowsBridge(fake_powershell(tmp_path, mode), timeout=1.0)
    with pytest.raises(BridgeError, match=message):
        bridge.status()
    bridge.close()


@pytest.mark.skipif(sys.platform == "win32", reason="the stand-in is a shell script")
def test_windows_bridge_reaps_a_stopped_powershell(tmp_path):
    bridge = WindowsBridge(fake_powershell(tmp_path))
    bridge.status()
    proc = bridge._proc
    bridge.close()
    assert proc.returncode is not None and proc.stdout.closed


def test_windows_bridge_without_windows(monkeypatch):
    monkeypatch.setattr("prada.music.bridge.find_powershell", lambda: None)
    with pytest.raises(BridgeError, match="Windows"):
        WindowsBridge().status()


def test_parse_status_skips_junk():
    status = parse_status({"players": [{"title": "no app"}, "junk", {"app": "X"}], "current": ""})
    assert [p.app for p in status.players] == ["X"]
    assert status.current is None


def test_bridge_script_ships_with_the_package():
    from prada.music.bridge import script

    text = script()
    assert "GlobalSystemMediaTransportControlsSessionManager" in text
