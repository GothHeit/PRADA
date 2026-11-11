import json
import random
import sys
import textwrap
import time

import pytest

from prada import config
from prada.cli import main
from prada.music import corpus as corpus_mod
from prada.music.bridge import FakeBridge
from prada.music.catalog import MIGRATIONS, STORE, Catalog, Track, key, parse_export
from prada.music.corpus import (
    FAKE_SONGS,
    CorpusCli,
    CorpusError,
    FakeCorpus,
    fit,
    to_linux,
)
from prada.music.library import Choice, Found, Library, find, mood_of
from prada.store import open_store

# The catalog


def song(path, title, artist="", album="", genre="", duration=200.0):
    return {"path": path, "title": title, "artist": artist, "album": album, "genre": genre,
            "duration": duration}  # fmt: skip


@pytest.fixture
def catalog():
    cat = Catalog(open_store(STORE, MIGRATIONS, ":memory:"))
    cat.replace(parse_export(FAKE_SONGS), now=1000.0)
    return cat


def test_parse_export_fills_gaps_and_skips_junk():
    tracks = parse_export(
        [
            {"path": "E:\\Music\\X\\03 No Tags.mp3"},
            {"path": "E:\\a.flac", "title": "A", "genre": ["Rock", "Pop"], "duration": "x"},
            {"title": "no path"},
            "junk",
            {"path": "E:\\a.flac", "title": "A again"},
        ]
    )
    assert [t.title for t in tracks] == ["03 No Tags", "A again"]
    assert parse_export([FAKE_SONGS[0]])[0].duration == 363.0
    assert parse_export([song("p", "t", genre=["Rock", "Pop"])])[0].genre == "Rock; Pop"


def test_catalog_survives_a_reopen(tmp_path):
    path = tmp_path / "prada.db"
    Catalog(open_store(STORE, MIGRATIONS, path)).replace([Track("p", "T", "A")], 5.0)
    again = Catalog(open_store(STORE, MIGRATIONS, path))
    assert len(again) == 1 and again.imported_at == 5.0


def test_catalog_sync_sees_another_process(tmp_path):
    path = tmp_path / "prada.db"
    node = Catalog(open_store(STORE, MIGRATIONS, path))
    Catalog(open_store(STORE, MIGRATIONS, path)).replace([Track("p", "T", "A")], 5.0)
    assert len(node) == 0
    node.sync()
    assert len(node) == 1


@pytest.mark.parametrize(
    "query, kind, name",
    [
        ("space dementia", "song", "Space Dementia"),
        ("muse space dementia", "song", "Space Dementia"),  # artist first
        ("Space Dementia Muse", "song", "Space Dementia"),
        ("space dementa", "song", "Space Dementia"),  # misheard
        ("newborn", "song", "New Born"),
        ("origin of symmetry", "album", "Origin of Symmetry"),
        ("muse", "artist", "Muse"),
        ("unison square garden", "artist", "UNISON SQUARE GARDEN"),
        ("cello suite no 1 prelude", "song", "Cello Suite No. 1: Prelude"),
    ],
)
def test_catalog_finds_by_name(catalog, query, kind, name):
    best = catalog.find(query)[0]
    assert (best.kind, best.name) == (kind, name)


def test_catalog_keeps_far_names_out(catalog):
    assert catalog.find("bohemian rhapsody") == []
    assert catalog.find("chess") == []
    assert [h.kind for h in catalog.find("muse", kind="song")] == []


def test_catalog_by_genre_matches_whole_words(catalog):
    assert {t.title for t in catalog.by_genre(["rock"])} >= {"Unravel", "Space Dementia"}
    assert catalog.by_genre(["roc"]) == []


def test_key_drops_case_accents_and_the():
    assert key("The Beatles") == "beatles"
    assert key("Évidências!") == "evidencias"


# Finding what to play


def test_find_a_song_plays_just_that_song(catalog):
    found = find(catalog, "space dementia")
    assert isinstance(found, Found)
    assert (found.label, found.kind, found.shuffle) == ("Space Dementia by Muse", "song", False)
    assert [t.title for t in found.tracks] == ["Space Dementia"]


def test_find_an_album_plays_in_order(catalog):
    found = find(catalog, "origin of symmetry")
    assert [t.title for t in found.tracks] == ["New Born", "Space Dementia"]
    assert found.shuffle is False
    assert find(catalog, "origin of symmetry", shuffle=True).shuffle is True


def test_find_an_artist_shuffles(catalog):
    found = find(catalog, "muse")
    assert (found.kind, found.shuffle, len(found.tracks)) == ("artist", True, 3)
    assert find(catalog, "muse", shuffle=False).shuffle is False


def test_a_single_is_the_song(catalog):
    found = find(catalog, "wave")  # the album "Wave" has only this song
    assert (found.kind, found.label) == ("song", "Wave by Antonio Carlos Jobim")


def test_a_title_track_asks_which(catalog):
    songs = [
        song("E:\\M\\Dark\\01 Intro.flac", "Intro", "Band", "Dark Side"),
        song("E:\\M\\Dark\\02 Dark Side.flac", "Dark Side", "Band", "Dark Side"),
    ]
    catalog.replace(parse_export(songs), 2000.0)
    choice = find(catalog, "dark side")
    assert isinstance(choice, Choice)
    assert choice.question() == "Dark Side: the song or the album?"
    assert choice.answer("the album").kind == "album"
    assert choice.answer("a música").kind == "song"
    assert choice.answer("o disco inteiro").kind == "album"
    assert choice.answer("never mind") is None
    assert find(catalog, "dark side", kind="album").kind == "album"


def test_find_a_genre_or_a_mood(catalog):
    rng = random.Random(1)
    rock = find(catalog, "rock", rng=rng)
    assert rock.kind == "genre" and rock.shuffle and len(rock.tracks) == 5
    calm = find(catalog, "", mood="calm", rng=rng)
    assert {t.title for t in calm.tracks} == {"Wave", "Cello Suite No. 1: Prelude"}
    assert calm.label == "something calm"
    assert len(find(catalog, "", mood="any").tracks) == len(FAKE_SONGS)


def test_find_says_what_is_missing(catalog):
    assert find(catalog, "Bohemian Rhapsody") == (
        "I couldn't find Bohemian Rhapsody in your Corpus library."
    )
    assert find(catalog, "", mood="sad") == "None of your Corpus songs has a genre that sounds sad."
    empty = Catalog(open_store(STORE, MIGRATIONS, ":memory:"))
    assert "haven't read it yet" in find(empty, "", mood="any")


@pytest.mark.parametrize(
    "text, mood",
    [
        ("calm", "calm"),
        ("pra relaxar", "calm"),
        ("calma", "calm"),
        ("to study to", "focus"),
        ("bem agitada", "energetic"),
        ("muse", None),
        ("", None),
    ],
)
def test_mood_of(text, mood):
    assert mood_of(text) == mood


# Keeping the catalog fresh


def test_library_reads_a_fake_corpus_at_once():
    library = Library(FakeCorpus())
    assert len(library.catalog()) == len(FAKE_SONGS)
    library.catalog()
    assert library.corpus.calls == [("export",)]  # fresh: not read again


def test_library_refreshes_a_stale_catalog(monkeypatch):
    library = Library(FakeCorpus())
    library.catalog()
    monkeypatch.setattr(time, "time", lambda: 1e12)
    library.catalog()
    assert library.corpus.calls == [("export",), ("export",)]


def test_library_reads_a_real_corpus_in_the_background(capsys):
    class Slow(CorpusCli):
        def export(self):
            raise CorpusError("I don't know where Corpus is")

    library = Library(Slow("/nowhere/Corpus.exe"), path=":memory:")
    assert len(library.catalog()) == 0  # never waits
    for _ in range(100):
        if not library._reading:
            break
        time.sleep(0.01)
    assert "Corpus' catalog not read (I don't know where Corpus is)" in capsys.readouterr().err
    library.catalog()  # a failure waits RETRY before the next try
    assert not library._reading


def test_music_import_cli(monkeypatch, capsys):
    monkeypatch.setenv("PRADA_FAKE_LLM", "1")
    assert main(["music", "import"]) == 0
    assert capsys.readouterr().out == f"Read {len(FAKE_SONGS)} songs from Corpus.\n"


def test_music_import_cli_without_corpus(monkeypatch, capsys):
    monkeypatch.delenv("PRADA_FAKE_LLM", raising=False)
    monkeypatch.setattr(corpus_mod.shutil, "which", lambda name: None)
    assert main(["music", "import"]) == 1
    assert "set corpus under [music]" in capsys.readouterr().err


# Corpus' command line


@pytest.fixture
def fake_exe(tmp_path, monkeypatch):
    """A stand-in for Corpus.exe: logs its arguments, writes the export, exits with
    $CODE; $HANG keeps it running like a Corpus that just started."""
    log = tmp_path / "calls.jsonl"
    exe = tmp_path / "Corpus.exe"
    exe.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import json, os, sys, time
            args = sys.argv[1:]
            with open({str(log)!r}, "a") as f:
                f.write(json.dumps(args) + "\\n")
            if args[:1] == ["--export-music"]:
                with open(args[1], "w", encoding="utf-8-sig") as f:
                    f.write(os.environ.get("EXPORT", "[]"))
            if os.environ.get("HANG"):
                time.sleep(30)
            sys.exit(int(os.environ.get("CODE", "0")))
            """
        )
    )
    exe.chmod(0o755)
    monkeypatch.setattr(corpus_mod, "to_windows", str)
    wintemp = tmp_path / "wintemp"
    wintemp.mkdir()
    monkeypatch.setattr(corpus_mod, "windows_temp", lambda: wintemp)

    def calls():
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    return str(exe), calls


def test_cli_exports_the_catalog(fake_exe, monkeypatch):
    exe, calls = fake_exe
    monkeypatch.setenv("EXPORT", json.dumps(FAKE_SONGS[:2]))
    assert CorpusCli(exe).export() == FAKE_SONGS[:2]
    assert calls()[0][0] == "--export-music"


def test_cli_play_queue_and_next(fake_exe):
    exe, calls = fake_exe
    cli = CorpusCli(exe)
    cli.play(["E:\\a.flac", "E:\\b.flac"], "play", shuffle=True)
    cli.play(["E:\\a.flac"], "play")
    cli.play(["E:\\c.flac"], "queue")
    cli.play(["E:\\d.flac"], "next")
    assert calls() == [
        ["--play", "--shuffle", "E:\\a.flac", "E:\\b.flac"],
        ["--play", "E:\\a.flac"],
        ["--queue", "E:\\c.flac"],
        ["--next", "E:\\d.flac"],
    ]


@pytest.mark.parametrize(
    "code, message", [("1", "exit code 1"), ("2", "didn't understand the command")]
)
def test_cli_exit_codes(fake_exe, monkeypatch, code, message):
    exe, _ = fake_exe
    monkeypatch.setenv("CODE", code)
    with pytest.raises(CorpusError, match=message):
        CorpusCli(exe).play(["E:\\a.flac"])


def test_cli_bad_export(fake_exe, monkeypatch):
    exe, _ = fake_exe
    monkeypatch.setenv("EXPORT", '{"not": "a list"}')
    with pytest.raises(CorpusError, match="isn't a list"):
        CorpusCli(exe).export()
    monkeypatch.setenv("EXPORT", "[{broken")
    with pytest.raises(CorpusError, match="couldn't be read"):
        CorpusCli(exe).export()


def test_cli_leaves_a_corpus_it_started_running(fake_exe, monkeypatch):
    exe, calls = fake_exe
    monkeypatch.setenv("HANG", "1")
    monkeypatch.setattr(corpus_mod, "PLAY_TIMEOUT", 0.3)
    monkeypatch.setattr(corpus_mod, "EXPORT_TIMEOUT", 0.3)
    CorpusCli(exe).play(["E:\\a.flac"])  # returns: Corpus started and keeps playing
    with pytest.raises(CorpusError, match="too long"):
        CorpusCli(exe).export()


def test_cli_without_corpus(monkeypatch):
    monkeypatch.setattr(corpus_mod.shutil, "which", lambda name: None)
    with pytest.raises(CorpusError, match=r"set corpus under \[music\]"):
        CorpusCli().play(["E:\\a.flac"])
    with pytest.raises(CorpusError, match="didn't start"):
        CorpusCli("/nowhere/Corpus.exe").play(["E:\\a.flac"])


def test_corpus_path_from_config(tmp_path, monkeypatch):
    (tmp_path / "config.toml").write_text(
        '[music]\ncorpus = "C:\\\\Apps\\\\Corpus\\\\Corpus.exe"\n'
    )
    if sys.platform != "win32":
        assert corpus_mod.find_corpus() == "/mnt/c/Apps/Corpus/Corpus.exe"
    (tmp_path / "config.toml").write_text("[music]\ncorpse = 1\n")
    with pytest.raises(ValueError, match="unknown"):
        config.music_settings()
    with pytest.raises(CorpusError, match="unknown"):
        CorpusCli().play(["E:\\a.flac"])


def test_paths_and_command_line_size(monkeypatch):
    if sys.platform != "win32":
        assert to_linux("D:\\Music\\a b.flac") == "/mnt/d/Music/a b.flac"
        assert to_linux("/usr/bin/x") == "/usr/bin/x"
        assert corpus_mod.to_windows("/mnt/e/Music/x.flac") == "E:\\Music\\x.flac"
    paths = ["E:\\" + "x" * 997] * 40
    assert len(fit(paths)) == corpus_mod.MAX_ARGS // 1003  # each path + quotes and a space


def test_fake_corpus_shows_up_in_the_media_controls():
    bridge = FakeBridge()
    fake = FakeCorpus(bridge=bridge)
    fake.play([FAKE_SONGS[1]["path"]])
    corpus = bridge.status().get("Corpus.Corpus")
    assert (corpus.title, corpus.state) == ("Space Dementia", "playing")
    fake.play([FAKE_SONGS[0]["path"]], "queue")  # queued: what plays doesn't change
    assert bridge.status().get("Corpus.Corpus").title == "Space Dementia"


# Playing from Corpus through the node


@pytest.fixture
def corpus_node(tmp_path):
    from prada.music.module import MusicModule
    from prada.music.spotify import FakeSpotify
    from prada.node import Node

    bridge, spotify = FakeBridge(), FakeSpotify()
    fake = FakeCorpus(bridge=bridge)
    music = MusicModule(
        bridge=bridge, spotify=spotify, corpus=fake, play_verify=0.3, step=0.01, verify=0.1
    )
    node = Node(modules=[music], db_path=tmp_path / "prada.db", background=False)
    return node, bridge, fake, spotify


def say(node, text):
    from prada.module import Request

    return node.handle(Request(text=text, session="s"))


def test_a_song_in_corpus_plays_there_and_spotify_pauses(corpus_node):
    node, bridge, fake, spotify = corpus_node
    reply = say(node, "Prada, play Muse Space Dementia")
    assert reply.text == "Playing Space Dementia by Muse in Corpus. I paused Spotify."
    assert fake.calls[-1] == ("play", [FAKE_SONGS[1]["path"]], False)
    assert bridge.status().get("Spotify.exe").state == "paused"
    assert spotify.searched == []
    outcome = reply.outcomes[0]
    assert outcome.ok and outcome.action["kind"] == "music.play_corpus"


@pytest.mark.parametrize(
    "text, call, said",
    [
        ("toca o álbum Origin of Symmetry", ("play", 2, False), "Origin of Symmetry by Muse"),
        ("shuffle Muse", ("play", 3, True), "Muse in Corpus on shuffle, 3 songs"),
        ("toca um rock", ("play", 5, True), "some rock in Corpus on shuffle"),
        ("play something calm", ("play", 2, True), "something calm in Corpus on shuffle"),
        ("add Unravel to the queue", ("queue", 1, False), "Added Unravel by TK"),
        ("coloca Unravel na fila", ("queue", 1, False), "Corpus' queue"),
        ("play Take a Bow next", ("next", 1, False), "Take a Bow by Muse plays next"),
    ],
)
def test_corpus_requests(corpus_node, text, call, said):
    node, _, fake, _ = corpus_node
    reply = say(node, text)
    mode, paths, shuffle = fake.calls[-1]
    assert (mode, len(paths), shuffle) == call
    assert said in reply.text and reply.outcomes[0].ok


def test_not_in_corpus_goes_to_spotify(corpus_node):
    node, bridge, fake, spotify = corpus_node
    say(node, "play Muse Space Dementia")  # Corpus plays
    reply = say(node, "play Hysteria")
    assert reply.text.startswith("Playing Hysteria by Muse on ")
    assert reply.text.endswith(" I paused Corpus.")
    assert bridge.status().get("Corpus.Corpus").state == "paused"  # one player at a time


def test_a_corpus_pause_that_didnt_take_is_never_claimed(corpus_node):
    node, bridge, _, _ = corpus_node
    say(node, "play Muse Space Dementia")  # Corpus plays
    bridge.stuck = True
    reply = say(node, "play Hysteria")
    assert reply.text.startswith("Playing Hysteria by Muse on ")
    assert "paused" not in reply.text


def test_a_named_app_decides(corpus_node):
    node, _, fake, spotify = corpus_node
    reply = say(node, "play Space Dementia on Spotify")
    assert "in Corpus" not in reply.text and spotify.searched
    assert not [c for c in fake.calls if c[0] != "export"]
    assert say(node, "play Hysteria no Corpus").text == (
        "I couldn't find hysteria in your Corpus library."
    )


def test_only_corpus_queues(corpus_node):
    node, _, _, spotify = corpus_node
    reply = say(node, "play Hysteria next")
    assert reply.text.endswith("I can only queue songs in Corpus.")
    assert spotify.searched == []


def test_a_title_track_question_and_answer(tmp_path):
    from prada.music.module import MusicModule
    from prada.node import Node
    from tutor.module import TutorModule

    songs = [
        song("E:\\M\\Dark\\01 Intro.flac", "Intro", "Band", "Dark Side"),
        song("E:\\M\\Dark\\02 Dark Side.flac", "Dark Side", "Band", "Dark Side"),
    ]
    bridge = FakeBridge()
    fake = FakeCorpus(songs, bridge=bridge)
    music = MusicModule(bridge=bridge, corpus=fake, play_verify=0.3)
    node = Node(modules=[TutorModule(), music], db_path=tmp_path / "p.db", background=False)
    assert say(node, "play Dark Side").text == "Dark Side: the song or the album?"
    reply = say(node, "o álbum")  # no router: she asked
    assert reply.module == "music" and "Playing Dark Side by Band in Corpus, 2 songs" in reply.text
    assert fake.calls[-1][1] == [s["path"] for s in songs]


def test_a_play_corpus_never_shows_is_not_claimed():
    from prada.music.module import MusicModule
    from prada.node import Node

    bridge = FakeBridge()
    music = MusicModule(
        bridge=bridge, corpus=FakeCorpus(bridge=bridge, silent=True), play_verify=0.2
    )
    node = Node(modules=[music], db_path=":memory:", background=False)
    reply = say(node, "play Space Dementia")
    assert "Corpus didn't start playing" in reply.text and not reply.outcomes[0].ok
    assert "Playing" not in reply.text
    assert bridge.status().get("Spotify.exe").playing  # nothing paused for nothing


def test_corpus_failing_is_said(tmp_path):
    from prada.music.module import MusicModule
    from prada.node import Node

    bridge = FakeBridge()
    fake = FakeCorpus(bridge=bridge)
    music = MusicModule(bridge=bridge, corpus=fake, play_verify=0.2)
    music.library().catalog()  # read while Corpus still works
    fake.fail = True
    node = Node(modules=[music], db_path=":memory:", background=False)
    reply = say(node, "play Space Dementia")
    assert "exit code 1" in reply.text and not reply.outcomes[0].ok


def test_corpus_playing_another_song_is_not_claimed(tmp_path):
    from prada.music.module import MusicModule, PlayCorpus, PlayCorpusExecutor

    music = MusicModule(bridge=FakeBridge())
    action = PlayCorpus(label="x", paths=["p"], titles=["Space Dementia"])
    result = {"after": vars(FakeBridge().players[1]) | {"state": "playing"}, "paused": []}
    with pytest.raises(Exception, match="Unravel by TK from Ling tosite sigure instead"):
        PlayCorpusExecutor(music).check(action, result, None)


@pytest.mark.parametrize(
    "text, mode, shuffle, query",
    [
        ("add Unravel to the queue please", "queue", None, "unravel"),
        ("queue up Unravel", "queue", None, "unravel"),
        ("coloca Unravel na fila", "queue", None, "unravel"),
        ("play Take a Bow next", "next", None, "take a bow"),
        ("toca Take a Bow depois dessa", "next", None, "take a bow"),
        ("play Muse on shuffle please", "play", True, "muse"),
        ("shuffle Muse", "play", True, "muse"),
        ("toca o álbum Origin of Symmetry em ordem", "play", False, "origin of symmetry"),
    ],
)
def test_wish_reads_queue_and_shuffle(text, mode, shuffle, query):
    from prada.music.play import wish

    found = wish(text)
    assert (found.mode, found.shuffle, found.query) == (mode, shuffle, query)
    rewritten = text.split()[0] in ("add", "queue", "coloca", "shuffle")
    assert found.queued == rewritten
    assert (wish(text, strict=True) is None) == rewritten  # the module decides those


def test_fake_mode_plays_from_a_fake_corpus(monkeypatch):
    from prada.music.module import MusicModule

    monkeypatch.setenv("PRADA_FAKE_LLM", "1")
    assert isinstance(MusicModule().corpus(), FakeCorpus)
    monkeypatch.delenv("PRADA_FAKE_LLM")
    assert isinstance(MusicModule().corpus(), CorpusCli)


def test_eval_never_reaches_real_corpus():
    from prada.eval import eval_modules

    music = next(m for m in eval_modules() if m.name == "music")
    assert isinstance(music.corpus(), FakeCorpus)


@pytest.mark.parametrize(
    "text",
    [
        "embaralha as cartas",
        "shuffle the deck",
        "shuffle my notes",
        "queue up the downloads",
        "add a reminder to my queue",
        "coloca o nome dele na fila",
        "bota o carro na fila",
    ],
)
def test_queue_and_shuffle_words_alone_never_skip_the_router(corpus_node, text):
    node, _, _, _ = corpus_node
    assert not node.modules["music"].wants(text)


@pytest.mark.parametrize(
    "text", ["add Unravel to the queue", "coloca Unravel na fila", "shuffle Muse", "queue Unravel"]
)
def test_queue_and_shuffle_skip_the_router_when_corpus_has_it(corpus_node, text):
    node, _, _, _ = corpus_node
    music = node.modules["music"]
    assert music.wants(text) and music.accepts(text)
    assert "queue" in music.description


def test_a_pause_that_didnt_take_is_not_claimed(tmp_path):
    from prada.music.module import MusicModule
    from prada.node import Node

    class Deaf(FakeBridge):
        def control(self, app, verb):  # says ok, Spotify keeps playing
            return True if verb == "pause" and app == "Spotify.exe" else super().control(app, verb)

    bridge = Deaf()
    music = MusicModule(
        bridge=bridge, corpus=FakeCorpus(bridge=bridge), play_verify=0.3, verify=0.1
    )
    node = Node(modules=[music], db_path=":memory:", background=False)
    reply = say(node, "play Space Dementia")
    assert reply.text == "Playing Space Dementia by Muse in Corpus."


def test_an_empty_export_keeps_the_catalog(capsys):
    fake = FakeCorpus()
    library = Library(fake)
    assert len(library.catalog()) == len(FAKE_SONGS)
    fake.songs = []
    with pytest.raises(CorpusError, match="kept the 7"):
        library.refresh(wait=True)
    assert len(library.open()) == len(FAKE_SONGS)


def test_a_title_ending_in_next_keeps_it(tmp_path):
    from prada.music.module import MusicModule
    from prada.node import Node

    bridge = FakeBridge()
    songs = [song("E:\\M\\Who\\01 Baba ORiley.flac", "Baba O'Riley", "The Who", "Who's Next")]
    fake = FakeCorpus(songs, bridge=bridge)
    music = MusicModule(bridge=bridge, corpus=fake, play_verify=0.3, verify=0.1)
    node = Node(modules=[music], db_path=":memory:", background=False)
    reply = say(node, "play Who's Next")
    assert fake.calls[-1][0] == "play" and "Baba O'Riley" in reply.text


def test_the_export_goes_through_windows_temp_and_is_removed(fake_exe, tmp_path, monkeypatch):
    exe, calls = fake_exe
    monkeypatch.setenv("EXPORT", json.dumps(FAKE_SONGS[:1]))
    assert CorpusCli(exe).export() == FAKE_SONGS[:1]
    dest = calls()[0][1]
    assert dest.startswith(str(tmp_path / "wintemp")) and dest.endswith("music.json")
    assert list((tmp_path / "wintemp").iterdir()) == []  # cleaned up


def test_temp_candidates():
    echoed = "C:\\Users\\KAFKA~1\\AppData\\Local\\Temp;C:\\Users\\kafka\\AppData\\Local\r\n"
    assert corpus_mod.temp_candidates(echoed) == [
        "C:\\Users\\KAFKA~1\\AppData\\Local\\Temp",
        "C:\\Users\\kafka\\AppData\\Local\\Temp",
    ]
    assert corpus_mod.temp_candidates("%TEMP%;%LOCALAPPDATA%") == []


@pytest.mark.skipif(sys.platform == "win32", reason="asks cmd.exe from WSL")
def test_windows_temp_asks_cmd_and_skips_unreachable_folders(tmp_path, monkeypatch):
    cmd = tmp_path / "cmd.exe"
    cmd.write_text("#!/bin/sh\nprintf 'C:\\\\Short~1;C:\\\\Local\\r\\n'\n")
    cmd.chmod(0o755)
    (tmp_path / "c" / "Local" / "Temp").mkdir(parents=True)
    monkeypatch.setattr(corpus_mod.shutil, "which", lambda name: str(cmd))
    monkeypatch.setattr(
        corpus_mod, "to_linux", lambda p: str(tmp_path / "c" / p[3:].replace("\\", "/"))
    )
    assert corpus_mod.windows_temp() == tmp_path / "c" / "Local" / "Temp"
    (tmp_path / "c" / "Local" / "Temp").rmdir()
    with pytest.raises(CorpusError, match="isn't reachable"):
        corpus_mod.windows_temp()


@pytest.mark.skipif(sys.platform == "win32", reason="asks cmd.exe from WSL")
def test_windows_temp_without_windows(monkeypatch):
    monkeypatch.setattr(corpus_mod.shutil, "which", lambda name: None)
    monkeypatch.setattr(corpus_mod, "CMD", "/nowhere/cmd.exe")
    with pytest.raises(CorpusError, match="no cmd.exe"):
        corpus_mod.windows_temp()
