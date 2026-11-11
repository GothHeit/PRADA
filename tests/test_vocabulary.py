import time

import pytest

from prada import client
from prada.music.bridge import FakeBridge
from prada.music.corpus import FakeCorpus
from prada.music.module import MusicModule
from prada.music.playlog import _iso
from prada.node import VOCABULARY_AGE, Node
from prada.server import make_server
from prada.vocabulary import memory_names, names_in, unique


def test_names_in_finds_runs_of_capitalized_words():
    assert names_in("You call your sister Ana Clara and love Toru Kitajima.") == [
        "Ana Clara",
        "Toru Kitajima",
    ]
    assert names_in("Ana is your sister. You play the cello on Monday.") == ["Ana"]
    assert names_in("Your band is UNISON SQUARE GARDEN, not Muse") == [
        "UNISON SQUARE GARDEN",
        "Muse",
    ]
    assert names_in("Você mora em São Paulo com a Bia.") == ["São Paulo", "Bia"]
    assert names_in("you have no names here. I do.") == []


def test_unique_drops_blanks_and_repeats_in_any_case():
    assert unique(["Muse", " muse ", "", "Toru  Kitajima", "Ana"], 2) == ["Muse", "Toru Kitajima"]


def test_memory_names_newest_first_and_only_live_facts():
    facts = [
        {"fact": "You like Valve.", "superseded_by": 3},
        {"fact": "Your sister is Ana.", "superseded_by": None},
        {"fact": "You like Velvet and Ana's cooking.", "superseded_by": None},
    ]
    assert memory_names(facts) == ["Velvet", "Ana"]


def _add_fact(node, fact, superseded_by=None):
    db = node._mem().db
    with db:
        episode = db.execute("INSERT INTO memory_episode (started, ended) VALUES (0, 0)").lastrowid
        turn = db.execute(
            "INSERT INTO memory_turn (episode, ts, role, content) VALUES (?, 0, 'user', ?)",
            (episode, fact),
        ).lastrowid
        db.execute(
            "INSERT INTO memory_fact (episode, turn, ts, fact, quote, superseded_by, model)"
            " VALUES (?, ?, 0, ?, ?, ?, 'fake')",
            (episode, turn, fact, fact, superseded_by),
        )


@pytest.fixture
def node():
    music = MusicModule(bridge=FakeBridge(), corpus=FakeCorpus())
    music.library().refresh()
    clock = [time.time()]
    n = Node(modules=[music], db_path=":memory:", background=False, clock=lambda: clock[0])
    n.clock = clock
    return n


def test_node_gathers_her_name_memory_and_music(node):
    _add_fact(node, "Your cello teacher is Marta Souza.")
    db = node._db(node.modules["music"])
    now = time.time()
    with db:
        for title, artist, seconds in [
            ("Unravel (TV Size)", "TK from Ling tosite sigure", 900),
            ("Space Dementia - Remastered", "Muse", 300),
        ]:
            db.execute(
                "INSERT INTO music_play (app, title, artist, started_at, seconds)"
                " VALUES ('Spotify.exe', ?, ?, ?, ?)",
                (title, artist, _iso(now - 3600), seconds),
            )
        db.execute(
            "INSERT INTO music_play (app, title, artist, started_at, seconds)"
            " VALUES ('Spotify.exe', 'Old Song', 'Old Band', ?, 9999)",
            (_iso(now - 400 * 86400),),
        )
    words = node.vocabulary()
    assert words[:6] == [
        "PRADA",
        "Marta Souza",
        "TK from Ling tosite sigure",
        "Unravel",
        "Muse",
        "Space Dementia",
    ]
    assert "Old Band" not in words  # listened to too long ago
    assert words.count("Muse") == 1 and "Yo-Yo Ma" in words  # catalog artists follow


def test_vocabulary_is_kept_a_while_and_never_waits_for_a_request(node):
    first = node.vocabulary()
    _add_fact(node, "Your friend is Kenji.")
    assert node.vocabulary() == first  # kept
    node.clock[0] += VOCABULARY_AGE + 1
    with node._lock:  # she is busy thinking: the last list, at once
        assert node.vocabulary() == first
    assert "Kenji" in node.vocabulary()


def test_vocabulary_over_http(node):
    server = make_server(node, 0)
    import threading

    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        assert client.RemoteSession(url).vocabulary() == node.vocabulary()
    finally:
        server.shutdown()
        server.server_close()
    with pytest.raises(client.NodeError):
        client.RemoteSession(url).vocabulary(timeout=0.5)
    assert client.LocalSession(node).vocabulary() == node.vocabulary()
