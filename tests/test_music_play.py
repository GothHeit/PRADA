import json
import sqlite3
import urllib.parse

import pytest

from prada.eval import eval_modules
from prada.module import Request
from prada.music.bridge import FakeBridge
from prada.music.module import MusicModule
from prada.music.play import Wish, fold, pick, wish
from prada.music.spotify import (
    Device,
    FakeSpotify,
    Hit,
    Now,
    Results,
    SpotifyApi,
    SpotifyError,
    choose,
    parse_now,
    parse_results,
    save_token,
)
from prada.node import Node


@pytest.mark.parametrize(
    "text, query, kind, app",
    [
        ("play Muse Space Dementia", "muse space dementia", "any", None),
        ("Prada, play Space Dementia by Muse on Spotify", "space dementia muse", "any", "spotify"),
        ("can you play Unravel please", "unravel", "any", None),
        ("toca Evidências do Chitãozinho e Xororó", "evidencias chitaozinho e xororo", "any", None),
        ("toca Garota de Ipanema", "garota de ipanema", "any", None),
        ("põe Unravel no Spotify", "unravel", "any", "spotify"),
        ("quero ouvir Muse", "muse", "any", None),
        ("play some Muse", "muse", "any", None),
        ("toca músicas do Muse", "muse", "artist", None),
        ("play songs by Muse", "muse", "artist", None),
        ("play the album Origin of Symmetry", "origin of symmetry", "album", None),
        ("play Origin of Symmetry album", "origin of symmetry", "album", None),
        ("toca o disco Origin of Symmetry", "origin of symmetry", "album", None),
        ("play my chill playlist", "chill", "playlist", None),
        ("toca a playlist Calm Vibes", "calm vibes", "playlist", None),
        ("toca a música Evidências", "evidencias", "track", None),
        ("play something calm", "calm", "mood", None),
        ("toca algo calmo", "calmo", "mood", None),
        ("toca música calma", "calma", "mood", None),
        ("toca Unravel no Corpus", "unravel", "any", "corpus"),
        ("play 残酷な天使のテーゼ", "残酷な天使のテーゼ", "any", None),
        ("toca uma música do Muse", "muse", "artist", None),
        ("bota pra tocar Muse", "muse", "any", None),
        ("coloca pra tocar Evidências", "evidencias", "any", None),
        ("play Back in Black", "back in black", "any", None),
        ("play Dumb by Nirvana", "dumb nirvana", "any", None),
    ],
)
def test_wishes(text, query, kind, app):
    assert wish(text) == Wish(query, kind, app)


@pytest.mark.parametrize(
    "text",
    [
        "let's play a game",
        "play with me",
        "play chess",
        "I want to play cello today",
        "play some music",
        "toca uma música",
        "tell me about Muse",
        "what is Space Dementia about",
        "play it again",
        "play back what I said",
        "play dumb",
        "play nice",
        "toca o sino",
        "coloca o alarme pra 7h",
    ],
)
def test_not_wishes(text):
    assert wish(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "put on some coffee",
        "listen to me",
        "quero ouvir sua opinião",
        "coloca isso nas notas",
        "bota isso na lista de compras",
        "põe sal na receita",
        "let me hear your opinion",
        "põe Unravel",  # the router may still send it to music
        "coloca Evidências do Chitãozinho e Xororó",  # this one too
        "coloca a foto do gato",
        "quero ouvir a opinião da Maria",
    ],
)
def test_everyday_verbs_skip_the_router_only_with_a_sign_of_music(text):
    music = MusicModule()
    assert not music.wants(text)


@pytest.mark.parametrize(
    "text",
    [
        "play Muse Space Dementia",
        "toca Evidências",
        "bota pra tocar Muse",
        "põe Unravel no Spotify",
        "coloca o disco Origin of Symmetry",
        "put on something calm",
    ],
)
def test_music_requests_skip_the_router(text):
    music = MusicModule()
    assert music.wants(text) and music.accepts(text)


def test_router_pick_with_an_everyday_verb_is_accepted():
    assert MusicModule().accepts("põe Unravel")


def test_fold_keeps_other_scripts():
    assert fold("Evidências!") == "evidencias"
    assert fold("ドラゴン") == "ドラゴン"  # dakuten kept
    assert fold("Чайковский") == "чайковский"


def hits(*rows, top=None):
    return Results([Hit(k, f"spotify:{k}:{i}", n, b) for i, (k, n, b) in enumerate(rows)], top)


SEARCH = hits(
    ("track", "Space Dementia", "Muse"),
    ("track", "Muse", "Some Band"),
    ("artist", "Muse", ""),
    ("album", "Origin of Symmetry", "Muse"),
    ("playlist", "Calm Vibes", "Spotify"),
)


def test_pick_the_track_named_with_its_artist():
    assert pick(Wish("muse space dementia"), SEARCH).name == "Space Dementia"
    assert pick(Wish("space dementia"), SEARCH).name == "Space Dementia"


def test_pick_a_bare_name_trusts_spotify_top_else_the_artist():
    assert pick(Wish("muse"), SEARCH).kind == "artist"
    top_is_track = Results(SEARCH.hits, top="spotify:track:1")
    assert pick(Wish("muse"), top_is_track).by == "Some Band"


def test_pick_by_kind_and_mood():
    assert pick(Wish("origin of symmetry", "album"), SEARCH).kind == "album"
    assert pick(Wish("muse", "artist"), SEARCH).kind == "artist"
    assert pick(Wish("calm", "mood"), SEARCH).name == "Calm Vibes"
    assert pick(Wish("chill", "playlist"), SEARCH) is None


def test_pick_forgives_a_small_slip_but_not_a_stranger():
    assert pick(Wish("muse space dementa"), SEARCH).name == "Space Dementia"
    assert pick(Wish("toxicity system of a down"), SEARCH) is None


def test_parse_results_skips_junk():
    found = parse_results(
        {
            "tracks": {"items": [{"uri": "u1", "name": "A", "artists": [{"name": "B"}]}, 3]},
            "playlists": {
                "items": [None, {"uri": "p", "name": "P", "owner": {"display_name": "O"}}]
            },
            "albums": "junk",
        }
    )
    assert [(h.kind, h.name, h.by) for h in found.hits] == [
        ("track", "A", "B"),
        ("playlist", "P", "O"),
    ]


def test_parse_now_and_empty_player():
    now = parse_now(
        {
            "is_playing": True,
            "item": {"uri": "u", "name": "T", "artists": [{"name": "A"}], "album": {"name": "L"}},
            "context": {"uri": "c"},
            "device": {"name": "PC"},
        }
    )
    assert now == Now("u", "T", "A", "L", "c", True, "PC")
    assert parse_now(None) == Now()


def test_choose_a_device():
    pc = Device("1", "PC", "Computer")
    phone = Device("2", "Phone", "Smartphone")
    assert choose([phone, pc]) is pc
    playing = Device("3", "Speaker", "Speaker", active=True)
    assert choose([pc, playing]) is playing
    assert choose([phone]) is None  # open the PC's app first
    assert choose([phone], anywhere=True) is phone and choose([], anywhere=True) is None


@pytest.fixture
def play_node(tmp_path):
    spotify = FakeSpotify()
    music = MusicModule(bridge=FakeBridge(), spotify=spotify, launch=0.2, step=0.01, verify=0.1)
    node = Node(modules=[music], db_path=tmp_path / "prada.db", background=False)
    return node, spotify


def test_play_searches_plays_and_checks(play_node, tmp_path):
    node, spotify = play_node
    reply = node.handle(Request(text="Prada, play Muse Space Dementia", session="s"))
    assert reply.module == "music"
    assert reply.text == "Playing Space Dementia by Muse on KAFKA-PC."
    assert spotify.searched == ["muse space dementia"]
    assert spotify.played == [("spotify:track:spacedementia", "pc")]
    db = sqlite3.connect(tmp_path / "prada.db")
    assert db.execute("SELECT kind, status FROM actionlog_entry").fetchone() == (
        "music.play",
        "done",
    )


def test_play_an_album_in_portuguese(play_node):
    node, spotify = play_node
    reply = node.handle(Request(text="toca o disco Origin of Symmetry", session="s"))
    assert reply.text == "Playing the album Origin of Symmetry by Muse on KAFKA-PC."
    assert spotify.playing.context == "spotify:album:originofsymmetry"


def test_nothing_close_is_never_played(play_node):
    node, spotify = play_node
    reply = node.handle(Request(text="play Toxicity", session="s"))
    assert "nothing close" in reply.text and not reply.outcomes[0].ok
    assert spotify.played == []


def test_a_play_that_changes_nothing_is_never_claimed(tmp_path):
    spotify = FakeSpotify(stuck=True)
    music = MusicModule(spotify=spotify, launch=0, step=0.01, verify=0.05)
    node = Node(modules=[music], db_path=tmp_path / "prada.db", background=False)
    reply = node.handle(Request(text="play Hysteria", session="s"))
    assert reply.text.startswith("I tried to play hysteria on Spotify, but")
    assert not reply.outcomes[0].ok


def test_same_title_by_another_artist_is_not_a_relinked_track():
    from prada.music.module import Play, PlayExecutor

    executor = PlayExecutor(MusicModule(spotify=FakeSpotify()))
    hit = Hit("track", "spotify:track:dl", "Hysteria", "Def Leppard")
    action = Play(query="hysteria def leppard")
    other = Now("spotify:track:muse", "Hysteria", "Muse", "Absolution", "", True)
    relinked = Now("spotify:track:dl2", "Hysteria", "Def Leppard", "Hysteria", "", True)
    with pytest.raises(Exception, match="didn't start"):
        executor.check(action, {"hit": vars(hit), "now": vars(other)}, None)
    executor.check(action, {"hit": vars(hit), "now": vars(relinked)}, None)


def test_she_opens_spotify_when_no_device_is_open(play_node):
    node, spotify = play_node
    spotify._open = False
    reply = node.handle(Request(text="play Unravel", session="s"))
    assert spotify.opened == 1
    assert reply.text == "Playing Unravel by TK from Ling tosite sigure on KAFKA-PC."


def test_an_online_phone_waits_until_the_pc_app_is_opened(play_node):
    node, spotify = play_node
    spotify._open = False
    spotify.others = [Device("phone", "Phone", "Smartphone")]
    reply = node.handle(Request(text="play Unravel", session="s"))
    assert spotify.opened == 1
    assert spotify.played == [("spotify:track:unravel", "pc")]
    assert reply.text.endswith("on KAFKA-PC.")


def test_the_phone_is_the_last_resort(tmp_path):
    phone = Device("phone", "Phone", "Smartphone")
    spotify = FakeSpotify(open=False, others=[phone])
    spotify.open_app = lambda: None  # the PC's app never shows up
    music = MusicModule(spotify=spotify, launch=0.05, step=0.01, verify=0.05)
    node = Node(modules=[music], db_path=tmp_path / "prada.db", background=False)
    reply = node.handle(Request(text="play Unravel", session="s"))
    assert spotify.played == [("spotify:track:unravel", "phone")]
    assert reply.text.endswith("on Phone.")


def test_no_device_even_after_opening(tmp_path):
    spotify = FakeSpotify(open=False)
    spotify.open_app = lambda: None  # Spotify never shows up
    music = MusicModule(spotify=spotify, launch=0.05, step=0.01)
    node = Node(modules=[music], db_path=tmp_path / "prada.db", background=False)
    reply = node.handle(Request(text="play Unravel", session="s"))
    assert "didn't show up as a device" in reply.text and spotify.played == []


def test_not_logged_in_says_how(tmp_path):
    music = MusicModule(spotify=FakeSpotify(logged_in=False))
    node = Node(modules=[music], db_path=tmp_path / "prada.db", background=False)
    reply = node.handle(Request(text="play Unravel", session="s"))
    assert "prada music login" in reply.text and not reply.outcomes[0].ok


def test_corpus_by_name_is_not_yet(play_node):
    node, spotify = play_node
    reply = node.handle(Request(text="toca Unravel no Corpus", session="s"))
    assert "only on Spotify" in reply.text and not reply.outcomes
    assert spotify.searched == []


def test_eval_never_reaches_real_spotify():
    music = next(m for m in eval_modules() if m.name == "music")
    assert isinstance(music.spotify(), FakeSpotify)


def test_fake_mode_never_reaches_real_spotify(monkeypatch, tmp_path):
    monkeypatch.setenv("PRADA_FAKE_LLM", "1")
    module = MusicModule()
    module.start(tmp_path / "prada.db")
    assert isinstance(module.spotify(), FakeSpotify)


# The Web API client, against a stand-in for Spotify's servers.


class Server:
    """Answers like accounts.spotify.com and api.spotify.com, and records each call."""

    def __init__(self):
        self.calls = []
        self.tokens = 0
        self.expired = False
        self.routes = {}

    def __call__(self, method, url, headers, body):
        self.calls.append((method, url, headers, body))
        if url.endswith("/api/token"):
            form = dict(urllib.parse.parse_qsl(body.decode()))
            if form.get("refresh_token") == "bad":
                return 400, b'{"error": "invalid_grant"}'
            self.tokens += 1
            answer = {"access_token": f"access{self.tokens}", "expires_in": 3600}
            if form["grant_type"] == "authorization_code" or self.tokens == 2:
                answer["refresh_token"] = f"refresh{self.tokens}"
            return 200, json.dumps(answer).encode()
        if self.expired:
            self.expired = False
            return 401, b'{"error": {"status": 401, "message": "expired"}}'
        path = urllib.parse.urlsplit(url).path.removeprefix("/v1")
        status, answer = self.routes.get((method, path), (404, {"error": {"message": "nope"}}))
        return status, json.dumps(answer).encode() if answer is not None else b""


@pytest.fixture
def api(tmp_path):
    server = Server()
    save_token(tmp_path / "spotify-token.json", "refresh0")
    return SpotifyApi("client", tmp_path / "spotify-token.json", http=server), server


def test_api_searches_with_a_fresh_token(api):
    api, server = api
    track = {"uri": "spotify:track:sd", "name": "Space Dementia", "artists": [{"name": "Muse"}]}
    server.routes[("GET", "/search")] = (200, {"tracks": {"items": [track]}})
    found = api.search("muse space dementia")
    assert found.hits[0].by == "Muse"
    token_call, search_call = server.calls
    assert "refresh_token=refresh0" in token_call[3].decode()
    assert search_call[2]["Authorization"] == "Bearer access1"
    assert "q=muse+space+dementia" in search_call[1]
    api.search("again")
    assert server.tokens == 1  # the access token is reused


def test_api_plays_a_track_or_a_context_on_a_device(api):
    api, server = api
    server.routes[("PUT", "/me/player/play")] = (204, None)
    pc = Device("pc1", "PC", "Computer")
    sd = Hit("track", "spotify:track:sd", "Space Dementia", "Muse", "OoS", "spotify:album:os")
    api.play(sd, pc)
    api.play(Hit("track", "spotify:track:x", "No Album"), pc)
    api.play(Hit("album", "spotify:album:os", "Origin of Symmetry"), pc)
    plays = [c for c in server.calls if c[0] == "PUT"]
    assert all("device_id=pc1" in c[1] for c in plays)
    assert [json.loads(c[3]) for c in plays] == [
        {"context_uri": "spotify:album:os", "offset": {"uri": "spotify:track:sd"}},
        {"uris": ["spotify:track:x"]},  # no album known: the bare track
        {"context_uri": "spotify:album:os"},
    ]


def test_parse_results_keeps_a_tracks_album_uri():
    track = {
        "uri": "spotify:track:sd",
        "name": "Space Dementia",
        "album": {"uri": "spotify:album:os", "name": "Origin of Symmetry"},
    }
    hit = parse_results({"tracks": {"items": [track]}}).hits[0]
    assert (hit.album, hit.album_uri) == ("Origin of Symmetry", "spotify:album:os")


def test_api_refreshes_once_on_401_and_keeps_a_new_refresh_token(api, tmp_path):
    api, server = api
    server.routes[("GET", "/me/player/devices")] = (200, {"devices": [{"id": "d", "name": "PC"}]})
    api.devices()
    server.expired = True
    assert [d.name for d in api.devices()] == ["PC"]
    assert server.tokens == 2
    saved = json.loads((tmp_path / "spotify-token.json").read_text())
    assert saved == {"refresh_token": "refresh2"}


def test_api_errors_are_plain(api, tmp_path):
    api, server = api
    server.routes[("PUT", "/me/player/play")] = (
        404,
        {"error": {"status": 404, "message": "x", "reason": "NO_ACTIVE_DEVICE"}},
    )
    with pytest.raises(SpotifyError, match="no Spotify device"):
        api.play(Hit("track", "u", "T"), Device("d", "PC"))
    save_token(tmp_path / "spotify-token.json", "bad")
    api._expires = 0
    with pytest.raises(SpotifyError, match="prada music login"):
        api.now()


def test_api_without_login(tmp_path):
    api = SpotifyApi("client", tmp_path / "none.json", http=Server())
    assert not api.ready()
    with pytest.raises(SpotifyError, match="prada music login"):
        api.now()
    assert not SpotifyApi("", tmp_path / "none.json", http=Server()).ready()


def test_login_url_uses_pkce(api):
    api, _ = api
    url = urllib.parse.urlsplit(api.login_url("verifier", "state"))
    query = dict(urllib.parse.parse_qsl(url.query))
    assert query["redirect_uri"] == "http://127.0.0.1:7725/callback"
    assert query["code_challenge_method"] == "S256" and "=" not in query["code_challenge"]
    assert "user-modify-playback-state" in query["scope"]


def test_finish_login_saves_the_refresh_token_privately(api, tmp_path):
    api, server = api
    api.finish_login("code", "verifier")
    path = tmp_path / "spotify-token.json"
    assert json.loads(path.read_text()) == {"refresh_token": "refresh1"}
    assert path.stat().st_mode & 0o077 == 0
    form = dict(urllib.parse.parse_qsl(server.calls[0][3].decode()))
    assert form["code_verifier"] == "verifier" and form["grant_type"] == "authorization_code"


def test_login_catches_the_browser_answer(api, monkeypatch):
    import threading
    import urllib.request

    from prada.music import spotify

    api, server = api
    seen = {}

    def browser(url):
        seen["state"] = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))["state"]

        def go():
            back = f"http://127.0.0.1:{port}/callback?code=abc&state={seen['state']}"
            urllib.request.urlopen(back, timeout=5).read()

        threading.Thread(target=go, daemon=True).start()

    monkeypatch.setattr(spotify, "open_url", browser)
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    spotify.login(api, wait=5, port=port)
    form = dict(urllib.parse.parse_qsl(server.calls[-1][3].decode()))
    assert form["code"] == "abc"
