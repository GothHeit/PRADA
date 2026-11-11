"""The situation: what her code sees each turn, rendered the same for both minds."""

import io

from prada import mind, situation
from prada.assistant import AssistantModule
from prada.module import Module, Reply, Request
from prada.music.bridge import BridgeError, FakeBridge, Player, Status
from prada.music.module import MusicModule, targets
from prada.node import Node
from prada.situation import Ability, Situation, Source
from prada.trace import Trace, describe
from prada.voice import loop
from prada.voice.fake import TextEars
from prada.wake import WakeModule
from prada.web.module import WebModule


def test_render_shows_each_part_only_when_known():
    s = Situation(0, players=None)
    assert situation.render(s).splitlines()[1:] == []  # nothing but the clock
    s = Situation(
        0,
        [situation.Player("Spotify", "playing", "Unravel", "TK")],
        last_talk=0,
        source=Source(via="voice", start="follow-up", ear="the PC microphone", language="en"),
        confirm=["Turn on the PC?"],
        done=["at 21:05 you asked me to pause Spotify, and I did"],
        since=30,
        broken=["turning on the PC: no PC address is set up"],
        abilities=[Ability("music", "play music", ["play Muse", "pause"])],
    )
    text = situation.render(s)
    assert "The user and PRADA last talked less than a minute ago." in text
    assert (
        "The user said this by voice (as a follow-up, without the wake word, heard by the PC "
        "microphone, in en)." in text
    )
    assert '- Spotify: playing "Unravel" by TK' in text
    assert 'Waiting for a yes or no: PRADA asked "Turn on the PC?"' in text
    assert "What PRADA did in the last 30 minutes (her action log):\n- at 21:05" in text
    assert "Not working right now:\n- turning on the PC" in text
    assert '- play music (e.g. "play Muse", "pause")' in text
    assert "never talked" in situation.render(Situation(0, first_talk=True))
    assert "typed" in situation.render(Situation(0, source=Source()))
    unsure = Source(via="voice", sure=0.3)
    assert "misheard" in situation.render(Situation(0, source=unsure))
    assert "misheard" not in situation.render(Situation(0, source=Source(via="voice", sure=0.9)))


def test_the_understand_step_reads_the_same_render_without_abilities():
    s = Situation(0, [], abilities=[Ability("music", "play music")], done=["paused"], since=30)
    user = mind.messages(s, [], "hi")[1]["content"]
    assert (
        situation.render(s)
        .replace("\n- play music", "")
        .replace(
            "\nWhat PRADA can do when the user asks plainly (her system does it and tells her how "
            "it went):",
            "",
        )
        in user
    )
    assert "play music" not in user and "- paused" in user


def test_node_situation_sees_players_offers_and_what_she_can_do(tmp_path):
    node = Node(
        [AssistantModule(), WebModule(), MusicModule(bridge=FakeBridge())],
        db_path=tmp_path / "p.db",
        background=False,
    )
    s = node.situation("s")
    assert [(p.app, p.state) for p in s.players] == [("Spotify", "playing"), ("Corpus", "paused")]
    assert s.first_talk and not s.offer
    assert {a.name for a in s.abilities} == {"web", "music"}  # the assistant has no ability line
    node.handle(Request(text="who won the match yesterday?", module="web", session="s"))
    assert node.situation("s").offer == "who won the match yesterday?"
    assert node.situation("other").offer == ""


def test_a_failing_state_provider_is_said_not_raised(tmp_path):
    class Down(FakeBridge):
        def status(self):
            raise BridgeError("no answer")

    class Broken(Module):
        name = "broken"
        description = "x"

        def observe(self, state, s):
            raise RuntimeError("oops")

    node = Node([MusicModule(bridge=Down()), Broken()], db_path=":memory:", background=False)
    s = node.situation(None)
    assert s.players is None
    assert s.broken == [
        "the music players can't be reached (no answer)",
        "broken: can't tell its state (oops)",
    ]


def test_a_held_action_and_the_action_log_show_up(tmp_path, monkeypatch):
    monkeypatch.setenv("PRADA_WAKE_MAC", "aa:bb:cc:dd:ee:ff")
    node = Node([AssistantModule(), WakeModule()], db_path=tmp_path / "p.db", background=False)
    node.handle(Request(text="wake up the PC", module="wake", session="s"))
    s = node.situation("s")
    assert len(s.confirm) == 1 and "PC" in s.confirm[0]
    node.handle(Request(text="no", session="s"))
    s = node.situation("s")
    assert s.confirm == [] and len(s.done) == 1  # the declined wake, in her words
    monkeypatch.delenv("PRADA_WAKE_MAC")
    monkeypatch.setenv("PRADA_CONFIG", str(tmp_path / "none.toml"))
    assert "no PC address" in " ".join(node.situation("s").broken)


def test_each_turn_shows_her_the_situation_and_the_trace_keeps_it(tmp_path):
    seen = []

    class Echo(Module):
        name = "echo"
        description = "x"
        ability = "echo things"

        def handle(self, request, ctx):
            seen.append(ctx.status)
            return Reply(module=self.name, text="ok")

    node = Node([Echo()], db_path=tmp_path / "p.db", background=False)
    source = Source(via="voice", start="wake word", language="pt", sure=0.2)
    node.handle(Request(text="oi", module="echo", session="s", source=source))
    lines = seen[0]
    assert any("after the wake word" in line and "in pt" in line for line in lines)
    assert any("misheard" in line for line in lines)
    assert "- echo things" in lines
    turn = Trace(node._traces()._db).last(1)[0]
    assert "after the wake word" in turn["situation"]
    assert "echo things" not in turn["situation"]  # the same every turn: not kept
    assert "saw:" in describe(turn)


def test_next_never_presses_a_video_that_isnt_playing():
    video = Player("Chrome", "A video", "", "", "paused")
    spotify = Player("Spotify.exe", "Unravel", "TK", "", "paused")
    assert [p.name for p in targets("next", Status([video, spotify], "Chrome"), None)] == [
        "Spotify"
    ]
    assert targets("next", Status([video], "Chrome"), None) == "Nothing's playing right now."


class _Session:
    def __init__(self):
        self.sources = []

    def send(self, text, speak=None, source=None, **options):
        self.sources.append(source)
        return Reply(text="ok")


class _Mouth:
    def chime(self):
        pass

    def say(self, text, barge=None):
        return True


def test_the_voice_loop_says_how_each_request_started():
    session = _Session()
    ears = TextEars(io.StringIO("hello\n"))
    loop.run(session, ears, _Mouth(), once=True)
    src = session.sources[0]
    assert (src.via, src.start, src.ear) == ("voice", "wake word", TextEars.ear)


def test_typed_requests_say_so(monkeypatch):
    from prada import client

    class Spy:
        sources = []

        def send(self, text, module=None, source=None, **options):
            self.sources.append(source)
            return Reply(text="ok")

    lines = iter(["hi"])

    def typed(prompt):
        try:
            return next(lines)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", typed)
    spy = Spy()
    client.repl(spy)
    assert spy.sources[0].via == "typed"
