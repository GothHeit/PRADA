import json
import threading
from typing import ClassVar, Literal

import pytest

from prada import client, llm, persona
from prada.actions import Action, Executor
from prada.assistant import AssistantModule
from prada.module import Module, Reply, Request
from prada.node import Node
from prada.notes import NotesModule
from prada.server import make_server
from prada.stream import FieldText, Sentences, rest
from prada.voice import loop

# --- decoding and sentences ---


def decode(chunks):
    field = FieldText("reply")
    return [field.feed(c) for c in chunks]


def test_field_text_decodes_as_it_arrives():
    assert decode(['{"rep', 'ly": "Hel', "lo", '"}']) == ["", "Hel", "lo", ""]


def test_field_text_waits_for_split_escapes():
    pieces = decode(['{"reply": "a\\', "nb \\u00", 'e9 \\"q\\"', '"}'])
    assert "".join(pieces) == 'a\nb é "q"'
    assert pieces[0] == "a"  # the cut escape is held back, not decoded wrong


def test_field_text_keeps_surrogate_pairs_whole():
    pieces = decode(['{"reply": "hi \\ud83d', '\\ude00 there"}'])
    assert pieces == ["hi ", "\U0001f600 there"]


def test_field_text_stops_at_the_closing_quote():
    assert "".join(decode(['{"reply": "done", "x": "nope"}'])) == "done"


def sentences(*chunks):
    said = []
    s = Sentences(said.append)
    for c in chunks:
        s.feed(c)
    return said


def test_sentences_come_out_as_they_finish():
    assert sentences("Hello there, Dana. How ", "are you today? I'm ", "fine") == [
        "Hello there, Dana.",
        "How are you today?",
    ]


def test_short_sentences_wait_for_the_next_one():
    assert sentences("Hi! Oh. It is a lovely day. ") == ["Hi! Oh. It is a lovely day."]


def test_a_number_at_the_end_is_not_a_sentence_yet():
    assert sentences("The price went up by 3.", "5 percent today. ") == [
        "The price went up by 3.5 percent today."
    ]


def test_japanese_sentences_end_without_spaces():
    assert sentences("今日はいい天気ですね。散歩に行きましょうか？") == [
        "今日はいい天気ですね。",
        "散歩に行きましょうか？",
    ]


def test_rest_is_what_was_not_said():
    text = "Hello there, Dana. How are you today? I'm fine."
    assert rest(text, ["Hello there, Dana.", "How are you today?"]) == "I'm fine."
    assert rest(text, []) == text
    assert rest(text, [text]) == ""
    assert rest(text, ["Something else entirely."]) == text  # said twice, never skipped


# --- through the node ---

LONG = "Hello there, Dana. It's nice to hear from you again. Shall we start?"


def streaming_model(monkeypatch, text=LONG):
    """A real-model stand-in that writes its JSON a few characters at a time."""
    monkeypatch.delenv("PRADA_FAKE_LLM", raising=False)
    calls = []

    def chat(messages, schema, model=llm.MODEL, keep_alive=None, on_text=None):
        out = json.dumps({"reply": text})
        calls.append(on_text is not None)
        if on_text is not None:
            for i in range(0, len(out), 5):
                on_text(out[i : i + 5])
        return out

    monkeypatch.setattr(llm, "chat", chat)
    return calls


def test_assistant_streams_sentences_through_the_node(monkeypatch):
    calls = streaming_model(monkeypatch)
    heard = []
    node = Node([AssistantModule()], db_path=":memory:", background=False)
    reply = node.handle(
        Request(text="hi", session="s"), speak=lambda kind, t: heard.append((kind, t))
    )
    assert heard == [("part", "Hello there, Dana."), ("part", "It's nice to hear from you again.")]
    assert reply.text == LONG and calls == [True]
    assert rest(reply.text, [t for _, t in heard]) == "Shall we start?"


def test_no_streaming_without_a_listener(monkeypatch):
    calls = streaming_model(monkeypatch)
    node = Node([AssistantModule()], db_path=":memory:", background=False)
    assert node.handle(Request(text="hi")).text == LONG
    assert calls == [False]


def test_fake_client_streams_its_whole_reply(monkeypatch):
    monkeypatch.setenv("PRADA_FAKE_LLM", "1")
    heard = []
    node = Node([AssistantModule()], db_path=":memory:", background=False)
    reply = node.handle(Request(text="hi. there you go."), speak=lambda k, t: heard.append(t))
    assert heard == ["(fake assistant) hi."]
    assert rest(reply.text, heard) == "there you go."


class Slow(Action):
    kind: Literal["slow.do"] = "slow.do"


class SlowExecutor(Executor):
    action = Slow

    def filler(self, action):
        return "One moment."

    def run(self, action, ctx):
        ctx.state["ran"] = ctx.state.get("ran", 0) + 1
        return {}

    def check(self, action, result, ctx):
        pass

    def done(self, action, result):
        return "Done it."


class SlowModule(Module):
    name = "slow"
    description = "slow things"
    executors: ClassVar = [SlowExecutor()]
    saw_say: ClassVar = []

    def handle(self, request, ctx):
        self.saw_say.append(ctx.say)
        return Reply(module=self.name, text="", actions=[Slow(), Slow()])


def test_filler_before_a_slow_action_once_and_no_early_text(monkeypatch):
    monkeypatch.setenv("PRADA_FAKE_LLM", "1")
    heard = []
    node = Node([SlowModule()], db_path=":memory:", background=False)
    reply = node.handle(Request(text="go", session="s"), speak=lambda k, t: heard.append((k, t)))
    assert heard == [("filler", "One moment.")]
    assert reply.text == "Done it."  # the filler is not part of what she said
    assert SlowModule.saw_say == [None]  # a module with actions never speaks early
    assert node._mem().recent(2)[-1]["content"] == "Done it."


def test_web_search_says_it_is_looking(monkeypatch):
    from prada.web.module import SearchExecutor, SearchWeb

    assert SearchExecutor(None).filler(SearchWeb(query="q", sources=["web"])) == (
        persona.looking_up()
    )


# --- over HTTP ---


@pytest.fixture
def served(monkeypatch):
    streaming_model(monkeypatch)
    node = Node([AssistantModule(), NotesModule()], db_path=":memory:", background=False)
    server = make_server(node, 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def test_remote_session_streams(served):
    session = client.RemoteSession(served)
    heard = []
    reply = session.send("/assistant hi", speak=lambda kind, t: heard.append((kind, t)))
    assert reply.text == LONG
    assert [t for _, t in heard] == ["Hello there, Dana.", "It's nice to hear from you again."]
    assert session.send("/assistant hi").text == LONG  # the plain endpoint is unchanged


def test_remote_stream_without_parts_still_replies(served):
    reply = client.RemoteSession(served).send("note that milk", speak=lambda *a: None)
    assert reply.text == persona.noted("milk")


# --- the voice loop ---


class Mouth:
    def __init__(self):
        self.said = []

    def chime(self):
        pass

    def say(self, text):
        self.said.append(text)


class Ears:
    def __init__(self, *texts, follow=None):
        self.texts = list(texts)
        self.follow = follow
        self.woken = False

    def wait_for_wake(self):
        if self.woken:
            return None
        self.woken = True
        return "hey"

    def listen(self, wait=None):
        if wait is not None and self.follow is not None:
            return self.follow
        return self.texts.pop(0) if self.texts else ""


class StreamingSession:
    def __init__(self, events, text):
        self.events, self.text, self.sent = events, text, 0

    def send(self, text, module=None, speak=None, **options):
        self.sent += 1
        for kind, line in self.events:
            speak(kind, line)
        return Reply(text=self.text)


def test_loop_speaks_early_and_only_the_rest_after():
    session = StreamingSession(
        [("filler", "Let me look that up."), ("part", "First sentence here.")],
        "First sentence here. And the rest.",
    )
    mouth = Mouth()
    loop.run(session, Ears("question"), mouth)
    assert mouth.said == ["Let me look that up.", "First sentence here.", "And the rest."]


def test_follow_ups_are_capped():
    session = StreamingSession([], "ok")
    loop.run(session, Ears("first", follow="the TV again"), Mouth(), follow_up=5.0)
    assert session.sent == 1 + loop.MAX_FOLLOW_UPS


def test_field_text_ignores_a_broken_escape():
    assert decode(['{"reply": "a \\uzzzz b"}']) == [""]
