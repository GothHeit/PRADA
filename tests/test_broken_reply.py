"""A broken or empty answer from the model never reaches the user as "{" or silence."""

import json

from prada import llm, persona
from prada.assistant import AssistantModule
from prada.module import Context, Request
from prada.node import Node
from prada.stream import rest


def scripted(answers):
    """A chat function giving `answers` in order (streamed in small pieces when asked)."""
    calls = []

    def chat(messages, schema, on_text=None, **kw):
        out = answers[len(calls)]
        calls.append(messages)
        if on_text is not None:
            for i in range(0, len(out), 4):
                on_text(out[i : i + 4])
        return out

    return chat, calls


def ok(text, offer=False):
    return json.dumps({"reply": text, "offer_search": offer})


def test_a_lone_brace_is_asked_again_with_a_changed_prompt():
    chat, calls = scripted(["{\n\n\n", ok("Valeu! Saudade é normal.")])
    reply = AssistantModule().handle(Request(text="Valeu!"), Context(chat=chat))
    assert reply.text == "Valeu! Saudade é normal." and reply.error is None
    assert len(calls) == 2
    assert calls[1][:-1] == calls[0] and calls[1][-1]["role"] == "system"


def test_an_empty_reply_is_asked_again():
    chat, calls = scripted([ok("   "), ok("Honestly? I like you.")])
    reply = AssistantModule().handle(Request(text="what do you think of me"), Context(chat=chat))
    assert reply.text == "Honestly? I like you." and len(calls) == 2


def test_broken_twice_gives_a_safe_line_never_an_error():
    chat, calls = scripted(["{", '{"reply": ""'])
    reply = AssistantModule().handle(Request(text="hi"), Context(chat=chat))
    assert reply.text == persona.lost_words() and reply.error is None and not reply.offer
    assert len(calls) == 2


def test_a_cut_off_answer_keeps_its_reply_and_drops_the_offer():
    chat, calls = scripted(['{"reply": "I am not sure who that is.", "offer_sea'])
    reply = AssistantModule().handle(Request(text="who is X"), Context(chat=chat))
    assert reply.text == "I am not sure who that is." and not reply.offer
    assert len(calls) == 1


def test_streamed_broken_answer_says_nothing_then_streams_the_retry(monkeypatch):
    chat, calls = scripted(["{\n\n", ok("Good evening, Dana. It is late already.")])
    monkeypatch.delenv("PRADA_FAKE_LLM", raising=False)
    monkeypatch.setattr(
        llm,
        "chat",
        lambda messages, schema, on_text=None, **kw: chat(messages, schema, on_text=on_text),
    )
    heard = []
    node = Node([AssistantModule()], db_path=":memory:", background=False)
    reply = node.handle(Request(text="hey", session="s"), speak=lambda k, t: heard.append(t))
    assert heard == ["Good evening, Dana."]
    assert rest(reply.text, heard) == "It is late already."


def test_streamed_cut_off_answer_matches_what_was_said(monkeypatch):
    chat, calls = scripted(['{"reply": "Hello there, Dana. And then I was cut'])
    monkeypatch.delenv("PRADA_FAKE_LLM", raising=False)
    monkeypatch.setattr(
        llm,
        "chat",
        lambda messages, schema, on_text=None, **kw: chat(messages, schema, on_text=on_text),
    )
    heard = []
    node = Node([AssistantModule()], db_path=":memory:", background=False)
    reply = node.handle(Request(text="hey", session="s"), speak=lambda k, t: heard.append(t))
    assert heard == ["Hello there, Dana."] and len(calls) == 1
    assert reply.text == "Hello there, Dana. And then I was cut"
    assert rest(reply.text, heard) == "And then I was cut"


def test_node_never_returns_a_brace(monkeypatch):
    monkeypatch.setattr(AssistantModule, "fake_chat", staticmethod(lambda **k: "{"))
    monkeypatch.setenv("PRADA_FAKE_LLM", "1")
    reply = Node([AssistantModule()], db_path=":memory:", background=False).handle(
        Request(text="hi", session="s")
    )
    assert reply.text == persona.lost_words() and reply.error is None
