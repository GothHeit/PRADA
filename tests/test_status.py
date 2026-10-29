import json
from datetime import datetime

import pytest

from prada import status
from prada.assistant import AssistantModule
from prada.module import Request
from prada.node import Node
from prada.notes import NotesModule, asks_to_save, note_text
from prada.web.module import WebModule


class Clock:
    def __init__(self):
        self.now = datetime(2026, 10, 29, 21, 5).timestamp()

    def __call__(self):
        return self.now


def test_status_block_round_trip():
    text = status.attach("hello", ["Now: x", "", "You last talked 2 hours ago."])
    assert text == "hello\n\n<status>\nNow: x\nYou last talked 2 hours ago.\n</status>"
    assert status.detach(text) == "hello"
    assert status.attach("hello", []) == "hello" == status.detach("hello")


def test_ago():
    assert status.ago(20) == "less than a minute ago"
    assert status.ago(60) == "1 minute ago"
    assert status.ago(3 * 3600 + 50) == "3 hours ago"
    assert status.ago(2 * 86400) == "2 days ago"
    assert status.since_line(None, 0) == "You have never talked before."


def spy(monkeypatch):
    seen = []

    def chat(messages, schema, model=None):
        seen.append(messages)
        return json.dumps({"reply": "ok"})

    monkeypatch.setattr(AssistantModule, "fake_chat", staticmethod(chat))
    monkeypatch.setenv("PRADA_FAKE_LLM", "1")
    return seen


def test_status_goes_last_and_the_prompt_start_stays_the_same(monkeypatch):
    seen = spy(monkeypatch)
    clock = Clock()
    node = Node([AssistantModule(), NotesModule()], db_path=":memory:", clock=clock)
    node.handle(Request(text="/assistant hi", session="s"))
    clock.now += 20 * 60
    node.handle(Request(text="/notes note that eggs", session="s"))
    node.handle(Request(text="/assistant what time is it?", session="s"))
    first, second = seen
    assert first[0] == second[0]  # the same system prompt, 20 minutes later
    assert second[1:3] == [  # the earlier turns carry no status
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
    ]
    last = second[-1]["content"]
    assert last.startswith("what time is it?\n\n<status>\n")
    assert "Now: Thursday 29 October 2026, 21:25" in last
    assert "You last talked less than a minute ago." in last
    assert "you asked me to save that note, and I did" in last
    assert "Thursday 29 October 2026, 21:05" in first[-1]["content"]
    assert "never talked before" in first[-1]["content"]
    assert "status" in first[0]["content"] and "date and time" in first[0]["content"]


def test_old_actions_stay_out_of_the_status(monkeypatch):
    seen = spy(monkeypatch)
    clock = Clock()
    node = Node([AssistantModule(), NotesModule()], db_path=":memory:", clock=clock)
    node.handle(Request(text="/notes note that eggs", session="s"))
    clock.now += 2 * 3600
    node.handle(Request(text="/assistant hi", session="s"))
    assert "action log" not in seen[-1][-1]["content"]


# --- explicit notes in other languages (they went to chat, which said "Anotado") ---


@pytest.mark.parametrize(
    "text,note",
    [
        ("anota: comprar ovos", "comprar ovos"),
        ("Anota aí que a reunião é às 3", "a reunião é às 3"),
        ("toma nota: dentista na sexta", "dentista na sexta"),
        ("você pode anotar que preciso de pão?", "preciso de pão?"),
        ("apunta: llamar a mamá", "llamar a mamá"),
        ("prends note que je dois appeler Marie", "je dois appeler Marie"),
        ("Write it down: is the dentist at 3?", "is the dentist at 3?"),
    ],
)
def test_explicit_notes_in_any_language_are_saved(monkeypatch, text, note):
    monkeypatch.setenv("PRADA_FAKE_LLM", "1")
    asked = []
    monkeypatch.setattr("prada.router.route", lambda *a, **k: asked.append(a) or "assistant")
    node = Node([AssistantModule(), NotesModule()], db_path=":memory:", background=False)
    reply = node.handle(Request(text=text, session="s"))
    assert reply.module == "notes" and asked == []  # no router model involved
    assert [o.ok for o in reply.outcomes] == [True]
    rows = node._dbs["notes"].execute("SELECT text FROM notes_note").fetchall()
    assert [r["text"] for r in rows] == [note]


@pytest.mark.parametrize(
    "text", ["anotado", "notebook is broken", "anotações?", "Noted.", "what did I note?"]
)
def test_words_that_only_look_like_notes(text):
    assert not asks_to_save(text)


def test_a_bare_ask_names_no_note():
    assert note_text("can you note that?") == ""


# --- web: a yes or no is hers only after an offer ---


def test_web_claims_yes_only_after_offering():
    web = WebModule()
    assert not web.claims("sure", {})
    assert web.claims("sure", {"offer": "price of a keyboard"})
    assert web.claims("search for keyboards", {})


def test_past_tense_is_not_a_save():
    assert not asks_to_save("anotou o número do dentista")


def test_a_lesson_keeps_a_french_sentence_that_starts_like_a_note(monkeypatch):
    from prada.node import default_modules

    monkeypatch.setenv("PRADA_FAKE_LLM", "1")
    node = Node(default_modules(), db_path=":memory:", background=False)
    node.handle(Request(text="/tutor Je voudrais un café.", session="s"))
    assert node.handle(Request(text="Notez bien la date du train.", session="s")).module == "tutor"
    assert node.handle(Request(text="anota: comprar ovos", session="s")).module == "notes"
