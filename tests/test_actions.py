from typing import Literal

import pytest

from prada import persona
from prada.actions import Action, ActionFailed, Executor
from prada.assistant import AssistantModule
from prada.module import Module, Reply, Request
from prada.node import Node
from prada.notes import NotesModule, SaveNote, SaveNoteExecutor, note_text
from prada.wake import WakeModule


class Ping(Action):
    kind: Literal["ping.send"] = "ping.send"

    def describe(self):
        return "ping"


class PingExecutor(Executor):
    action = Ping

    def __init__(self, fail_run=False, fail_check=False):
        self.fail_run, self.fail_check, self.ran = fail_run, fail_check, 0

    def run(self, action, ctx):
        self.ran += 1
        if self.fail_run:
            raise OSError("network down")
        return {"pong": True}

    def check(self, action, result, ctx):
        if self.fail_check:
            raise ActionFailed("no pong came back")

    def done(self, action, result):
        return "Pinged."


class Pinger(Module):
    name = "ping"
    description = "Pings."

    def __init__(self, executor=None, asks=None):
        self.executors = [executor] if executor else []
        self.asks = asks or [Ping()]

    def handle(self, request, ctx):
        return Reply(module=self.name, text="Let me see.", actions=self.asks)


def test_verified_action_is_claimed():
    reply = Node([Pinger(PingExecutor())]).handle(Request(text="go"))
    assert reply.text == "Let me see. Pinged."
    assert [(o.ok, o.action, o.data) for o in reply.outcomes] == [
        (True, {"kind": "ping.send"}, {"pong": True})
    ]


@pytest.mark.parametrize(
    "executor, reason",
    [
        (PingExecutor(fail_run=True), "network down"),
        (PingExecutor(fail_check=True), "no pong came back"),
    ],
)
def test_failed_action_is_never_claimed(executor, reason):
    reply = Node([Pinger(executor)]).handle(Request(text="go"))
    assert reply.error is None and "Pinged" not in reply.text
    assert reply.text == "Let me see. " + persona.failed("ping", reason)
    assert not reply.outcomes[0].ok and reply.outcomes[0].error == reason


def test_actions_outside_the_modules_executors_are_refused():
    other = SaveNote(text="sneaky")
    executor = PingExecutor()
    reply = Node([Pinger(executor, asks=[other])]).handle(Request(text="go"))
    assert not reply.outcomes[0].ok and executor.ran == 0
    assert "allowed" in reply.text and "wrote that down" not in reply.text


def test_actions_are_not_sent_to_clients():
    reply = Node([Pinger(PingExecutor())]).handle(Request(text="go"))
    dumped = Reply.model_validate(reply.model_dump(mode="json"))
    assert "actions" not in reply.model_dump() and dumped.outcomes == reply.outcomes


def test_notes_are_saved_checked_and_read_back():
    node = Node([NotesModule()])
    reply = node.handle(Request(text="note that we're out of milk", session="s"))
    assert reply.text == persona.noted("we're out of milk") and reply.outcomes[0].ok
    node.handle(Request(text="write this down: call mum", session="s"))
    reply = node.handle(Request(text="what are my notes?", session="s"))
    assert reply.text == "Here are your notes: 1) we're out of milk; 2) call mum."
    assert reply.outcomes == []


def test_note_that_does_not_stick_is_reported(monkeypatch):
    monkeypatch.setattr(
        SaveNoteExecutor, "check", lambda *a: (_ for _ in ()).throw(ActionFailed("gone"))
    )
    reply = Node([NotesModule()]).handle(Request(text="note buy eggs"))
    assert reply.text == persona.failed("save that note", "gone")


def test_note_without_memory_fails(monkeypatch):
    monkeypatch.setattr("prada.node.open_store", lambda *a: (_ for _ in ()).throw(OSError("disk")))
    reply = Node([NotesModule()]).handle(Request(text="note buy eggs"))
    assert not reply.outcomes[0].ok and "wrote that down" not in reply.text


@pytest.mark.parametrize(
    "text, note",
    [
        ("note that the keys are in the drawer", "the keys are in the drawer"),
        ("Take a note: dentist on Friday", "dentist on Friday"),
        ("please write this down, buy bread", "buy bread"),
        ("jot down 42", "42"),
        ("buy bread", "buy bread"),
        ("note", ""),
    ],
)
def test_note_text(text, note):
    assert note_text(text) == note


def test_empty_note_asks_and_does_nothing():
    reply = Node([NotesModule()]).handle(Request(text="take a note"))
    assert reply.outcomes == [] and "what should I write down" in reply.text


def test_wake_is_an_action(monkeypatch):
    sent = []
    monkeypatch.setattr("prada.wol.send_magic_packet", lambda *a: sent.append(a))
    monkeypatch.setenv("PRADA_WAKE_MAC", "aa:bb:cc:dd:ee:ff")
    reply = Node([WakeModule()]).handle(Request(text="wake up"))
    assert reply.text == persona.waking() and reply.outcomes[0].ok and len(sent) == 1


def test_failed_wake_is_not_claimed(monkeypatch):
    def refuse(*a):
        raise OSError("network is unreachable")

    monkeypatch.setattr("prada.wol.send_magic_packet", refuse)
    monkeypatch.setenv("PRADA_WAKE_MAC", "aa:bb:cc:dd:ee:ff")
    reply = Node([WakeModule()]).handle(Request(text="wake up"))
    assert reply.text == persona.failed("wake your PC up", "network is unreachable")


def test_modules_share_the_conversation(monkeypatch):
    seen = []

    def chat(messages, schema, model=None):
        seen.append(messages)
        return '{"reply": "You asked me to note the milk."}'

    monkeypatch.setattr(AssistantModule, "fake_chat", staticmethod(chat))
    monkeypatch.setenv("PRADA_FAKE_LLM", "1")
    node = Node([AssistantModule(), NotesModule()])
    node.handle(Request(text="/notes note that milk", session="s"))
    node.handle(Request(text="/assistant what did I just ask?", session="s"))
    node.handle(Request(text="/assistant and before that?", session="s"))
    assert seen[0][1:] == [
        {"role": "user", "content": "note that milk"},
        {"role": "assistant", "content": persona.noted("milk")},
        {"role": "user", "content": "what did I just ask?"},
    ]
    assert seen[1][-2] == {"role": "assistant", "content": "You asked me to note the milk."}
