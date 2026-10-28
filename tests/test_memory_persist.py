"""Persistent conversation memory: history across restarts, episodes and their summaries."""

import json
import sqlite3

import pytest

from prada import llm, memory, persona
from prada.assistant import AssistantModule
from prada.module import Context, Request
from prada.node import MAX_TURNS, SUMMARY_RETRY, Node
from prada.notes import NotesModule


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now

    def later(self, seconds):
        self.now += seconds


def _memory_canned(messages, schema, model=None):
    raise AssertionError("never called: tests replace the memory's client")


def pick(fake, summary_chat):
    """The summary chat for the memory's client, the canned one for everything else."""
    return summary_chat if fake is _memory_canned else fake


def run_pending(mem, chat, model):
    for episode, transcript in mem.jobs():
        mem.store(episode, "" if transcript is None else memory.summarize(chat, transcript), model)


def node(tmp_path, clock, **kw):
    return Node([AssistantModule(), NotesModule()], db_path=tmp_path / "p.db", clock=clock, **kw)


def ask(n, text, sid="s"):
    return n.handle(Request(text=text, module="assistant", session=sid))


def spy(monkeypatch):
    seen = []

    def chat(messages, schema, model=None):
        seen.append(messages)
        if "summary" in json.dumps(schema):
            return json.dumps({"summary": "I learned that Dana likes tea."})
        return '{"reply": "ok"}'

    monkeypatch.setattr(AssistantModule, "fake_chat", staticmethod(chat))
    monkeypatch.setattr("prada.node.memory_fake_chat", chat)
    monkeypatch.setenv("PRADA_FAKE_LLM", "1")
    return seen


def test_conversation_survives_a_restart(tmp_path, monkeypatch):
    seen = spy(monkeypatch)
    clock = Clock()
    ask(node(tmp_path, clock), "my name is Dana")
    clock.later(60)
    ask(node(tmp_path, clock), "what is my name?", sid="another session")
    assert seen[-1][1:] == [
        {"role": "user", "content": "my name is Dana"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "what is my name?"},
    ]


def test_a_long_silence_starts_a_new_conversation(tmp_path, monkeypatch):
    seen = spy(monkeypatch)
    clock = Clock()
    n = node(tmp_path, clock, background=False)
    ask(n, "my name is Dana")
    clock.later(memory.EPISODE_GAP + 1)
    ask(n, "hello again")
    assert seen[-1][1:] == [{"role": "user", "content": "hello again"}]


def test_history_is_capped_at_max_turns(tmp_path, monkeypatch):
    seen = spy(monkeypatch)
    clock = Clock()
    n = node(tmp_path, clock)
    for i in range(MAX_TURNS):
        ask(n, f"m{i}")
        clock.later(1)
    assert len(seen[-1]) == 2 + MAX_TURNS  # system + the cap + the new message
    assert seen[-1][1]["content"] != "m0"


def test_finished_conversations_are_summarized_and_remembered(tmp_path, monkeypatch):
    seen = spy(monkeypatch)
    monkeypatch.setattr(llm, "fake_enabled", lambda: False)  # summaries kept, as with a real model
    monkeypatch.setattr("prada.node.Node._chat", lambda self, fake: fake)
    clock = Clock()
    n = node(tmp_path, clock, background=False)
    for text in ["my name is Dana", "I like tea"]:
        ask(n, text)
    n.summarize_pending()  # still going: nothing to summarize
    assert n._mem().remembered() == ""
    clock.later(memory.EPISODE_GAP + 1)
    ask(n, "hello again")
    n.summarize_pending()
    assert "I learned that Dana likes tea." in n._mem().remembered()
    ask(n, "do you remember me?")
    system = seen[-1][0]["content"]
    assert system.startswith(persona.CHARACTER)
    assert "What you remember of earlier conversations" in system
    assert "Dana likes tea" in system
    summary_prompt = next(m for m in seen if "remember of the conversation" in m[0]["content"])
    assert summary_prompt[0]["content"].startswith(persona.CHARACTER)
    assert "User: my name is Dana" in summary_prompt[1]["content"]


def test_short_conversations_are_skipped_and_fake_summaries_never_loaded(tmp_path):
    clock = Clock()
    from prada.store import open_store

    mem = memory.Memory(open_store("memory", memory.MIGRATIONS, ":memory:"), clock)
    mem.add("hi", "hello")
    clock.later(memory.EPISODE_GAP + 1)
    mem.add("a", "b")
    mem.add("c", "d")
    clock.later(memory.EPISODE_GAP + 1)
    mem.add("e", "f")
    calls = []

    def chat(messages, schema):
        calls.append(1)
        return json.dumps({"summary": "S"})

    run_pending(mem, chat, "fake")
    assert len(calls) == 1  # the one-exchange episode was skipped
    assert mem.remembered() == ""  # a fake summary is never loaded
    assert not mem.has_pending()


def test_summary_failure_is_retried_later(tmp_path):
    from prada.store import open_store

    clock = Clock()
    mem = memory.Memory(open_store("memory", memory.MIGRATIONS, ":memory:"), clock)
    for _ in range(2):
        mem.add("a", "b")
    clock.later(memory.EPISODE_GAP + 1)

    def down(messages, schema):
        raise ConnectionError("ollama is down")

    with pytest.raises(ConnectionError):
        run_pending(mem, down, "m")
    assert mem.has_pending()
    run_pending(mem, lambda messages, schema: json.dumps({"summary": "S"}), "m")
    assert not mem.has_pending() and "S" in mem.remembered()


def test_remembered_respects_the_budget_and_keeps_the_newest():
    from prada.store import open_store

    clock = Clock()
    mem = memory.Memory(open_store("memory", memory.MIGRATIONS, ":memory:"), clock)
    for i in range(5):
        for _ in range(2):
            mem.add("a", "b")
        clock.later(memory.EPISODE_GAP + 1)
        run_pending(mem, lambda messages, schema, i=i: json.dumps({"summary": f"s{i}" * 20}), "m")
    got = mem.remembered(budget=150)
    assert len(got) <= 150
    assert "s4" in got and "s0" not in got
    assert got.index("s3") < got.index("s4")  # oldest first


def test_unopenable_store_falls_back_to_ram(tmp_path, monkeypatch):
    seen = spy(monkeypatch)
    blocker = tmp_path / "file"
    blocker.write_text("x")
    n = Node([AssistantModule()], db_path=blocker / "p.db")
    ask(n, "one")
    ask(n, "two")
    assert len(seen[-1]) == 4


def test_context_memory_defaults_empty():
    assert Context(chat=lambda **k: "").memory == ""


def test_failed_action_drops_the_modules_own_text_and_merges_duplicates():
    from typing import Literal

    from prada.actions import Action, ActionFailed, Executor
    from prada.module import Module, Reply

    class Boom(Action):
        kind: Literal["t.boom"] = "t.boom"

        def describe(self):
            return "boom"

    class Ex(Executor):
        action = Boom

        def run(self, a, ctx):
            raise ActionFailed("no luck.")

        def check(self, a, r, ctx): ...

        def done(self, a, r):
            return "done"

    class T(Module):
        name = "t"
        description = "t"
        executors = [Ex()]

        def handle(self, request, ctx):
            return Reply(module="t", text="I did it.", actions=[Boom(), Boom()])

    reply = Node([T()], db_path=":memory:").handle(Request(text="x", module="t"))
    assert reply.text == persona.failed("boom", "no luck")
    assert "did it" not in reply.text and ".." not in reply.text
    assert len(reply.outcomes) == 2


def test_done_raising_counts_as_failed():
    from prada.actions import Action, Executor, execute

    class A(Action):
        kind: str = "t.a"

    class Ex(Executor):
        action = A

        def run(self, a, ctx):
            return {}

        def check(self, a, r, ctx): ...

        def done(self, a, r):
            raise ValueError("bad wording")

    out = execute(A(), Ex(), Context(chat=lambda **k: ""))
    assert not out.ok and "bad wording" in out.said


def test_failing_summaries_are_not_retried_at_once(tmp_path, monkeypatch):
    spy(monkeypatch)
    monkeypatch.setattr(llm, "fake_enabled", lambda: False)
    calls = []

    def down(messages, schema):
        calls.append(1)
        raise ConnectionError("ollama busy")

    monkeypatch.setattr("prada.node.memory_fake_chat", _memory_canned)
    monkeypatch.setattr("prada.node.Node._chat", lambda self, fake: pick(fake, down))
    clock = Clock()
    n = node(tmp_path, clock, background=False)
    for _ in range(2):
        ask(n, "x")
    clock.later(memory.EPISODE_GAP + 1)
    ask(n, "y")
    n.summarize_pending()
    n.summarize_pending()
    assert len(calls) == 1  # backed off
    clock.later(SUMMARY_RETRY + 1)
    n.summarize_pending()
    assert len(calls) == 2


def test_summarizing_does_not_hold_the_node_lock(tmp_path, monkeypatch):
    spy(monkeypatch)
    monkeypatch.setattr(llm, "fake_enabled", lambda: False)
    clock = Clock()
    n = node(tmp_path, clock, background=False)

    def chat(messages, schema):
        assert n._lock.acquire(blocking=False)  # a request could run right now
        n._lock.release()
        return json.dumps({"summary": "S"})

    monkeypatch.setattr("prada.node.memory_fake_chat", _memory_canned)
    monkeypatch.setattr("prada.node.Node._chat", lambda self, fake: pick(fake, chat))
    for _ in range(2):
        ask(n, "x")
    clock.later(memory.EPISODE_GAP + 1)
    ask(n, "y")
    n.summarize_pending()
    assert n._mem().remembered().endswith("S")


def test_a_memory_failure_does_not_hide_the_reply(tmp_path, monkeypatch):
    spy(monkeypatch)
    n = node(tmp_path, Clock())

    def broken(self, said, replied):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(memory.Memory, "add", broken)
    reply = ask(n, "hi")
    assert reply.error is None and reply.text == "ok"
