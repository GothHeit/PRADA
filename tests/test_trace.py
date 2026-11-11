import json
import sqlite3
from typing import Literal

from prada import cli, llm, trace
from prada.actions import Action, Executor, Risk
from prada.module import Module, Reply, Request
from prada.node import Node
from prada.store import open_store


class Ping(Action):
    kind: Literal["ping.send"] = "ping.send"

    def describe(self):
        return "ping the box"


class PingExecutor(Executor):
    action = Ping

    def __init__(self, risk=Risk.SAFE):
        self.risk = risk

    def run(self, action, ctx):
        return {}

    def check(self, action, result, ctx):
        pass

    def done(self, action, result):
        return "Pinged."


class Pinger(Module):
    """Asks the model once (schema "Thought"), then asks for a ping."""

    name = "ping"
    description = "Pings."

    def __init__(self, risk=Risk.SAFE):
        self.executors = [PingExecutor(risk)]

    def handle(self, request, ctx):
        ctx.chat(messages=[{"role": "user", "content": request.text}], schema={"title": "Thought"})
        return Reply(module=self.name, text="On it.", actions=[Ping()])


class Talker(Module):
    name = "talk"
    description = "Talks."

    def handle(self, request, ctx):
        return Reply(module=self.name, text="Hello.")


def route_to(name):
    def chat(messages, schema, on_text=None):
        if schema.get("title") == "Thought":
            return "{}"
        return json.dumps({"reasoning": "test", "module": name})

    return chat


def make(tmp_path, *modules, route="ping"):
    return Node(
        list(modules) or [Talker(), Pinger()],
        db_path=tmp_path / "p.db",
        background=False,
        client=lambda fake: route_to(route),
    )


def test_routed_turn_records_path_calls_actions_and_reply(tmp_path):
    n = make(tmp_path)
    reply = n.handle(Request(text="ping it please", session="s"))
    assert reply.text == "On it. Pinged."
    (turn,) = n._traces().last()
    assert turn["heard"] == "ping it please" and turn["session"] == "s"
    assert turn["module"] == "ping" and turn["reply"] == "On it. Pinged."
    assert turn["path"] == ["router picked ping"]
    assert [c["prompt"] for c in turn["calls"]] == ["router/Route", "ping/Thought"]
    assert all(c["ms"] >= 0 for c in turn["calls"])
    assert [(a["kind"], a["status"]) for a in turn["actions"]] == [("ping.send", "done")]
    assert turn["total_ms"] >= 0 and turn["error"] is None


def test_held_action_and_the_answer_are_both_recorded(tmp_path):
    n = make(tmp_path, Talker(), Pinger(Risk.IRREVERSIBLE))
    n.handle(Request(text="ping it", session="s"))
    n.handle(Request(text="yes", session="s"))
    first, second = n._traces().last()
    assert [a["status"] for a in first["actions"]] == ["asked"]
    assert second["path"] == ["answer to a held question (yes, code)"]
    assert [a["status"] for a in second["actions"]] == ["done"]
    assert second["calls"] == []  # settled by code, no model


def test_named_module_and_errors_are_recorded(tmp_path):
    n = make(tmp_path)
    n.handle(Request(text="hi", module="talk"))
    n.handle(Request(text="hi", module="nope"))
    named, missing = n._traces().last()
    assert named["path"] == ["sent to talk by name"] and named["calls"] == []
    assert missing["error"] == "no module named 'nope'"


def test_store_error_never_changes_the_reply(tmp_path, capsys):
    n = make(tmp_path)

    class Broken:
        def add(self, *args):
            raise sqlite3.OperationalError("disk full")

    n._trace = Broken()
    assert n.handle(Request(text="hi", module="talk")).text == "Hello."
    assert "trace: could not record this turn" in capsys.readouterr().err


def test_token_counts_come_from_the_model_reply():
    usage = {}
    token = llm.USAGE.set(usage)
    try:
        llm._count({"message": {"content": "{}"}, "prompt_eval_count": 812, "eval_count": 9})
    finally:
        llm.USAGE.reset(token)
    assert usage == {"tokens_in": 812, "tokens_out": 9}
    llm._count({"prompt_eval_count": 1})  # nobody listening: nothing happens


def test_only_recent_turns_are_kept(monkeypatch):
    now = [1_000_000.0]
    log = trace.Trace(open_store("trace", trace.MIGRATIONS, ":memory:"), lambda: now[0])
    log.add(trace.Turn("old", started=now[0]), None, "", None)
    now[0] += trace.KEEP_DAYS * 86400 + 60
    monkeypatch.setattr(trace, "MAX_TURNS", 2)
    for text in ("a", "b", "c"):
        log.add(trace.Turn(text, started=now[0]), None, "", None)
    assert [t["heard"] for t in log.last()] == ["b", "c"]


def test_case_line_takes_earlier_turns_of_the_same_conversation():
    now = [1_000_000.0]
    log = trace.Trace(open_store("trace", trace.MIGRATIONS, ":memory:"), lambda: now[0])
    for text, session, gap in [
        ("long ago", "s", 0),
        ("play muse", "s", trace.BEFORE_GAP + 60),
        ("other room", "t", 5),
        ("skip this one", "s", 5),
    ]:
        now[0] += gap
        log.add(trace.Turn(text, session, started=now[0]), "music", "", None)
    turn = log.last(1)[0]
    line = json.loads(trace.case_line(turn, log.before(turn)))
    assert line == {
        "category": "routing",
        "text": "skip this one",
        "before": ["play muse"],
        "module": "music",
    }


def test_prada_trace_shows_turns_and_prints_a_case(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PRADA_DATA_DIR", str(tmp_path))
    assert cli.main(["trace"]) == 0
    assert "No turns recorded yet." in capsys.readouterr().out
    n = Node([Talker(), Pinger()], background=False, client=lambda fake: route_to("ping"))
    n.handle(Request(text="ping it please", session="s"))
    assert cli.main(["trace"]) == 0
    out = capsys.readouterr().out
    assert "heard: 'ping it please'" in out and "router picked ping" in out
    assert "router/Route" in out and "ping the box [ping.send]: done" in out
    turn_id = n._traces().last()[0]["id"]
    assert cli.main(["trace", "--case", str(turn_id)]) == 0
    case = json.loads(capsys.readouterr().out)
    assert case == {"category": "routing", "text": "ping it please", "module": "ping"}
    assert cli.main(["trace", "--case", "999"]) == 1
