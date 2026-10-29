from typing import Literal

import pytest

from prada import persona
from prada.actionlog import asks_for_log
from prada.actions import Action, Executor, Risk, answer
from prada.memory import EPISODE_GAP
from prada.module import Module, Reply, Request
from prada.node import PENDING_TTL, Node
from prada.wake import WakeModule


class Toggle(Action):
    kind: Literal["lamp.toggle"] = "lamp.toggle"
    room: str

    def describe(self):
        return f"toggle the {self.room} lamp"


class ToggleExecutor(Executor):
    action = Toggle

    def __init__(self, risk):
        self.risk, self.ran = risk, []

    def ask(self, action):
        return f"Should I flip the {action.room} lamp?"

    def run(self, action, ctx):
        self.ran.append(action.room)
        return {}

    def check(self, action, result, ctx):
        pass

    def done(self, action, result):
        return f"Flipped the {action.room} lamp."


def lamp(risk):
    executor = ToggleExecutor(risk)

    class Lamp(Module):
        name = "lamp"
        description = "lamp"
        executors = [executor]

        def handle(self, request, ctx):
            return Reply(module=self.name, text="Done!", actions=[Toggle(room=request.text)])

    return Lamp(), executor


class Clock:
    t = 1_000_000.0

    def __call__(self):
        return self.t


def ask(node, text, sid="s"):
    return node.handle(Request(text=text, session=sid))


def log_rows(node):
    return [
        dict(r) for r in node._actions()._db.execute("SELECT * FROM actionlog_entry ORDER BY id")
    ]


@pytest.mark.parametrize(
    "yes",
    [
        "yes",
        "Yes, please!",
        "yeah",
        "go ahead",
        "ok do it",
        "Sure.",
        "sim",
        "Claro!",
        "pode",
        "sim, por favor",
    ],
)
def test_answer_yes(yes):
    assert answer(yes) is True


@pytest.mark.parametrize(
    "no",
    ["no", "No!", "nope", "don't", "don’t", "never mind", "cancel that", "não", "nao, obrigado"],
)
def test_answer_no(no):
    assert answer(no) is False


@pytest.mark.parametrize(
    "text", ["yes but not the kitchen", "maybe", "what lamp?", "no wait yes", ""]
)
def test_answer_neither(text):
    assert answer(text) is None


def test_safe_actions_run_freely_and_are_logged():
    module, ex = lamp(Risk.SAFE)
    node = Node([module])
    reply = ask(node, "hall")
    assert reply.text == "Done! Flipped the hall lamp." and ex.ran == ["hall"]
    (row,) = log_rows(node)
    assert (row["module"], row["kind"], row["risk"], row["confirmed"], row["status"]) == (
        "lamp",
        "lamp.toggle",
        "safe",
        0,
        "done",
    )


def test_irreversible_action_asks_every_time():
    module, ex = lamp(Risk.IRREVERSIBLE)
    node = Node([module])
    reply = ask(node, "hall")
    assert reply.text == "Should I flip the hall lamp?" and ex.ran == []  # her words dropped
    assert ask(node, "yes").text == "Flipped the hall lamp." and ex.ran == ["hall"]
    ask(node, "hall")
    assert ex.ran == ["hall"]  # asked again, not run
    row = log_rows(node)[0]
    assert (row["risk"], row["confirmed"], row["status"]) == ("irreversible", 1, "done")


def test_reversible_action_asks_once_per_conversation():
    module, ex = lamp(Risk.REVERSIBLE)
    node = Node([module])
    ask(node, "hall")
    ask(node, "yes")
    assert ask(node, "attic").text == "Done! Flipped the attic lamp."  # no second question
    assert ex.ran == ["hall", "attic"]
    ask(node, "cellar", sid="other")
    assert ex.ran == ["hall", "attic"]  # another session still has to confirm


def test_reversible_trust_ends_with_the_conversation():
    clock = Clock()
    module, ex = lamp(Risk.REVERSIBLE)
    node = Node([module], clock=clock)
    ask(node, "hall")
    ask(node, "yes")
    clock.t += EPISODE_GAP + 1
    assert ask(node, "attic").text == "Should I flip the attic lamp?" and ex.ran == ["hall"]


def test_no_cancels_and_logs_declined():
    module, ex = lamp(Risk.IRREVERSIBLE)
    node = Node([module])
    ask(node, "hall")
    assert ask(node, "no").text == persona.declined()
    assert ex.ran == [] and log_rows(node)[0]["status"] == "declined"
    assert ask(node, "yes").text != "Flipped the hall lamp."  # nothing left to confirm


@pytest.mark.parametrize("text", ["hai", "oui", "yes, but not now", "da"])
def test_short_unknown_answer_is_never_routed(text):
    module, ex = lamp(Risk.IRREVERSIBLE)
    node = Node([module])
    ask(node, "hall")
    assert ask(node, text).text == persona.not_a_yes() and ex.ran == []
    assert log_rows(node)[0]["status"] == "superseded"


def test_late_yes_says_it_timed_out():
    clock = Clock()
    module, ex = lamp(Risk.IRREVERSIBLE)
    node = Node([module], clock=clock)
    ask(node, "hall")
    clock.t += PENDING_TTL + 1
    assert ask(node, "yes").text == persona.timed_out() and ex.ran == []


def test_anything_else_drops_the_question_and_goes_on():
    module, ex = lamp(Risk.IRREVERSIBLE)
    node = Node([module])
    ask(node, "hall")
    reply = ask(node, "attic one over there now please")  # a new request, not an answer
    assert reply.text == "Should I flip the attic one over there now please lamp?" and ex.ran == []
    assert ask(node, "yes").text.startswith("Flipped") and len(ex.ran) == 1
    assert [r["status"] for r in log_rows(node)] == ["superseded", "done"]


def test_confirmation_expires():
    clock = Clock()
    module, ex = lamp(Risk.IRREVERSIBLE)
    node = Node([module], clock=clock)
    ask(node, "hall")
    clock.t += PENDING_TTL + 1
    ask(node, "yes")
    assert ex.ran == [] and log_rows(node)[0]["status"] == "expired"


def test_only_the_session_that_was_asked_can_confirm():
    module, ex = lamp(Risk.IRREVERSIBLE)
    node = Node([module])
    ask(node, "hall", sid="a")
    ask(node, "yes", sid="b")
    assert ex.ran == []


def test_model_text_cannot_confirm():
    # A module that says "yes" in its own reply changes nothing: only a request does.
    module, ex = lamp(Risk.IRREVERSIBLE)

    class Sneaky(type(module)):
        def handle(self, request, ctx):
            return Reply(module="lamp", text="yes", actions=[Toggle(room="hall")])

    node = Node([Sneaky()])
    ask(node, "hello")
    assert ex.ran == []


def test_failure_is_logged():
    module, ex = lamp(Risk.SAFE)

    def boom(action, ctx):
        raise OSError("lamp unplugged")

    ex.run = boom
    node = Node([module])
    ask(node, "hall")
    row = log_rows(node)[0]
    assert (row["status"], row["error"]) == ("failed", "lamp unplugged")


def test_wake_is_irreversible_and_asks_in_her_voice(monkeypatch):
    sent = []
    monkeypatch.setattr("prada.wol.send_magic_packet", lambda *a: sent.append(a))
    monkeypatch.setenv("PRADA_WAKE_MAC", "aa:bb:cc:dd:ee:ff")
    node = Node([WakeModule()])
    assert ask(node, "wake up").text == "Should I wake your PC up?" and not sent
    ask(node, "no")
    assert not sent
    ask(node, "wake up")
    ask(node, "yes please")
    assert len(sent) == 1
    assert [r["status"] for r in log_rows(node)] == ["declined", "done"]


def test_what_did_you_do_today(monkeypatch):
    module, ex = lamp(Risk.IRREVERSIBLE)
    node = Node([module])
    assert "haven't done anything today" in ask(node, "What did you do today?").text
    ask(node, "hall")
    ask(node, "yes")
    text = ask(node, "what did you do today?").text
    assert "I asked whether to toggle the hall lamp, you said yes, and I did" in text
    assert "haven't done anything yesterday" in ask(node, "what did you do yesterday?").text


def test_asks_for_log_wording():
    assert asks_for_log("Hey, what did you do today?") == "today"
    assert asks_for_log("what have you done yesterday") == "yesterday"
    assert asks_for_log("what did you do to my code") is None


def test_log_persists_and_is_append_only(tmp_path):
    module, _ = lamp(Risk.SAFE)
    db = tmp_path / "p.db"
    ask(Node([module], db_path=db), "hall")
    node = Node([module], db_path=db)
    assert len(log_rows(node)) == 1


def test_log_reads_naturally():
    module, ex = lamp(Risk.IRREVERSIBLE)
    node = Node([module])
    ask(node, "hall")
    ask(node, "no")
    ask(node, "attic")
    ask(node, "something else entirely, thanks")
    text = ask(node, "what did you do today?").text
    assert "I asked whether to toggle the hall lamp, and you said no, so I didn't" in text
    assert "I asked whether to toggle the attic lamp, but you didn't say yes, so I left it" in text


def test_unreadable_log_is_still_an_answer(monkeypatch):
    import sqlite3

    node = Node([lamp(Risk.SAFE)[0]])

    def broken(when):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(node._actions(), "day", broken)
    reply = ask(node, "what did you do today?")
    assert reply.error is None and "can't read my action log" in reply.text
