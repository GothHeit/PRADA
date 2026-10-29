"""Typed actions: how a module asks for something to be done, and how the node does it.

A module never acts on the world itself. It returns `Action`s in `Reply.actions`; the node
looks each one up among the executors of the module that asked (anything else is refused),
runs it, checks the result, and only then lets PRADA say it happened. The sentence about an
action always comes from its `Outcome`: `Executor.done` when the check passed, otherwise
`persona.failed`, so a failure can never be reported as done.
"""

import re
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, Field

from prada import persona

if TYPE_CHECKING:
    from prada.module import Context


class Risk(StrEnum):
    """How much an executor can hurt, declared by the executor and enforced by the node.

    SAFE runs freely. REVERSIBLE asks once per conversation and kind, then runs freely for
    the rest of it. IRREVERSIBLE (and anything that reaches outside the house) asks every
    time. The model never decides this: it only chooses an action.
    """

    SAFE = "safe"
    REVERSIBLE = "reversible"
    IRREVERSIBLE = "irreversible"


# English and Portuguese.
_YES = re.compile(
    r"(?:yes|yeah|yep|yup|sure|ok|okay|please|go ahead|do it|go for it|confirm|confirmed"
    r"|that's right|that is right|correct|affirmative"
    r"|sim|claro|pode|pode sim|pode ser|isso|isso mesmo|com certeza|beleza|manda ver|faz isso)"
    r"(?:[ ,]+(?:please|do it|go ahead|prada|thanks|thank you|yes|sure|okay|ok"
    r"|sim|claro|pode|por favor|obrigado|obrigada|valeu))*",
    re.IGNORECASE,
)
_NO = re.compile(
    r"(?:no|nope|nah|cancel|stop|don't|do not|never ?mind|not now|wait|abort|negative"
    r"|não|nao|cancela|para|espera|deixa|deixa pra lá|agora não|não precisa)"
    r"(?:[ ,]+(?:please|thanks|thank you|prada|don't|do not|do it|stop|cancel|that|it"
    r"|não|nao|obrigado|obrigada|valeu|cancela|precisa))*",
    re.IGNORECASE,
)


def answer(text: str) -> bool | None:
    """True / False if `text` is a plain yes / no to a question, else None. Read from what
    the person said, never from model output; anything longer or vaguer is not an answer."""
    t = text.lower().replace("\u2019", "'")  # don’t
    t = re.sub(r"[^\w' ]+", " ", t).strip()
    t = re.sub(r"\s+", " ", t)
    if _YES.fullmatch(t):
        return True
    if _NO.fullmatch(t):
        return False
    return None


class Action(BaseModel):
    """Something a module wants done. Subclasses fix `kind` ("<module>.<verb>") and add
    the fields the executor needs."""

    kind: str

    def describe(self) -> str:
        """What PRADA tried to do, for the failure sentence ("I tried to <describe>...")."""
        return "do that"


class Outcome(BaseModel):
    """What happened when the node ran an action."""

    action: dict = Field(..., description="The action as the module asked for it")
    ok: bool = Field(..., description="True only if the action ran and its result was checked")
    said: str = Field("", description="What PRADA says about it")
    data: dict = Field(default_factory=dict, description="The executor's result")
    error: str | None = None


class ActionFailed(Exception):
    """Raised by an executor when the action did not happen (or can't be shown to have)."""


class Executor:
    """Runs one kind of action. `run` does it and returns its result, `check` confirms it
    against the world (raise ActionFailed if it doesn't hold), `done` says it happened."""

    action: ClassVar[type[Action]]
    risk: ClassVar[Risk] = Risk.SAFE

    def ask(self, action: Action) -> str:
        """The question PRADA puts to the person before a risky action runs, with this
        action's real values."""
        return f"Should I {action.describe()}?"

    def run(self, action: Action, ctx: "Context") -> dict:
        raise NotImplementedError

    def check(self, action: Action, result: dict, ctx: "Context") -> None:
        raise NotImplementedError

    def done(self, action: Action, result: dict) -> str:
        raise NotImplementedError


def kind_of(action_type: type[Action]) -> str:
    return action_type.model_fields["kind"].default


def execute(action: Action, executor: Executor | None, ctx: "Context") -> Outcome:
    """Run, check and describe one action; any failure becomes a not-ok Outcome."""
    asked = action.model_dump(mode="json")
    if executor is None or not isinstance(action, executor.action):
        reason = "that isn't something I'm allowed to do"
        return Outcome(
            action=asked, ok=False, said=persona.failed(action.describe(), reason), error=reason
        )
    try:
        result = executor.run(action, ctx)
        executor.check(action, result, ctx)
        said = executor.done(action, result)
    except Exception as e:
        reason = str(e) or type(e).__name__
        return Outcome(
            action=asked, ok=False, said=persona.failed(action.describe(), reason), error=reason
        )
    return Outcome(action=asked, ok=True, said=said, data=result)
