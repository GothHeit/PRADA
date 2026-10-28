"""Typed actions: how a module asks for something to be done, and how the node does it.

A module never acts on the world itself. It returns `Action`s in `Reply.actions`; the node
looks each one up among the executors of the module that asked (anything else is refused),
runs it, checks the result, and only then lets PRADA say it happened. The sentence about an
action always comes from its `Outcome`: `Executor.done` when the check passed, otherwise
`persona.failed`, so a failure can never be reported as done.
"""

from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, Field

from prada import persona

if TYPE_CHECKING:
    from prada.module import Context


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
    except Exception as e:
        reason = str(e) or type(e).__name__
        return Outcome(
            action=asked, ok=False, said=persona.failed(action.describe(), reason), error=reason
        )
    return Outcome(action=asked, ok=True, said=executor.done(action, result), data=result)
