"""The module contract: how a capability (tutor, wake, ...) plugs into PRADA.

A module is a class with a `name`, a one-line `description` of what it handles (the router
reads it), and `handle(request, ctx) -> Reply`. The node owns everything shared and hands
it over in `Context`: the model client, the module's own tables in the shared store,
per-session state and the conversation so far.

A module that wants something done returns typed `Action`s in `Reply.actions` instead of
doing it, and lists the `Executor`s it may use (see prada.actions). The node runs them after
`handle` and adds what happened to the reply, so `Reply.text` must never claim an action
itself.
"""

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar

from pydantic import BaseModel, Field

from prada.actions import Action, Executor, Outcome

Chat = Callable[..., str]


class Request(BaseModel):
    text: str
    module: str | None = Field(None, description="Skip the router and send to this module")
    session: str | None = Field(None, description="Conversation id; state is kept per session")
    options: dict[str, str] = Field(default_factory=dict, description="Module options, e.g. lang")


class Cut(BaseModel):
    """She was interrupted: of her reply `text`, only `heard` was said out loud."""

    text: str
    heard: str = ""


class Reply(BaseModel):
    module: str | None = None
    text: str = Field("", description="What PRADA says back")
    data: dict = Field(default_factory=dict, description="The module's structured result")
    error: str | None = None
    # What the module asks the node to do; run by the node, never sent to clients.
    actions: list[Action] = Field(default_factory=list, exclude=True)
    outcomes: list[Outcome] = Field(default_factory=list, description="Actions the node ran")
    # A message she offered to look up online; a plain yes next searches it (the node
    # hands it to the web module). Never sent to clients.
    offer: str = Field("", exclude=True)


@dataclass
class Context:
    chat: Chat
    db: sqlite3.Connection | None = None  # None when the store could not be opened
    state: dict = field(default_factory=dict)  # this module's state for the session
    # The session's conversation with PRADA as chat messages, whichever module answered;
    # read-only for modules (the node appends each finished turn).
    turns: list[dict] = field(default_factory=list)
    # What PRADA remembers of earlier conversations (episode summaries, newest first,
    # within a budget); "" when there is nothing yet. Read-only.
    memory: str = ""
    # Searches her memory of earlier conversations for a message and returns the few best
    # matches as lines of text ("" if none); the tutor never calls it. Read-only.
    recall: Callable[[str], str] = lambda text: ""  # noqa: E731
    # What the node's own code knows right now (date and time, when they last talked,
    # actions that ran or were dropped), one line each, for prada.status.attach. Read-only.
    status: list[str] = field(default_factory=list)
    # Speaks a finished sentence of the reply right away (voice clients), or None. Given
    # only to modules without executors, whose text the node never changes, and every
    # sentence said must be the start of `Reply.text`, in order.
    say: Callable[[str], None] | None = None


class Module:
    name: ClassVar[str]
    description: ClassVar[str]
    # Append-only SQL migrations for the module's tables (see prada.store).
    migrations: ClassVar[list[str]] = []
    # Canned chat used under PRADA_FAKE_LLM; None if the module never calls the model.
    fake_chat: ClassVar[Chat | None] = None
    # The actions this module may ask for; the node refuses any other.
    executors: ClassVar[list[Executor]] = []
    # Put before her reply in the conversation memory when one of the module's actions ran,
    # so later turns and summaries know where it came from (e.g. "[from a web search]").
    memory_note: ClassVar[str] = ""

    def claims(self, text: str, state: dict) -> bool:
        """True when `text` plainly continues a conversation with this module, so the router
        model is skipped. Only asked of the module the session is already talking to;
        `state` is the module's state for the session (read-only here)."""
        return False

    def wants(self, text: str) -> bool:
        """True when `text` is an explicit command for this module ("note that ..."), so it
        goes here without the router model, unless the current module claims it."""
        return False

    def accepts(self, text: str) -> bool:
        """False when the router model picked this module for `text` but the module can tell
        from the words that it isn't for it; the node then sends it to the fallback module.
        Not asked when the request names the module."""
        return True

    def handle(self, request: Request, ctx: Context) -> Reply:
        raise NotImplementedError
