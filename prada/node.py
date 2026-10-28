"""The home node: hosts the modules, owns the shared store and model client, keeps sessions.

`Node.handle(request)` routes the request (unless it names a module), runs the module, then
runs the actions the module asked for (see prada.actions) and always returns a Reply: module
errors come back in `Reply.error` instead of raising. Calls are serialized, so modules never
share the model or a database connection concurrently. Each finished turn is added to the
session's shared conversation, so every module sees what PRADA has said, whoever said it.
"""

import sqlite3
import sys
import threading
from collections import OrderedDict
from collections.abc import Callable

from prada import actions, llm, router
from prada.module import Context, Module, Reply, Request
from prada.store import open_store

KEEP_ALIVE = "30m"  # how long Ollama keeps the model loaded between requests
MAX_SESSIONS = 64
MAX_TURNS = 20  # messages of shared conversation kept per session


def default_modules() -> list[Module]:
    """The built-in modules; the first (the assistant) is the router's fallback."""
    from prada.assistant import AssistantModule
    from prada.notes import NotesModule
    from prada.wake import WakeModule
    from tutor.module import TutorModule

    return [AssistantModule(), TutorModule(), WakeModule(), NotesModule()]


class Node:
    def __init__(self, modules: list[Module] | None = None, db_path=None):
        self.modules = {m.name: m for m in (modules or default_modules())}
        self._db_path = db_path
        self._dbs: dict[str, sqlite3.Connection | None] = {}
        self._sessions: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()

    def handle(self, request: Request, gone: Callable[[], bool] | None = None) -> Reply:
        """`gone()` is asked once the node is free: True means whoever sent the request has
        stopped waiting, so it is dropped instead of acting on it (e.g. a stale wake)."""
        with self._lock:
            name = request.module
            if gone is not None and gone():
                return Reply(module=name, error="the client went away; request dropped")
            try:
                session = self._session(request.session)
                if name is None:
                    name, request = self._route(request, session.get("current"))
                module = self.modules.get(name)
                if module is None:
                    return Reply(module=name, error=f"no module named {name!r}")
                ctx = Context(
                    chat=self._chat(module.fake_chat),
                    db=self._db(module),
                    state=session.setdefault(name, {}),
                    turns=list(session.get("turns", [])),
                )
                reply = self._act(module, module.handle(request, ctx), ctx)
                session["current"] = name
                self._remember(session, request.text, reply)
                return reply
            except Exception as e:
                return Reply(module=name, error=f"{name or 'router'}: {e}")

    def _act(self, module: Module, reply: Reply, ctx: Context) -> Reply:
        """Run the reply's actions with the module's own executors; what PRADA says about
        each one comes from its outcome, after the result was checked."""
        if not reply.actions:
            return reply
        allowed = {actions.kind_of(e.action): e for e in module.executors}
        outcomes = [actions.execute(a, allowed.get(a.kind), ctx) for a in reply.actions]
        text = " ".join(t for t in [reply.text, *(o.said for o in outcomes)] if t)
        return reply.model_copy(update={"text": text, "actions": [], "outcomes": outcomes})

    def _remember(self, session: dict, said: str, reply: Reply) -> None:
        if reply.error or not reply.text:
            return
        turns = session.setdefault("turns", [])
        turns.append({"role": "user", "content": said})
        turns.append({"role": "assistant", "content": reply.text})
        del turns[:-MAX_TURNS]

    def _route(self, request: Request, current: str | None) -> tuple[str, Request]:
        if hit := router.explicit(request.text, self.modules):
            name, text = hit
            return name, request.model_copy(update={"text": text})
        modules = list(self.modules.values())
        if len(modules) == 1:
            return modules[0].name, request
        if current in self.modules and self.modules[current].claims(request.text):
            return current, request  # no router call mid-conversation
        return router.route(request.text, modules, self._chat(router.fake_chat), current), request

    def _chat(self, fake):
        return llm.client(fake, keep_alive=KEEP_ALIVE)

    def _session(self, sid: str | None) -> dict:
        if sid is None:
            return {}
        if sid in self._sessions:
            self._sessions.move_to_end(sid)
        else:
            self._sessions[sid] = {}
            if len(self._sessions) > MAX_SESSIONS:
                self._sessions.popitem(last=False)
        return self._sessions[sid]

    def _db(self, module: Module) -> sqlite3.Connection | None:
        if not module.migrations:
            return None
        if module.name not in self._dbs:
            try:
                self._dbs[module.name] = open_store(module.name, module.migrations, self._db_path)
            except (sqlite3.Error, OSError, RuntimeError) as e:
                print(
                    f"{module.name}: memory unavailable, nothing will be logged ({e})",
                    file=sys.stderr,
                )
                self._dbs[module.name] = None
        return self._dbs[module.name]
