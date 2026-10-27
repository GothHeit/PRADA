"""The home node: hosts the modules, owns the shared store and model client, keeps sessions.

`Node.handle(request)` routes the request (unless it names a module), runs the module and
always returns a Reply: module errors come back in `Reply.error` instead of raising. Calls
are serialized, so modules never share the model or a database connection concurrently.
"""

import sqlite3
import sys
import threading
from collections import OrderedDict
from collections.abc import Callable

from prada import llm, router
from prada.module import Context, Module, Reply, Request
from prada.store import open_store

KEEP_ALIVE = "30m"  # how long Ollama keeps the model loaded between requests
MAX_SESSIONS = 64


def default_modules() -> list[Module]:
    """The built-in modules; the first (the assistant) is the router's fallback."""
    from prada.assistant import AssistantModule
    from prada.wake import WakeModule
    from tutor.module import TutorModule

    return [AssistantModule(), TutorModule(), WakeModule()]


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
                )
                reply = module.handle(request, ctx)
                session["current"] = name
                return reply
            except Exception as e:
                return Reply(module=name, error=f"{name or 'router'}: {e}")

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
