"""The home node: hosts the modules, owns the shared store and model client, keeps sessions.

`Node.handle(request)` routes the request (unless it names a module), runs the module, then
runs the actions the module asked for (see prada.actions) and always returns a Reply: module
errors come back in `Reply.error` instead of raising. Calls are serialized, so modules never
share the model or a database connection concurrently. Each finished turn is added to the
shared conversation, so every module sees what PRADA has said, whoever said it. The
conversation lives in the store (prada.memory), so it survives restarts; summaries of finished
conversations are written in the background and loaded into `Context.memory`.
"""

import sqlite3
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Callable

from prada import actions, llm, router
from prada.memory import MIGRATIONS as MEMORY_MIGRATIONS
from prada.memory import Memory, summarize
from prada.memory import fake_chat as memory_fake_chat
from prada.module import Context, Module, Reply, Request
from prada.store import open_store

KEEP_ALIVE = "30m"  # how long Ollama keeps the model loaded between requests
MAX_SESSIONS = 64
SUMMARY_RETRY = 5 * 60  # seconds before trying again after a summary failed
MAX_TURNS = 20  # messages of the current conversation given to a module


def default_modules() -> list[Module]:
    """The built-in modules; the first (the assistant) is the router's fallback."""
    from prada.assistant import AssistantModule
    from prada.notes import NotesModule
    from prada.wake import WakeModule
    from tutor.module import TutorModule

    return [AssistantModule(), TutorModule(), WakeModule(), NotesModule()]


class Node:
    def __init__(
        self,
        modules: list[Module] | None = None,
        db_path=None,
        clock=time.time,
        background: bool = True,
    ):
        """`background`: write episode summaries in a thread after a reply (tests turn it off
        and call `summarize_pending`)."""
        self._clock = clock
        self._background = background
        self._memory: Memory | None = None
        self._summarizing = False
        self._retry_at = 0.0
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
                memory = self._mem()
                if name is None:
                    name, request = self._route(request, session.get("current"))
                module = self.modules.get(name)
                if module is None:
                    return Reply(module=name, error=f"no module named {name!r}")
                ctx = Context(
                    chat=self._chat(module.fake_chat),
                    db=self._db(module),
                    state=session.setdefault(name, {}),
                    turns=memory.recent(MAX_TURNS),
                    memory=memory.remembered(),
                )
                reply = self._act(module, module.handle(request, ctx), ctx)
                session["current"] = name
                self._remember(memory, request.text, reply)
                return reply
            except Exception as e:
                return Reply(module=name, error=f"{name or 'router'}: {e}")
            finally:
                self._summarize_later()

    def _act(self, module: Module, reply: Reply, ctx: Context) -> Reply:
        """Run the reply's actions with the module's own executors; what PRADA says about
        each one comes from its outcome, after the result was checked."""
        if not reply.actions:
            return reply
        allowed = {actions.kind_of(e.action): e for e in module.executors}
        outcomes = [actions.execute(a, allowed.get(a.kind), ctx) for a in reply.actions]
        said = list(dict.fromkeys(o.said for o in outcomes if o.said))  # one line per kind
        # The module's own words come first, but never next to a failure: it may have
        # written them assuming the action would work.
        lead = [] if any(not o.ok for o in outcomes) else [reply.text]
        text = " ".join(t for t in [*lead, *said] if t)
        return reply.model_copy(update={"text": text, "actions": [], "outcomes": outcomes})

    def _remember(self, memory: Memory, said: str, reply: Reply) -> None:
        if reply.error or not reply.text:
            return
        try:
            memory.add(said, reply.text)
        except sqlite3.Error as e:
            # The actions already ran: the user must still hear what happened.
            print(f"memory: could not keep this turn ({e})", file=sys.stderr)

    def _mem(self) -> Memory:
        """The node's conversation memory; if the store can't be opened it lives in RAM for
        this run (PRADA still holds a conversation, it just won't survive a restart)."""
        if self._memory is None:
            try:
                db = open_store("memory", MEMORY_MIGRATIONS, self._db_path)
            except (sqlite3.Error, OSError, RuntimeError) as e:
                print(f"memory: conversations won't be kept after a restart ({e})", file=sys.stderr)
                db = open_store("memory", MEMORY_MIGRATIONS, ":memory:")
            self._memory = Memory(db, self._clock)
        return self._memory

    def summarize_pending(self) -> None:
        """Write the summary of every finished conversation that has none. One run at a time;
        the model call is made without the node lock, so requests never wait for it. After a
        failure nothing is retried for `SUMMARY_RETRY` seconds."""
        with self._lock:
            if self._summarizing or self._clock() < self._retry_at:
                return
            memory = self._mem()
            jobs = memory.jobs()
            if not jobs:
                return
            self._summarizing = True
            model = "fake" if llm.fake_enabled() else llm.MODEL
            chat = self._chat(memory_fake_chat)
        try:
            for episode, transcript in jobs:
                summary = "" if transcript is None else summarize(chat, transcript)
                with self._lock:
                    memory.store(episode, summary, model)
        except Exception as e:
            print(f"memory: could not summarize a conversation ({e})", file=sys.stderr)
            with self._lock:
                self._retry_at = self._clock() + SUMMARY_RETRY
        finally:
            with self._lock:
                self._summarizing = False

    def _summarize_later(self) -> None:
        if not self._background or self._summarizing or self._clock() < self._retry_at:
            return
        try:
            pending = self._mem().has_pending()
        except Exception:
            return
        if pending:
            threading.Thread(target=self.summarize_pending, daemon=True).start()

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
