"""The home node: hosts the modules, owns the shared store and model client, keeps sessions.

`Node.handle(request)` routes the request (unless it names a module), runs the module, then
runs the actions the module asked for (see prada.actions) and always returns a Reply: module
errors come back in `Reply.error` instead of raising. Calls are serialized, so modules never
share the model or a database connection concurrently. Each finished turn is added to the
shared conversation, so every module sees what PRADA has said, whoever said it. The
conversation lives in the store (prada.memory), so it survives restarts; summaries of finished
conversations are written in the background and loaded into `Context.memory`.

The node, not the model, decides whether an action may run: each executor declares a
`Risk` (prada.actions). A risky action is held, PRADA asks about it with the real values, and
only the person's next request can confirm it (a plain yes) or cancel it (a no, anything
else, or `PENDING_TTL` seconds of silence). A reversible kind, once confirmed, stays trusted
for the rest of that conversation (`EPISODE_GAP` of silence ends it). Every action's end is appended to the action log
(prada.actionlog).
"""

import sqlite3
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from prada import actionlog, actions, llm, persona, router
from prada.actionlog import MIGRATIONS as LOG_MIGRATIONS
from prada.actionlog import ActionLog
from prada.actions import Risk
from prada.memory import EPISODE_GAP, Memory, summarize
from prada.memory import MIGRATIONS as MEMORY_MIGRATIONS
from prada.memory import fake_chat as memory_fake_chat
from prada.module import Context, Module, Reply, Request
from prada.store import open_store

KEEP_ALIVE = "30m"  # how long Ollama keeps the model loaded between requests
MAX_SESSIONS = 64
SUMMARY_RETRY = 5 * 60  # seconds before trying again after a summary failed
MAX_TURNS = 20  # messages of the current conversation given to a module
PENDING_TTL = 60  # seconds a question about a risky action stays open
SHORT_ANSWER_WORDS = 4  # a reply this short to a held question is never routed as chat


@dataclass
class Pending:
    """Actions held until the person answers; `risks` parallels `held`."""

    module: str
    held: list[actions.Action]
    asked_at: float


def default_modules() -> list[Module]:
    """The built-in modules; the first (the assistant) is the router's fallback."""
    from prada.assistant import AssistantModule
    from prada.notes import NotesModule
    from prada.wake import WakeModule
    from prada.web.module import WebModule
    from tutor.module import TutorModule

    return [AssistantModule(), TutorModule(), WakeModule(), NotesModule(), WebModule()]


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
        self._log: ActionLog | None = None
        self._pending: dict[str | None, Pending] = {}  # by session id
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
                now = self._clock()
                if now - session.get("_seen", now) > EPISODE_GAP:
                    session.pop("_trusted", None)  # a new conversation confirms again
                session["_seen"] = now
                memory = self._mem()
                if (reply := self._answer(request, session, memory)) is not None:
                    name = reply.module
                    self._remember(memory, request.text, reply)
                    return reply
                if name is None:
                    name, request = self._route(request, session.get("current"))
                module = self.modules.get(name)
                if module is None:
                    return Reply(module=name, error=f"no module named {name!r}")
                ctx = self._context(module, session, memory)
                reply = self._act(
                    module, module.handle(request, ctx), ctx, request.session, session
                )
                session["current"] = name
                self._remember(memory, request.text, reply)
                return reply
            except Exception as e:
                return Reply(module=name, error=f"{name or 'router'}: {e}")
            finally:
                self._summarize_later()

    def _context(self, module: Module, session: dict, memory: Memory) -> Context:
        return Context(
            chat=self._chat(module.fake_chat),
            db=self._db(module),
            state=session.setdefault(module.name, {}),
            turns=memory.recent(MAX_TURNS),
            memory=memory.remembered(),
            recall=lambda text: memory.recall(text, MAX_TURNS),
        )

    def _answer(self, request: Request, session: dict, memory: Memory) -> Reply | None:
        """Handle a request that answers a held action or asks about the action log; None
        when it is neither and should be routed as usual. A held action is settled by the
        very next request: yes runs it, anything else drops it."""
        pending = self._pending.pop(request.session, None)
        if pending is not None:
            module = self.modules[pending.module]
            fresh = self._clock() - pending.asked_at <= PENDING_TTL
            yes = actions.answer(request.text) if fresh else None
            if yes:
                ctx = self._context(module, session, memory)
                self._trusted(session).update(a.kind for a in pending.held)
                outcomes = [self._do(module, a, ctx, confirmed=True) for a in pending.held]
                said = " ".join(dict.fromkeys(o.said for o in outcomes if o.said))
                return Reply(module=module.name, text=said, outcomes=outcomes)
            status = "declined" if yes is False else "superseded" if fresh else "expired"
            for a in pending.held:
                self._record(module, a, status=status)
            if yes is False:
                return Reply(module=module.name, text=persona.declined())
            # A short reply is probably a yes or no in words she doesn't know ("sim", "yes,
            # but not now") or a late yes. Routed as chat, the model could answer "done"
            # to her own question, so settle it here: nothing ran, and she says so.
            if len(request.text.split()) <= SHORT_ANSWER_WORDS:
                text = persona.timed_out() if not fresh else persona.not_a_yes()
                return Reply(module=module.name, text=text)
        if request.module is None and (when := actionlog.asks_for_log(request.text)):
            try:
                text = self._actions().day(when)
            except sqlite3.Error as e:
                print(f"actions: could not read the action log ({e})", file=sys.stderr)
                text = "I can't read my action log right now, so I can't tell you."
            return Reply(module="actions", text=text)
        return None

    @staticmethod
    def _trusted(session: dict) -> set[str]:
        """The reversible kinds this conversation confirmed. Kept in the session, so it is
        bounded by `MAX_SESSIONS` (a request without a session trusts nothing)."""
        return session.setdefault("_trusted", set())

    def _act(self, module: Module, reply: Reply, ctx: Context, sid, session: dict) -> Reply:
        """Run the reply's actions with the module's own executors; what PRADA says about
        each one comes from its outcome, after the result was checked. Actions whose risk
        needs the person's yes are held and asked about instead."""
        if not reply.actions:
            return reply
        allowed = {actions.kind_of(e.action): e for e in module.executors}
        outcomes, held = [], []
        for a in reply.actions:
            ex = allowed.get(a.kind)
            if ex is not None and self._needs_yes(ex, a, session):
                held.append(a)
            else:
                outcomes.append(self._do(module, a, ctx, confirmed=False))
        said = [o.said for o in outcomes if o.said]
        if held:
            self._pending[sid] = Pending(module.name, held, self._clock())
            said += [allowed[a.kind].ask(a) for a in held]
        # The module's own words come first, but never next to a failure or a question: it
        # may have written them assuming the action would work.
        lead = [] if held or any(not o.ok for o in outcomes) else [reply.text]
        text = " ".join(t for t in [*lead, *dict.fromkeys(said)] if t)  # one line per kind
        return reply.model_copy(update={"text": text, "actions": [], "outcomes": outcomes})

    def _needs_yes(self, executor: actions.Executor, action: actions.Action, session) -> bool:
        if executor.risk is Risk.SAFE or not isinstance(action, executor.action):
            return False  # an action the executor can't run is refused, not asked about
        if executor.risk is Risk.REVERSIBLE:
            return action.kind not in self._trusted(session)
        return True

    def _do(self, module: Module, action: actions.Action, ctx: Context, confirmed: bool):
        allowed = {actions.kind_of(e.action): e for e in module.executors}
        executor = allowed.get(action.kind)
        outcome = actions.execute(action, executor, ctx)
        refused = executor is None or not isinstance(action, executor.action)
        status = "refused" if refused else "done" if outcome.ok else "failed"
        self._record(module, action, executor, status, confirmed, outcome.error)
        return outcome

    def _record(self, module, action, executor=None, status="done", confirmed=False, error=None):
        if executor is None:
            executor = next((e for e in module.executors if isinstance(action, e.action)), None)
        risk = executor.risk if executor else Risk.SAFE
        try:
            self._actions().add(
                module.name,
                action.model_dump(mode="json"),
                action.describe(),
                risk.value,
                confirmed,
                status,
                error,
            )
        except sqlite3.Error as e:
            # The action already ran (or didn't): the user must still hear what happened.
            print(f"actions: could not log this action ({e})", file=sys.stderr)

    def _actions(self) -> ActionLog:
        if self._log is None:
            try:
                db = open_store("actionlog", LOG_MIGRATIONS, self._db_path)
            except (sqlite3.Error, OSError, RuntimeError) as e:
                print(f"actions: the action log won't be kept ({e})", file=sys.stderr)
                db = open_store("actionlog", LOG_MIGRATIONS, ":memory:")
            self._log = ActionLog(db, self._clock)
        return self._log

    def _remember(self, memory: Memory, said: str, reply: Reply) -> None:
        if reply.error or not reply.text:
            return
        text = reply.text
        module = self.modules.get(reply.module)
        if module is not None and module.memory_note and any(o.ok for o in reply.outcomes):
            text = f"{module.memory_note} {text}"
        try:
            memory.add(said, text)
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
        name = router.route(request.text, modules, self._chat(router.fake_chat), current)
        if not self.modules[name].accepts(request.text):
            name = modules[0].name  # e.g. a question the model sent to notes
        return name, request

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
