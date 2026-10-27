"""The tutor as a PRADA module: one practice exchange per request, logged to shared memory."""

import sqlite3
import sys

from prada.module import Context, Module, Reply, Request
from tutor import fake_client, memory
from tutor.languages import detect_language, get_language
from tutor.orchestrator import run_turn

MAX_HISTORY = 20  # messages kept per session, so the system prompt never gets truncated


class TutorModule(Module):
    name = "tutor"
    description = (
        "Language practice: corrects a sentence the user wrote in Russian, French or Japanese "
        "and keeps a practice conversation going. Not for questions in English."
    )
    migrations = memory.MIGRATIONS
    fake_chat = staticmethod(fake_client.chat)

    def claims(self, text: str) -> bool:
        return detect_language(text) is not None

    def handle(self, request: Request, ctx: Context) -> Reply:
        # An explicit --lang wins; then what the text itself is written in (script, or French
        # words); then the session's language. Anything unclear is practised as French.
        lang = (
            request.options.get("lang")
            or detect_language(request.text)
            or ctx.state.get("lang")
            or "fr"
        )
        lang = get_language(lang).code
        if ctx.state.get("lang") != lang:
            ctx.state.update(lang=lang, history=[])
        history = ctx.state["history"]
        turn = run_turn(request.text, lang, history, chat=ctx.chat)
        if ctx.db is not None:
            try:
                memory.log_exchange(ctx.db, lang, request.text, turn)
            except sqlite3.Error as e:
                print(f"tutor: could not log ({e})", file=sys.stderr)
        history.append({"role": "user", "content": request.text})
        history.append({"role": "assistant", "content": turn.spoken})
        del history[:-MAX_HISTORY]
        return Reply(module=self.name, text=turn.spoken, data=turn.model_dump(mode="json"))
