"""The notes module: short notes the user dictates, saved to the shared store and read back.

Saving is a typed action: the node writes the note, reads it back to check it, and only then
does PRADA say it is written down. No model call: the wording is matched directly.
"""

import re
from typing import Literal

from prada import persona
from prada.actions import Action, ActionFailed, Executor
from prada.module import Context, Module, Reply, Request

MIGRATIONS = [
    """
    CREATE TABLE notes_note (
        id INTEGER PRIMARY KEY,
        ts TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
        text TEXT NOT NULL
    );
    """
]
SHOWN = 10  # most recent notes read back

_SAVE = re.compile(
    r"^(?:please\s+)?(?:(?:take|make|write)\s+(?:a\s+|this\s+)?note(?:\s+(?:that|of))?"
    r"|note(?:\s+that)?|(?:write|jot|put)\s+(?:this|that|it)?\s*down)\s*[:,-]?\s*",
    re.IGNORECASE,
)
_LIST = re.compile(
    r"^\s*(?:(?:my\s+)?notes|(?:what(?:'s|\s+is|\s+are|\s+did\s+i)|list|read|show|tell\s+me)\b.*\bnot(?:e|es|ed)\b.*)\W*$",
    re.IGNORECASE,
)


class SaveNote(Action):
    kind: Literal["notes.save"] = "notes.save"
    text: str

    def describe(self) -> str:
        return "save that note"


class SaveNoteExecutor(Executor):
    action = SaveNote

    def run(self, action: SaveNote, ctx: Context) -> dict:
        if ctx.db is None:
            raise ActionFailed("my memory isn't available right now")
        with ctx.db:
            cur = ctx.db.execute("INSERT INTO notes_note (text) VALUES (?)", (action.text,))
        return {"id": cur.lastrowid}

    def check(self, action: SaveNote, result: dict, ctx: Context) -> None:
        row = ctx.db.execute("SELECT text FROM notes_note WHERE id = ?", (result["id"],)).fetchone()
        if row is None or row["text"] != action.text:
            raise ActionFailed("the note didn't stick")

    def done(self, action: SaveNote, result: dict) -> str:
        return persona.noted(action.text)


def note_text(text: str) -> str:
    """The note in "note that we're out of milk" ("we're out of milk")."""
    return _SAVE.sub("", text.strip(), count=1).strip()


class NotesModule(Module):
    name = "notes"
    description = (
        'Saves a short note the user dictates ("note that...", "write this down...") and '
        "reads their notes back."
    )
    migrations = MIGRATIONS
    executors = [SaveNoteExecutor()]

    def handle(self, request: Request, ctx: Context) -> Reply:
        if not request.text.strip() or _LIST.match(request.text):
            return Reply(module=self.name, text=self._read(ctx))
        text = note_text(request.text)
        if not text:
            return Reply(module=self.name, text="Sure, what should I write down?")
        return Reply(module=self.name, actions=[SaveNote(text=text)])

    def _read(self, ctx: Context) -> str:
        if ctx.db is None:
            return "I can't reach my memory right now, so I can't read your notes."
        rows = ctx.db.execute(
            "SELECT text FROM notes_note ORDER BY id DESC LIMIT ?", (SHOWN,)
        ).fetchall()
        if not rows:
            return "You don't have any notes yet."
        listed = "; ".join(f"{i}) {r['text']}" for i, r in enumerate(reversed(rows), 1))
        return f"Here are your notes: {listed}."
