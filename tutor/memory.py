"""The tutor's tables in PRADA's shared memory: a log of every exchange."""

from prada import llm
from prada.store import open_store
from tutor.schema import TutorTurn

MIGRATIONS = [
    """
    CREATE TABLE tutor_exchange (
        id INTEGER PRIMARY KEY,
        ts TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
        language TEXT NOT NULL,
        model TEXT NOT NULL,
        user_text TEXT NOT NULL,
        is_correct INTEGER NOT NULL,
        error_type TEXT,
        correction TEXT,
        explanation TEXT,
        spoken TEXT NOT NULL
    );
    CREATE INDEX tutor_exchange_lang ON tutor_exchange (language, error_type);
    """
]


def open_memory(path=None):
    return open_store("tutor", MIGRATIONS, path)


def log_exchange(db, language: str, user_text: str, turn: TutorTurn) -> None:
    model = "fake" if llm.fake_enabled() else llm.model_name()
    with db:
        db.execute(
            "INSERT INTO tutor_exchange "
            "(language, model, user_text, is_correct, error_type, correction, explanation, spoken) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                language,
                model,
                user_text,
                int(turn.is_correct),
                turn.error_type.value if turn.error_type else None,
                turn.correction,
                turn.explanation_en,
                turn.spoken,
            ),
        )


def stats(db, language: str | None = None) -> dict:
    """Totals and per-category error counts for real (non-fake) exchanges."""
    where = "WHERE model != 'fake'" + (" AND language = ?" if language else "")
    args = (language,) if language else ()
    total, correct = db.execute(
        f"SELECT COUNT(*), COALESCE(SUM(is_correct), 0) FROM tutor_exchange {where}", args
    ).fetchone()
    rows = db.execute(
        f"SELECT error_type, COUNT(*) AS n FROM tutor_exchange {where} AND error_type IS NOT NULL "
        "GROUP BY error_type ORDER BY n DESC, error_type",
        args,
    ).fetchall()
    return {"total": total, "correct": correct, "errors": {r["error_type"]: r["n"] for r in rows}}


def format_stats(s: dict) -> str:
    if not s["total"]:
        return "No practice logged yet."
    lines = [f"{s['total']} sentences, {s['correct']} correct"]
    lines += [f"  {name}: {n}" for name, n in s["errors"].items()]
    return "\n".join(lines)
