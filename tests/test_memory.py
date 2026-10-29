import pytest

from prada import config, store
from prada.cli import main
from tutor import memory
from tutor.schema import TutorTurn


def make_turn(correct=True, error_type=None):
    return TutorTurn(
        reasoning="r",
        is_correct=correct,
        error_type=error_type,
        original="x",
        correction=None if correct else "y",
        explanation_en=None if correct else "because",
        reply_target="?",
        spoken="ok ?",
    )


def test_migrations_run_once_and_are_versioned(tmp_path):
    path = tmp_path / "sub" / "p.db"
    store.open_store("m", ["CREATE TABLE a (x)"], path).close()
    db = store.open_store("m", ["CREATE TABLE a (x)", "CREATE TABLE b (y)"], path)
    assert db.execute("SELECT version FROM schema_versions WHERE module='m'").fetchone()[0] == 2
    db.execute("SELECT * FROM b")


def test_newer_schema_is_refused(tmp_path):
    path = tmp_path / "p.db"
    store.open_store("m", ["CREATE TABLE a (x)"], path).close()
    with pytest.raises(RuntimeError):
        store.open_store("m", [], path)


def test_data_dir_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "x"))
    monkeypatch.delenv("PRADA_DATA_DIR", raising=False)
    assert store.data_dir() == tmp_path / "x" / "prada"
    monkeypatch.setenv("PRADA_DATA_DIR", str(tmp_path / "d"))
    assert store.db_path() == tmp_path / "d" / "prada.db"


def test_stats_counts_errors_and_ignores_fake(monkeypatch):
    db = memory.open_memory(":memory:")
    monkeypatch.delenv("PRADA_FAKE_LLM", raising=False)
    memory.log_exchange(db, "ru", "a", make_turn())
    memory.log_exchange(db, "ru", "b", make_turn(False, "case"))
    memory.log_exchange(db, "ru", "c", make_turn(False, "case"))
    memory.log_exchange(db, "fr", "d", make_turn(False, "article"))
    monkeypatch.setenv("PRADA_FAKE_LLM", "1")
    memory.log_exchange(db, "ru", "e", make_turn(False, "aspect"))
    assert memory.stats(db) == {"total": 4, "correct": 1, "errors": {"case": 2, "article": 1}}
    assert memory.stats(db, "fr")["errors"] == {"article": 1}
    assert "case: 2" in memory.format_stats(memory.stats(db))
    assert memory.format_stats(memory.stats(db, "ja")) == "No practice logged yet."


def test_cli_logs_and_shows_stats(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PRADA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PRADA_FAKE_LLM", "1")
    lines = iter(["Я читаю книгу."])

    def fake_input(_prompt):
        try:
            return next(lines)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", fake_input)
    assert main(["tutor"]) == 0
    assert main(["tutor", "stats"]) == 0
    assert "No practice logged yet." in capsys.readouterr().out
    rows = (
        memory.open_memory(tmp_path / "prada.db").execute("SELECT * FROM tutor_exchange").fetchall()
    )
    assert len(rows) == 1 and rows[0]["model"] == "fake"


def test_wake_table_must_be_table(monkeypatch, tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text("wake = 1\n")
    monkeypatch.setenv("PRADA_CONFIG", str(cfg))
    with pytest.raises(ValueError):
        config.wake_settings("aa:bb:cc:dd:ee:ff", None, None)


@pytest.mark.parametrize(
    "argv", [["tutor", "stats", "--lang", "fr"], ["tutor", "--lang", "fr", "stats"]]
)
def test_stats_accepts_lang_either_side(monkeypatch, tmp_path, capsys, argv):
    monkeypatch.setenv("PRADA_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("PRADA_FAKE_LLM", raising=False)
    db = memory.open_memory()
    memory.log_exchange(db, "fr", "a", make_turn(False, "article"))
    memory.log_exchange(db, "ru", "b", make_turn(False, "case"))
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "article: 1" in out and "case" not in out


def test_exchange_logs_the_brain_model(monkeypatch):
    db = memory.open_memory(":memory:")
    monkeypatch.delenv("PRADA_FAKE_LLM", raising=False)
    monkeypatch.setenv("PRADA_LLM_MODEL", "qwen3.5:9b")
    memory.log_exchange(db, "ru", "a", make_turn())
    assert db.execute("SELECT model FROM tutor_exchange").fetchone()[0] == "qwen3.5:9b"
