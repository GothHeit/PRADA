"""Keyword search over past conversations: index, recall, exclusions, migration, CJK."""

import json
import sqlite3

from prada import memory, search
from prada.assistant import AssistantModule
from prada.memory import EPISODE_GAP, Memory
from prada.module import Context, Request
from prada.node import MAX_TURNS, Node
from prada.store import open_store
from tutor.module import TutorModule


def make(clock=None, path=":memory:"):
    now = [1_000_000.0]
    clock = clock or (lambda: now[0])
    mem = Memory(open_store("memory", memory.MIGRATIONS, path), clock)
    return mem, now


def talk(mem, now, said, replied="ok", gap=EPISODE_GAP + 60):
    now[0] += gap
    mem.add(said, replied)


def test_accents_and_case_are_ignored():
    mem, now = make()
    talk(mem, now, "Meu café favorito é de Açaí")
    talk(mem, now, "unrelated")
    talk(mem, now, "x")
    assert "café favorito" in mem.recall("qual meu CAFE favorito?")
    assert "Açaí" in mem.recall("acai")


def test_nothing_matches_gives_empty_and_stopwords_alone_find_nothing():
    mem, now = make()
    talk(mem, now, "my cat is called Miso")
    talk(mem, now, "later")
    assert mem.recall("tell me about dogs") == ""
    assert mem.recall("what is the you") == ""
    assert mem.recall("hi") == ""


def test_one_shared_common_word_is_not_a_memory():
    mem, now = make()
    talk(mem, now, "I love my morning coffee with oat milk", "Nice choice")
    talk(mem, now, "filler")
    talk(mem, now, "filler")
    assert mem.recall("what should I cook for dinner today? maybe something with milk") == ""
    assert "oat milk" in mem.recall("should I cook with oat milk today?")  # two terms shared
    assert "oat milk" in mem.recall("milk")  # a one-term query still hits


def test_finds_old_turns_by_both_speakers_and_orders_oldest_first():
    mem, now = make()
    talk(mem, now, "I adopted a cat named Miso", "Lovely, a cat named Miso")
    talk(mem, now, "weather?", "I can't see weather")
    talk(mem, now, "what about the other cat Pixel", "ok")
    talk(mem, now, "filler", "filler")
    talk(mem, now, "filler", "filler")
    out = mem.recall("tell me about my cat Miso and Pixel").splitlines()
    assert out[0].startswith("- ") and "you said" in out[0] and "Miso" in out[0]
    assert "I said" in out[1] and "Miso" in out[1]  # same moment: your message first
    assert any("I said" in line for line in out)
    assert any("Pixel" in line for line in out)
    days = [line[2:12] for line in out]
    assert days == sorted(days)


def test_best_matches_win_and_are_capped():
    mem, now = make()
    for i in range(10):
        talk(mem, now, f"lasagna recipe number {i}")
    talk(mem, now, "lasagna with spinach and ricotta cheese")
    talk(mem, now, "end")
    talk(mem, now, "end")
    out = mem.recall("lasagna ricotta spinach").splitlines()
    assert len(out) <= memory.RECALL_HITS
    assert any("ricotta" in line for line in out)


def test_budget_and_line_length():
    mem, now = make()
    for i in range(6):
        talk(mem, now, "zebra " + "x" * 2000 + str(i))
    talk(mem, now, "end")
    out = mem.recall("zebra")
    assert len(out) <= memory.RECALL_CHARS + memory.RECALL_LINE
    assert all(len(line) < memory.RECALL_LINE + 60 for line in out.splitlines())


def test_current_conversation_and_tail_are_not_repeated():
    mem, now = make()
    talk(mem, now, "old pelican story", "pelicans are big")  # finished, unsummarized: the tail
    talk(mem, now, "new pelican question", "yes", gap=EPISODE_GAP + 60)
    # the tail would be the old conversation now, but the live one is `recent`
    assert "new pelican" not in mem.recall("pelican", recent=MAX_TURNS)
    assert "new pelican" in mem.recall("pelican", recent=0)
    # make the old one the tail: finished, newest unsummarized
    now[0] += EPISODE_GAP + 60
    tail = mem.remembered()
    assert "new pelican question" in tail
    got = mem.recall("pelican", recent=MAX_TURNS)  # the newest is in the tail already
    assert "new pelican" not in got and "old pelican story" in got


def test_older_part_of_a_long_conversation_is_found():
    mem, now = make()
    talk(mem, now, "my sister Lúcia will visit in May", gap=10)
    for i in range(MAX_TURNS):
        talk(mem, now, f"chatter {i}", gap=10)
    assert "Lúcia" in mem.recall("when does Lucia visit", recent=MAX_TURNS)


def test_summaries_are_found_unless_already_loaded():
    mem, now = make()
    for _ in range(2):
        talk(mem, now, "a")
        talk(mem, now, "b")
        talk(mem, now, "c")
        talk(mem, now, "d")
    now[0] += EPISODE_GAP + 60
    jobs = mem.jobs()
    assert jobs
    for ep, _ in jobs:
        mem.store(ep, "You are learning Russian cases and I promised a quiz.", "m")
    got = mem.recall("russian quiz")
    assert got == ""  # both summaries are already in `remembered`
    assert "Russian cases" in mem.remembered()
    # push the older ones out of the loaded window: now they can be searched
    for k in range(memory.SUMMARIES_LOADED):
        talk(mem, now, "p")
        talk(mem, now, "q")
        talk(mem, now, "r")
        talk(mem, now, "s")
        now[0] += EPISODE_GAP + 60
        for ep, _ in mem.jobs():
            mem.store(ep, f"filler {k}", "m")
    got = mem.recall("russian quiz")
    assert "what I wrote down" in got and "Russian cases" in got


def test_fake_summaries_are_not_searchable():
    mem, now = make()
    for t in "abcd":
        talk(mem, now, t)
    now[0] += EPISODE_GAP + 60
    for ep, _ in mem.jobs():
        mem.store(ep, "(fake summary) quokka", "fake")
    assert mem.recall("quokka") == ""


def test_older_database_is_backfilled(tmp_path):
    path = tmp_path / "old.db"
    db = open_store("memory", memory.MIGRATIONS, path)
    db.execute(
        "INSERT INTO memory_episode (started, ended, summary, model) VALUES (1, 2, 'I met Zoltán', 'm')"
    )
    db.execute(
        "INSERT INTO memory_turn (episode, ts, role, content) VALUES (1, 1, 'user', 'my dentist is Zoltán')"
    )
    db.commit()
    db.close()
    mem, _ = make(clock=lambda: 10**9, path=path)  # opening it builds the index
    out = mem.recall("dentist")
    assert "Zoltan" not in out and "Zoltán" in out
    assert [h.kind for h in mem.index.search("zoltan", 5)].count("summary") == 1
    mem2, _ = make(clock=lambda: 10**9, path=path)  # a second open adds nothing twice
    assert mem2.recall("dentist").count("dentist") == 1


def test_without_fts5_nothing_breaks(monkeypatch):
    class NoFts(search.KeywordIndex):
        def __init__(self, db):
            self.db, self.available = db, False

    monkeypatch.setattr(memory, "KeywordIndex", NoFts)
    mem, now = make()
    talk(mem, now, "hello kangaroo")
    assert mem.recall("kangaroo") == ""
    assert mem.recent(5)


def test_japanese_phrases_and_their_limits():
    mem, now = make()
    talk(mem, now, "日本語を勉強しています", "いいですね")
    talk(mem, now, "猫が好きです", "かわいい")
    talk(mem, now, "end")
    talk(mem, now, "end")
    assert "日本語" in mem.recall("日本語は難しい")  # pairs 日本, 本語 match
    assert "猫" in mem.recall("猫")  # a lone kanji is a term
    assert mem.recall("が") == ""  # a lone kana is a particle
    assert mem.recall("本日") == ""  # the characters must be consecutive
    assert "日本語" in mem.recall("Japanese: 日本語")  # mixed with Latin
    assert search.terms("Я читаю книгу") == ["читаю", "книгу"]  # Cyrillic works as words


def test_search_terms():
    assert search.terms("What's my Café, 2024?") == ["café"]
    assert search.terms("the of an") == []


# --- through the node ---


class Spy:
    """Records the system prompt the assistant sent."""

    def __init__(self):
        self.system = None

    def __call__(self, messages, schema, model=None):
        self.system = messages[0]["content"]
        return json.dumps({"reply": "ok"})


def test_assistant_gets_found_lines_and_tutor_does_not_search(monkeypatch):
    n = Node([AssistantModule(), TutorModule()], db_path=":memory:", background=False)
    n._mem()  # real memory
    clock = [1_000_000.0]
    n._clock = lambda: clock[0]
    n._memory.clock = n._clock
    n._memory.add("my turtle is called Donatello", "Cute name")
    clock[0] += EPISODE_GAP + 60
    n._memory.add("filler", "filler")
    clock[0] += EPISODE_GAP + 60
    n._memory.add("filler2", "filler")
    clock[0] += EPISODE_GAP + 60

    spy = Spy()
    monkeypatch.setattr(n, "_chat", lambda fake: spy)
    reply = n.handle(Request(text="what was my turtle called?", module="assistant"))
    assert reply.error is None
    assert "Found by searching your memory" in spy.system and "Donatello" in spy.system
    assert "never add details" in spy.system

    spy.system = None
    calls = []
    memory_recall = n._memory.recall
    monkeypatch.setattr(n._memory, "recall", lambda *a: calls.append(a) or memory_recall(*a))
    n.handle(Request(text="turtle", module="tutor", options={"lang": "fr"}))
    assert calls == []


def test_assistant_without_matches_has_no_found_section():
    spy = Spy()
    AssistantModule().handle(Request(text="hello there"), Context(chat=spy))
    assert "\n\nFound by searching" not in spy.system


def test_context_recall_defaults_empty():
    assert Context(chat=lambda **k: "").recall("anything") == ""


def test_tail_keeps_whole_lines():
    mem, now = make()
    for i in range(memory.TAIL_MESSAGES // 2):
        talk(mem, now, "u" * 400 + str(i), "a" * 400, gap=10)
    now[0] += EPISODE_GAP + 60
    lines = mem._tail().splitlines()[1:]
    assert lines and all(
        line.startswith(("User: ", "PRADA: ", f"{memory.persona.NAME}: ")) for line in lines
    )


# --- topics: a turn is also found by its conversation's summary ---


def camera_talk(mem, now):
    talk(mem, now, "which camera should I get for the trip?", "the Fuji or the Sony")
    talk(mem, now, "the lighter one", "that's the Fuji", gap=10)
    talk(mem, now, "ok, let's do that", "done, it's on your list", gap=10)
    now[0] += EPISODE_GAP + 60
    (ep,) = [ep for ep, transcript in mem.jobs() if transcript]
    mem.store(ep, "You chose the Fuji camera for the trip because it is lighter.", "m")
    return ep


def bury(mem, now):
    """Push the summaries written so far out of what `remembered` loads."""
    for k in range(memory.SUMMARIES_LOADED):
        for t in "pqrs":
            talk(mem, now, t)
        now[0] += EPISODE_GAP + 60
        for ep, _ in mem.jobs():
            mem.store(ep, f"filler {k}", "m")


def test_vague_lines_are_found_by_their_conversations_topic():
    mem, now = make()
    talk(mem, now, "ok, let's do that", "done")
    assert not mem.index.search("fuji camera", 10)  # no summary yet, nothing to find it by
    ep = camera_talk(mem, now)
    hits = [h for h in mem.index.search("fuji camera", 50) if h.kind == "turn"]
    vague = [h for h in hits if h.text == "ok, let's do that"]
    assert len(vague) == 1 and vague[0].episode == ep  # the older one has no topic
    assert vague[0].matched == 2 and vague[0].own == 0
    assert hits[0].own > 0  # a turn's own words count before its topic
    rows = mem.db.execute("SELECT content FROM memory_turn WHERE episode = ?", (ep,)).fetchall()
    assert "ok, let's do that" in [r[0] for r in rows]  # the stored turn is unchanged


def test_recall_takes_one_vague_line_per_conversation():
    mem, now = make()
    camera_talk(mem, now)
    # the summary is loaded, so its vague lines aren't repeated; the turn naming the
    # camera (own word "fuji", topic "camera") still is
    got = mem.recall("which camera did I pick, the fuji?")
    assert "that's the Fuji" in got and "let's do that" not in got
    bury(mem, now)
    got = mem.recall("what did I decide about the trip camera")
    vague = [line for line in got.splitlines() if not any(w in line for w in ("trip", "camera"))]
    assert "what I wrote down" in got and len(vague) == 1


def test_fake_summaries_give_no_topic():
    mem, now = make()
    for t in ("ok", "sure", "fine", "yes"):
        talk(mem, now, t, gap=10)
    now[0] += EPISODE_GAP + 60
    for ep, _ in mem.jobs():
        mem.store(ep, "(fake summary) wombat", "fake")
    assert mem.index.search("wombat", 5) == []


def test_index_built_before_topics_is_rebuilt(tmp_path):
    path = tmp_path / "old.db"
    db = open_store("memory", memory.MIGRATIONS, path)
    db.execute("CREATE VIRTUAL TABLE memory_turn_fts USING fts5(content)")
    db.execute(
        "INSERT INTO memory_episode (started, ended, summary, model) "
        "VALUES (1, 2, 'We planned the Lisbon trip', 'm')"
    )
    db.execute(
        "INSERT INTO memory_turn (episode, ts, role, content) VALUES (1, 1, 'user', 'sounds good')"
    )
    db.execute("INSERT INTO memory_turn_fts (rowid, content) VALUES (1, 'sounds good')")
    db.commit()
    db.close()
    mem, _ = make(clock=lambda: 10**9, path=path)
    hits = mem.index.search("lisbon trip", 5)
    assert [(h.kind, h.own) for h in hits if h.kind == "turn"] == [("turn", 0)]
    mem2, _ = make(clock=lambda: 10**9, path=path)  # reopening keeps it, adds nothing twice
    assert len(mem2.index.search("sounds good lisbon", 5)) == 2


def test_contractions_are_not_search_terms():
    assert search.terms("don't forget kafka's camera, I won't") == ["forget", "kafka", "camera"]


def test_broken_index_on_open_only_disables_search(monkeypatch):
    def broken(self):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(search.KeywordIndex, "backfill", broken)
    mem, now = make()
    talk(mem, now, "hello kangaroo")
    assert mem.recall("kangaroo") == "" and mem.recent(5)


def test_tail_lines_not_shown_can_still_be_found():
    mem, now = make()
    talk(mem, now, "my locksmith is called Bartholomew", "noted", gap=10)
    for i in range(3):
        talk(mem, now, "x" * 600 + str(i), "y", gap=10)
    now[0] += EPISODE_GAP + 60
    assert "Bartholomew" not in mem.remembered()  # cut from the tail by its budget
    assert "Bartholomew" in mem.recall("what is my locksmith called, Bartholomew?")


def test_a_long_vague_conversation_does_not_hide_real_lines():
    mem, now = make()
    talk(mem, now, "My guitar teacher is called Renata. " + "She lives near the park. " * 10)
    for t in "abc":
        talk(mem, now, t, gap=10)
    now[0] += EPISODE_GAP + 60
    for ep, transcript in mem.jobs():
        if transcript:
            mem.store(ep, "We talked about the weather.", "m")
    for i in range(30):
        talk(mem, now, f"ok sounds good {i}", "sure", gap=10)
    now[0] += EPISODE_GAP + 60
    for ep, transcript in mem.jobs():
        if transcript:
            mem.store(ep, "You planned guitar practice with the teacher for the guitar exam.", "m")
    assert "Renata" in mem.recall("who is my guitar teacher")
