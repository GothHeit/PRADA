"""Evidence facts, search by meaning, hybrid recall and how recalled memories are phrased."""

import json
import time
from datetime import datetime

import numpy as np
import pytest

from prada import config, llm, memory, status, vectors
from prada.assistant import AssistantModule
from prada.memory import EPISODE_GAP, Fact, Memory
from prada.module import Request
from prada.node import Node
from prada.store import open_store

# A tiny "meaning" space: words in the same group are close, other words far apart.
GROUPS = [{"cello", "instrument", "violin", "concerto"}, {"anime", "naruto", "frieren"}]


def synonyms(texts, query=False):
    out = []
    for text in texts:
        vec = np.zeros(len(GROUPS) + 1)
        for word in text.lower().replace("?", " ").replace(",", " ").split():
            for i, group in enumerate(GROUPS):
                vec[i] += word in group
        vec[-1] = 0.01  # nothing in common: still a vector, far from everything
        out.append(vec.tolist())
    return out


def make(embed=None):
    now = [1_000_000.0]
    embedder = vectors.Embedder("syn", embed, 0.5) if embed else None
    mem = Memory(open_store("memory", memory.MIGRATIONS, ":memory:"), lambda: now[0], embedder)
    return mem, now


def talk(mem, now, said, replied="ok", gap=EPISODE_GAP + 60):
    now[0] += gap
    mem.add(said, replied)


def move_on(mem, now):
    """Two more conversations, so the ones before are neither live nor her shown tail."""
    talk(mem, now, "end")
    talk(mem, now, "end")


def finish(mem, now, facts_by_text=None, model="m"):
    """End the conversation and store what a model would have found in each one."""
    now[0] += EPISODE_GAP + 60
    for ep, transcript in mem.jobs():
        known = [k["id"] for k in mem.known_facts(transcript)]
        facts = [f for text, fs in (facts_by_text or {}).items() if text in transcript for f in fs]
        mem.store(ep, "", model, facts, known)


def embed_all(mem):
    while items := mem.vectors.missing(50):
        mem.vectors.add(items, mem.vectors.embed([t for *_, t in items]))


# --- facts with evidence ---


def test_a_fact_is_kept_with_the_quote_and_when_it_was_said():
    mem, now = make()
    talk(mem, now, "By the way, I've played the cello since I was eight!")
    said_at = now[0]
    good = Fact(fact="You play the cello.", quote="i've played the CELLO since i was eight")
    invented = Fact(fact="You play the violin.", quote="I love my violin")
    finish(mem, now, {"cello": [good, invented]})
    (fact,) = mem.facts()
    assert fact["fact"] == "You play the cello." and fact["ts"] == said_at
    assert fact["superseded_by"] is None
    assert "since i was eight" in fact["quote"].lower()


def test_a_quote_from_her_own_words_is_no_evidence():
    mem, now = make()
    talk(mem, now, "what should I play next?", "You love Elgar, so the concerto")
    finish(mem, now, {"Elgar": [Fact(fact="You love Elgar.", quote="You love Elgar")]})
    assert mem.facts() == []


def test_a_correction_supersedes_but_never_erases():
    mem, now = make()
    talk(mem, now, "my favorite anime is Naruto")
    finish(mem, now, {"Naruto": [Fact(fact="Your favorite anime is Naruto.", quote="Naruto")]})
    talk(mem, now, "I changed my mind, my favorite anime is Frieren now")
    (old,) = mem.known_facts()
    fix = Fact(
        fact="Your favorite anime is Frieren.",
        quote="my favorite anime is Frieren",
        replaces=old["id"],
    )
    finish(mem, now, {"Frieren": [fix]})
    first, second = mem.facts()
    assert first["superseded_by"] == second["id"] and second["superseded_by"] is None
    assert [k["fact"] for k in mem.known_facts()] == ["Your favorite anime is Frieren."]
    got = mem.recall("what is my favorite anime?")
    assert got.count("\n") == 0  # one memory: the newest, with what it changed
    assert "you told me your favorite anime is Frieren" in got
    assert "That changed what you had said" in got and "Naruto" in got


def test_replaces_must_name_a_fact_she_was_shown():
    mem, now = make()
    talk(mem, now, "my sister is Ana")
    finish(mem, now, {"Ana": [Fact(fact="Your sister is Ana.", quote="my sister is Ana")]})
    talk(mem, now, "my brother is Rui")
    (ana,) = mem.known_facts()
    rui = Fact(fact="Your brother is Rui.", quote="my brother is Rui", replaces=ana["id"])
    now[0] += EPISODE_GAP + 60
    ((ep, _),) = mem.jobs()
    mem.store(ep, "", "m", [rui], known=[])  # not shown: the model made the number up
    assert all(f["superseded_by"] is None for f in mem.facts())


def test_fake_facts_are_never_searched():
    mem, now = make()
    talk(mem, now, "my dog is called Bolt")
    finish(
        mem, now, {"Bolt": [Fact(fact="Your dog is Bolt.", quote="my dog is called Bolt")]}, "fake"
    )
    assert len(mem.facts()) == 1
    assert mem.known_facts() == [] and not [
        h for h in mem.index.search("dog bolt", 5) if h.kind == "fact"
    ]


def test_known_facts_prefers_the_ones_the_conversation_touches(monkeypatch):
    monkeypatch.setattr(memory, "KNOWN_FACTS", 3)
    mem, now = make()
    talk(mem, now, "my cat is Miso")
    finish(mem, now, {"Miso": [Fact(fact="Your cat is Miso.", quote="my cat is Miso")]})
    for i in range(4):
        talk(mem, now, f"I like color{i} a lot")
        finish(
            mem, now, {f"color{i}": [Fact(fact=f"You like color{i}.", quote=f"I like color{i}")]}
        )
    picked = [k["fact"] for k in mem.known_facts("User: is Miso a good cat name?")]
    assert len(picked) == 3 and "Your cat is Miso." in picked


def test_a_fact_replaces_the_message_it_quotes_in_recall():
    mem, now = make()
    talk(mem, now, "my sister Ana starts a new job in Porto")
    finish(
        mem,
        now,
        {
            "Porto": [
                Fact(
                    fact="Your sister Ana works in Porto.",
                    quote="my sister Ana starts a new job in Porto",
                )
            ]
        },
    )
    move_on(mem, now)
    # A fact her profile doesn't show (a passing one) is recalled instead of its message.
    mem.db.execute("UPDATE memory_fact SET kind = 'moment'")
    got = mem.recall("where does my sister Ana work?")
    assert "in your words" in got and got.count("starts a new job") == 1


# --- search by meaning ---


def test_meaning_finds_what_shares_no_word():
    mem, now = make(synonyms)
    talk(mem, now, "I practiced the cello all night", "Bravo")
    talk(mem, now, "what's for dinner?", "Pasta")
    move_on(mem, now)
    assert mem.recall("which instrument?") == ""  # not embedded yet: keywords only
    embed_all(mem)
    got = mem.recall("which instrument?")
    assert "cello" in got and "dinner" not in got


def test_both_rankings_are_merged():
    mem, now = make(synonyms)
    talk(mem, now, "my cello teacher is strict")  # words and meaning
    talk(mem, now, "the concerto was long")  # meaning only
    talk(mem, now, "my teacher is strict about homework")  # words only
    move_on(mem, now)
    embed_all(mem)
    got = mem.recall("is my cello teacher strict?").splitlines()
    assert len(got) == 3 and "concerto" in got[1] and "homework" in got[2]


def test_a_dead_embedding_model_falls_back_to_keywords(capsys):
    calls = []

    def broken(texts, query=False):
        calls.append(query)
        if query:
            raise ConnectionError("ollama is not running")
        return synonyms(texts)

    mem, now = make(broken)
    talk(mem, now, "I adopted a cat named Miso")
    embed_all(mem)
    for _ in range(2):
        assert "Miso" in mem.recall("tell me about Miso the cat")
    assert calls.count(True) == 1  # backed off after the failure
    assert capsys.readouterr().err.count("searching by meaning is off") == 1


def test_a_cut_reply_is_embedded_again():
    mem, now = make(synonyms)
    talk(mem, now, "play something", "Here is a long story about a violin")
    embed_all(mem)
    assert mem.vectors.missing(10) == []
    mem.cut("Here is a long story about a violin", "Here is")
    assert [k for k, *_ in mem.vectors.missing(10)] == ["turn"]


def test_vectors_are_kept_per_model():
    mem, now = make(synonyms)
    talk(mem, now, "cello")
    embed_all(mem)
    other = Memory(mem.db, mem.clock, vectors.Embedder("other", synonyms))
    assert len(other.vectors.missing(10)) == 2


def test_fake_embedder_and_settings(monkeypatch, tmp_path):
    a, b, c = vectors.fake_embed(["cello practice", "practice cello tonight", "pizza"])
    assert np.dot(a, b) > 0 and np.dot(a, c) == 0
    monkeypatch.setenv("PRADA_FAKE_LLM", "1")
    assert vectors.embedder().model == "fake"
    monkeypatch.delenv("PRADA_FAKE_LLM")
    assert vectors.embedder().model == config.DEFAULT_EMBED_MODEL
    path = tmp_path / "config.toml"
    monkeypatch.setenv("PRADA_CONFIG", str(path))
    path.write_text('[memory]\nembed_model = ""\n')
    assert vectors.embedder() is None
    path.write_text("[memory]\nmin_similarity = 0.7\n")
    assert config.memory_settings().min_similarity == 0.7
    for bad in ("min_similarity = 3", 'min_similarity = "x"', "colour = 1", "embed_model = 1"):
        path.write_text(f"[memory]\n{bad}\n")
        with pytest.raises(ValueError):
            config.memory_settings()


def test_nomic_gets_its_task_prefixes(monkeypatch):
    import ollama

    seen = []
    monkeypatch.setattr(ollama, "embed", lambda **kw: seen.append(kw) or {"embeddings": [[1.0]]})
    vectors.ollama_embed("nomic-embed-text")(["hi"], True)
    vectors.ollama_embed("bge-m3")(["hi"], False)
    assert seen[0]["input"] == ["search_query: hi"] and seen[1]["input"] == ["hi"]


# --- how memories are said ---


def at(*args):
    return datetime(*args).timestamp()


@pytest.mark.parametrize(
    "then, said",
    [
        (at(2026, 11, 6, 9), "this morning"),
        (at(2026, 11, 6, 2), "last night"),  # read at 21:00, after the night ended
        (at(2026, 11, 5, 19), "yesterday evening"),
        (at(2026, 11, 5, 1), "last night"),
        (at(2026, 11, 3, 15), "on Tuesday afternoon"),
        (at(2026, 10, 28, 10), "last week, one morning"),
        (at(2026, 10, 9, 20), "about four weeks ago, one evening"),
        (at(2026, 8, 6, 8), "about three months ago, one morning"),
        (at(2024, 11, 1, 14), "about two years ago, one afternoon"),
    ],
)
def test_perceived_time(then, said):
    assert status.perceived(then, at(2026, 11, 6, 21)) == said


def test_recalled_lines_read_like_memories():
    mem, now = make()
    talk(mem, now, "I adopted a cat named Miso", "Lovely, a cat named Miso")
    now[0] += 90 * 24 * 3600
    move_on(mem, now)
    got = mem.recall("tell me about my cat Miso")
    assert got.startswith("About three months ago, one ") and 'you said: "I adopted' in got
    assert "I said:" in got
    assert "-" not in got.split(",")[0] and str(time.localtime(now[0]).tm_year) not in got


# --- the node: background indexing and facts from the summarizer ---


def test_node_indexes_and_finds_by_meaning(tmp_path, monkeypatch):
    monkeypatch.setenv("PRADA_FAKE_LLM", "1")
    monkeypatch.setattr(vectors, "fake_embed", synonyms)
    now = [1_000_000.0]
    n = Node([AssistantModule()], db_path=tmp_path / "p.db", clock=lambda: now[0], background=False)
    for text in ("I practiced the cello all night", "end", "end"):
        n.handle(Request(text=text, module="assistant", session="s"))
        now[0] += EPISODE_GAP + 60
    n.index_pending()
    assert n._mem().vectors.missing(10) == []
    assert "cello" in n._mem().recall("which instrument?")


def test_node_index_failure_waits_before_trying_again(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PRADA_FAKE_LLM", "1")
    calls = []

    def broken(texts, query=False):
        calls.append(1)
        raise ConnectionError("no model")

    monkeypatch.setattr(vectors, "fake_embed", broken)
    n = Node([AssistantModule()], db_path=tmp_path / "p.db", background=False)
    n.handle(Request(text="hi", module="assistant", session="s"))
    n.index_pending()
    n.index_pending()
    assert len(calls) == 1 and "could not index" in capsys.readouterr().err


def test_node_summarizer_sees_known_facts_and_stores_new_ones(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "fake_enabled", lambda: False)
    monkeypatch.setattr("prada.node.Node._chat", lambda self, fake: fake)
    monkeypatch.setattr("prada.node.vectors.embedder", lambda keep_alive=None: None)
    prompts = []

    def chat(messages, schema, model=None, on_text=None):
        if schema.get("title") != "Summary":
            return '{"reply": "ok"}'
        prompts.append(messages[1]["content"])
        if "Naruto" in messages[1]["content"] and "Frieren" not in messages[1]["content"]:
            facts = [
                {"fact": "Your favorite anime is Naruto.", "quote": "my favorite anime is Naruto"}
            ]
        else:
            facts = [
                {"fact": "Your favorite anime is Frieren.", "quote": "Frieren now", "replaces": 1}
            ]
        return json.dumps({"summary": "We talked about anime.", "facts": facts})

    monkeypatch.setattr(AssistantModule, "fake_chat", staticmethod(chat))
    monkeypatch.setattr("prada.node.memory_fake_chat", chat)
    now = [1_000_000.0]
    n = Node([AssistantModule()], db_path=tmp_path / "p.db", clock=lambda: now[0], background=False)
    for text in ("my favorite anime is Naruto", "Frieren now, I changed my mind"):
        n.handle(Request(text=text, module="assistant", session="s"))
        now[0] += EPISODE_GAP + 60
        n.summarize_pending()
    assert "1: Your favorite anime is Naruto." in prompts[1]
    first, second = n._mem().facts()
    assert first["superseded_by"] == second["id"]
    # One exchange each: facts, but no summary; the profile shows only the newest version.
    assert n._mem().remembered() == (
        "What you know about them (they told you; still true):\n"
        "- Other things they told you: Your favorite anime is Frieren."
    )


def test_perceived_tonight_while_the_night_lasts():
    assert status.perceived(at(2026, 11, 6, 1), at(2026, 11, 6, 3)) == "earlier tonight"
    assert status.perceived(at(2026, 11, 6, 23), at(2026, 11, 6, 23, 30)) == "earlier tonight"


def test_a_short_quote_must_be_whole_words():
    mem, now = make()
    talk(mem, now, "I went there yesterday")
    finish(mem, now, {"there": [Fact(fact="You went.", quote="the")]})
    assert mem.facts() == []
    talk(mem, now, "日本語を勉強しています")
    finish(mem, now, {"日本語": [Fact(fact="You study Japanese.", quote="日本語を勉強")]})
    assert len(mem.facts()) == 1


def test_odd_sized_vectors_are_dropped_not_the_rest():
    mem, now = make(synonyms)
    for text in ("cello", "violin", "concerto"):
        talk(mem, now, text)
    embed_all(mem)
    first = mem.db.execute("SELECT kind, ref FROM memory_vector ORDER BY rowid").fetchone()
    with mem.db:
        mem.db.execute(
            "UPDATE memory_vector SET vec = ? WHERE kind = ? AND ref = ?",
            (np.zeros(7, dtype=np.float32).tobytes(), *first),
        )
    mem.vectors._load()
    assert len(mem.vectors._keys) == 5 and mem.vectors._matrix.shape[1] == len(GROUPS) + 1
