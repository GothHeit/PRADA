"""Correcting what she got wrong: by voice (`Memory.correct` via the node) and with the
`prada memory` developer tool."""

import json

import pytest

from prada import memory, status, vectors
from prada.assistant import AssistantModule
from prada.cli import main
from prada.memory import EPISODE_GAP, Fact, Memory
from prada.module import Request
from prada.node import Node
from prada.store import open_store


def make():
    now = [1_000_000.0]
    embedder = vectors.Embedder("fake", vectors.fake_embed, 0.1)
    mem = Memory(open_store("memory", memory.MIGRATIONS, ":memory:"), lambda: now[0], embedder)
    return mem, now


def talk(mem, now, said, replied="ok", gap=EPISODE_GAP + 60):
    now[0] += gap
    mem.add(said, replied)


def finish(mem, now, facts_by_text, model="m", summary=""):
    now[0] += EPISODE_GAP + 60
    for ep, transcript in mem.jobs():
        known = [k["id"] for k in mem.known_facts(transcript)]
        facts = [f for text, fs in facts_by_text.items() if text in transcript for f in fs]
        mem.store(ep, summary, model, facts, known)


def embed_all(mem):
    while items := mem.vectors.missing(50):
        mem.vectors.add(items, mem.vectors.embed([t for *_, t in items]))


def misheard():
    """She stored "Valve" where they had said "velvet"."""
    mem, now = make()
    talk(mem, now, "my new jacket is Valve, so soft", "A Valve jacket sounds cozy!")
    talk(mem, now, "and what do you think of it?", "Valve suits you.", gap=60)
    talk(mem, now, "thanks", "anytime", gap=60)
    fact = Fact(fact="Your new jacket is Valve.", quote="my new jacket is Valve")
    finish(mem, now, {"jacket": [fact]}, summary="You showed me your Valve jacket.")
    embed_all(mem)
    return mem, now


def test_a_correction_fixes_facts_messages_and_summaries_in_place():
    mem, now = misheard()
    assert mem.correct('"Valve"', "velvet", "you misheard, it's velvet, not Valve") == 5
    (fact,) = mem.facts()
    assert fact["fact"] == "Your new jacket is velvet." and fact["superseded_by"] is None
    assert fact["quote"] == "my new jacket is velvet"
    turns = [r["content"] for r in mem.db.execute("SELECT content FROM memory_turn")]
    assert not any("Valve" in t for t in turns) and "A velvet jacket sounds cozy!" in turns
    assert "velvet jacket" in mem.remembered() and "Valve" not in mem.remembered()
    # Searches find the new word and no longer the old one; vectors are written again.
    assert mem.find("valve") == []
    assert "velvet" in mem.recall("tell me about my velvet jacket")
    assert "Valve" not in mem.recall("my Valve jacket")
    assert mem.vectors.missing(50)


def test_the_right_word_must_be_in_their_message():
    mem, _ = misheard()
    assert mem.correct("Valve", "velvet", "no, that's wrong, not Valve") == 0
    assert mem.correct("Valve", "corduroy", "no, it's velvet, not Valve") == 0
    assert mem.facts()[0]["fact"] == "Your new jacket is Valve."


def test_only_whole_words_and_real_changes():
    mem, now = make()
    talk(mem, now, "I play the cello", "The cellos are lovely.")
    assert mem.correct("cello", "cello", "cello, you misheard") == 0  # nothing to change
    assert mem.correct("ab", "abc", "abc, you misheard") == 0  # too short to be sure
    assert mem.correct("the", "tea", "tea, you misheard") == 0  # too common to be sure
    assert mem.correct("cello", "viola", "you misheard, the viola") == 1  # not "cellos"
    contents = [r[0] for r in mem.db.execute("SELECT content FROM memory_turn ORDER BY id")]
    assert contents == ["I play the viola", "The cellos are lovely."]
    assert mem.correct("valve", "velvet", "velvet, not valve") == 0  # nothing to fix
    assert mem.correct("viola", r"\1 velvet", r"\1 velvet, you misheard") == 1  # just text
    assert mem.db.execute("SELECT content FROM memory_turn").fetchone()[0].endswith(r"\1 velvet")


def test_only_the_last_few_conversations_are_rewritten():
    mem, now = make()
    talk(mem, now, "my favorite game studio is Valve", "Half-Life fan!")
    fact = Fact(fact="Your favorite studio is Valve.", quote="favorite game studio is Valve")
    finish(mem, now, {"studio": [fact]})
    for _ in range(memory.FIX_EPISODES - 1):
        talk(mem, now, "hello")
    talk(mem, now, "my jacket is Valve", "nice")
    assert mem.correct("Valve", "velvet", "it's velvet, you misheard") == 1
    assert mem.facts()[0]["fact"] == "Your favorite studio is Valve."  # too old to be it
    contents = [r[0] for r in mem.db.execute("SELECT content FROM memory_turn ORDER BY id")]
    assert (
        contents[0] == "my favorite game studio is Valve" and contents[-2] == "my jacket is velvet"
    )


def test_a_change_of_mind_is_not_a_correction():
    mem, now = make()
    talk(mem, now, "my favorite game studio is Valve", "Half-Life fan!")
    talk(mem, now, "my cat is Mochi", "cute", gap=60)
    for said in (
        "I don't like Valve that much anymore, FromSoftware is my favorite now",
        "I don't like Mochi anymore, Miso is my favorite",
    ):
        assert mem.correct("Valve", "FromSoftware", said) == 0
        assert mem.correct("Mochi", "Miso", said) == 0
    assert (
        mem.correct("Mochi", "Miso", "você ouviu errado, não Mochi, Miso") == 1
    )  # Portuguese "not X"


def test_the_wrong_word_must_be_what_she_was_given():
    mem, now = make()
    talk(mem, now, "my jacket is soft", "nice")
    assert mem.correct("Valve", "velvet", "you misheard, velvet, not Valve") == 0  # never had it
    talk(mem, now, "my jacket is Valve", "nice")
    talk(mem, now, "hi again", "hello!")  # a new conversation: Valve is in her tail
    assert mem.correct("Valve", "velvet", "you misheard, it's velvet") == 1


def test_a_fix_that_touches_too_much_changes_nothing(capsys):
    mem, now = make()
    for i in range(memory.FIX_MAX):
        talk(mem, now, f"Valve {i}", "Valve!", gap=60)
    with pytest.raises(memory.TooBroad):
        mem.correct("Valve", "velvet", "you misheard, velvet, not Valve")
    assert mem.find("velvet") == []
    mem, now = make()
    talk(mem, now, "my jacket is Valve")
    assert mem.correct("Valve", "velvet", "velvet, not Valve", manual=True) == 1
    assert "it was: my jacket is Valve" in capsys.readouterr().err


def test_a_replaced_fact_stays_as_it_was():
    mem, now = make()
    talk(mem, now, "my jacket is Valve")
    finish(mem, now, {"jacket": [Fact(fact="Your jacket is Valve.", quote="my jacket is Valve")]})
    talk(mem, now, "I gave the jacket away")
    finish(
        mem,
        now,
        {
            "away": [
                Fact(fact="You gave your jacket away.", quote="gave the jacket away", replaces=1)
            ]
        },
    )
    mem.correct("Valve", "velvet", "it's velvet, not Valve")
    old, new = mem.facts()
    assert old["superseded_by"] == new["id"] and old["fact"] == "Your jacket is Valve."


def test_edit_and_forget_facts():
    mem, now = make()
    talk(mem, now, "my favorite anime is Naruto")
    finish(
        mem, now, {"Naruto": [Fact(fact="Your favorite anime is Naruto.", quote="anime is Naruto")]}
    )
    talk(mem, now, "Frieren now, I changed my mind")
    finish(
        mem,
        now,
        {"Frieren": [Fact(fact="Your favorite is Frieren.", quote="Frieren now", replaces=1)]},
    )
    assert mem.edit_fact(2, "Your favorite anime is Frieren.")
    assert mem.facts()[1]["fact"] == "Your favorite anime is Frieren."
    assert mem.facts()[1]["quote"] == "Frieren now"
    assert "favorite anime is Frieren" in mem.recall("favorite anime")
    assert not mem.edit_fact(99, "nothing")
    assert sorted(mem.forget_fact(2)) == [1, 2]
    assert mem.facts() == [] and mem.forget_fact(2) == []
    assert mem.recall("favorite anime Naruto Frieren").count("you told me") == 0


def corrector(monkeypatch, wrong="Valve", right="velvet", reply="Oops, I misheard you!"):
    def chat(messages, schema, model=None, on_text=None):
        if schema.get("title") == "Answer":
            text = status.detach(messages[-1]["content"])
            fix = "misheard" in text
            return json.dumps(
                {
                    "reply": reply if fix else "Sounds nice.",
                    "offer_search": False,
                    "wrong": wrong if fix else "",
                    "right": right if fix else "",
                }
            )
        return json.dumps({"summary": "", "facts": []})

    monkeypatch.setattr(AssistantModule, "fake_chat", staticmethod(chat))


def test_by_voice_she_says_it_only_when_her_memory_changed(monkeypatch, tmp_path):
    corrector(monkeypatch)
    n = Node([AssistantModule()], db_path=tmp_path / "p.db", background=False)
    ask = lambda text: n.handle(Request(text=text, module="assistant", session="s"))  # noqa: E731
    ask("my new jacket is Valve")
    reply = ask("it's velvet, you misheard me")
    assert reply.text == "Oops, I misheard you! I fixed it in my memory: velvet, not Valve."
    assert reply.data == {"fixed": 1} and "correction" not in reply.model_dump()
    turns = n._mem().recent(10)
    assert turns[0]["content"] == "my new jacket is velvet"
    assert turns[2]["content"] == "it's velvet, you misheard me"  # their words, as said
    # Nothing to fix (already fixed): no claim that anything changed.
    reply = ask("it's velvet, you misheard me")
    assert reply.text == "Oops, I misheard you!" and reply.data == {}


def test_a_guess_that_is_not_their_words_changes_nothing(monkeypatch, tmp_path):
    corrector(monkeypatch, right="corduroy")
    n = Node([AssistantModule()], db_path=tmp_path / "p.db", background=False)
    n.handle(Request(text="my new jacket is Valve", module="assistant", session="s"))
    reply = n.handle(Request(text="you misheard me", module="assistant", session="s"))
    assert reply.text == "Oops, I misheard you!"
    assert n._mem().recent(10)[0]["content"] == "my new jacket is Valve"


def test_prada_memory_tool(capsys):
    mem = Memory(open_store("memory", memory.MIGRATIONS))
    mem.add("my jacket is Valve", "nice")
    episode = mem.db.execute("SELECT id FROM memory_episode").fetchone()[0]
    mem.store(episode, "", "m", [Fact(fact="Your jacket is Valve.", quote="my jacket is Valve")])
    assert main(["memory"]) == 0
    assert "fact 1 (" in (out := capsys.readouterr().out) and "Your jacket is Valve." in out
    assert main(["memory", "search", "valve"]) == 0
    out = capsys.readouterr().out
    assert "fact 1" in out and "message 1" in out
    assert main(["memory", "fix", "Valve", "velvet"]) == 0
    assert "Fixed 2 place(s)" in capsys.readouterr().out
    assert main(["memory", "edit", "1", "Your jacket is soft velvet."]) == 0
    assert main(["memory", "--all"]) == 0
    assert "soft velvet" in capsys.readouterr().out
    assert main(["memory", "forget", "1"]) == 0
    assert main(["memory", "forget", "1"]) == 1
    assert main(["memory"]) == 0
    assert "no facts yet" in capsys.readouterr().out
    assert main(["memory", "search", "nothing here"]) == 0
    assert "Nothing in her memory" in capsys.readouterr().out


def test_a_text_about_the_mistake_keeps_both_words():
    mem, now = make()
    talk(mem, now, "my jacket is Valve", "nice")
    talk(mem, now, "no, velvet, not Valve", "Velvet, got it.", gap=60)
    assert mem.correct("Valve", "velvet", "velvet, you misheard") == 1
    contents = [r[0] for r in mem.db.execute("SELECT content FROM memory_turn ORDER BY id")]
    assert contents[:3] == ["my jacket is velvet", "nice", "no, velvet, not Valve"]


def test_by_voice_a_fix_too_broad_is_said_not_done(monkeypatch, tmp_path):
    corrector(monkeypatch)
    n = Node([AssistantModule()], db_path=tmp_path / "p.db", background=False)
    for i in range(memory.FIX_MAX + 1):
        n.handle(Request(text=f"Valve {i}", module="assistant", session="s"))
    reply = n.handle(Request(text="velvet, you misheard", module="assistant", session="s"))
    assert reply.text.endswith('"Valve" is in too many places, so I left it.')
    assert "fixed" not in reply.data and n._mem().find("velvet")[0][1]["role"] == "user"


@pytest.mark.parametrize(
    "said, pairs",
    [
        ("it's velvet, not valve, you misheard me", [("valve", "velvet")]),
        ("you misheard, her name is Miso, not Mochi", [("Mochi", "Miso")]),
        ("not valve, velvet", [("valve", "velvet")]),
        ("I said velvet not valve", [("valve", "velvet")]),
        ("my favorite is Toru Kitajima, not Toro Kitajima", [("Toro", "Toru")]),
        ("é veludo e não Valve", [("Valve", "veludo")]),
        ("não Valve, veludo", [("Valve", "veludo")]),
        ("I like Miso now, not Mochi", []),
        ("I don't like Valve that much anymore, FromSoftware is my favorite now", []),
        ("it is not good", []),
        ("not valve, you misheard me", []),
    ],
)
def test_corrections_are_read_from_their_words(said, pairs):
    assert memory.corrections(said) == pairs


def test_by_voice_the_fix_needs_no_help_from_the_model(monkeypatch, tmp_path):
    corrector(monkeypatch, wrong="", right="")  # the model never names the mistake
    n = Node([AssistantModule()], db_path=tmp_path / "p.db", background=False)
    n.handle(Request(text="my cat is called Mochi", module="assistant", session="s"))
    reply = n.handle(
        Request(text="you misheard, her name is Miso, not Mochi", module="assistant", session="s")
    )
    assert reply.text.endswith("I fixed it in my memory: Miso, not Mochi.")
    assert n._mem().recent(10)[0]["content"] == "my cat is called Miso"


def test_a_module_that_does_not_fix_memory_never_does(monkeypatch, tmp_path):
    corrector(monkeypatch)
    monkeypatch.setattr(AssistantModule, "fixes_memory", False)  # e.g. the tutor
    n = Node([AssistantModule()], db_path=tmp_path / "p.db", background=False)
    n.handle(Request(text="my jacket is Valve", module="assistant", session="s"))
    reply = n.handle(
        Request(text="velvet, not Valve, you misheard", module="assistant", session="s")
    )
    assert "fixed" not in reply.data
    assert n._mem().recent(10)[0]["content"] == "my jacket is Valve"


@pytest.mark.parametrize(
    "said",
    [
        "I prefer tea, not coffee",
        "I'll take the train, not the bus",
        "I want the blue one, not the red one",
        "I play cello, not violin",
        "it's spicy, not sweet",
        "buy oat milk, not cow milk",
        "it was my sister, not my mom who called",
        "let's watch Frieren tonight, not Naruto",
        "it's Python, not Java, that I use at work",
        "eu quero pizza, não hambúrguer",
        "vamos de carro, não de ônibus",
        "quero o vermelho e não o azul",
        "that's wrong, it's tea not coffee",
        "like I said, I prefer tea, not coffee",
        "I said the train, not the bus",
        "como eu disse, quero pizza, não hambúrguer",
    ],
)
def test_a_preference_is_never_a_fix(monkeypatch, tmp_path, said):
    corrector(monkeypatch, wrong="", right="")
    n = Node([AssistantModule()], db_path=tmp_path / "p.db", background=False)
    before = (
        "I drink coffee every morning by train or the bus with blue and red cups, my mom, cow "
        "milk, sweet violin, Naruto, Java, hambúrguer, de ônibus, o azul"
    )
    n.handle(Request(text=before, module="assistant", session="s"))
    reply = n.handle(Request(text=said, module="assistant", session="s"))
    assert "fixed" not in reply.data
    assert n._mem().recent(10)[0]["content"] == before


def test_i_said_counts_only_right_before_the_word():
    mem, now = make()
    talk(mem, now, "my jacket is Valve", "nice")
    assert mem.correct("Valve", "velvet", "as I said velvet, not Valve") == 0
    assert mem.correct("Valve", "velvet", "I said velvet, not Valve") == 1
    talk(mem, now, "my jacket is Valve", "nice", gap=60)
    assert mem.correct("Valve", "veludo", "eu disse veludo, não Valve") == 1
