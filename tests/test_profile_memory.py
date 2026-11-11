"""Her profile of the person: lasting facts, grouped, deduplicated, within a budget."""

from prada import memory
from prada.memory import EPISODE_GAP, PROFILE_CHARS, Fact, Memory, Summary
from prada.store import open_store

HEAD = "What you know about them (they told you; still true):"


def make():
    now = [1_000_000.0]
    mem = Memory(open_store("memory", memory.MIGRATIONS, ":memory:"), lambda: now[0])
    return mem, now


def tell(mem, now, said, fact, kind=None, model="m", replaces=None):
    """One conversation where they say `said`, and the fact the summarizer found in it."""
    now[0] += EPISODE_GAP + 60
    mem.add(said, "ok")
    now[0] += EPISODE_GAP + 60
    for ep, transcript in mem.jobs():
        known = [k["id"] for k in mem.known_facts(transcript)]
        facts = [Fact(fact=fact, quote=said, kind=kind, replaces=replaces)]
        mem.store(ep, "", model, facts, known)


def test_profile_groups_lasting_facts_in_a_fixed_order():
    mem, now = make()
    tell(mem, now, "I am planning a trip to Tokyo in December", "You plan to visit Tokyo.", "plans")
    tell(mem, now, "I play the cello", "You play the cello.", "self")
    tell(mem, now, "I am so tired today", "You are tired today.", "moment")
    tell(mem, now, "my sister Ana lives in Porto", "Your sister Ana lives in Porto.", "people")
    tell(mem, now, "I study computer engineering", "You study computer engineering.", "work")
    tell(mem, now, "I love Unison Square Garden", "You love Unison Square Garden.", "tastes")
    tell(mem, now, "I live in Recife", "You live in Recife.", "self")
    tell(mem, now, "fake people say fake things", "You are fake.", "self", model="fake")
    assert mem.remembered() == "\n".join(
        [
            HEAD,
            "- Who they are: You play the cello. You live in Recife.",
            "- People in their life: Your sister Ana lives in Porto.",
            "- Tastes: You love Unison Square Garden.",
            "- Work and study: You study computer engineering.",
            "- Plans: You plan to visit Tokyo.",
        ]
    )


def test_a_change_of_mind_shows_only_what_is_true_now():
    mem, now = make()
    tell(mem, now, "my favorite anime is Naruto", "Your favorite anime is Naruto.", "tastes")
    tell(mem, now, "Frieren now", "Your favorite anime is Frieren.", "tastes", replaces=1)
    assert "Frieren" in mem.remembered() and "Naruto" not in mem.remembered()


def test_facts_from_before_kinds_get_their_own_group():
    mem, now = make()
    tell(mem, now, "I play the cello", "You play the cello.")
    assert mem.remembered() == HEAD + "\n- Other things they told you: You play the cello."


def test_near_duplicates_show_once_saying_the_most():
    mem, now = make()
    tell(mem, now, "I play the cello", "You play the cello.", "self")
    tell(mem, now, "I play cello in an orchestra", "You play cello in an orchestra.", "self")
    tell(mem, now, "my cat is Mochi", "Your cat is called Mochi.", "self")
    tell(mem, now, "my cat Mochi", "Your cat is called Mochi!", "work")  # same words: skipped
    profile = mem.remembered()
    assert "You play cello in an orchestra." in profile and "the cello" not in profile
    assert profile.count("Mochi") == 1


def test_a_fact_said_again_is_not_stored_twice_unless_it_replaces():
    mem, now = make()
    tell(mem, now, "I play the cello", "You play the cello.", "self")
    tell(mem, now, "I really play the cello", "You play cello.", "self")
    assert len(mem.facts()) == 1
    tell(mem, now, "I play the cello again", "You play the cello.", "self", replaces=1)
    assert len(mem.facts()) == 2 and mem.facts()[0]["superseded_by"] == 2


def test_a_change_the_summarizer_did_not_mark_as_replacing_is_never_lost():
    changes = [
        ("Você gosta de café.", "Você não gosta de café."),
        ("You are not vegetarian.", "You are vegetarian."),
        ("You don't play cello anymore.", "You play cello."),
        (
            "Your sister works at the Porto city hospital.",
            "Your sister works at the Lisbon city hospital.",
        ),
    ]
    for old, new in changes:
        mem, now = make()
        tell(mem, now, "first time I say it", old, "self")
        tell(mem, now, "second time I say it", new, "self")  # no `replaces`
        assert len(mem.facts()) == 2
        assert new in mem.remembered()  # the newest is never hidden behind the older one


def test_a_kind_that_is_not_a_word_is_dropped_too():
    raw = '{"summary": "s", "facts": [{"fact": "f", "quote": "q", "kind": ["self"]}]}'
    assert Summary.model_validate_json(raw).facts[0].kind is None


def test_profile_keeps_the_newest_within_its_budget_and_summaries_share_the_rest():
    mem, now = make()
    for i in range(40):
        tell(mem, now, f"remember code word{i} please", f"Your code is word{i} number {i}.", "self")
    with mem.db:
        mem.db.execute("UPDATE memory_episode SET summary = ?", ("We talked. " * 20,))
    text = mem.remembered()
    profile, _, summaries = text.partition("\n\n")
    assert len(profile) <= PROFILE_CHARS and "word39" in profile and "word0 " not in profile
    assert len(text) <= memory.MEMORY_CHARS and summaries.startswith("- ")


def test_recall_does_not_repeat_what_the_profile_shows():
    mem, now = make()
    tell(
        mem,
        now,
        "my sister Ana starts a new job in Porto",
        "Your sister Ana works in Porto.",
        "people",
    )
    now[0] += EPISODE_GAP + 60
    mem.add("end", "ok")
    now[0] += EPISODE_GAP + 60
    mem.add("end", "ok")
    assert "Porto" in mem.remembered()
    assert mem.recall("where does my sister Ana work? the new job in Porto") == ""


def test_recall_still_tells_a_change_the_profile_has_no_past_for():
    mem, now = make()
    tell(mem, now, "my favorite anime is Naruto", "Your favorite anime is Naruto.", "tastes")
    tell(
        mem,
        now,
        "my favorite anime is Frieren now",
        "Your favorite anime is Frieren.",
        "tastes",
        replaces=1,
    )
    got = mem.recall("what was my favorite anime before?")
    assert "Frieren" in got and "Naruto" in got


def test_a_kind_the_model_made_up_is_dropped_not_the_summary():
    raw = '{"summary": "s", "facts": [{"fact": "f", "quote": "q", "kind": "hobby"}]}'
    assert Summary.model_validate_json(raw).facts[0].kind is None
    assert "moment" in str(Summary.model_json_schema())
