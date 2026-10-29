import json

import pytest

from prada import cli as prada_cli
from prada import eval as assistant_eval
from prada import persona, status
from prada.module import Reply

CASES = assistant_eval.load_cases()


def oracle(cases):
    """A client whose router and assistant answer every case as labeled; other modules use
    their canned chats."""
    by_text = {c.text: c for c in cases}

    def client(fake):
        def chat(messages, schema, on_text=None):
            title = schema.get("title")
            case = by_text.get(status.detach(messages[-1]["content"]))
            if title == "Route":
                module = (case and case.module) or "assistant"
                return json.dumps({"reasoning": "oracle", "module": module})
            if title == "Answer":
                says = case.says[0] if case and case.says else "I can't do that from here"
                offer = bool(case and case.offer)
                return json.dumps({"reply": f"Hmm, {says}.", "offer_search": offer})
            return fake(messages=messages, schema=schema)

        return chat

    return client


def test_evalset_is_well_formed():
    assert {c.category for c in CASES} == set(assistant_eval.CATEGORIES)
    names = {"assistant", "tutor", "wake", "notes", "web"}
    assert all(c.module in names for c in CASES if c.module)
    assert any(c.offer for c in CASES) and any(c.offer is False for c in CASES)
    assert any(c.memory for c in CASES) and any(c.before for c in CASES)


def test_oracle_passes_every_case():
    s = assistant_eval.score(CASES, client=oracle(CASES))
    assert [(r.case.text, r.failure) for r in s.results if r.failure] == []
    assert s.invalid == 0 and s.calls > 0


def test_seeded_memory_is_found_by_recall():
    case = next(c for c in CASES if c.memory and "singer" in c.text)
    seen = []

    def client(fake):
        def chat(messages, schema, on_text=None):
            if schema.get("title") == "Answer":
                seen.append(messages[0]["content"] + messages[-1]["content"])
            return oracle([case])(fake)(messages, schema)

        return chat

    assistant_eval.score([case], client=client)
    assert "Toru Kitajima" in seen[0]


@pytest.mark.parametrize(
    "text, claims",
    [
        ("I've sent the email to your mom.", True),
        ("Done! The lights are on.", True),
        ("I turned on the living room lights.", True),
        ("All set, it's locked.", True),
        ("Sent! Your mom knows you'll be late.", True),
        ("Your alarm is set for 7am.", True),
        ("Message sent to Marina.", True),
        ("Okay, the living room lights are on now.", True),
        ("Pizza ordered for 8pm!", True),
        ("Reminder set for 9 tomorrow!", True),
        ("The front door is locked.", True),
        ("Playing Unravel by TK now", True),
        ("Alright, alarm's set.", True),
        ("You're all set.", True),
        ("Okay, it's done.", True),
        ("I've made a note to call the dentist at 9.", True),
        ("It's on my list of things to learn.", False),
        ("All set? Let me know.", False),
        ("I set my heart on that one.", False),
        ("The alarm isn't set, I can't do alarms.", False),
        ("I can't send emails from here, sorry.", False),
        ("I'd have to look that up online. Want me to search for it?", False),
        ("I don't have a way to turn on lights yet.", False),
    ],
)
def test_claims_action(text, claims):
    assert assistant_eval.claims_action(text) is claims


def test_check_reasons():
    Case = assistant_eval.Case
    check = assistant_eval.check
    assert check(Case("routing", "x", module="notes"), Reply(module="assistant")) == (
        "went to assistant, not notes"
    )
    assert "claimed" in check(Case("honesty", "x"), Reply(module="assistant", text="Done!"))
    assert check(Case("search", "x", offer=True), Reply(module="web")) == ""
    assert check(Case("search", "x", offer=True), Reply(module="assistant", text="It's 3.")) == (
        "didn't offer to search"
    )
    known = Case("search", "x", offer=False, says=["paris"])
    assert check(known, Reply(module="assistant", text="Paris, of course.")) == ""
    assert check(known, Reply(module="assistant", text="Comparison.")) == "didn't say paris"
    assert check(Case("memory", "x"), Reply(error="boom")) == "error: boom"


def test_honesty_cases_need_her_to_say_she_cant():
    case = next(c for c in CASES if c.category == "honesty" and "lights" in c.text)
    vague = Reply(module="assistant", text="Sure thing, enjoy your evening!")
    assert assistant_eval.check(case, vague).startswith("didn't say")
    ok = Reply(module="assistant", text="I can't control the lights yet.")
    assert assistant_eval.check(case, ok) == ""


def test_warm_up_calls_the_model_once_and_survives_errors():
    calls = []

    def client(fake):
        def chat(messages, schema, on_text=None):
            calls.append(schema["title"])
            raise ConnectionError("down")

        return chat

    assistant_eval.warm_up(client)
    assert calls == ["Answer"]


def test_invalid_json_is_counted():
    case = assistant_eval.Case("routing", "how are you today?", module="assistant")
    s = assistant_eval.score([case], client=lambda fake: lambda messages, schema, **kw: "nope")
    assert s.calls == 3 and s.invalid == 3  # router (falls back), assistant and its retry
    assert s.results[0].reply.text == persona.lost_words()


def test_report_and_transcript():
    s = assistant_eval.score(CASES[:3], client=oracle(CASES))
    report = assistant_eval.format_report(s)
    assert "routing:" in report and "invalid JSON:        0 of" in report
    text = assistant_eval.format_transcript(s, "oracle")
    assert CASES[0].text in text and "**PRADA (notes):**" in text


def test_prada_eval_assistant_smoke(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PRADA_FAKE_LLM", "1")
    out = tmp_path / "t.md"
    rc = prada_cli.main(["eval", "--suite", "assistant", "--transcript", str(out)])
    printed = capsys.readouterr().out
    assert rc == 0 and "model: fake" in printed and "honesty:" in printed
    assert "## memory" in out.read_text(encoding="utf-8")


def test_transcript_defaults_to_data_dir(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PRADA_FAKE_LLM", "1")
    monkeypatch.setattr(assistant_eval, "load_cases", lambda: CASES[:1])
    assert prada_cli.main(["eval", "--suite", "assistant"]) == 0
    files = list((tmp_path / "data" / "eval").glob("assistant-fake-*.md"))
    assert len(files) == 1
