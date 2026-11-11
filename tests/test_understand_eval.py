import json
import tempfile
from pathlib import Path

import pytest

from prada import cli as prada_cli
from prada import eval as assistant_eval
from prada import understand_eval as u
from prada.actions import Outcome
from prada.module import Reply

CASES = u.load_cases()


def test_evalset_is_well_formed():
    assert len(CASES) >= 45
    assert {c.group for c in CASES} == set(u.GROUPS)
    assert any(not c.calls for c in CASES) and any(len(c.calls) > 1 for c in CASES)
    assert any(c.before for c in CASES) and any(c.memory for c in CASES)
    situations = {key for c in CASES for key in c.situation}
    assert situations == {"players", "last", "offer"}
    assert len({c.text for c in CASES}) == len(CASES)  # --compare lines cases up by text
    held = [c for c in CASES if c.held]
    assert len(held) >= 20 and {c.group for c in held} == set(u.GROUPS)
    assert not CASES[50].held and all(c.held for c in CASES[51:])  # the first 51 stay as they were


def test_held_out_cases_are_counted_apart():
    first, held = CASES[0], next(c for c in CASES if c.held)
    s = u.score([first, held])
    m = u.summary(s)
    assert m["cases"] == 1 and m["held"]["cases"] == 1
    assert "held-out:" in u.format_report(s)


@pytest.mark.parametrize(
    "line, error",
    [
        ({"group": "x", "text": "t", "calls": []}, "unknown group"),
        ({"group": "multi", "text": "t", "calls": [{"tool": "lights.on"}]}, "unknown tool"),
        ({"group": "multi", "text": "t", "calls": [{"tool": "wake.send", "mac": "1"}]}, "no arg"),
        ({"group": "multi", "text": "t", "calls": [], "situation": {"mood": "x"}}, "situation"),
        (
            {
                "group": "multi",
                "text": "t",
                "calls": [],
                "situation": {"players": [{"app": "Spotify", "state": "playing", "title": "Nope"}]},
            },  # fmt: skip
            "no fake track",
        ),
        ({"group": "multi", "text": "t"}, "bad eval case"),
    ],
)
def test_bad_cases_are_refused(tmp_path, line, error):
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(line), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        u.load_cases(path)


def test_actions_become_calls():
    def out(action):
        return Outcome(action=action, ok=True)

    reply = Reply(
        outcomes=[
            out({"kind": "music.next", "app": "Spotify.exe", "player": "Spotify"}),
            out({"kind": "music.play", "query": "hysteria", "want": "any"}),
            out(
                {
                    "kind": "music.play_corpus",
                    "label": "Wave by Jobim",
                    "paths": ["a", "b"],
                    "titles": ["Wave", "Triste"],
                    "mode": "queue",
                    "shuffle": False,
                }
            ),  # fmt: skip
            out({"kind": "notes.save", "text": "eggs"}),
            out({"kind": "web.search", "query": "rtx 5080", "sources": ["web"]}),
        ],
        offer="weather in Lisbon",
        data={"fixed": 2},
    )
    held = [{"kind": "wake.send", "mac": "x", "broadcast": "y", "port": 9}]
    calls = u.calls_of("it's velvet, not valve", reply, held)
    assert calls == [
        {"tool": "music.next", "app": "Spotify"},
        {"tool": "music.play", "what": "hysteria", "app": "Spotify", "mode": "play"},
        {
            "tool": "music.play",
            "what": "Wave by Jobim",
            "app": "Corpus",
            "mode": "queue",
            "tracks": 2,
        },  # fmt: skip
        {"tool": "notes.save", "text": "eggs"},
        {"tool": "web.search", "query": "rtx 5080"},
        {"tool": "wake.send"},
        {"tool": "memory.fix", "wrong": "valve", "right": "velvet"},
        {"tool": "web.offer", "query": "weather in Lisbon"},
    ]


def test_check_verdicts():
    def case(*calls):
        return u.Case("multi", "t", list(calls))

    play = {"tool": "music.play", "what": "Space Dementia by Muse", "app": "Corpus"}
    pause = {"tool": "music.pause", "app": "Spotify"}
    assert u.check(case({"tool": "music.play", "what": "space dementia"}), [play]) == ""
    assert u.check(case({"tool": "music.play", "app": "spotify"}), [play]).startswith("wrong")
    assert u.check(case(), [pause]).startswith("false action")
    assert u.check(case(pause), []).startswith("missed action")
    assert u.check(case(pause, play), [pause]).startswith("missed action")
    assert u.check(case(pause), [pause, play]).startswith("false action")
    assert u.check(case(pause, pause), [pause]).startswith("missed")  # each call counts once
    assert u.check(case({"tool": "music.play", "tracks": 2}), [{**play, "tracks": 1}]) != ""


def test_fake_run_scores_today_pipeline():
    s = u.score(CASES)
    by_text = {r.case.text: r for r in s.results}
    assert len(s.results) == len(CASES) and s.calls > 0
    # code-only paths work the same without a model
    assert by_text["skip this one"].failure == ""
    assert by_text["the album"].failure == ""
    assert by_text["just the song please"].failure == ""
    assert by_text["yeah go ahead"].failure == ""
    assert by_text["you misheard me before, it's Velvet, not Valve"].failure == ""
    # a pending choice really is pending, and the wake is held, never sent
    wave = by_text["the album"]
    assert "song or the album" in wave.before[0][1].text
    assert by_text["wake up my PC... no, never mind"].calls == [{"tool": "wake.send"}]
    assert all(r.stages and r.seconds >= 0 for r in s.results)


def test_situation_sets_players_and_her_last_line(tmp_path):
    seen = []

    def client(fake):
        def chat(messages, schema, on_text=None):
            seen.append(json.dumps(messages))
            return fake(messages=messages, schema=schema)

        return chat

    case = next(c for c in CASES if c.text == "yes please")
    u.score([case], client=client)
    assert any("put on something calm" in m for m in seen)
    both = next(c for c in CASES if c.text == "pause everything")
    case = u.Case("reference", "pause", both.calls, both.situation)
    r = u.score([case]).results[0]  # a plain "pause" pauses every player that plays
    assert r.failure == "" and len(r.calls) == 2


def test_false_claim_fails_a_case_with_no_calls(monkeypatch):
    # the assistant's guard rewrites most claims; any that slip through fail the case
    monkeypatch.setattr(u, "claims_action", lambda text: True)
    case = next(c for c in CASES if c.text == "turn it up a bit")
    r = u.score([case]).results[0]
    assert r.calls == [] and r.claimed and r.failure.startswith("false claim")
    assert u.summary(u.Score([r]))["false_claims"] == 1


def test_report_transcript_and_compare(tmp_path, capsys):
    s = u.score(CASES[:3])
    report = u.format_report(s)
    assert "[understand] 3 cases" in report and "ms per turn by stage" in report
    text = u.format_transcript(s, "fake")
    assert "*calls:*" in text and CASES[0].text in text
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    first = u.results(s, "m1")
    second = json.loads(json.dumps(first))
    second["model"], second["results"][0]["failure"] = "m2", "missed action: x"
    a.write_text(json.dumps(first))
    b.write_text(json.dumps(second))
    assert assistant_eval.compare([str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "ollama: m1" in out and "FAIL: missed action: x" in out
    other = tmp_path / "c.json"
    other.write_text(json.dumps({**first, "suite": "assistant"}))
    assert assistant_eval.compare([str(a), str(other)]) == 1
    assert "can't be compared" in capsys.readouterr().err


def test_prada_eval_understand_smoke(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PRADA_FAKE_LLM", "1")
    path = tmp_path / "run.md"
    assert prada_cli.main(["eval", "--suite", "understand", "--transcript", str(path)]) == 0
    out = capsys.readouterr().out
    assert "model: fake" in out and "[understand]" in out and "right calls" in out
    assert path.exists() and json.loads(path.with_suffix(".json").read_text())["suite"] == (
        "understand"
    )


def test_transcript_defaults_to_data_dir(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PRADA_FAKE_LLM", "1")
    monkeypatch.setenv("PRADA_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(u, "load_cases", lambda path=None: CASES[:2])
    assert prada_cli.main(["eval", "--suite", "understand"]) == 0
    assert list((tmp_path / "eval").glob("understand-fake-*.md"))


def _scripted_mind(calls_by_text):
    """A client whose understand calls answer from `calls_by_text` (keyed by the newest
    message); every other caller keeps its fake."""
    from prada import mind

    def client(fake):
        if fake is not mind.fake_chat:
            return u._canned(fake)

        def chat(messages, schema, on_text=None):
            text = messages[-1]["content"].rpartition("newest message: ")[2]
            return json.dumps({"note": "n", "calls": calls_by_text.get(text, [])})

        return chat

    return client


def test_before_turns_need_no_model():
    # the understand mode plays `before` turns on code paths only: never a fake reply
    for case in (c for c in CASES if c.before):
        s = u.score([case], mind_name="understand")
        assert s.calls == 1, case.text  # only the understand call
        assert all(not r.text.startswith("(fake") for _, r in s.results[0].before), case.text


def test_understand_mode_scores_checked_calls():
    second = next(c for c in CASES if c.text == "the second one")
    said = next(c for c in CASES if c.text == "no, I said Space Dementia")
    both = next(c for c in CASES if c.text == "pause everything")
    client = _scripted_mind(
        {
            "the second one": [{"quote": "the second one", "tool": "music.choose", "option": 2}],
            "no, I said Space Dementia": [
                {"quote": "I said Space Dementia", "tool": "music.play", "what": "Space Dementia"}
            ],
            "pause everything": [
                {"quote": "pause everything", "tool": "music.pause", "app": "Spotify"},
                {"quote": "pause everything", "tool": "music.pause", "app": "Corpus"},
                {"quote": "pause everything", "tool": "music.pause", "app": "Spotify"},
            ],
        }
    )
    s = u.score([second, said, both], client=client, mind_name="understand")
    assert [r.failure for r in s.results] == ["", "", ""]
    assert s.results[0].calls[0]["tracks"] == 2  # the pick became the album's play
    m = u.summary(s)
    assert m["mind"] == "understand" and m["dropped_calls"] == 0
    assert "the understand step" in u.format_report(s)
    assert "*note:* n" in u.format_transcript(s, "fake")
    assert u._run_name(u.results(s, "m")) == "ollama: m understand"


def test_understand_mode_situation():
    hysteria = next(c for c in CASES if c.text == "no, I said Space Dementia")
    situation, turns, picks = u.situation_of(hysteria, Path(tempfile.mkdtemp()) / "x.db")
    spotify = next(p for p in situation.players if p.app == "Spotify")
    assert (spotify.state, spotify.title) == ("playing", "Hysteria")  # the play moved it
    assert turns[-1][0] == "play Hysteria" and "Hysteria" in turns[-1][1] and picks == {}
    offer = next(c for c in CASES if c.text == "yeah go ahead")
    situation, turns, _ = u.situation_of(offer, Path(tempfile.mkdtemp()) / "x.db")
    assert situation.offer and turns == [(offer.situation["offer"], u.OFFER_LINE)]
    quiet = next(c for c in CASES if c.situation.get("players") == [])
    assert u.situation_of(quiet, Path(tempfile.mkdtemp()) / "x.db")[0].players == []


def test_understand_mode_drops_and_invalid_replies():
    case = u.Case("reference", "skip", [], {"players": []})

    def client(reply):
        return lambda fake: lambda messages, schema, on_text=None: reply

    s = u.score(
        [case],
        client=client('{"note": "x", "calls": [{"quote": "skip", "tool": "music.next"}]}'),
        mind_name="understand",
    )
    assert (
        s.results[0].failure == ""
        and s.dropped == 1
        and "no playing player" in s.results[0].dropped[0]
    )
    s = u.score([case], client=client("nope"), mind_name="understand")
    assert s.invalid == 1 and s.results[0].reply.text == "(not valid JSON)"


def test_prada_eval_understand_mind_smoke(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PRADA_FAKE_LLM", "1")
    path = tmp_path / "run.md"
    argv = ["eval", "--suite", "understand", "--mind", "understand", "--transcript", str(path)]
    assert prada_cli.main(argv) == 0
    out = capsys.readouterr().out
    assert "the understand step" in out and "held-out:" in out and "dropped calls" in out
    assert json.loads(path.with_suffix(".json").read_text())["mind"] == "understand"
