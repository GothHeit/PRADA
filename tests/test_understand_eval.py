import json

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
