import json

from prada import mind
from prada.mind import Option, Player, Situation

PLAYERS = [Player("Spotify", "playing", "Unravel", "TK"), Player("Corpus", "paused", "Wave")]


def scripted(*calls, note="the user wants something"):
    seen = []

    def chat(messages, schema):
        seen.append(messages)
        return json.dumps({"note": note, "calls": list(calls)})

    return chat, seen


def run(text, *calls, situation=None, turns=()):
    chat, _ = scripted(*calls)
    return mind.understand(chat, situation or Situation(0, PLAYERS), list(turns), text)


def test_quote_must_be_the_users_words():
    assert mind.quote_ok("Toca Evidências!", ["toca evidencias por favor"])
    assert mind.quote_ok("skip  this", ["Skip this one"])
    assert not mind.quote_ok("play Muse", ["could you put on something"])
    assert not mind.quote_ok("", ["anything"]) and not mind.quote_ok("?", ["?"])
    assert not mind.quote_ok("play", ["I played tennis today"])
    assert not mind.quote_ok("no", ["I know it"])
    assert mind.quote_ok("space dementia", ["play spacedementia"])
    u = run("put on Muse", {"quote": "play Muse", "tool": "music.play", "what": "Muse"})
    assert u.calls == [] and "not the user's words" in u.dropped[0]


def test_valid_calls_are_kept_with_their_arguments():
    u = run(
        "put on some Muse and write down eggs",
        {"quote": "put on some Muse", "tool": "music.play", "what": "Muse", "kind": "artist"},
        {"quote": "write down eggs", "tool": "notes.save", "text": "eggs", "what": "junk"},
    )
    assert u.note == "the user wants something" and u.dropped == []
    assert [(c.tool, c.args) for c in u.calls] == [
        ("music.play", {"what": "Muse", "kind": "artist", "app": "", "mode": "play"}),
        ("notes.save", {"text": "eggs"}),  # arguments of other tools are ignored
    ]


def test_quote_may_come_from_the_previous_turn():
    turns = [("play Wave", "Wave: the song or the album?")]
    s = Situation(0, PLAYERS, "Wave: the song or the album?", [Option("song"), Option("album")])
    u = run("two", {"quote": "two", "tool": "music.choose", "option": 2}, situation=s, turns=turns)
    assert [(c.tool, c.args) for c in u.calls] == [("music.choose", {"option": 2})]
    u = run("yes", {"quote": "play Wave", "tool": "music.play", "what": "Wave"}, turns=turns)
    assert len(u.calls) == 1


def test_player_calls_follow_the_situation():
    def one(tool, app=None, players=PLAYERS):
        call = {"quote": "do it", "tool": tool} | ({"app": app} if app else {})
        u = run("do it", call, situation=Situation(0, players))
        return (u.calls[0].args["app"] if u.calls else None), u.dropped

    assert one("music.pause") == ("Spotify", [])  # the one player playing
    assert one("music.resume") == ("Corpus", [])  # the one paused
    assert one("music.pause", "Corpus")[1][0].endswith("Corpus is paused")  # already done
    assert one("music.next", players=[])[1][0].endswith("no playing player")
    both = [Player("Spotify", "playing"), Player("Corpus", "playing")]
    assert one("music.pause", players=both)[1][0].endswith("which player?")
    assert one("music.pause", "Corpus", players=both) == ("Corpus", [])
    assert one("music.pause", "Spotify", players=[Player("Corpus", "playing")])[1]


def test_other_checks_drop_calls():
    u = run("pick it", {"quote": "pick it", "tool": "music.choose", "option": 1})
    assert u.calls == [] and "no such option" in u.dropped[0]
    u = run("search", {"quote": "search", "tool": "web.search"})
    assert "no query" in u.dropped[0]
    said = "you misheard, it's Velvet"
    fix = {"quote": said, "tool": "memory.fix", "wrong": "Valve", "right": "Velvet"}
    assert len(run(said, fix).calls) == 1
    assert "not in the newest" in run(said, fix | {"right": "Velour"}).dropped[0]
    assert "same word" in run(said, fix | {"wrong": "velvet"}).dropped[0]
    play = {"quote": "play it", "tool": "music.play", "app": "Winamp"}
    assert run("play it", play).dropped  # not an app the schema allows


def test_duplicates_and_bad_replies():
    call = {"quote": "skip", "tool": "music.next"}
    assert len(run("skip", call, call).calls) == 1
    for reply in ("not json", '{"calls": []}', '{"note": "x", "calls": "none"}'):
        u = mind.understand(lambda messages, schema, reply=reply: reply, Situation(0), [], "hi")
        assert not u.valid and u.calls == []


def test_prompt_situation_and_schema():
    chat, seen = scripted()
    s = Situation(
        0, PLAYERS, "Wave: the song or the album?", [Option("the song: Wave", 1, "Corpus")]
    )
    s.offer, s.remembered = "rain in Porto", ["I play the cello"]
    mind.understand(chat, s, [("hello", "hi there")], "and now?")
    system, user = seen[0][0]["content"], seen[0][1]["content"]
    assert system == mind.prompt() and all(t.name in system for t in mind.TOOLS)
    assert (
        '- Spotify: playing "Unravel" by TK' in user
        and "1. the song: Wave (1 song, Corpus)" in user
    )
    assert "rain in Porto" in user and "- I play the cello" in user
    assert "User: hello\nPRADA: hi there" in user
    assert user.endswith("The user's newest message: and now?")
    assert "none open" in mind.render(Situation(0))
    schema = mind.schema()
    assert list(schema["properties"]) == ["note", "calls"]  # the note is written first
    tools = schema["properties"]["calls"]["items"]["properties"]["tool"]["enum"]
    assert tools == [t.name for t in mind.TOOLS]


def test_fake_chat_and_nothing_at_runtime_uses_it():
    chat = lambda messages, schema: mind.fake_chat(messages, schema)  # noqa: E731
    u = mind.understand(chat, Situation(0, PLAYERS), [], "play Muse")
    assert [(c.tool, c.args["what"]) for c in u.calls] == [("music.play", "Muse")]
    assert mind.understand(chat, Situation(0, PLAYERS), [], "hello").calls == []
    from pathlib import Path

    import prada

    for name in ("node.py", "server.py", "assistant.py", "module.py"):
        source = (Path(prada.__file__).parent / name).read_text(encoding="utf-8")
        assert "prada.mind" not in source and "import mind" not in source
