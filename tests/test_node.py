import json
import re
import socket
import threading

import pytest

from prada import client, config, persona, router
from prada.assistant import AssistantModule
from prada.cli import main
from prada.module import Context, Module, Reply, Request
from prada.node import Node
from prada.server import make_server
from prada.wake import WakeModule
from tutor.languages import detect_language, script_language
from tutor.module import TutorModule


class Echo(Module):
    name = "echo"
    description = "Repeats the message."
    migrations = ["CREATE TABLE echo_seen (text TEXT)"]

    def handle(self, request: Request, ctx: Context) -> Reply:
        ctx.state["count"] = ctx.state.get("count", 0) + 1
        ctx.db.execute("INSERT INTO echo_seen VALUES (?)", (request.text,))
        return Reply(module=self.name, text=request.text, data={"count": ctx.state["count"]})


class Boom(Module):
    name = "boom"
    description = "Always fails."

    def handle(self, request, ctx):
        raise RuntimeError("kaboom")


@pytest.fixture
def node(monkeypatch):
    monkeypatch.setenv("PRADA_FAKE_LLM", "1")
    return Node([Echo(), Boom()], db_path=":memory:")


def test_explicit_module_keeps_state_per_session(node):
    send = node.handle
    assert send(Request(text="a", module="echo", session="s1")).data == {"count": 1}
    assert send(Request(text="b", module="echo", session="s1")).data == {"count": 2}
    assert send(Request(text="c", module="echo", session="s2")).data == {"count": 1}
    assert node._dbs["echo"].execute("SELECT COUNT(*) FROM echo_seen").fetchone()[0] == 3


def test_errors_come_back_in_the_reply(node):
    assert node.handle(Request(text="x", module="boom")).error == "boom: kaboom"
    assert "no module" in node.handle(Request(text="x", module="nope")).error


def test_routing(node):
    assert node.handle(Request(text="make it go boom")).module == "boom"
    assert node.handle(Request(text="anything else")).module == "echo"
    reply = node.handle(Request(text="/echo hi there"))
    assert (reply.module, reply.text) == ("echo", "hi there")


def test_router_prompt_and_schema():
    modules = [Echo(), Boom()]
    assert router.route_schema(["echo", "boom"])["properties"]["module"]["enum"] == ["echo", "boom"]
    prompt = router.build_prompt(modules, current="boom")
    assert "- echo: Repeats the message." in prompt and "pick echo" in prompt
    assert "conversation so far is with boom" in prompt
    assert router.explicit("/nope x", {"echo"}) is None


@pytest.mark.parametrize("answer", ['{"reasoning": "r", "module": "ghost"}', "not json"])
def test_router_falls_back_to_first_module(answer):
    chat = lambda messages, schema: answer  # noqa: E731
    assert router.route("hi", [Echo(), Boom()], chat) == "echo"


def test_router_passes_current_module():
    seen = {}

    def chat(messages, schema):
        seen["system"] = messages[0]["content"]
        return json.dumps({"reasoning": "r", "module": "boom"})

    n = Node([Echo(), Boom()], db_path=":memory:")
    n._chat = lambda fake: chat
    n.handle(Request(text="x", module="echo", session="s"))
    assert n.handle(Request(text="y", session="s")).error == "boom: kaboom"
    assert "with echo" in seen["system"]


def test_sessions_are_capped(node, monkeypatch):
    monkeypatch.setattr("prada.node.MAX_SESSIONS", 2)
    for sid in "abc":
        node.handle(Request(text="x", module="echo", session=sid))
    assert list(node._sessions) == ["b", "c"]


@pytest.mark.parametrize(
    "text,lang", [("Я читаю", "ru"), ("本を読む", "ja"), ("je lis", None), ("123", None)]
)
def test_script_language(text, lang):
    assert script_language(text) == lang


def test_tutor_module_language_history_and_log(monkeypatch):
    monkeypatch.setenv("PRADA_FAKE_LLM", "1")
    n = Node([TutorModule()], db_path=":memory:")
    reply = n.handle(Request(text="Je lis un livre.", session="s"))
    assert reply.module == "tutor" and reply.text.endswith("Qu'est-ce que tu aimes lire ?")
    assert reply.data["is_correct"] is True
    n.handle(Request(text="encore", session="s"))  # Latin text keeps the session's French
    assert len(n._sessions["s"]["tutor"]["history"]) == 4
    assert n.handle(Request(text="Я читаю.", session="s")).text.endswith("читать?")
    assert n._sessions["s"]["tutor"]["lang"] == "ru"
    assert len(n._sessions["s"]["tutor"]["history"]) == 2
    forced = n.handle(Request(text="Я читаю.", session="s", options={"lang": "ja"}))
    assert forced.text.endswith("好きですか？")
    rows = n._dbs["tutor"].execute("SELECT language FROM tutor_exchange ORDER BY id").fetchall()
    assert [r[0] for r in rows] == ["fr", "fr", "ru", "ja"]


def test_unknown_language_is_an_error_reply(monkeypatch):
    monkeypatch.setenv("PRADA_FAKE_LLM", "1")
    n = Node([TutorModule()], db_path=":memory:")
    assert "unknown language" in n.handle(Request(text="x", options={"lang": "xx"})).error


def test_unopenable_store_still_answers(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PRADA_FAKE_LLM", "1")
    blocker = tmp_path / "file"
    blocker.write_text("")
    n = Node([TutorModule()], db_path=blocker / "prada.db")
    assert n.handle(Request(text="Я читаю.")).error is None
    assert "memory unavailable" in capsys.readouterr().err


def test_wake_module(monkeypatch):
    sent = []
    monkeypatch.setattr("prada.wol.send_magic_packet", lambda *a: sent.append(a))
    monkeypatch.setenv("PRADA_WAKE_MAC", "aa:bb:cc:dd:ee:ff")
    node = Node([WakeModule()])
    node.handle(Request(text="wake up", session="s"))
    reply = node.handle(Request(text="yes", session="s"))
    assert reply.error is None and sent == [("aa:bb:cc:dd:ee:ff", "255.255.255.255", 9)]


@pytest.mark.parametrize(
    "text, wake",
    [
        ("wake up", True),
        ("turn on my computer", True),
        ("liga o pc", True),
        ("wake the PC from sleep", True),
        ("my computer is off, can you turn it on?", True),
        ("start my pc", True),
        ("start up the computer", True),
        ("ligue o computador", True),
        ("shut down the pc", False),
        ("put the computer to sleep", False),
        ("I'm leaving now, I'll turn you off for a moment.", False),
        ("turn up the volume", False),
        ("Do you know who my favorite singer is?", False),
    ],
)
def test_wake_accepts_only_turning_the_pc_on(text, wake):
    assert WakeModule().accepts(text) is wake


def test_router_pick_of_wake_for_a_goodbye_goes_to_the_fallback(monkeypatch):
    monkeypatch.setattr("prada.router.route", lambda *a, **k: "wake")
    node = Node([AssistantModule(), WakeModule()])
    reply = node.handle(Request(text="I'm leaving, I'll turn you off now", session="s"))
    assert reply.module == "assistant" and not reply.actions


def test_wake_without_a_mac_explains_in_her_words(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("PRADA_WAKE_MAC", raising=False)
    monkeypatch.setenv("PRADA_CONFIG", str(tmp_path / "none.toml"))
    reply = Node([WakeModule()]).handle(Request(text="wake up", session="s"))
    assert reply.error is None and reply.text == persona.cant_wake()
    assert "wake: no MAC" in capsys.readouterr().err


@pytest.fixture
def served(node):
    server = make_server(node, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def test_remote_session_round_trip(served):
    assert client.ping(served) == ["echo", "boom"]
    session = client.RemoteSession(served)
    assert session.send("hi", "echo").data == {"count": 1}
    assert session.send("again", "echo").data == {"count": 2}
    assert session.send("x", "boom").error == "boom: kaboom"


def test_server_rejects_bad_requests(served):
    import urllib.error
    import urllib.request

    def post(path, body, ctype):
        req = urllib.request.Request(served + path, data=body, headers={"Content-Type": ctype})
        try:
            client._opener.open(req)
        except urllib.error.HTTPError as e:
            return e.code

    assert post("/v1/request", b"text=hi", "application/x-www-form-urlencoded") == 415
    assert post("/v1/request", b'{"module": "echo"}', "application/json") == 400
    assert post("/elsewhere", b"{}", "application/json") == 404


def test_remote_session_reports_lost_node():
    with pytest.raises(client.NodeError):
        client.RemoteSession("http://127.0.0.1:1", timeout=1).send("hi")


def test_connect_falls_back_to_in_process(monkeypatch, capsys):
    monkeypatch.delenv("PRADA_NODE")
    monkeypatch.delenv("PRADA_FAKE_LLM", raising=False)
    monkeypatch.setenv("PRADA_NODE_PORT", "1")
    assert isinstance(client.connect(), client.LocalSession)
    assert "no node running" in capsys.readouterr().err


def test_connect_uses_running_node(monkeypatch, served):
    monkeypatch.delenv("PRADA_NODE")
    monkeypatch.delenv("PRADA_FAKE_LLM", raising=False)
    session = client.connect(int(served.rsplit(":", 1)[1]))
    assert isinstance(session, client.RemoteSession)


def test_node_port_precedence(monkeypatch, tmp_path):
    monkeypatch.delenv("PRADA_NODE_PORT", raising=False)
    assert config.node_port() == 7723
    (tmp_path / "config.toml").write_text("[node]\nport = 8000\n")
    assert config.node_port() == 8000
    monkeypatch.setenv("PRADA_NODE_PORT", "8001")
    assert config.node_port() == 8001
    assert config.node_port(8002) == 8002


def test_cli_chat_routes(monkeypatch, capsys):
    monkeypatch.setenv("PRADA_FAKE_LLM", "1")
    lines = iter(["tutor 本を読みます。", "what time is it", "wake the pc"])

    def fake_input(_):
        try:
            return next(lines)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", fake_input)
    assert main(["chat"]) == 0
    captured = capsys.readouterr()
    assert "何を読むのが好きですか？" in captured.out
    assert "(fake assistant) what time is it" in captured.out
    assert "wake: no MAC" in captured.err


def test_server_rejects_foreign_host(served):
    import http.client

    port = int(served.rsplit(":", 1)[1])
    for host, status in [(f"evil.example:{port}", 421), (f"localhost:{port}", 200)]:
        conn = http.client.HTTPConnection("127.0.0.1", port)
        conn.request("GET", "/health", headers={"Host": host})
        assert conn.getresponse().status == status
        conn.close()
    conn = http.client.HTTPConnection("127.0.0.1", port)
    conn.request(
        "POST",
        "/v1/request",
        body=b'{"text": "x", "module": "echo"}',
        headers={"Host": "evil.example", "Content-Type": "application/json"},
    )
    assert conn.getresponse().status == 421


def test_stalled_client_is_dropped(served, monkeypatch):
    import socket

    from prada import server

    monkeypatch.setattr(server._Handler, "timeout", 0.2)
    port = int(served.rsplit(":", 1)[1])
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        sock.sendall(
            b"POST /v1/request HTTP/1.1\r\nHost: 127.0.0.1:%d\r\n"
            b"Content-Type: application/json\r\nContent-Length: 100\r\n\r\n" % port
        )
        assert sock.recv(1024) == b""  # the server gives up and closes the connection


def test_tutor_history_is_capped(monkeypatch):
    monkeypatch.setenv("PRADA_FAKE_LLM", "1")
    monkeypatch.setattr("tutor.module.MAX_HISTORY", 4)
    n = Node([TutorModule()], db_path=":memory:")
    for text in ["un", "deux", "trois"]:
        n.handle(Request(text=text, session="s", options={"lang": "fr"}))
    history = n._sessions["s"]["tutor"]["history"]
    assert [m["content"] for m in history if m["role"] == "user"] == ["deux", "trois"]


@pytest.mark.parametrize(
    "text,lang",
    [
        ("Je suis allé a la plage hier.", "fr"),
        ("Il fait beau aujourd'hui.", "fr"),
        ("Elle est très content.", "fr"),
        ("café", "fr"),
        ("Я читаю книгу.", "ru"),
        ("私は本を読みます。", "ja"),
        ("What is the weather like today?", None),
        ("how do I install python", None),
        ("hello", None),
        ("123", None),
    ],
)
def test_detect_language(text, lang):
    assert detect_language(text) == lang


def test_english_goes_to_the_assistant_and_french_stays_with_the_tutor(monkeypatch):
    monkeypatch.setenv("PRADA_FAKE_LLM", "1")
    n = Node(db_path=":memory:")
    calls = []
    real = router.route
    monkeypatch.setattr(router, "route", lambda *a, **k: calls.append(a[0]) or real(*a, **k))
    send = lambda text: n.handle(Request(text=text, session="s"))  # noqa: E731
    assert send("What is the weather like?").module == "assistant"
    assert send("tutor Je lis un livre.").module == "tutor"
    assert send("Il fait beau aujourd'hui.").module == "tutor"
    assert calls == ["What is the weather like?", "tutor Je lis un livre."]  # none for the 3rd
    assert send("and what is the capital of Peru?").module == "assistant"


def test_french_text_overrides_the_session_language(monkeypatch):
    monkeypatch.setenv("PRADA_FAKE_LLM", "1")
    n = Node([TutorModule()], db_path=":memory:")
    n.handle(Request(text="Я читаю.", session="s"))
    assert (
        n.handle(Request(text="Je lis un livre.", session="s"))
        .data["reply_target"]
        .startswith("Qu'est")
    )


def test_tutor_prompt_lists_categories_only_where_defined():
    from tutor.languages import get_language
    from tutor.prompts import build_system_prompt

    assert "tense_mood:" in build_system_prompt(get_language("fr"))
    assert "counter:" in build_system_prompt(get_language("ja"))
    assert "Choose error_type" not in build_system_prompt(get_language("ru"))


def test_cli_health(served, capsys):
    port = served.rsplit(":", 1)[1]
    assert main(["health", "--port", port]) == 0
    assert "PRADA node up" in capsys.readouterr().out


def test_cli_health_no_node(capsys):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]  # closed again before the probe: nothing listens there
    assert main(["health", "--port", str(free)]) == 1
    assert "no node answering" in capsys.readouterr().err


def test_persona_is_shared_by_assistant_wake_and_tutor(monkeypatch):
    from prada import persona
    from tutor.languages import get_language
    from tutor.prompts import build_system_prompt

    seen = []

    def chat(messages, schema, model=None):
        seen.append(messages[0]["content"])
        return '{"reply": "hi"}'

    AssistantModule().handle(Request(text="hello"), Context(chat=chat))
    assert seen[0].startswith(persona.CHARACTER)

    monkeypatch.setattr("prada.wol.send_magic_packet", lambda *a: None)
    monkeypatch.setenv("PRADA_WAKE_MAC", "aa:bb:cc:dd:ee:ff")
    node = Node([WakeModule()])
    node.handle(Request(text="wake up", session="s"))
    reply = node.handle(Request(text="yes", session="s"))
    assert reply.text == persona.waking()

    prompt = build_system_prompt(get_language("fr"))
    assert persona.TONE in prompt and not re.search(r"\{\w+\}", prompt)


def test_remote_session_reports_http_errors(served):
    with pytest.raises(client.NodeError, match="refused the request.*404: not found"):
        client.RemoteSession(served + "/elsewhere").send("hi", "echo")


def test_request_from_a_client_that_hung_up_is_dropped(node, served):
    port = int(served.rsplit(":", 1)[1])
    body = b'{"text": "late", "module": "echo"}'
    with node._lock:  # the node is busy with someone else's request
        sock = socket.create_connection(("127.0.0.1", port))
        sock.sendall(
            b"POST /v1/request HTTP/1.1\r\nHost: 127.0.0.1:%d\r\n"
            b"Content-Type: application/json\r\nContent-Length: %d\r\n\r\n%s"
            % (port, len(body), body)
        )
        threading.Event().wait(0.3)  # the server has read the request and waits on the lock
        sock.close()
    threading.Event().wait(0.3)  # the server takes the lock, sees the hang-up, drops it
    assert node.handle(Request(text="next", module="echo")).error is None
    seen = node._dbs["echo"].execute("SELECT text FROM echo_seen").fetchall()
    assert [tuple(row) for row in seen] == [("next",)]


def test_node_skips_requests_nobody_waits_for(node):
    reply = node.handle(Request(text="x", module="echo"), gone=lambda: True)
    assert "went away" in reply.error and "echo" not in node._dbs
