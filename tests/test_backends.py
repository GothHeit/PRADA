import io
import json
import urllib.error

import pytest

from prada import backends
from prada import cli as prada_cli
from prada import eval as assistant_eval
from tutor import eval as tutor_eval

OPENAI = '[backends.openai]\nurl = "https://api.example.com/v1/"\nmodel = "m1"\n'


def write_config(tmp_path, text):
    (tmp_path / "config.toml").write_text(text, encoding="utf-8")


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def opener(replies, sent):
    """Answers each request with the next reply: a dict (200) or an int (HTTP error)."""

    def open_(request, timeout):
        sent.append(request)
        reply = replies.pop(0)
        if isinstance(reply, int):
            body = io.BytesIO(json.dumps({"error": {"message": "slow down"}}).encode())
            raise urllib.error.HTTPError(request.full_url, reply, "err", {}, body)
        return FakeResponse(json.dumps(reply).encode())

    return open_


def answer(text):
    return {"choices": [{"message": {"content": text}}]}


def test_settings_key_from_env_and_url_trimmed(monkeypatch, tmp_path):
    write_config(tmp_path, OPENAI + 'api_key_env = "MY_KEY"\n')
    monkeypatch.setenv("MY_KEY", "sk-secret")
    b = backends.settings("openai")
    assert (b.url, b.model, b.api_key, b.json) == (
        "https://api.example.com/v1",
        "m1",
        "sk-secret",
        "schema",
    )
    assert "sk-secret" not in repr(b)
    assert backends.names() == ["openai"]


@pytest.mark.parametrize(
    "text, error",
    [
        ("", "no \\[backends.openai\\]"),
        ('[backends.openai]\nurl = "http://api.example.com"\n', "https"),
        (OPENAI + 'api_key_env = "NOPE_KEY"\n', "NOPE_KEY is not set"),
        (OPENAI + 'json = "yaml"\n', "schema"),
        (OPENAI + "temprature = 1\n", "unknown"),
        ('[backends.openai]\nmodel = "m"\n', "base URL"),
    ],
)
def test_bad_settings(tmp_path, text, error):
    write_config(tmp_path, text)
    with pytest.raises(ValueError, match=error):
        backends.settings("openai")


def test_local_http_server_needs_no_key(tmp_path):
    write_config(tmp_path, '[backends.local]\nurl = "http://127.0.0.1:8080/v1"\n')
    assert backends.settings("local").api_key == ""


def test_chat_sends_schema_and_key(tmp_path):
    b = backends.Backend("openai", "https://api.example.com/v1", "m1", "sk-x")
    sent, pieces = [], []
    schema = {"title": "Answer", "type": "object", "required": ["reply"]}
    text = backends.chat(
        b, "m1", [{"role": "user", "content": "hi"}], schema, pieces.append,
        opener=opener([answer('{"reply": "hello"}')], sent),
    )  # fmt: skip
    assert text == '{"reply": "hello"}' and pieces == [text]
    request = sent[0]
    assert request.full_url == "https://api.example.com/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer sk-x"
    body = json.loads(request.data)
    assert body["model"] == "m1" and "temperature" not in body  # reasoning models reject it
    assert body["response_format"]["json_schema"]["schema"] == schema
    assert body["response_format"]["json_schema"]["name"] == "Answer"


def test_json_object_mode_puts_the_schema_in_the_prompt():
    b = backends.Backend("x", "https://x", json="object", temperature=0.0)
    sent = []
    schema = {"title": "Answer", "properties": {"reply": {"type": "string"}}}
    backends.chat(
        b, "m", [{"role": "user", "content": "hi"}], schema, opener=opener([answer("{}")], sent)
    )
    body = json.loads(sent[0].data)
    assert body["response_format"] == {"type": "json_object"} and body["temperature"] == 0
    first = body["messages"][0]
    assert first["role"] == "system" and json.dumps(schema) in first["content"]
    assert "JSON" in first["content"] and body["messages"][1]["content"] == "hi"


def test_temperature_setting(tmp_path):
    write_config(tmp_path, OPENAI)
    assert backends.settings("openai").temperature is None
    write_config(tmp_path, OPENAI + "temperature = 0\n")
    assert backends.settings("openai").temperature == 0.0
    write_config(tmp_path, OPENAI + 'temperature = "low"\n')
    with pytest.raises(ValueError, match="temperature"):
        backends.settings("openai")


def test_redirects_are_refused_and_the_key_stays_home():
    """A redirect would resend the key elsewhere (and the POST as a GET): never follow one."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(("POST", self.path, self.headers.get("Authorization")))
            self.send_response(302)
            self.send_header("Location", "/steal")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            seen.append(("GET", self.path, self.headers.get("Authorization")))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/v1"
        b = backends.Backend("local", url, api_key="sk-secret")
        with pytest.raises(RuntimeError, match="redirected the request \\(check url\\)"):
            backends.post(b, {})
    finally:
        server.shutdown()
        server.server_close()
    assert [s[:2] for s in seen] == [("POST", "/v1/chat/completions")]


def test_rate_limits_are_retried_then_reported(monkeypatch):
    b = backends.Backend("openai", "https://api.example.com/v1")
    waits, sent = [], []
    reply = backends.post(b, {}, opener=opener([429, 503, answer("{}")], sent), sleep=waits.append)
    assert reply == answer("{}") and len(sent) == 3 and len(waits) == 2
    with pytest.raises(RuntimeError, match="HTTP 401: slow down"):
        backends.post(b, {}, opener=opener([401], []), sleep=waits.append)
    with pytest.raises(RuntimeError, match="HTTP 429"):
        backends.post(b, {}, opener=opener([429] * 4, []), sleep=lambda s: None)


def test_unreachable_is_one_line():
    def down(request, timeout):
        raise urllib.error.URLError("no route")

    b = backends.Backend("openai", "https://api.example.com/v1")
    with pytest.raises(RuntimeError, match="could not reach openai"):
        backends.post(b, {}, opener=down)


def test_assistant_eval_with_a_backend(monkeypatch, tmp_path, capsys):
    """Every model call goes to the backend; the run says which one, in the report and files."""
    monkeypatch.delenv("PRADA_FAKE_LLM", raising=False)
    write_config(tmp_path, OPENAI)
    case = assistant_eval.Case("routing", "how are you today?", module="assistant")
    monkeypatch.setattr(assistant_eval, "load_cases", lambda: [case])
    calls = []

    def chat(backend, model, messages, schema, on_text=None, opener=None):
        calls.append((backend.name, model, schema.get("title")))
        return json.dumps({"module": "assistant", "reply": "Doing well!", "spoken": "Doing well!"})

    monkeypatch.setattr(backends, "chat", chat)
    out = tmp_path / "run.md"
    argv = ["eval", "--suite", "assistant", "--backend", "openai", "--model", "m2"]
    assert prada_cli.main(argv + ["--transcript", str(out)]) == 0
    printed = capsys.readouterr()
    assert "model: openai m2" in printed.out and "sending the cases to openai" in printed.err
    assert calls and {c[:2] for c in calls} == {("openai", "m2")}
    assert "backend: openai" in out.read_text(encoding="utf-8")
    data = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert (data["backend"], data["model"], data["think"]) == ("openai", "m2", None)
    assert data["cases"][0]["text"] == case.text


def test_backend_under_fake_llm_sends_nothing(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PRADA_FAKE_LLM", "1")
    write_config(tmp_path, OPENAI)
    monkeypatch.setattr(assistant_eval, "load_cases", lambda: [])
    monkeypatch.setattr(backends, "post", lambda *a, **k: pytest.fail("sent"))
    argv = ["eval", "--suite", "assistant", "--backend", "openai"]
    assert prada_cli.main(argv + ["--transcript", str(tmp_path / "t.md")]) == 0
    assert "model: openai m1 (fake: nothing sent)" in capsys.readouterr().out
    assert json.loads((tmp_path / "t.json").read_text())["backend"] == "fake"


def test_backend_errors_are_one_line(tmp_path, capsys):
    assert prada_cli.main(["eval", "--suite", "assistant", "--backend", "nope"]) == 1
    assert "no [backends.nope]" in capsys.readouterr().err
    write_config(tmp_path, OPENAI)
    assert prada_cli.main(["eval", "--backend", "openai", "--think", "on"]) == 1
    assert "--think only applies to Ollama" in capsys.readouterr().err
    write_config(tmp_path, '[backends.openai]\nurl = "https://x"\n')
    assert prada_cli.main(["eval", "--backend", "openai"]) == 1
    assert "pass --model" in capsys.readouterr().err


def test_tutor_eval_with_a_backend(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PRADA_FAKE_LLM", "1")
    write_config(tmp_path, OPENAI)
    assert tutor_eval.main(["--lang", "fr", "--backend", "openai"]) == 0
    printed = capsys.readouterr().out
    assert "model: openai m1 (fake: nothing sent)" in printed and "[fr]" in printed


def test_runtime_never_uses_a_backend(monkeypatch, tmp_path):
    """A configured backend changes nothing outside evals: the brain stays on Ollama."""
    from prada import llm

    monkeypatch.delenv("PRADA_FAKE_LLM", raising=False)
    write_config(tmp_path, OPENAI + 'api_key = "sk-x"\n')
    monkeypatch.setattr(backends, "post", lambda *a, **k: pytest.fail("sent to the cloud"))
    seen = []
    monkeypatch.setattr(llm, "chat", lambda messages, schema, **kw: seen.append(kw) or "{}")
    llm.client(lambda **kw: "{}")([{"role": "user", "content": "hi"}], {})
    assert seen and seen[0]["model"] == "gemma4:12b"


def run_data(backend, model, failures):
    cases = [
        {"category": "routing", "text": t, "module": "x", "failure": f, "seconds": 1.0}
        for t, f in failures.items()
    ]
    passed = sum(not f for f in failures.values())
    return {
        "suite": "assistant", "backend": backend, "model": model, "think": None,
        "categories": {"routing": [passed, len(cases)]}, "false_claims": 0,
        "invalid": 0, "calls": 4, "seconds_per_turn": 1.5, "cases": cases,
    }  # fmt: skip


def test_compare_runs_side_by_side(tmp_path, capsys):
    a = run_data("ollama", "gemma4:12b", {"one": "", "two": "went to web, not notes"})
    b = run_data("openai", "m1", {"one": "", "two": ""})
    paths = []
    for name, data in (("a", a), ("b", b)):
        paths.append(tmp_path / f"{name}.json")
        paths[-1].write_text(json.dumps(data), encoding="utf-8")
    assert prada_cli.main(["eval", "--compare", *map(str, paths)]) == 0
    out = capsys.readouterr().out
    assert "ollama: gemma4:12b" in out and "openai: m1" in out
    assert "1/2 (50%)" in out and "2/2 (100%)" in out
    assert "'two'" in out and "FAIL: went to web" in out and "'one'" not in out


def test_compare_rejects_other_files(tmp_path, capsys):
    bad = tmp_path / "x.json"
    bad.write_text("[]", encoding="utf-8")
    assert assistant_eval.compare([str(bad)]) == 1
    assert "x.json" in capsys.readouterr().err
    partial = tmp_path / "y.json"
    partial.write_text(json.dumps({"suite": "assistant", "backend": "b", "model": "m"}))
    assert assistant_eval.compare([str(partial)]) == 1
    assert "not an assistant eval results file" in capsys.readouterr().err


def test_results_file_next_to_a_json_transcript():
    from pathlib import Path

    assert assistant_eval.results_path(Path("a/run.md")) == Path("a/run.json")
    assert assistant_eval.results_path(Path("a/run.json")) == Path("a/run.results.json")
