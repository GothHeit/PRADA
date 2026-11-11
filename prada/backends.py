"""Cloud model backends, for evals only: score a hosted model on the same cases as a local one.

PRADA herself never uses these. Chat, voice and the node always talk to Ollama; only
`prada eval --backend NAME` sends the eval cases (and the persona prompt) to a provider.
A backend is any OpenAI-compatible chat completions API (OpenAI, Gemini, OpenRouter, Groq,
Mistral, a local llama.cpp or LM Studio server, ...), set up in the config file:

    [backends.openai]
    url = "https://api.openai.com/v1"
    model = "gpt-5-mini"             # default model; --model overrides it
    api_key_env = "OPENAI_API_KEY"   # the environment variable that holds the key
    # api_key = "sk-..."             # or the key itself, in this local file only
    json = "schema"                  # "schema" (structured output) or "object" (JSON mode)
    # temperature = 0                # matches Ollama's; reasoning models reject it, so unset

Stdlib HTTP only (urllib), so no provider SDKs.
"""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlsplit

from prada import config

JSON_MODES = ("schema", "object")
TIMEOUT = 120  # seconds per request: a slow reasoning model can take a while
RETRIES = 3  # extra tries on a rate limit or a server error
MAX_WAIT = 30  # seconds, the longest pause between tries
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")


@dataclass(frozen=True)
class Backend:
    """One `[backends.NAME]` table."""

    name: str
    url: str  # base URL; requests go to <url>/chat/completions
    model: str = ""
    api_key: str = ""  # never printed
    json: str = "schema"
    temperature: float | None = None  # None = not sent (the provider's default)

    def __repr__(self) -> str:  # keep the key out of tracebacks and logs
        return f"Backend(name={self.name!r}, url={self.url!r}, model={self.model!r})"


_KEYS = {"url", "model", "api_key", "api_key_env", "json", "temperature"}


def names() -> list[str]:
    """The backends set up in the config file."""
    table = config.load_config().get("backends", {})
    return sorted(table) if isinstance(table, dict) else []


def settings(name: str) -> Backend:
    """The `[backends.NAME]` table, checked; raises ValueError with what to fix."""
    where = config.config_path()
    tables = config.load_config().get("backends", {})
    if not isinstance(tables, dict):
        raise ValueError(f"[backends] in {where} must be a table")
    if name not in tables:
        known = ", ".join(sorted(tables)) or "none"
        raise ValueError(f"no [backends.{name}] in {where} (set up: {known})")
    file = tables[name]
    if not isinstance(file, dict):
        raise ValueError(f"[backends.{name}] in {where} must be a table")
    if unknown := sorted(set(file) - _KEYS):
        raise ValueError(f"unknown [backends.{name}] setting(s) in {where}: {', '.join(unknown)}")
    url = file.get("url")
    if not isinstance(url, str) or not url.strip():
        raise ValueError(f"[backends.{name}] url in {where} must be the API's base URL")
    url = url.strip().rstrip("/")
    parts = urlsplit(url)
    if not (parts.scheme == "https" or (parts.scheme == "http" and parts.hostname in LOCAL_HOSTS)):
        raise ValueError(f"[backends.{name}] url must be https (http only for this computer)")
    model = file.get("model", "")
    if not isinstance(model, str):
        raise ValueError(f"[backends.{name}] model in {where} must be a model name")
    mode = file.get("json", "schema")
    if mode not in JSON_MODES:
        raise ValueError(f'[backends.{name}] json in {where} must be "schema" or "object"')
    temperature = file.get("temperature")
    if temperature is not None:
        if isinstance(temperature, bool) or not isinstance(temperature, int | float):
            raise ValueError(f"[backends.{name}] temperature in {where} must be a number")
        temperature = float(temperature)
    return Backend(name, url, model.strip(), _key(name, file), mode, temperature)


def _key(name: str, file: dict) -> str:
    if env := file.get("api_key_env"):
        if not isinstance(env, str):
            raise ValueError(f"[backends.{name}] api_key_env must be an environment variable name")
        if key := os.environ.get(env, "").strip():
            return key
        raise ValueError(f"[backends.{name}]: the environment variable {env} is not set")
    key = file.get("api_key", "")
    if not isinstance(key, str):
        raise ValueError(f"[backends.{name}] api_key must be a string")
    return key.strip()  # "" is fine for a local server without keys


def _schema_name(schema: dict) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(schema.get("title") or "reply"))[:64]


def _format(backend: Backend, schema: dict) -> dict:
    if backend.json == "object":
        return {"type": "json_object"}
    # strict mode would need every property required and no extra keys: pydantic's
    # schemas don't promise that, so the provider follows the schema without it.
    return {
        "type": "json_schema",
        "json_schema": {"name": _schema_name(schema), "schema": schema, "strict": False},
    }


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect would resend the key to another URL (and turn the POST into a GET)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # urllib then raises the 3xx as an HTTPError


_open = urllib.request.build_opener(_NoRedirect).open


def _wait(error: urllib.error.HTTPError, attempt: int) -> float:
    try:
        seconds = float(error.headers.get("Retry-After", ""))
    except (TypeError, ValueError):
        seconds = 2.0 * 2**attempt
    return min(max(seconds, 0.0), MAX_WAIT)


def _error(error: urllib.error.HTTPError) -> str:
    try:
        body = json.loads(error.read().decode("utf-8", "replace"))
        detail = body.get("error", body)
        detail = detail.get("message", detail) if isinstance(detail, dict) else detail
    except (ValueError, OSError, AttributeError):
        detail = error.reason
    return f"HTTP {error.code}: {str(detail)[:200]}"


def post(backend: Backend, payload: dict, opener=_open, sleep=time.sleep):
    """POSTs to <url>/chat/completions; retries rate limits and server errors."""
    headers = {"Content-Type": "application/json", "User-Agent": "prada-eval"}
    if backend.api_key:
        headers["Authorization"] = f"Bearer {backend.api_key}"
    data = json.dumps(payload).encode("utf-8")
    for attempt in range(RETRIES + 1):
        request = urllib.request.Request(
            f"{backend.url}/chat/completions", data=data, headers=headers, method="POST"
        )
        try:
            with opener(request, timeout=TIMEOUT) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if 300 <= e.code < 400:
                raise RuntimeError(f"{backend.name} redirected the request (check url)") from None
            if attempt < RETRIES and (e.code == 429 or e.code >= 500):
                sleep(_wait(e, attempt))
                continue
            raise RuntimeError(f"{backend.name} refused the request ({_error(e)})") from None
        except (urllib.error.URLError, TimeoutError) as e:
            reason = getattr(e, "reason", e)
            raise RuntimeError(f"could not reach {backend.name} ({reason})") from None
        except ValueError:
            raise RuntimeError(f"{backend.name} sent a reply that isn't JSON") from None
    raise AssertionError("unreachable")


def chat(backend: Backend, model: str, messages, schema, on_text=None, opener=None) -> str:
    """Same contract as `llm.chat`: the model's reply text (JSON for `schema`)."""
    if backend.json == "object":  # JSON mode alone doesn't say which fields: tell the model
        rule = f"Reply with one JSON object following this JSON schema: {json.dumps(schema)}"
        messages = [{"role": "system", "content": rule}, *messages]
    payload = {"model": model, "messages": messages, "response_format": _format(backend, schema)}
    if backend.temperature is not None:
        payload["temperature"] = backend.temperature
    reply = post(backend, payload, **({"opener": opener} if opener else {}))
    try:
        text = reply["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"{backend.name} sent no message: {str(reply)[:200]}") from None
    if on_text is not None:
        on_text(text)
    return text


def client(name: str, model: str | None = None):
    """`(model, factory)` for an eval: `factory(fake)` gives a chat function like
    `llm.client`'s, or the fake under PRADA_FAKE_LLM (then nothing is sent)."""
    from prada import llm

    backend = settings(name)
    model = model or backend.model
    if not model:
        raise ValueError(
            f"no model for backend {name}: pass --model or set [backends.{name}] model"
        )
    if llm.fake_enabled():
        return model, lambda fake: llm.client(fake)
    print(f"eval: sending the cases to {name} ({backend.url})", file=sys.stderr)
    remote = lambda messages, schema, on_text=None: chat(  # noqa: E731
        backend, model, messages, schema, on_text
    )
    return model, lambda fake: remote
