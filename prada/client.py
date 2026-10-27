"""Talk to PRADA: through the home node when it is running, in-process otherwise.

`connect()` returns a session whose `send(text, module=None, **options)` gives a Reply.
Under PRADA_FAKE_LLM or PRADA_NODE=local it always runs in-process, so tests and smoke
runs never reach a real node or model.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

from prada import config, llm
from prada.module import Reply, Request
from prada.server import HOST

# Never send localhost traffic through a configured HTTP proxy.
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class NodeError(Exception):
    """The node could not be reached or answered with an HTTP error."""


class LocalSession:
    def __init__(self, node=None):
        from prada.node import Node

        self.node = node or Node()
        self.id = uuid.uuid4().hex

    def send(self, text: str, module: str | None = None, **options: str) -> Reply:
        request = Request(text=text, module=module, session=self.id, options=options)
        return self.node.handle(request)


class RemoteSession:
    def __init__(self, url: str, timeout: float = 600):
        self.url = url
        self.timeout = timeout
        self.id = uuid.uuid4().hex

    def send(self, text: str, module: str | None = None, **options: str) -> Reply:
        request = Request(text=text, module=module, session=self.id, options=options)
        req = urllib.request.Request(
            f"{self.url}/v1/request",
            data=request.model_dump_json().encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with _opener.open(req, timeout=self.timeout) as response:
                return Reply.model_validate_json(response.read())
        except urllib.error.HTTPError as e:
            raise NodeError(
                f"the PRADA node at {self.url} refused the request ({_reason(e)})"
            ) from e
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise NodeError(f"lost the PRADA node at {self.url} ({e})") from e


def _reason(e: urllib.error.HTTPError) -> str:
    """The node's own error message for an HTTP error, else the status line."""
    try:
        return f"HTTP {e.code}: {json.load(e)['error']}"
    except (OSError, ValueError, KeyError, TypeError):
        return f"HTTP {e.code} {e.reason}"


def node_url(port: int | None = None) -> str:
    return f"http://{HOST}:{config.node_port(port)}"


def ping(url: str, timeout: float = 0.5) -> list[str] | None:
    """The node's module names, or None if no node answers at `url`."""
    try:
        with _opener.open(f"{url}/health", timeout=timeout) as response:
            return json.load(response)["modules"]
    except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError):
        return None


def wait_for_node(url: str, seconds: float = 0) -> list[str] | None:
    """Like `ping`, but keeps trying for up to `seconds` (a node that is still starting)."""
    deadline = time.monotonic() + seconds
    while (modules := ping(url)) is None and time.monotonic() < deadline:
        time.sleep(0.5)
    return modules


def connect(port: int | None = None) -> LocalSession | RemoteSession:
    if llm.fake_enabled() or os.environ.get("PRADA_NODE") == "local":
        return LocalSession()
    url = node_url(port)
    if ping(url) is not None:
        return RemoteSession(url)
    print(
        "prada: no node running (start one with `prada node`); working in-process", file=sys.stderr
    )
    return LocalSession()


def repl(session, module: str | None = None, **options: str) -> int:
    """Read lines from stdin, print PRADA's replies; returns an exit code."""
    while True:
        try:
            text = input("> ")
        except (KeyboardInterrupt, EOFError):
            print()
            return 0
        if not text.strip():
            continue
        try:
            reply = session.send(text, module, **options)
        except NodeError as e:
            print(f"prada: {e}", file=sys.stderr)
            return 1
        if reply.error:
            print(f"prada: {reply.error}", file=sys.stderr)
        else:
            print(reply.text)
