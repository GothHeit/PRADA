"""Talk to PRADA: through the home node when it is running, in-process otherwise.

`connect()` returns a session whose `send(text, module=None, speak=None, **options)` gives a
Reply; `speak(kind, text)` hears what is said before it is done (see prada.node.Node.handle).
`cut(text, heard)` tells the node a reply was interrupted after `heard` (see Node.cut).
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
from prada.module import Cut, Reply, Request
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

    def send(self, text: str, module: str | None = None, speak=None, **options: str) -> Reply:
        request = Request(text=text, module=module, session=self.id, options=options)
        return self.node.handle(request, speak=speak)

    def cut(self, text: str, heard: str) -> bool:
        return self.node.cut(Cut(text=text, heard=heard))


class RemoteSession:
    def __init__(self, url: str, timeout: float = 600):
        self.url = url
        self.timeout = timeout
        self.id = uuid.uuid4().hex

    def send(self, text: str, module: str | None = None, speak=None, **options: str) -> Reply:
        request = Request(text=text, module=module, session=self.id, options=options)
        req = urllib.request.Request(
            f"{self.url}/v1/{'request' if speak is None else 'stream'}",
            data=request.model_dump_json().encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with _opener.open(req, timeout=self.timeout) as response:
                if speak is None:
                    return Reply.model_validate_json(response.read())
                for line in response:
                    event = json.loads(line)
                    if "reply" in event:
                        return Reply.model_validate(event["reply"])
                    speak(event["kind"], event["say"])
                raise ValueError("the stream ended without a reply")
        except urllib.error.HTTPError as e:
            raise NodeError(
                f"the PRADA node at {self.url} refused the request ({_reason(e)})"
            ) from e
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise NodeError(f"lost the PRADA node at {self.url} ({e})") from e

    def cut(self, text: str, heard: str) -> bool:
        req = urllib.request.Request(
            f"{self.url}/v1/cut",
            data=Cut(text=text, heard=heard).model_dump_json().encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with _opener.open(req, timeout=self.timeout) as response:
                return bool(json.load(response)["cut"])
        except urllib.error.HTTPError as e:
            raise NodeError(
                f"the PRADA node at {self.url} refused the request ({_reason(e)})"
            ) from e
        except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError) as e:
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
            for line in sources(reply):
                print(line)


def sources(reply: Reply) -> list[str]:
    """What a web search sent out and the sources her reply used, for the chat screen."""
    lines = []
    for o in reply.outcomes:
        if o.ok and "sources" in o.data:
            searched = ", ".join(o.data.get("searched", []))
            lines.append(f"  searched {searched} for: {o.data.get('query', '')}")
            lines += [f"  [{s['id']}] {s['title']}: {s['url']}" for s in o.data["sources"]]
    return lines
