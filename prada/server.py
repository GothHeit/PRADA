"""The home node's transport: HTTP + JSON on localhost, standard library only.

    GET  /health      -> {"status": "ok", "modules": [...]}
    POST /v1/request  -> body: Request JSON, response: Reply JSON
    POST /v1/stream   -> body: Request JSON, response: NDJSON lines, sent as they happen:
                         {"say": text, "kind": "part"|"filler"} ... then {"reply": Reply JSON}
    POST /v1/cut      -> body: Cut JSON (she was interrupted), response: {"cut": bool}

It binds 127.0.0.1 only. POSTs must be application/json, which a web page can't send to
another origin without a CORS preflight this server never answers, and the Host header must
name 127.0.0.1 or localhost, so a DNS-rebinding page can't pass as same-origin.
"""

import json
import select
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from pydantic import ValidationError

from prada import llm
from prada.module import Cut, Request
from prada.node import Node

HOST = "127.0.0.1"
MAX_BODY = 1 << 20


class _Handler(BaseHTTPRequestHandler):
    node: Node  # set by make_server
    timeout = 30  # seconds a client may stall before its thread is freed
    hung_up = False
    streaming = False  # a stream's headers are sent

    def _host_ok(self) -> bool:
        port = self.server.server_port
        if self.headers.get("Host") in (f"{HOST}:{port}", f"localhost:{port}"):
            return True
        self._send(421, {"error": "unexpected Host header"})
        return False

    def do_GET(self):
        if not self._host_ok():
            return
        if self.path != "/health":
            return self._send(404, {"error": "not found"})
        self._send(200, {"status": "ok", "modules": list(self.node.modules)})

    def do_POST(self):
        if not self._host_ok():
            return
        if self.path not in ("/v1/request", "/v1/stream", "/v1/cut"):
            return self._send(404, {"error": "not found"})
        if self.headers.get_content_type() != "application/json":
            return self._send(415, {"error": "expected application/json"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if not 0 <= length <= MAX_BODY:
            return self._send(413, {"error": "missing or oversized Content-Length"})
        try:
            body = self.rfile.read(length)
            if self.path == "/v1/cut":
                return self._send(200, {"cut": self.node.cut(Cut.model_validate_json(body))})
            request = Request.model_validate_json(body)
        except ValidationError as e:
            return self._send(400, {"error": f"bad request: {e.error_count()} invalid field(s)"})
        if self.path == "/v1/stream":
            reply = self.node.handle(request, self._client_gone, self._say)
            return self._line({"reply": reply.model_dump(mode="json")})
        reply = self.node.handle(request, self._client_gone)
        if not self.hung_up:
            self._send(200, reply.model_dump(mode="json"))

    def _say(self, kind: str, text: str) -> None:
        self._line({"say": text, "kind": kind})

    def _line(self, body: dict) -> None:
        """One NDJSON line of a stream, written at once; headers go before the first. The
        response has no length, so the connection closes after it."""
        if self.hung_up:
            return
        try:
            if not self.streaming:
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.end_headers()
                self.streaming = self.close_connection = True
            self.wfile.write(json.dumps(body, ensure_ascii=False).encode() + b"\n")
            self.wfile.flush()
        except OSError:
            self.hung_up = True  # the node finishes the request; nobody hears the rest

    def _client_gone(self) -> bool:
        """True if the client hung up while its request waited (it has nothing left to send,
        so a readable socket with no data means the connection was closed)."""
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            self.hung_up = bool(readable) and not self.connection.recv(1, socket.MSG_PEEK)
        except OSError:
            self.hung_up = True
        if self.hung_up:
            self.close_connection = True
        return self.hung_up

    def _send(self, status: int, body: dict) -> None:
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        pass


def make_server(node: Node, port: int) -> ThreadingHTTPServer:
    handler = type("Handler", (_Handler,), {"node": node})
    server = ThreadingHTTPServer((HOST, port), handler)
    server.daemon_threads = True
    return server


def serve(port: int, node: Node | None = None) -> None:
    node = node or Node()
    brain = "fake" if llm.fake_enabled() else llm.model_name()  # a bad [llm] table fails here
    with make_server(node, port) as server:
        modules = ", ".join(node.modules)
        print(f"PRADA node on http://{HOST}:{server.server_port} ({modules}; brain {brain})")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print()
