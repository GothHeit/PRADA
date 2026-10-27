"""The home node's transport: HTTP + JSON on localhost, standard library only.

    GET  /health      -> {"status": "ok", "modules": [...]}
    POST /v1/request  -> body: Request JSON, response: Reply JSON

It binds 127.0.0.1 only. POSTs must be application/json, which a web page can't send to
another origin without a CORS preflight this server never answers, and the Host header must
name 127.0.0.1 or localhost, so a DNS-rebinding page can't pass as same-origin.
"""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from pydantic import ValidationError

from prada.module import Request
from prada.node import Node

HOST = "127.0.0.1"
MAX_BODY = 1 << 20


class _Handler(BaseHTTPRequestHandler):
    node: Node  # set by make_server
    timeout = 30  # seconds a client may stall before its thread is freed

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
        if self.path != "/v1/request":
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
            request = Request.model_validate_json(self.rfile.read(length))
        except ValidationError as e:
            return self._send(400, {"error": f"bad request: {e.error_count()} invalid field(s)"})
        self._send(200, self.node.handle(request).model_dump(mode="json"))

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
    with make_server(node, port) as server:
        print(f"PRADA node on http://{HOST}:{server.server_port} ({', '.join(node.modules)})")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print()
