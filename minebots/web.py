"""Minimal web server exposing the reasoning agent."""
from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Tuple

from .agent import MiniAgent

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_INDEX_HTML = (_TEMPLATE_DIR / "index.html").read_bytes()
_CONTENT_TYPE_HTML = "text/html; charset=utf-8"
_CONTENT_TYPE_JSON = "application/json; charset=utf-8"


class _RequestHandler(BaseHTTPRequestHandler):
    agent = MiniAgent()

    def do_GET(self) -> None:  # noqa: N802 (http-server naming convention)
        if self.path in {"/", "/index.html"}:
            self._send_response(HTTPStatus.OK, _CONTENT_TYPE_HTML, _INDEX_HTML)
        else:
            self._send_response(HTTPStatus.NOT_FOUND, _CONTENT_TYPE_HTML, b"<h1>404 Not Found</h1>")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/chat":
            self._send_response(HTTPStatus.NOT_FOUND, _CONTENT_TYPE_JSON, b"{}")
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw_payload = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw_payload.decode("utf-8")) if raw_payload else {}
        except json.JSONDecodeError:
            self._send_response(
                HTTPStatus.BAD_REQUEST,
                _CONTENT_TYPE_JSON,
                json.dumps({"error": "invalid json"}).encode("utf-8"),
            )
            return

        message = str(payload.get("message", "")).strip()
        response = self.agent.respond(message)

        data = json.dumps(
            {
                "user_message": response.user_message,
                "reasoning": response.reasoning,
                "final_answer": response.final_answer,
                "history": response.final_answer and self.agent.export_history(),
            }
        ).encode("utf-8")
        self._send_response(HTTPStatus.OK, _CONTENT_TYPE_JSON, data)

    def log_message(self, format: str, *args) -> None:  # noqa: A003 - keep signature
        """Silence default stdout logging to keep the console clean."""

        return

    def _send_response(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def create_server(host: str = "127.0.0.1", port: int = 0) -> Tuple[ThreadingHTTPServer, Thread]:
    """Create and start the HTTP server in a background thread.

    The returned server is already serving requests; callers should remember to
    invoke :meth:`shutdown` and :meth:`server_close` when finished.
    """

    server = ThreadingHTTPServer((host, port), _RequestHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the development server until interrupted."""

    server, thread = create_server(host, port)
    try:
        thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Minebots demo server")
    parser.add_argument("--host", default="127.0.0.1", help="Address to bind (default: 127.0.0.1)")
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind (default: 8000)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args.host, args.port)
