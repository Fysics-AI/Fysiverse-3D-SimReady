#!/usr/bin/env python3
from __future__ import annotations

import json
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit


class JsonWorker:
    """Base class for simple JSON HTTP workers."""

    def health(self) -> dict[str, Any]:
        return {"status": "ok"}

    def handle(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(f"Unhandled path: {path}")


def run_json_server(worker: JsonWorker, host: str, port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        server_version = "JsonWorkerHTTP/1.0"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, status_code: int, body: dict[str, Any]) -> None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_error(self, exc: BaseException) -> None:
            self._send_json(
                500,
                {
                    "status": "error",
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )

        def do_GET(self) -> None:
            try:
                path = urlsplit(self.path).path
                if path != "/health":
                    raise ValueError(f"Unsupported GET path: {path}")
                self._send_json(200, worker.health())
            except BaseException as exc:
                self._send_error(exc)

        def do_POST(self) -> None:
            try:
                path = urlsplit(self.path).path
                content_length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(content_length) if content_length else b""
                payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
                if not isinstance(payload, dict):
                    raise ValueError("POST body must be a JSON object")
                self._send_json(200, worker.handle(path, payload))
            except BaseException as exc:
                self._send_error(exc)

    server = ThreadingHTTPServer((host, port), Handler)
    server.serve_forever()
