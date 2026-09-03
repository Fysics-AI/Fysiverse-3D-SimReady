#!/usr/bin/env python3
"""Static preview server for the Fysiverse 3DGS checker."""

from __future__ import annotations

import argparse
import mimetypes
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class PreviewHandler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".html": "text/html; charset=utf-8",
        ".js": "text/javascript; charset=utf-8",
        ".mjs": "text/javascript; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".wasm": "application/wasm",
        ".glb": "model/gltf-binary",
        ".gltf": "model/gltf+json",
        ".ksplat": "application/octet-stream",
        ".splat": "application/octet-stream",
        ".ply": "application/octet-stream",
    }

    def guess_type(self, path: str) -> str:
        # SimpleHTTPRequestHandler translates URLs before this call, but keep
        # the MIME lookup query-safe so /page.html?manifest=...json stays HTML.
        clean_path = path.split("?", 1)[0].split("#", 1)[0]
        suffix = Path(clean_path).suffix.lower()
        if suffix in self.extensions_map:
            return self.extensions_map[suffix]
        return mimetypes.guess_type(clean_path)[0] or "application/octet-stream"

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()

    handler = lambda *a, **kw: PreviewHandler(*a, directory=args.directory, **kw)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving {args.directory} at http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
