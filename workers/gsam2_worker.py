#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
if str(WORKFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_ROOT))

from system_infer.gsam2_infer import GSAM2Infer
from workers.common_http import JsonWorker, run_json_server


class GSAM2Worker(JsonWorker):
    def __init__(self, gpu: int) -> None:
        self.gpu = gpu
        self.infer = GSAM2Infer(gpu=gpu)

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "worker": "gsam2", "gpu": self.gpu}

    def handle(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if path == "/segment":
            return self.infer.segment(payload)
        raise ValueError(f"Unsupported path: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GSAM2 JSON worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_json_server(GSAM2Worker(gpu=args.gpu), args.host, args.port)


if __name__ == "__main__":
    main()
