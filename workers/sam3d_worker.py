#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
if str(WORKFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_ROOT))

from system_infer.sam3d_infer import SAM3DInfer
from workers.common_http import JsonWorker, run_json_server


class SAM3DWorker(JsonWorker):
    def __init__(self, args: argparse.Namespace) -> None:
        self.gpu = args.gpu
        self.infer = SAM3DInfer(config=args.config, compile=args.compile, gpu=args.gpu)

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "worker": "sam3d", "gpu": self.gpu}

    def handle(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if path == "/generate":
            return self.infer.generate(payload)
        raise ValueError(f"Unsupported path: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM3D JSON worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18085)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument(
        "--config",
        default=os.environ.get("SAM3D_CONFIG", str(Path(__file__).resolve().parents[1] / "third_party" / "SAM3D" / "runtime_configs" / "pipeline_hf_cache_moge_modelpt.yaml")),
    )
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_json_server(SAM3DWorker(args), args.host, args.port)


if __name__ == "__main__":
    main()
