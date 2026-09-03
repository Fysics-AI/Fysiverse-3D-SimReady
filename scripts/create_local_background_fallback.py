#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


def label_mask(path: Path, size: tuple[int, int], dilate: int) -> np.ndarray:
    mask = Image.open(path)
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    mask = mask.convert("L")
    if dilate > 0:
        kernel = max(3, int(dilate) * 2 + 1)
        mask = mask.filter(ImageFilter.MaxFilter(kernel))
    return np.asarray(mask) > 0


def diffuse_fill(image: np.ndarray, mask: np.ndarray, max_iter: int = 800) -> np.ndarray:
    filled = image.astype(np.float32).copy()
    known = ~mask
    active = mask.copy()
    if not known.any():
        return image.copy()

    for _ in range(max_iter):
        if not active.any():
            break
        accum = np.zeros_like(filled)
        count = np.zeros(mask.shape, dtype=np.float32)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            shifted_known = np.roll(known, shift=(dy, dx), axis=(0, 1))
            shifted_value = np.roll(filled, shift=(dy, dx), axis=(0, 1))
            if dy < 0:
                shifted_known[dy:, :] = False
            elif dy > 0:
                shifted_known[:dy, :] = False
            if dx < 0:
                shifted_known[:, dx:] = False
            elif dx > 0:
                shifted_known[:, :dx] = False
            use = active & shifted_known
            accum[use] += shifted_value[use]
            count[use] += 1.0
        update = active & (count > 0)
        if not update.any():
            break
        filled[update] = accum[update] / count[update, None]
        known[update] = True
        active[update] = False

    if active.any():
        mean_color = filled[known].mean(axis=0) if known.any() else np.array([128, 128, 128], dtype=np.float32)
        filled[active] = mean_color
    return np.clip(filled, 0, 255).astype(np.uint8)


def local_inpaint(image_path: Path, mask_path: Path, output_path: Path, dilate: int, radius: float) -> dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    size = image.size
    mask = label_mask(mask_path, size, dilate=dilate)
    rgb = np.asarray(image).copy()

    method = "diffuse_fill"
    try:
        import cv2  # type: ignore

        mask_u8 = (mask.astype(np.uint8) * 255)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        inpainted = cv2.inpaint(bgr, mask_u8, float(radius), cv2.INPAINT_TELEA)
        result = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
        method = "opencv_telea"
    except Exception:
        result = diffuse_fill(rgb, mask)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result).save(output_path)
    return {
        "status": "ok",
        "backend": "local_inpaint",
        "method": method,
        "input": str(image_path),
        "mask": str(mask_path),
        "output": str(output_path),
        "image_size": [int(size[0]), int(size[1])],
        "mask_pixels": int(mask.sum()),
        "mask_ratio": float(mask.mean()),
        "dilate": int(dilate),
        "radius": float(radius),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a local inpainted background fallback from an instance mask.")
    parser.add_argument("--session", type=Path, default=None)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--mask", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--dilate", type=int, default=9)
    parser.add_argument("--radius", type=float, default=5.0)
    return parser


def resolve(args: argparse.Namespace) -> tuple[Path, Path, Path, Path | None]:
    if args.session is not None:
        session = args.session.expanduser().resolve()
        input_path = args.input or session / "input" / "image.png"
        mask_path = args.mask or session / "input" / "mask_label.png"
        output_path = args.output or session / "input" / "background.png"
        report_path = args.report or session / "input" / "background_edit_report.json"
    else:
        if args.input is None or args.mask is None or args.output is None:
            raise ValueError("--input, --mask, and --output are required without --session")
        input_path = args.input
        mask_path = args.mask
        output_path = args.output
        report_path = args.report
    return input_path.expanduser().resolve(), mask_path.expanduser().resolve(), output_path.expanduser().resolve(), report_path.expanduser().resolve() if report_path else None


def main() -> int:
    args = build_parser().parse_args()
    try:
        input_path, mask_path, output_path, report_path = resolve(args)
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        payload = local_inpaint(input_path, mask_path, output_path, args.dilate, args.radius)
        if report_path is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    except Exception as exc:
        payload = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        try:
            _, _, _, report_path = resolve(args)
            if report_path is not None:
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
