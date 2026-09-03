#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Iterable
from urllib import error, request

import numpy as np
from PIL import Image


WORKFLOW_ROOT = Path(__file__).resolve().parent
SESSIONS_ROOT = WORKFLOW_ROOT / "sessions"
BLENDER_BIN = os.environ.get("BLENDER_BIN", "blender")


def now_session_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_rgb_image(image: Any, path: Path) -> Path:
    ensure_dir(path.parent)
    if isinstance(image, Image.Image):
        pil = image.convert("RGB")
    elif isinstance(image, np.ndarray):
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        pil = Image.fromarray(arr, mode="RGB")
    else:
        pil = Image.open(image).convert("RGB")
    pil.save(path)
    return path


def mask_to_label_array(mask: Any, threshold: int = 10) -> np.ndarray:
    if mask is None:
        raise ValueError("mask is required")
    if isinstance(mask, np.ndarray):
        arr = mask
    else:
        arr = np.asarray(Image.open(mask))
    if arr.ndim == 2:
        if arr.dtype != np.uint8:
            arr = np.asarray(arr, dtype=np.int64)
        unique = np.unique(arr)
        if len(unique) > 2 or int(unique.max(initial=0)) > 1:
            return arr.astype(np.int32)
        return (arr > 0).astype(np.int32)
    if arr.ndim == 3 and arr.shape[2] == 4 and arr[..., 3].max() > 0:
        return (arr[..., 3] > threshold).astype(np.int32)
    rgb = arr[..., :3].astype(np.uint8)
    nonzero = rgb.max(axis=-1) > threshold
    colors = np.unique(rgb[nonzero].reshape(-1, 3), axis=0) if nonzero.any() else np.empty((0, 3), dtype=np.uint8)
    label = np.zeros(rgb.shape[:2], dtype=np.int32)
    for idx, color in enumerate(colors, start=1):
        if int(color.max()) <= threshold:
            continue
        label[(rgb == color).all(axis=-1)] = idx
    if label.max() == 0 and nonzero.any():
        label[nonzero] = 1
    return label


def save_label_mask(label: np.ndarray, path: Path) -> Path:
    ensure_dir(path.parent)
    label = np.asarray(label)
    max_label = int(label.max(initial=0))
    if max_label <= 255:
        Image.fromarray(label.astype(np.uint8), mode="L").save(path)
    else:
        Image.fromarray(label.astype(np.uint16), mode="I;16").save(path)
    return path


def palette(index: int) -> tuple[int, int, int]:
    hue = (index * 137.508) % 360.0
    c = 255.0
    x = c * (1.0 - abs((hue / 60.0) % 2.0 - 1.0))
    if hue < 60:
        rgb = (c, x, 0)
    elif hue < 120:
        rgb = (x, c, 0)
    elif hue < 180:
        rgb = (0, c, x)
    elif hue < 240:
        rgb = (0, x, c)
    elif hue < 300:
        rgb = (x, 0, c)
    else:
        rgb = (c, 0, x)
    return tuple(int(round(v)) for v in rgb)


def label_ids_by_area(label: np.ndarray, order: str = "index") -> list[int]:
    ids = [int(v) for v in np.unique(label) if int(v) > 0]
    if order == "largest":
        ids.sort(key=lambda v: int((label == v).sum()), reverse=True)
    elif order == "smallest":
        ids.sort(key=lambda v: int((label == v).sum()))
    else:
        ids.sort()
    return ids


def save_color_mask(label: np.ndarray, path: Path) -> Path:
    ensure_dir(path.parent)
    label = np.asarray(label)
    out = np.zeros((*label.shape, 3), dtype=np.uint8)
    for idx, label_id in enumerate(label_ids_by_area(label, order="index")):
        out[label == label_id] = np.asarray(palette(idx), dtype=np.uint8)
    Image.fromarray(out, mode="RGB").save(path)
    return path


def write_mask_crops_and_manifest(
    *,
    session_dir: Path,
    scene_id: str,
    image_path: Path,
    label_path: Path,
    manifest_path: Path,
    crop_dir: Path | None = None,
) -> Path:
    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image, dtype=np.uint8)
    label = mask_to_label_array(label_path)
    if label.shape != rgb.shape[:2]:
        raise ValueError(f"mask/image size mismatch: mask={label.shape[::-1]} image={image.size}")

    crop_root = ensure_dir(crop_dir or (session_dir / "segmentation" / "crops"))
    objects: list[dict[str, Any]] = []
    for object_index, label_id in enumerate(label_ids_by_area(label, order="index")):
        mask = label == label_id
        area = int(mask.sum())
        if area <= 0:
            continue
        ys, xs = np.where(mask)
        x1 = int(xs.min())
        y1 = int(ys.min())
        x2 = int(xs.max())
        y2 = int(ys.max())
        mask_name = f"mask_{int(label_id):03d}"
        crop_path = crop_root / f"{mask_name}_rgb_crop.png"
        Image.fromarray(rgb[y1 : y2 + 1, x1 : x2 + 1], mode="RGB").save(crop_path)
        objects.append(
            {
                "object_index": int(object_index),
                "sam3d_input_index": int(object_index),
                "mask_id": int(label_id),
                "mask_name": mask_name,
                "bbox_2d_xyxy": [x1, y1, x2, y2],
                "bbox_2d_xywh": [x1, y1, int(x2 - x1 + 1), int(y2 - y1 + 1)],
                "area_pixels": area,
                "crop_rgb": str(crop_path),
            }
        )

    existing: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    manifest = {
        **existing,
        "schema": "fysiverse_final_scene_manifest.v1",
        "status": existing.get("status") or "segmentation_prepared",
        "session_id": scene_id,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "inputs": {
            **(existing.get("inputs") or {}),
            "image": str(image_path),
            "label_mask": str(label_path),
            "session_dir": str(session_dir),
        },
        "segmentation": {
            **(existing.get("segmentation") or {}),
            "crop_dir": str(crop_root),
            "num_objects": len(objects),
        },
        "objects": objects,
    }
    ensure_dir(manifest_path.parent)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def write_instance_masks(
    image_path: Path,
    label_path: Path,
    out_dir: Path,
    *,
    rgba_objects: bool = True,
    mask_suffix: str = "",
    order: str = "index",
) -> list[dict[str, Any]]:
    ensure_dir(out_dir)
    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image, dtype=np.uint8)
    label = mask_to_label_array(label_path)
    if label.shape != rgb.shape[:2]:
        raise ValueError(f"mask/image size mismatch: mask={label.shape[::-1]} image={image.size}")
    objects: list[dict[str, Any]] = []
    for out_idx, label_id in enumerate(label_ids_by_area(label, order=order)):
        m = label == label_id
        area = int(m.sum())
        if area <= 0:
            continue
        ys, xs = np.where(m)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
        mask_name = f"{out_idx:03d}{mask_suffix}.png" if mask_suffix else f"{out_idx}.png"
        mask_path = out_dir / mask_name
        if rgba_objects:
            rgba = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
            rgba[..., :3] = np.where(m[..., None], rgb, 0)
            rgba[..., 3] = np.where(m, 255, 0).astype(np.uint8)
            Image.fromarray(rgba, mode="RGBA").save(mask_path)
        else:
            Image.fromarray(np.where(m, 255, 0).astype(np.uint8), mode="L").save(mask_path)
        objects.append({"index": out_idx, "label_id": label_id, "mask": str(mask_path), "area": area, "bbox": bbox})
    return objects


def prepare_scenegen_input(session_dir: Path, scene_id: str, image_path: Path, label_path: Path) -> Path:
    root = ensure_dir(session_dir / "scenegen_input" / "masked_images_test" / scene_id)
    Image.open(image_path).convert("RGB").save(root / "scene.jpg")
    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    label = mask_to_label_array(label_path)
    for out_idx, label_id in enumerate(label_ids_by_area(label, order="index")):
        m = label == label_id
        rgba = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
        rgba[..., :3] = np.where(m[..., None], rgb, 0)
        rgba[..., 3] = np.where(m, 255, 0).astype(np.uint8)
        Image.fromarray(rgba, mode="RGBA").save(root / f"{out_idx:03d}.png")
        Image.fromarray(np.where(m, 255, 0).astype(np.uint8), mode="L").save(root / f"{out_idx:03d}_mask.png")
    union = (label > 0).astype(np.uint8) * 255
    masked = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    masked[..., :3] = np.where(union[..., None] > 0, rgb, 0)
    masked[..., 3] = union
    Image.fromarray(masked, mode="RGBA").save(root / "masked_scene.png")
    return root


def prepare_sam3d_input(
    session_dir: Path,
    scene_id: str,
    image_path: Path,
    label_path: Path,
    manifest_path: Path | None = None,
    min_mask_area: int = 32,
    min_mask_bbox_size: int = 4,
) -> Path:
    root = ensure_dir(session_dir / "sam3d_input" / scene_id)
    shutil.copy2(image_path, root / "image.png")
    for stale_mask in root.glob("*.png"):
        if stale_mask.stem.isdigit():
            stale_mask.unlink(missing_ok=True)
    manifest_objects: list[dict[str, Any]] = []
    if manifest_path is not None and manifest_path.is_file():
        try:
            manifest_objects = list(json.loads(manifest_path.read_text(encoding="utf-8")).get("objects") or [])
        except Exception:
            manifest_objects = []

    if not manifest_objects:
        seed_manifest_path = manifest_path or (root / "mask_manifest.json")
        write_mask_crops_and_manifest(
            session_dir=session_dir,
            scene_id=scene_id,
            image_path=image_path,
            label_path=label_path,
            manifest_path=seed_manifest_path,
        )
        manifest_objects = list(json.loads(seed_manifest_path.read_text(encoding="utf-8")).get("objects") or [])

    label = mask_to_label_array(label_path)
    objects: list[dict[str, Any]] = []
    skipped_objects: list[dict[str, Any]] = []
    for src_obj in sorted(manifest_objects, key=lambda item: int(item.get("sam3d_input_index", item.get("object_index", 0)))):
        fallback_label_id = int(src_obj.get("object_index", len(objects) + len(skipped_objects))) + 1
        label_id = int(src_obj.get("mask_id", fallback_label_id))
        m = label == label_id
        area = int(m.sum())
        if area <= 0:
            skipped = dict(src_obj)
            skipped.update({"mask_id": int(label_id), "skip_reason": "empty_mask", "area_pixels": area})
            skipped_objects.append(skipped)
            continue
        ys, xs = np.where(m)
        bbox_width = int(xs.max() - xs.min() + 1)
        bbox_height = int(ys.max() - ys.min() + 1)
        if area < int(min_mask_area) or min(bbox_width, bbox_height) < int(min_mask_bbox_size):
            skipped = dict(src_obj)
            skipped.update(
                {
                    "mask_id": int(label_id),
                    "skip_reason": "too_small_for_sam3d",
                    "area_pixels": area,
                    "bbox_2d_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                    "bbox_2d_xywh": [int(xs.min()), int(ys.min()), bbox_width, bbox_height],
                    "min_mask_area": int(min_mask_area),
                    "min_mask_bbox_size": int(min_mask_bbox_size),
                }
            )
            skipped_objects.append(skipped)
            continue
        out_idx = len(objects)
        mask_path = root / f"{out_idx}.png"
        Image.fromarray(np.where(m, 255, 0).astype(np.uint8), mode="L").save(mask_path)
        obj = dict(src_obj)
        obj.update(
            {
                "object_index": int(src_obj.get("object_index", out_idx)),
                "sam3d_input_index": int(out_idx),
                "mask_id": int(label_id),
                "mask_name": src_obj.get("mask_name") or f"mask_{int(label_id):03d}",
                "sam3d_input_mask": str(mask_path),
            }
        )
        objects.append(
            obj
        )
    if not objects:
        raise ValueError(
            f"No valid SAM3D masks after filtering empty/tiny masks "
            f"(min_mask_area={min_mask_area}, min_mask_bbox_size={min_mask_bbox_size})"
        )
    manifest = {
        "schema": "fysiverse_final_scene_manifest.v1",
        "status": "sam3d_input_prepared",
        "session_id": scene_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "inputs": {
            "image": str(image_path),
            "label_mask": str(label_path),
            "sam3d_input_dir": str(root),
        },
        "objects": objects,
        "skipped_objects": skipped_objects,
        "sam3d_input_filter": {
            "min_mask_area": int(min_mask_area),
            "min_mask_bbox_size": int(min_mask_bbox_size),
            "kept": len(objects),
            "skipped": len(skipped_objects),
        },
    }
    input_manifest = root / "mask_manifest.json"
    manifest["manifest_path"] = str(input_manifest)
    input_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if manifest_path is not None and manifest_path != input_manifest:
        ensure_dir(manifest_path.parent)
        manifest["manifest_path"] = str(manifest_path)
        manifest["internal_mask_manifest"] = str(input_manifest)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return root


def http_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 3600) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST" if payload is not None else "GET",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        message = f"HTTP {exc.code} from {url}: {body}"
        try:
            parsed = json.loads(body)
            message = json.dumps(parsed, ensure_ascii=False, indent=2)
        except Exception:
            pass
        raise RuntimeError(message) from exc


def find_free_port(start: int, host: str = "127.0.0.1", count: int = 200) -> int:
    for port in range(start, start + count):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"no free port from {start} to {start + count - 1}")


def glb_to_blend(glb_path: Path, blend_path: Path, blender_bin: str = BLENDER_BIN) -> Path:
    if not glb_path.is_file():
        raise FileNotFoundError(glb_path)
    ensure_dir(blend_path.parent)
    script = (
        "import bpy, sys\n"
        "bpy.ops.object.select_all(action='SELECT')\n"
        "bpy.ops.object.delete()\n"
        f"bpy.ops.import_scene.gltf(filepath={str(glb_path)!r})\n"
        f"bpy.ops.wm.save_as_mainfile(filepath={str(blend_path)!r})\n"
    )
    subprocess.run([blender_bin, "-b", "--python-expr", script], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return blend_path


def collect_existing(paths: Iterable[str | Path | None]) -> list[str]:
    out = []
    for path in paths:
        if not path:
            continue
        p = Path(path)
        if p.exists():
            out.append(str(p))
    return out
