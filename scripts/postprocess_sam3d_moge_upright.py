#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import trimesh


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
MOGE_ROOT = Path(os.environ.get("MOGE_ROOT", str(WORKFLOW_ROOT / "third_party" / "MoGe")))
MOGE_CKPT = Path(os.environ.get("MOGE_CKPT", str(WORKFLOW_ROOT / "checkpoints" / "moge-2-vitl-normal" / "model.pt")))
BLENDER_BIN = os.environ.get("BLENDER_BIN", "blender")
AXIS_TO_VECTOR = {
    "x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    "-x": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    "-y": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "-z": np.array([0.0, 0.0, -1.0], dtype=np.float64),
}


def timing_start() -> float:
    return time.monotonic()


def timing_elapsed(start: float) -> dict[str, float]:
    return {"duration_s": round(max(0.0, time.monotonic() - float(start)), 6)}
SCENE_TO_BLENDER_TRANSFORMS = {
    "identity": np.eye(3, dtype=np.float64),
    # SAM3D writes PyTorch3D scene coordinates directly into GLB. Blender's
    # glTF importer converts GLB Y-up coordinates to Blender Z-up coordinates:
    #   (x, y, z)_scene -> (x, -z, y)_blender
    "pytorch3d_y_up_to_blender_z_up": np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    ),
}
OPENCV_CAMERA_TO_PYTORCH3D_SCENE = np.diag([-1.0, -1.0, 1.0]).astype(np.float64)

if str(WORKFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_ROOT))
if str(MOGE_ROOT) not in sys.path:
    sys.path.insert(0, str(MOGE_ROOT))

try:
    from pipeline_utils import glb_to_blend
except Exception:  # pragma: no cover
    glb_to_blend = None


def normalize(vec: np.ndarray, name: str = "vector") -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm < 1e-10:
        raise ValueError(f"{name} must be a non-zero finite 3D vector")
    return arr / norm


def axis_vector(axis: str) -> np.ndarray:
    key = str(axis).strip().lower()
    if key not in AXIS_TO_VECTOR:
        raise ValueError(f"unsupported axis {axis!r}; expected one of {sorted(AXIS_TO_VECTOR)}")
    return AXIS_TO_VECTOR[key].copy()


def scene_to_blender_matrix(name: str) -> np.ndarray:
    key = str(name).strip().lower()
    if key not in SCENE_TO_BLENDER_TRANSFORMS:
        raise ValueError(f"unsupported scene-to-blender transform {name!r}; expected one of {sorted(SCENE_TO_BLENDER_TRANSFORMS)}")
    return SCENE_TO_BLENDER_TRANSFORMS[key].copy()


def transform_direction(matrix: np.ndarray, vector: np.ndarray, *, name: str) -> np.ndarray:
    return normalize(np.asarray(matrix, dtype=np.float64).reshape(3, 3) @ normalize(vector, name), name)


def orient_towards(vector: np.ndarray, reference: np.ndarray, *, name: str) -> np.ndarray:
    out = normalize(vector, name)
    ref = normalize(reference, "reference")
    if float(np.dot(out, ref)) < 0.0:
        out = -out
    return out


def up_axis_index(up_vector: np.ndarray) -> int:
    up = normalize(up_vector, "up vector")
    return int(np.argmax(np.abs(up)))


def horizontal_axis_indices(up_vector: np.ndarray) -> list[int]:
    vertical = up_axis_index(up_vector)
    return [idx for idx in range(3) if idx != vertical]


def numpy_to_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return numpy_to_json(value.tolist())
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): numpy_to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [numpy_to_json(v) for v in value]
    return value


def unique_base(output_dir: Path, stem: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(10000):
        suffix = "" if idx == 0 else f"_{idx:03d}"
        base = output_dir / f"{stem}{suffix}"
        if not base.with_suffix(".blend").exists() and not Path(f"{base}_report.json").exists():
            return base
    raise RuntimeError(f"could not reserve output path in {output_dir}")


def unique_report_path(output_dir: Path, stem: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(10000):
        suffix = "" if idx == 0 else f"_{idx:03d}"
        path = output_dir / f"{stem}{suffix}_report.json"
        if not path.exists():
            return path
    raise RuntimeError(f"could not reserve report path in {output_dir}")


def resize_image(image: np.ndarray, resize_max: int | None) -> tuple[np.ndarray, float]:
    if resize_max is None or resize_max <= 0:
        return image, 1.0
    h, w = image.shape[:2]
    scale = min(1.0, float(resize_max) / float(max(h, w)))
    if scale >= 1.0:
        return image, 1.0
    out = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return out, scale


def load_mask(mask_path: Path | None, size: tuple[int, int]) -> np.ndarray | None:
    if mask_path is None or not mask_path.is_file():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask[..., 0]
    h, w = size
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def as_uint8_mask(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask).astype(bool).astype(np.uint8) * 255)


def normalize_to_uint8(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    if valid is not None:
        finite &= np.asarray(valid).astype(bool)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(arr[finite], [2.0, 98.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(arr[finite]))
        hi = float(np.max(arr[finite]))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    out[~finite] = 0.0
    return (out * 255.0).astype(np.uint8)


def normal_to_rgb(normal: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(normal, dtype=np.float32)
    rgb = np.clip((arr + 1.0) * 0.5, 0.0, 1.0)
    if valid is not None:
        rgb[~np.asarray(valid).astype(bool)] = 0.0
    return (rgb * 255.0).astype(np.uint8)


def load_debug_image(image_path: Path, moge: dict[str, Any], size: tuple[int, int]) -> np.ndarray:
    image = moge.get("image")
    if image is not None:
        arr = np.asarray(image)
        if arr.ndim == 3:
            if arr.shape[:2] != size:
                arr = cv2.resize(arr, (size[1], size[0]), interpolation=cv2.INTER_AREA)
            return arr[..., :3].astype(np.uint8)
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return np.zeros((size[0], size[1], 3), dtype=np.uint8)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != size:
        rgb = cv2.resize(rgb, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    return rgb


def write_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2BGR))


def draw_arrow_panel(
    *,
    path: Path,
    vectors: dict[str, np.ndarray],
    title: str,
    size: int = 720,
) -> None:
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    center = np.array([size // 2, size // 2], dtype=np.float64)
    scale = size * 0.34
    colors = {
        "target_up": (35, 160, 70),
        "source_up_blender": (220, 60, 45),
        "raw_source_up_blender": (240, 150, 30),
        "source_up_scene": (40, 95, 220),
        "raw_source_up_scene": (120, 70, 200),
    }
    cv2.line(canvas, (60, size // 2), (size - 60, size // 2), (210, 210, 210), 2)
    cv2.line(canvas, (size // 2, 60), (size // 2, size - 60), (210, 210, 210), 2)
    cv2.putText(canvas, title, (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
    y = 82
    for name, vec in vectors.items():
        arr = normalize(np.asarray(vec, dtype=np.float64), name)
        # Front view: horizontal X, vertical up component. This is intended as a
        # quick sanity check of tilt magnitude, not a full 3D projection.
        endpoint = center + np.array([arr[0], -arr[2]]) * scale
        color = colors.get(name, (80, 80, 80))
        bgr = (int(color[2]), int(color[1]), int(color[0]))
        cv2.arrowedLine(canvas, tuple(center.astype(int)), tuple(endpoint.astype(int)), bgr, 4, cv2.LINE_AA, tipLength=0.08)
        cv2.putText(canvas, f"{name}: [{arr[0]:+.3f}, {arr[1]:+.3f}, {arr[2]:+.3f}]", (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, bgr, 1, cv2.LINE_AA)
        y += 26
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def write_ground_normal_debug(
    *,
    output_dir: Path,
    image_path: Path,
    moge: dict[str, Any],
    foreground_mask: np.ndarray | None,
    ground_mask: np.ndarray | None,
    ground_mask_path: Path | None,
    coordinate_system: str,
    ground_report: dict[str, Any],
    ground_normal_cv: np.ndarray | None,
    raw_source_up_scene: np.ndarray,
    source_up_scene: np.ndarray,
    raw_source_up_blender: np.ndarray,
    source_up_blender: np.ndarray,
    target_up: np.ndarray,
    scene_to_blender: np.ndarray,
    source_up_flip_for_target: bool,
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any]]:
    debug_dir = output_dir / f"{args.output_name}_ground_normal_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    valid = np.asarray(moge["mask"]).astype(bool)
    h, w = valid.shape
    rgb = load_debug_image(image_path, moge, (h, w))
    write_rgb(debug_dir / "00_input_rgb.png", rgb)
    cv2.imwrite(str(debug_dir / "01_moge_valid_mask.png"), as_uint8_mask(valid))
    finite_mask = np.asarray(ground_report.get("sample_info", {}).get("finite_mask", np.isfinite(moge["points"]).all(axis=-1))).astype(bool)
    cv2.imwrite(str(debug_dir / "02_moge_finite_points_mask.png"), as_uint8_mask(finite_mask))
    if foreground_mask is not None:
        cv2.imwrite(str(debug_dir / "03_label_foreground_mask.png"), as_uint8_mask(foreground_mask))
    if ground_mask is not None:
        cv2.imwrite(str(debug_dir / "03_gsam_ground_mask.png"), as_uint8_mask(ground_mask))
        ground_overlay = rgb.copy()
        ground_bool = np.asarray(ground_mask).astype(bool)
        ground_overlay[ground_bool] = (0.35 * ground_overlay[ground_bool] + 0.65 * np.array([0, 210, 230])).astype(np.uint8)
        write_rgb(debug_dir / "03_gsam_ground_mask_overlay.png", ground_overlay)
    sample_info = ground_report.get("sample_info") or {}
    for name, filename in [
        ("initial_bottom_roi", "04_initial_bottom_fraction_roi.png"),
        ("foreground_excluded_roi", "05_after_foreground_exclusion_roi.png"),
        ("final_ground_roi", "06_final_ground_candidate_roi.png"),
    ]:
        value = sample_info.get(name)
        if isinstance(value, np.ndarray):
            cv2.imwrite(str(debug_dir / filename), as_uint8_mask(value))
            overlay = rgb.copy()
            overlay[value.astype(bool)] = (0.35 * overlay[value.astype(bool)] + 0.65 * np.array([40, 220, 80])).astype(np.uint8)
            write_rgb(debug_dir / filename.replace(".png", "_overlay.png"), overlay)

    depth = np.asarray(moge.get("depth", moge["points"][..., 2]), dtype=np.float32)
    depth_u8 = normalize_to_uint8(depth, valid)
    cv2.imwrite(str(debug_dir / "07_moge_depth_normalized.png"), cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO))
    normal = moge.get("normal")
    if normal is not None:
        cv2.imwrite(str(debug_dir / "08_moge_normal_rgb.png"), cv2.cvtColor(normal_to_rgb(np.asarray(normal), valid), cv2.COLOR_RGB2BGR))

    sample_coords = sample_info.get("sample_coords_yx")
    sample_overlay = rgb.copy()
    if isinstance(sample_coords, np.ndarray) and len(sample_coords) > 0:
        coords = sample_coords.astype(np.int32)
        max_draw = min(len(coords), 12000)
        if len(coords) > max_draw:
            stride = max(1, len(coords) // max_draw)
            coords = coords[::stride][:max_draw]
        for y, x in coords:
            if 0 <= int(y) < h and 0 <= int(x) < w:
                cv2.circle(sample_overlay, (int(x), int(y)), 1, (255, 0, 0), -1)
    write_rgb(debug_dir / "09_sampled_ground_points_overlay.png", sample_overlay)

    draw_arrow_panel(
        path=debug_dir / "10_blender_up_vector_panel.png",
        title="MoGe estimated up vectors, Blender front view (X vs Z)",
        vectors={
            "raw_source_up_blender": raw_source_up_blender,
            "source_up_blender": source_up_blender,
            "target_up": target_up,
        },
    )
    draw_arrow_panel(
        path=debug_dir / "11_scene_up_vector_panel.png",
        title="MoGe scene-space up vectors, front view",
        vectors={
            "raw_source_up_scene": raw_source_up_scene,
            "source_up_scene": source_up_scene,
        },
    )

    sample_points = None
    if isinstance(sample_coords, np.ndarray) and len(sample_coords) > 0:
        sample_points = np.asarray(moge["points"])[sample_coords[:, 0], sample_coords[:, 1]]
    npz_payload: dict[str, Any] = {
        "points": np.asarray(moge["points"], dtype=np.float32),
        "valid_mask": valid,
        "finite_mask": finite_mask,
        "depth": depth,
        "raw_source_up_scene": raw_source_up_scene,
        "source_up_scene": source_up_scene,
        "raw_source_up_blender": raw_source_up_blender,
        "source_up_blender": source_up_blender,
        "target_up": target_up,
        "scene_to_blender_matrix": scene_to_blender,
    }
    if foreground_mask is not None:
        npz_payload["foreground_mask"] = foreground_mask.astype(bool)
    if ground_mask is not None:
        npz_payload["ground_mask"] = ground_mask.astype(bool)
    if isinstance(sample_coords, np.ndarray):
        npz_payload["sample_coords_yx"] = sample_coords.astype(np.int32)
    if sample_points is not None:
        npz_payload["sample_points"] = np.asarray(sample_points, dtype=np.float32)
    for key in ("initial_bottom_roi", "foreground_excluded_roi", "final_ground_roi", "manual_ground_mask_roi"):
        value = sample_info.get(key)
        if isinstance(value, np.ndarray):
            npz_payload[key] = value.astype(bool)
    if normal is not None:
        npz_payload["normal"] = np.asarray(normal, dtype=np.float32)
    np.savez_compressed(debug_dir / "ground_normal_debug_arrays.npz", **npz_payload)

    plane_singular_values = ground_report.get("plane_fit_singular_values")
    flatness_ratio = None
    if isinstance(plane_singular_values, np.ndarray) and plane_singular_values.size >= 3:
        flatness_ratio = float(plane_singular_values[-1] / max(float(plane_singular_values[0]), 1e-12))
    summary = {
        "debug_dir": str(debug_dir),
        "coordinate_system": coordinate_system,
        "image_size_wh": [int(w), int(h)],
        "ground_mask_path": str(ground_mask_path) if ground_mask_path else None,
        "ground_mask_used_for_normal_estimation": ground_mask is not None,
        "bottom_fraction": float(args.bottom_fraction),
        "max_ground_samples": int(args.max_ground_samples),
        "normal_agreement_deg": float(args.normal_agreement_deg),
        "normal_weight": float(args.normal_weight),
        "scene_to_blender_transform": args.scene_to_blender_transform,
        "scene_to_blender_matrix": scene_to_blender,
        "ground_normal_cv": ground_normal_cv,
        "raw_source_up_scene": raw_source_up_scene,
        "source_up_scene": source_up_scene,
        "raw_source_up_blender": raw_source_up_blender,
        "source_up_blender": source_up_blender,
        "target_up": target_up,
        "source_up_blender_flipped_to_target": bool(source_up_flip_for_target),
        "rotation_angle_to_target_deg": math.degrees(math.acos(float(np.clip(np.dot(normalize(source_up_blender), normalize(target_up)), -1.0, 1.0)))),
        "plane_fit_flatness_ratio_s3_over_s1": flatness_ratio,
        "sample_info": {
            key: value
            for key, value in sample_info.items()
            if key not in {"valid_mask", "finite_mask", "initial_bottom_roi", "foreground_excluded_roi", "final_ground_roi", "manual_ground_mask_roi", "sample_coords_yx"}
        },
        "plane_normal_cv": ground_report.get("plane_normal_cv"),
        "plane_normal_scene": ground_report.get("plane_normal_scene"),
        "normal_vote_cv": ground_report.get("normal_vote_cv"),
        "normal_vote_scene": ground_report.get("normal_vote_scene"),
        "normal_vote_count": ground_report.get("normal_vote_count"),
        "plane_fit_center_cv": ground_report.get("plane_fit_center_cv"),
        "plane_fit_center_scene": ground_report.get("plane_fit_center_scene"),
        "plane_fit_singular_values": plane_singular_values,
        "files": {
            "input_rgb": str(debug_dir / "00_input_rgb.png"),
            "valid_mask": str(debug_dir / "01_moge_valid_mask.png"),
            "finite_mask": str(debug_dir / "02_moge_finite_points_mask.png"),
            "label_foreground_mask": str(debug_dir / "03_label_foreground_mask.png") if foreground_mask is not None else None,
            "gsam_ground_mask": str(debug_dir / "03_gsam_ground_mask.png") if ground_mask is not None else None,
            "gsam_ground_mask_overlay": str(debug_dir / "03_gsam_ground_mask_overlay.png") if ground_mask is not None else None,
            "initial_bottom_roi": str(debug_dir / "04_initial_bottom_fraction_roi.png"),
            "after_foreground_exclusion_roi": str(debug_dir / "05_after_foreground_exclusion_roi.png"),
            "final_ground_candidate_roi": str(debug_dir / "06_final_ground_candidate_roi.png"),
            "depth": str(debug_dir / "07_moge_depth_normalized.png"),
            "normal_rgb": str(debug_dir / "08_moge_normal_rgb.png") if normal is not None else None,
            "sample_overlay": str(debug_dir / "09_sampled_ground_points_overlay.png"),
            "blender_vector_panel": str(debug_dir / "10_blender_up_vector_panel.png"),
            "scene_vector_panel": str(debug_dir / "11_scene_up_vector_panel.png"),
            "arrays_npz": str(debug_dir / "ground_normal_debug_arrays.npz"),
        },
    }
    (debug_dir / "ground_normal_debug.json").write_text(
        json.dumps(numpy_to_json(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return debug_dir, summary


GROUND_SAMPLE_ARRAY_KEYS = {
    "valid_mask",
    "finite_mask",
    "initial_bottom_roi",
    "foreground_excluded_roi",
    "final_ground_roi",
    "manual_ground_mask_roi",
    "sample_coords_yx",
}


def compact_ground_report_for_json(report: dict[str, Any]) -> dict[str, Any]:
    out = dict(report)
    sample_info = dict(out.get("sample_info") or {})
    for key in sorted(GROUND_SAMPLE_ARRAY_KEYS):
        value = sample_info.pop(key, None)
        if isinstance(value, np.ndarray):
            sample_info[f"{key}_shape"] = list(value.shape)
            if value.dtype == np.bool_:
                sample_info[f"{key}_true_count"] = int(value.sum())
            elif key == "sample_coords_yx":
                sample_info[f"{key}_count"] = int(len(value))
    out["sample_info"] = sample_info
    return out


def infer_moge(
    image_path: Path,
    *,
    device: str,
    checkpoint: Path,
    resize_max: int,
    resolution_level: int,
    num_tokens: int | None,
    use_fp16: bool,
) -> dict[str, np.ndarray]:
    from moge.model.v2 import MoGeModel

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(image_path)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image, resize_scale = resize_image(image, resize_max)
    torch_device = torch.device(device)
    model = MoGeModel.from_pretrained(str(checkpoint)).to(torch_device).eval()
    input_image = torch.tensor(image / 255.0, dtype=torch.float32, device=torch_device).permute(2, 0, 1)
    output = model.infer(input_image, resolution_level=resolution_level, num_tokens=num_tokens, use_fp16=use_fp16)
    result = {
        "image": image,
        "points": output["points"].detach().cpu().numpy(),
        "depth": output["depth"].detach().cpu().numpy(),
        "mask": output["mask"].detach().cpu().numpy().astype(bool),
        "intrinsics": output["intrinsics"].detach().cpu().numpy(),
        "resize_scale": np.array(resize_scale, dtype=np.float32),
    }
    if "normal" in output:
        result["normal"] = output["normal"].detach().cpu().numpy()
    torch.cuda.empty_cache()
    return result


def load_moge_cache(cache_path: Path) -> dict[str, Any]:
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    data = np.load(cache_path, allow_pickle=False)
    points = data["points"] if "points" in data else data["pointmap"] if "pointmap" in data else data["pcd"]
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"invalid MoGe cache points shape: {points.shape}")
    valid = data["mask"].astype(bool) if "mask" in data else np.isfinite(points).all(axis=-1)
    if valid.shape != points.shape[:2]:
        valid = np.isfinite(points).all(axis=-1)
    result: dict[str, Any] = {
        "points": points,
        "depth": np.asarray(data["depth"], dtype=np.float32) if "depth" in data else points[..., 2],
        "mask": valid,
        "intrinsics": np.asarray(data["intrinsics"], dtype=np.float32) if "intrinsics" in data else None,
        "resize_scale": np.array(1.0, dtype=np.float32),
        "coordinate_system": str(data["coordinate_system"]) if "coordinate_system" in data else "unknown",
        "source": str(data["source"]) if "source" in data else "moge_cache",
        "cache_path": str(cache_path),
    }
    if "image" in data:
        result["image"] = np.asarray(data["image"])
    if "normal" in data:
        result["normal"] = np.asarray(data["normal"], dtype=np.float32)
    return result


def robust_plane_normal(points: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    center = np.median(points, axis=0)
    centered = points - center
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    normal = normalize(vh[-1], "plane normal")
    return normal, {"center": center, "singular_values": singular_values}


def choose_ground_samples(
    points: np.ndarray,
    normal: np.ndarray | None,
    valid: np.ndarray,
    *,
    foreground_mask: np.ndarray | None,
    ground_mask: np.ndarray | None,
    bottom_fraction: float,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    h, w = valid.shape
    yy = np.arange(h, dtype=np.float32)[:, None]
    finite = np.isfinite(points).all(axis=-1)
    initial_bottom_roi = valid & finite & (yy >= h * (1.0 - bottom_fraction))
    ground_mask_roi = None
    if ground_mask is not None:
        ground_mask_roi = np.asarray(ground_mask).astype(bool) & valid & finite
        roi = ground_mask_roi.copy()
        foreground_excluded_roi = roi.copy()
        roi_stage = "manual_ground_mask"
    else:
        roi = initial_bottom_roi.copy()
        if foreground_mask is not None:
            roi &= ~foreground_mask
        foreground_excluded_roi = roi.copy()
        roi_stage = "bottom_fraction_without_foreground" if foreground_mask is not None else "bottom_fraction"
    if int(roi.sum()) < 128:
        if ground_mask is not None and int(roi.sum()) > 2:
            roi_stage = "manual_ground_mask_small"
        else:
            roi = valid & finite & (yy >= h * 0.45)
            if foreground_mask is not None:
                roi &= ~foreground_mask
            roi_stage = "fallback_lower_55_percent"
            if int(roi.sum()) < 128:
                roi = valid & finite
                roi_stage = "fallback_all_valid"
    coords = np.argwhere(roi)
    if len(coords) == 0:
        raise RuntimeError("MoGe produced no valid ground candidate pixels")
    rng = np.random.default_rng(seed)
    if len(coords) > max_samples:
        coords = coords[rng.choice(len(coords), size=max_samples, replace=False)]
    sample_points = points[coords[:, 0], coords[:, 1]].astype(np.float64)
    sample_normals = None
    if normal is not None:
        sample_normals = normal[coords[:, 0], coords[:, 1]].astype(np.float64)
        normal_valid = np.isfinite(sample_normals).all(axis=1) & (np.linalg.norm(sample_normals, axis=1) > 1e-6)
        sample_normals = sample_normals[normal_valid]
    return sample_points, sample_normals, {
        "candidate_pixels": int(roi.sum()),
        "sampled_points": int(len(sample_points)),
        "image_size": [int(w), int(h)],
        "bottom_fraction": float(bottom_fraction),
        "foreground_excluded": foreground_mask is not None,
        "roi_stage": roi_stage,
        "valid_mask": valid.astype(bool),
        "finite_mask": finite.astype(bool),
        "initial_bottom_roi": initial_bottom_roi.astype(bool),
        "foreground_excluded_roi": foreground_excluded_roi.astype(bool),
        "final_ground_roi": roi.astype(bool),
        "manual_ground_mask_roi": ground_mask_roi.astype(bool) if ground_mask_roi is not None else None,
        "sample_coords_yx": coords.astype(np.int32),
    }


def estimate_ground_normal_cv(
    points: np.ndarray,
    normal: np.ndarray | None,
    valid: np.ndarray,
    foreground_mask: np.ndarray | None,
    ground_mask: np.ndarray | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    sample_points, sample_normals, sample_info = choose_ground_samples(
        points,
        normal,
        valid,
        foreground_mask=foreground_mask,
        ground_mask=ground_mask,
        bottom_fraction=args.bottom_fraction,
        max_samples=args.max_ground_samples,
        seed=args.seed,
    )
    plane_normal, plane_fit = robust_plane_normal(sample_points)
    if float(np.dot(plane_normal, np.array([0.0, -1.0, 0.0]))) < 0.0:
        plane_normal = -plane_normal
    normal_vote = None
    normal_vote_count = 0
    if sample_normals is not None and len(sample_normals) > 32:
        sample_normals = sample_normals / np.maximum(np.linalg.norm(sample_normals, axis=1, keepdims=True), 1e-8)
        if float(np.median(sample_normals @ plane_normal)) < 0.0:
            sample_normals = -sample_normals
        aligned = sample_normals[sample_normals @ plane_normal > math.cos(math.radians(args.normal_agreement_deg))]
        if len(aligned) > 32:
            normal_vote = normalize(np.median(aligned, axis=0), "normal vote")
            normal_vote_count = int(len(aligned))
            ground_normal = normalize((1.0 - args.normal_weight) * plane_normal + args.normal_weight * normal_vote, "ground normal")
        else:
            ground_normal = plane_normal
    else:
        ground_normal = plane_normal
    return ground_normal, {
        "sample_info": sample_info,
        "plane_normal_cv": plane_normal,
        "normal_vote_cv": normal_vote,
        "normal_vote_count": normal_vote_count,
        "normal_weight": float(args.normal_weight),
        "plane_fit_center_cv": plane_fit["center"],
        "plane_fit_singular_values": plane_fit["singular_values"],
    }


def estimate_ground_normal_scene(
    points: np.ndarray,
    normal: np.ndarray | None,
    valid: np.ndarray,
    foreground_mask: np.ndarray | None,
    ground_mask: np.ndarray | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    sample_points, sample_normals, sample_info = choose_ground_samples(
        points,
        normal,
        valid,
        foreground_mask=foreground_mask,
        ground_mask=ground_mask,
        bottom_fraction=args.bottom_fraction,
        max_samples=args.max_ground_samples,
        seed=args.seed,
    )
    plane_normal, plane_fit = robust_plane_normal(sample_points)
    if float(np.dot(plane_normal, np.array([0.0, 1.0, 0.0]))) < 0.0:
        plane_normal = -plane_normal
    normal_vote = None
    normal_vote_count = 0
    if sample_normals is not None and len(sample_normals) > 32:
        sample_normals = sample_normals / np.maximum(np.linalg.norm(sample_normals, axis=1, keepdims=True), 1e-8)
        if float(np.median(sample_normals @ plane_normal)) < 0.0:
            sample_normals = -sample_normals
        aligned = sample_normals[sample_normals @ plane_normal > math.cos(math.radians(args.normal_agreement_deg))]
        if len(aligned) > 32:
            normal_vote = normalize(np.median(aligned, axis=0), "scene normal vote")
            normal_vote_count = int(len(aligned))
            ground_normal = normalize((1.0 - args.normal_weight) * plane_normal + args.normal_weight * normal_vote, "scene ground normal")
        else:
            ground_normal = plane_normal
    else:
        ground_normal = plane_normal
    return ground_normal, {
        "sample_info": sample_info,
        "plane_normal_scene": plane_normal,
        "normal_vote_scene": normal_vote,
        "normal_vote_count": normal_vote_count,
        "normal_weight": float(args.normal_weight),
        "plane_fit_center_scene": plane_fit["center"],
        "plane_fit_singular_values": plane_fit["singular_values"],
    }


def cv_to_pytorch3d_scene_normal(normal_cv: np.ndarray) -> np.ndarray:
    return transform_direction(OPENCV_CAMERA_TO_PYTORCH3D_SCENE, normal_cv, name="opencv-to-pytorch3d scene normal")


def rotation_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = normalize(axis, "rotation axis")
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def rotation_between_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = normalize(source, "source")
    target = normalize(target, "target")
    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    norm = float(np.linalg.norm(cross))
    if norm < 1e-10:
        if dot > 0:
            return np.eye(3, dtype=np.float64)
        helper = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(helper, source))) > 0.9:
            helper = np.array([0.0, 0.0, 1.0])
        return rotation_from_axis_angle(np.cross(source, helper), math.pi)
    return rotation_from_axis_angle(cross / norm, math.atan2(norm, dot))


def snap_meshes_to_ground(scene: trimesh.Scene, ground_y: float, up_vector: np.ndarray) -> tuple[trimesh.Scene, list[dict[str, Any]]]:
    up = normalize(up_vector, "ground up vector")
    up_idx = up_axis_index(up)
    out = trimesh.Scene()
    reports: list[dict[str, Any]] = []
    used_names: set[str] = set()
    for idx, geom in enumerate(scene.dump(concatenate=False)):
        if not isinstance(geom, trimesh.Trimesh) or len(geom.vertices) == 0:
            continue
        mesh = geom.copy()
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        min_ground = float(np.min(vertices @ up))
        delta = up * (float(ground_y) - min_ground)
        mesh.vertices = vertices + delta
        vertices_after = np.asarray(mesh.vertices, dtype=np.float64)
        name = f"object_{idx:03d}"
        while name in used_names:
            name = f"object_{idx:03d}_{len(used_names):03d}"
        used_names.add(name)
        out.add_geometry(mesh, geom_name=name, node_name=name)
        reports.append(
            {
                "index": idx,
                "name": name,
                "up_vector": up,
                "up_axis_index": up_idx,
                "min_ground_before": min_ground,
                "translation": delta,
                "translation_y": float(delta[1]),
                "translation_up_axis": float(delta[up_idx]),
                "translation_along_up": float(np.dot(delta, up)),
                "min_ground_after": float(np.min(vertices_after @ up)),
                "bbox_before": np.asarray(geom.bounds, dtype=np.float64),
                "bbox_after": np.asarray(mesh.bounds, dtype=np.float64),
            }
        )
    if not reports:
        raise RuntimeError("bbox snap found no mesh geometry")
    return out, reports


def overlap_1d(a_min: float, a_max: float, b_min: float, b_max: float) -> float:
    return max(0.0, min(a_max, b_max) - max(a_min, b_min))


def bbox_volume(bounds: np.ndarray) -> float:
    extent = np.maximum(np.asarray(bounds, dtype=np.float64)[1] - np.asarray(bounds, dtype=np.float64)[0], 0.0)
    return float(np.prod(extent))


def bbox_overlap_report(mesh_i: trimesh.Trimesh, mesh_j: trimesh.Trimesh) -> dict[str, Any]:
    bi = np.asarray(mesh_i.bounds, dtype=np.float64)
    bj = np.asarray(mesh_j.bounds, dtype=np.float64)
    overlaps = np.array([overlap_1d(float(bi[0, a]), float(bi[1, a]), float(bj[0, a]), float(bj[1, a])) for a in range(3)], dtype=np.float64)
    volume = float(np.prod(overlaps))
    min_volume = max(min(bbox_volume(bi), bbox_volume(bj)), 1e-12)
    return {"overlap_xyz": overlaps, "overlap_volume": volume, "overlap_volume_ratio_to_smaller": volume / min_volume, "bounds_i": bi, "bounds_j": bj}


def scene_mesh_entries(scene: trimesh.Scene) -> list[tuple[str, trimesh.Trimesh]]:
    entries: list[tuple[str, trimesh.Trimesh]] = []
    for idx, geom in enumerate(scene.dump(concatenate=False)):
        if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0:
            entries.append((f"object_{idx:03d}", geom.copy()))
    return entries


def bbox_pair_reports(entries: list[tuple[str, trimesh.Trimesh]], *, eps: float, min_overlap_volume_ratio: float = 0.0) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            overlap_info = bbox_overlap_report(entries[i][1], entries[j][1])
            overlaps = overlap_info["overlap_xyz"]
            if np.any(overlaps <= eps):
                continue
            if float(overlap_info["overlap_volume_ratio_to_smaller"]) < float(min_overlap_volume_ratio):
                continue
            item = dict(overlap_info)
            item["indices"] = [i, j]
            item["pair"] = [entries[i][0], entries[j][0]]
            reports.append(item)
    return reports


def connected_components_from_pairs(pair_reports: list[dict[str, Any]], count: int) -> list[list[int]]:
    parent = list(range(count))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for pair in pair_reports:
        i, j = [int(v) for v in pair["indices"]]
        union(i, j)
    groups: dict[int, list[int]] = {}
    for idx in range(count):
        groups.setdefault(find(idx), []).append(idx)
    return [members for members in groups.values() if len(members) > 1]


def pack_indices_along_x(
    entries: list[tuple[str, trimesh.Trimesh]],
    translations: list[np.ndarray],
    indices: list[int],
    *,
    margin: float,
    reason: str,
    iteration: int,
    up_vector: np.ndarray,
) -> list[dict[str, Any]]:
    horizontal_axes = horizontal_axis_indices(up_vector)
    primary_axis = horizontal_axes[0]
    secondary_axis = horizontal_axes[1]
    ordered = sorted(
        set(int(i) for i in indices),
        key=lambda idx: (
            float(np.asarray(entries[idx][1].bounds, dtype=np.float64).mean(axis=0)[primary_axis]),
            float(np.asarray(entries[idx][1].bounds, dtype=np.float64).mean(axis=0)[secondary_axis]),
            entries[idx][0],
        ),
    )
    if len(ordered) < 2:
        return []
    bounds = {idx: np.asarray(entries[idx][1].bounds, dtype=np.float64) for idx in ordered}
    centers = {idx: float(bounds[idx].mean(axis=0)[primary_axis]) for idx in ordered}
    widths = {idx: max(float(bounds[idx][1, primary_axis] - bounds[idx][0, primary_axis]), 1e-8) for idx in ordered}
    group_center = 0.5 * (min(float(bounds[idx][0, primary_axis]) for idx in ordered) + max(float(bounds[idx][1, primary_axis]) for idx in ordered))
    total_width = sum(widths[idx] for idx in ordered) + float(margin) * float(len(ordered) - 1)
    cursor = group_center - 0.5 * total_width
    actions: list[dict[str, Any]] = []
    for idx in ordered:
        desired_center = cursor + 0.5 * widths[idx]
        d = desired_center - centers[idx]
        cursor += widths[idx] + float(margin)
        if abs(d) <= 1e-12:
            continue
        delta = np.zeros(3, dtype=np.float64)
        delta[primary_axis] = d
        mesh = entries[idx][1]
        mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) + delta
        translations[idx] += delta
        actions.append({"iteration": iteration, "reason": reason, "name": entries[idx][0], "translation": delta, "primary_axis": primary_axis, "center_before": centers[idx], "center_after": desired_center})
    return actions


def separate_overlapping_bboxes(
    scene: trimesh.Scene,
    *,
    margin: float,
    max_iters: int,
    min_overlap_volume_ratio: float,
    eps: float,
    up_vector: np.ndarray,
) -> tuple[trimesh.Scene, list[dict[str, Any]]]:
    entries = scene_mesh_entries(scene)
    if len(entries) < 2:
        return scene, [{"initial_overlaps": [], "pairs": [], "fallback_actions": [], "final_overlaps": [], "objects": []}]
    translations = [np.zeros(3, dtype=np.float64) for _ in entries]
    horizontal_axes = horizontal_axis_indices(up_vector)
    initial_overlaps = bbox_pair_reports(entries, eps=eps, min_overlap_volume_ratio=min_overlap_volume_ratio)
    pair_reports: list[dict[str, Any]] = []
    for iteration in range(max(0, int(max_iters))):
        moved = False
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                mesh_i = entries[i][1]
                mesh_j = entries[j][1]
                overlap_info = bbox_overlap_report(mesh_i, mesh_j)
                overlaps = overlap_info["overlap_xyz"]
                if np.any(overlaps <= eps):
                    continue
                if float(overlap_info["overlap_volume_ratio_to_smaller"]) < float(min_overlap_volume_ratio):
                    continue
                ci = np.asarray(mesh_i.bounds, dtype=np.float64).mean(axis=0)
                cj = np.asarray(mesh_j.bounds, dtype=np.float64).mean(axis=0)
                axis_idx = min(horizontal_axes, key=lambda a: float(overlaps[a]))
                direction = np.zeros(3, dtype=np.float64)
                direction[axis_idx] = 1.0 if ci[axis_idx] >= cj[axis_idx] else -1.0
                amount = 0.5 * (float(overlaps[axis_idx]) + margin)
                axis = "xyz"[axis_idx]
                separation_overlap = float(overlaps[axis_idx])
                delta_i = direction * amount
                delta_j = -direction * amount
                mesh_i.vertices = np.asarray(mesh_i.vertices, dtype=np.float64) + delta_i
                mesh_j.vertices = np.asarray(mesh_j.vertices, dtype=np.float64) + delta_j
                translations[i] += delta_i
                translations[j] += delta_j
                moved = True
                pair_reports.append(
                    {
                        "iteration": iteration,
                        "pair": [entries[i][0], entries[j][0]],
                        "axis": axis,
                        "separation_axis_overlap": float(separation_overlap),
                        "overlap_volume": float(overlap_info["overlap_volume"]),
                        "overlap_xyz": overlaps,
                        "overlap_volume_ratio_to_smaller": float(overlap_info["overlap_volume_ratio_to_smaller"]),
                        "margin": float(margin),
                        "translation_i": delta_i,
                        "translation_j": delta_j,
                    }
                )
        if not moved:
            break
    fallback_actions: list[dict[str, Any]] = []
    for fallback_iteration in range(10):
        remaining = bbox_pair_reports(entries, eps=eps, min_overlap_volume_ratio=min_overlap_volume_ratio)
        if not remaining:
            break
        before_count = len(fallback_actions)
        for component in connected_components_from_pairs(remaining, len(entries)):
            fallback_actions.extend(pack_indices_along_x(entries, translations, component, margin=margin, reason="component_pack_residual_overlap", iteration=fallback_iteration, up_vector=up_vector))
        if len(fallback_actions) == before_count:
            break
    final_overlaps = bbox_pair_reports(entries, eps=eps, min_overlap_volume_ratio=min_overlap_volume_ratio)
    if final_overlaps:
        fallback_actions.extend(pack_indices_along_x(entries, translations, list(range(len(entries))), margin=margin, reason="global_pack_residual_overlap", iteration=0, up_vector=up_vector))
        final_overlaps = bbox_pair_reports(entries, eps=eps, min_overlap_volume_ratio=min_overlap_volume_ratio)
    out = trimesh.Scene()
    for (name, mesh), _translation in zip(entries, translations):
        out.add_geometry(mesh, geom_name=name, node_name=name)
    reports = [{"name": name, "total_translation": translation, "bbox_after": np.asarray(mesh.bounds, dtype=np.float64)} for (name, mesh), translation in zip(entries, translations)]
    return out, [{"initial_overlaps": initial_overlaps, "pairs": pair_reports, "fallback_actions": fallback_actions, "final_overlaps": final_overlaps, "remaining_overlap_count": len(final_overlaps), "objects": reports}]


def transform_scene(
    scene: trimesh.Scene,
    source_up: np.ndarray,
    target_up: np.ndarray,
    ground_y: float,
    *,
    translate_scene_to_ground: bool,
    snap_bbox_to_ground: bool,
    separate_overlaps: bool,
    overlap_margin: float,
    overlap_iters: int,
    min_overlap_volume_ratio: float,
    overlap_eps: float,
    rotation_mode: str,
) -> tuple[trimesh.Scene, dict[str, Any]]:
    meshes = [m for m in scene.dump() if isinstance(m, trimesh.Trimesh) and len(m.vertices) > 0]
    if not meshes:
        raise RuntimeError("scene contains no mesh vertices")
    vertices = np.concatenate([np.asarray(mesh.vertices, dtype=np.float64) for mesh in meshes], axis=0)
    if rotation_mode == "none":
        rotation = np.eye(3, dtype=np.float64)
    elif rotation_mode == "upright":
        rotation = rotation_between_vectors(source_up, target_up)
    else:
        raise ValueError(f"unsupported rotation_mode: {rotation_mode}")
    rotated = vertices @ rotation.T
    min_ground = float(np.min(rotated @ target_up))
    translation = target_up * (float(ground_y) - min_ground) if translate_scene_to_ground else np.zeros(3, dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    out = scene.copy()
    out.apply_transform(transform)
    bbox_snap_reports: list[dict[str, Any]] = []
    if snap_bbox_to_ground:
        out, bbox_snap_reports = snap_meshes_to_ground(out, ground_y, target_up)
    overlap_reports: list[dict[str, Any]] = []
    if separate_overlaps:
        out, overlap_reports = separate_overlapping_bboxes(out, margin=overlap_margin, max_iters=overlap_iters, min_overlap_volume_ratio=min_overlap_volume_ratio, eps=overlap_eps, up_vector=target_up)
    angle = 0.0 if rotation_mode == "none" else math.degrees(math.acos(float(np.clip(np.dot(normalize(source_up), normalize(target_up)), -1.0, 1.0))))
    return out, {
        "rotation_mode": rotation_mode,
        "source_up": source_up,
        "target_up": target_up,
        "rotation_angle_deg": angle,
        "rotation_matrix": rotation,
        "translation": translation,
        "matrix_4x4": transform,
        "min_ground_before_translation": min_ground,
        "translate_scene_to_ground": bool(translate_scene_to_ground),
        "bbox_snap_to_ground": bool(snap_bbox_to_ground),
        "bbox_snap_objects": bbox_snap_reports,
        "bbox_separate_overlaps": bool(separate_overlaps),
        "bbox_overlap_margin": float(overlap_margin),
        "bbox_overlap_iters": int(overlap_iters),
        "bbox_min_overlap_volume_ratio": float(min_overlap_volume_ratio),
        "bbox_overlap_eps": float(overlap_eps),
        "bbox_overlap_reports": overlap_reports,
    }


def blend_to_glb(blend_path: Path, glb_path: Path, blender_bin: str) -> None:
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    script = (
        "import bpy\n"
        f"bpy.ops.wm.open_mainfile(filepath={str(blend_path)!r})\n"
        f"bpy.ops.export_scene.gltf(filepath={str(glb_path)!r}, export_format='GLB')\n"
    )
    subprocess.run([blender_bin, "-b", "--python-expr", script], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def export_stage(*, scene: trimesh.Scene, output_dir: Path, output_name: str, report: dict[str, Any], blender_bin: str) -> dict[str, str]:
    output_base = unique_base(output_dir, output_name)
    output_blend = output_base.with_suffix(".blend")
    report_json = Path(f"{output_base}_report.json")
    if glb_to_blend is None:
        raise RuntimeError("pipeline_utils.glb_to_blend is unavailable")
    with tempfile.TemporaryDirectory(prefix="sam3d_moge_stage_export_") as tmp_dir:
        output_glb = Path(tmp_dir) / f"{output_base.name}.glb"
        scene.export(output_glb, file_type="glb")
        glb_to_blend(output_glb, output_blend, blender_bin)
    outputs = {"blend": str(output_blend), "report": str(report_json)}
    report["outputs"] = outputs
    report_json.write_text(json.dumps(numpy_to_json(report), ensure_ascii=False, indent=2), encoding="utf-8")
    return outputs


def export_stage_preserve_blend(
    *,
    input_blend: Path,
    output_dir: Path,
    output_name: str,
    stage_report: dict[str, Any],
    blender_bin: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    output_base = unique_base(output_dir, output_name)
    output_blend = output_base.with_suffix(".blend")
    report_json = Path(f"{output_base}_report.json")
    plan_json = Path(f"{output_base}_plan.json")
    plan = {
        "stage": stage_report["stage"],
        "input": stage_report["input"],
        "moge": stage_report["moge"],
        "transform": stage_report["transform"],
        "snap_bbox_to_ground": bool(stage_report["transform"].get("bbox_snap_to_ground")),
        "separate_overlaps": bool(stage_report["transform"].get("bbox_separate_overlaps")),
        "ground_y": float(stage_report["transform"].get("ground_y", stage_report["transform"].get("target_ground_y", 0.0))),
        "bbox_snap_mode": str(stage_report["transform"].get("bbox_snap_mode", "per_object")),
        "support_adjust": bool(stage_report["transform"].get("support_adjust", False)),
        "support_require_scene_graph": bool(stage_report["transform"].get("support_require_scene_graph", False)),
        "support_gap": float(stage_report["transform"].get("support_gap", 0.001)),
        "support_xy_overlap_ratio": float(stage_report["transform"].get("support_xy_overlap_ratio", 0.15)),
        "support_max_gap": float(stage_report["transform"].get("support_max_gap", -1.0)),
        "support_max_gap_ratio": float(stage_report["transform"].get("support_max_gap_ratio", 0.03)),
        "support_max_penetration": float(stage_report["transform"].get("support_max_penetration", -1.0)),
        "support_max_penetration_ratio": float(stage_report["transform"].get("support_max_penetration_ratio", 0.01)),
        "support_min_lower_area_ratio": float(stage_report["transform"].get("support_min_lower_area_ratio", 0.25)),
        "support_collision_method": str(stage_report["transform"].get("support_collision_method", stage_report["transform"].get("overlap_collision_method", "convex_hull_sat"))),
        "bbox_overlap_margin": float(stage_report["transform"].get("bbox_overlap_margin", 0.01)),
        "bbox_overlap_iters": int(stage_report["transform"].get("bbox_overlap_iters", 32)),
        "bbox_min_overlap_volume_ratio": float(stage_report["transform"].get("bbox_min_overlap_volume_ratio", 0.0)),
        "bbox_overlap_eps": float(stage_report["transform"].get("bbox_overlap_eps", 1e-8)),
        "overlap_collision_method": str(stage_report["transform"].get("overlap_collision_method", "bbox")),
        "convex_hull_max_vertices": int(stage_report["transform"].get("convex_hull_max_vertices", 8000)),
        "convex_decomposition_method": str(stage_report["transform"].get("convex_decomposition_method", "single_convex_hull")),
        "coacd_conda_bin": str(stage_report["transform"].get("coacd_conda_bin", stage_report["transform"].get("nvdiffrast_conda_bin", os.environ.get("CONDA_BIN", "conda")))),
        "coacd_env": str(stage_report["transform"].get("coacd_env", stage_report["transform"].get("nvdiffrast_env", os.environ.get("FYSIVERSE_CONDA_ENV", "fysiverse-scene")))),
        "coacd_timeout": int(stage_report["transform"].get("coacd_timeout", 120)),
        "coacd_workers": int(stage_report["transform"].get("coacd_workers", 4)),
        "coacd_source_max_faces": int(stage_report["transform"].get("coacd_source_max_faces", 5000)),
        "coacd_source_simplification_backend": str(stage_report["transform"].get("coacd_source_simplification_backend", "fast_simplification")),
        "coacd_source_simplification_agg": float(stage_report["transform"].get("coacd_source_simplification_agg", 7.0)),
        "coacd_blender_source_decimate": bool(stage_report["transform"].get("coacd_blender_source_decimate", False)),
        "coacd_max_convex_parts": int(stage_report["transform"].get("coacd_max_convex_parts", 8)),
        "coacd_threshold": float(stage_report["transform"].get("coacd_threshold", 0.05)),
        "coacd_preprocess_mode": str(stage_report["transform"].get("coacd_preprocess_mode", "auto")),
        "coacd_preprocess_resolution": int(stage_report["transform"].get("coacd_preprocess_resolution", 50)),
        "coacd_resolution": int(stage_report["transform"].get("coacd_resolution", 2000)),
        "coacd_mcts_nodes": int(stage_report["transform"].get("coacd_mcts_nodes", 20)),
        "coacd_mcts_iterations": int(stage_report["transform"].get("coacd_mcts_iterations", 80)),
        "coacd_mcts_max_depth": int(stage_report["transform"].get("coacd_mcts_max_depth", 3)),
        "coacd_max_ch_vertex": int(stage_report["transform"].get("coacd_max_ch_vertex", 256)),
        "coacd_apx_mode": str(stage_report["transform"].get("coacd_apx_mode", "ch")),
        "coacd_seed": int(stage_report["transform"].get("coacd_seed", 0)),
        "coacd_real_metric": bool(stage_report["transform"].get("coacd_real_metric", False)),
        "coacd_merge": bool(stage_report["transform"].get("coacd_merge", True)),
        "coacd_decimate": bool(stage_report["transform"].get("coacd_decimate", False)),
        "convex_collision_eps": float(stage_report["transform"].get("convex_collision_eps", 1e-6)),
        "convex_min_horizontal_axis": float(stage_report["transform"].get("convex_min_horizontal_axis", 0.25)),
    }
    scene_graph = stage_report["transform"].get("scene_graph")
    if scene_graph:
        plan["scene_graph"] = str(scene_graph)
    plan_json.write_text(json.dumps(numpy_to_json(plan), ensure_ascii=False, indent=2), encoding="utf-8")
    cmd = [
        blender_bin,
        "-b",
        "--python",
        str(WORKFLOW_ROOT / "scripts" / "blender_apply_moge_stage.py"),
        "--",
        "--input",
        str(input_blend),
        "--plan",
        str(plan_json),
        "--output",
        str(output_blend),
        "--report",
        str(report_json),
        "--pack-textures",
    ]
    proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    outputs = {"blend": str(output_blend), "report": str(report_json), "plan": str(plan_json)}
    final_report = json.loads(report_json.read_text(encoding="utf-8"))
    final_report["blender_stdout"] = proc.stdout
    final_report["note"] = "MoGe stage applied directly in Blender to preserve source materials/textures."
    report_json.write_text(json.dumps(numpy_to_json(final_report), ensure_ascii=False, indent=2), encoding="utf-8")
    return outputs, final_report


def add_original_input_camera_to_stage(
    *,
    blend_path: Path,
    stage_report_path: Path,
    camera_report_path: Path,
    camera_optimization_path: Path | None,
    blender_bin: str,
) -> tuple[dict[str, str], dict[str, Any], str]:
    camera_json = blend_path.with_name(f"{blend_path.stem}_original_input_camera.json")
    cmd = [
        blender_bin,
        "-b",
        "--python",
        str(WORKFLOW_ROOT / "scripts" / "blender_add_original_camera.py"),
        "--",
        "--input",
        str(blend_path),
        "--camera-report",
        str(camera_report_path),
        "--stage-report",
        str(stage_report_path),
        "--output-json",
        str(camera_json),
        "--camera-name",
        "OriginalInputCamera",
        "--set-active",
        "--set-render-resolution",
    ]
    if camera_optimization_path is not None and camera_optimization_path.is_file():
        cmd.extend(["--camera-optimization", str(camera_optimization_path)])
    proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    camera_data = json.loads(camera_json.read_text(encoding="utf-8"))
    outputs = {"camera_json": str(camera_json)}
    return outputs, camera_data, proc.stdout


def run_command(cmd: list[str], *, cwd: Path) -> str:
    proc = subprocess.run(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        command = " ".join(str(part) for part in cmd)
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {command}\n{proc.stdout}")
    return proc.stdout


def nvdiffrast_python_command(args: argparse.Namespace) -> list[str]:
    if os.environ.get("CONDA_DEFAULT_ENV") == str(args.nvdiffrast_env):
        return [sys.executable]
    return [args.conda_bin, "run", "--no-capture-output", "-n", args.nvdiffrast_env, "python"]


def label_ids_from_mask(mask_path: Path) -> list[int]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(mask_path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    ids = sorted(int(v) for v in np.unique(mask) if int(v) > 0)
    return ids


def write_optimization_manifest(
    *,
    output_path: Path,
    image_path: Path,
    mask_path: Path,
    blend_path: Path,
    result_dir: Path,
) -> Path:
    objects = [
        {
            "mask_id": int(mask_id),
            "mask_name": f"mask_{int(mask_id):03d}",
            "final_3d_object_name": f"mask_{int(mask_id):03d}_object",
        }
        for mask_id in label_ids_from_mask(mask_path)
    ]
    payload = {
        "schema": "fysiverse_moge_stage_optimization_manifest.v1",
        "status": "ok",
        "inputs": {
            "image": str(image_path),
            "label_mask": str(mask_path),
            "result_dir": str(result_dir),
        },
        "outputs": {
            "final_blend": str(blend_path),
        },
        "objects": objects,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(numpy_to_json(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def optimize_grounded_stage(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    grounded_blend: Path,
    grounded_report: Path,
    rotated_report: Path,
    image_path: Path,
    mask_path: Path | None,
) -> tuple[dict[str, str], dict[str, Any]]:
    total_start = timing_start()
    timings: dict[str, dict[str, float]] = {}
    if mask_path is None or not mask_path.is_file():
        raise FileNotFoundError("Differentiable object optimization requires --mask label mask")

    prefix = f"{args.output_name}_differentiable"
    manifest_path = output_dir / f"{prefix}_manifest.json"
    camera_mesh = output_dir / f"{prefix}_camera_mesh.npz"
    camera_report = output_dir / f"{prefix}_camera_pose_optimization.json"
    camera_preview = output_dir / f"{prefix}_camera_pose_optimization_preview.png"
    object_mesh = output_dir / f"{prefix}_object_mesh.npz"
    object_report = output_dir / f"{prefix}_object_pose_optimization.json"
    object_preview_dir = output_dir / f"{prefix}_object_pose_optimization_previews"
    optimized_blend = output_dir / f"{args.output_name}_optimized.blend"

    write_optimization_manifest(
        output_path=manifest_path,
        image_path=image_path,
        mask_path=mask_path,
        blend_path=grounded_blend,
        result_dir=output_dir,
    )

    step_start = timing_start()
    camera_export_stdout = run_command(
        [
            args.blender_bin,
            "-b",
            "--python",
            str(WORKFLOW_ROOT / "scripts" / "blender_export_nvdiffrast_mesh.py"),
            "--",
            "--input",
            str(grounded_blend),
            "--output",
            str(camera_mesh),
            "--frame",
            "1",
        ],
        cwd=WORKFLOW_ROOT,
    )
    timings["camera_mesh_export"] = timing_elapsed(step_start)
    nvdiffrast_python = nvdiffrast_python_command(args)
    camera_opt_cmd = [
        *nvdiffrast_python,
        str(WORKFLOW_ROOT / "scripts" / "optimize_camera_pose_nvdiffrast.py"),
        "--mesh",
        str(camera_mesh),
        "--target-mask",
        str(mask_path),
        "--camera-report",
        str(rotated_report),
        "--stage-report",
        str(grounded_report),
        "--output-json",
        str(camera_report),
        "--preview-output",
        str(camera_preview),
        "--input-image",
        str(image_path),
        "--steps",
        str(args.camera_opt_steps),
        "--max-side",
        str(args.camera_opt_max_side),
        "--rot-lr",
        str(args.camera_opt_rot_lr),
        "--trans-lr",
        str(args.camera_opt_trans_lr),
    ]
    step_start = timing_start()
    camera_opt_stdout = run_command(camera_opt_cmd, cwd=WORKFLOW_ROOT)
    timings["camera_pose_optimization"] = timing_elapsed(step_start)

    step_start = timing_start()
    object_export_stdout = run_command(
        [
            args.blender_bin,
            "-b",
            "--python",
            str(WORKFLOW_ROOT / "scripts" / "blender_export_nvdiffrast_mesh.py"),
            "--",
            "--input",
            str(grounded_blend),
            "--output",
            str(object_mesh),
            "--frame",
            "1",
        ],
        cwd=WORKFLOW_ROOT,
    )
    timings["object_mesh_export"] = timing_elapsed(step_start)
    object_opt_cmd = [
        *nvdiffrast_python,
        str(WORKFLOW_ROOT / "scripts" / "optimize_object_poses_nvdiffrast.py"),
        "--mesh",
        str(object_mesh),
        "--target-mask",
        str(mask_path),
        "--final-manifest",
        str(manifest_path),
        "--camera-optimization",
        str(camera_report),
        "--output-json",
        str(object_report),
        "--preview-dir",
        str(object_preview_dir),
        "--input-image",
        str(image_path),
        "--workers",
        str(args.object_pose_workers),
        "--max-side",
        str(args.object_pose_max_side),
        "--position-steps",
        str(args.object_position_steps),
        "--yaw-steps",
        str(args.object_yaw_steps),
        "--scale-steps",
        str(args.object_scale_steps),
        "--translation-mode",
        str(args.object_translation_mode),
        "--translation-initial-grid",
        str(args.object_translation_initial_grid),
        "--center-weight",
        str(args.object_center_weight),
        "--area-weight",
        str(args.object_area_weight),
        "--yaw-initial-samples",
        str(args.object_yaw_initial_samples),
        "--scale-initial-samples",
        str(args.object_scale_initial_samples),
        "--phase-accept-iou-drop",
        str(args.object_phase_accept_iou_drop),
        "--position-lr",
        str(args.object_position_lr),
        "--yaw-lr",
        str(args.object_yaw_lr),
        "--scale-lr",
        str(args.object_scale_lr),
        "--max-translation",
        str(args.object_max_translation),
        "--max-yaw-deg",
        str(args.object_max_yaw_deg),
        "--max-scale-delta",
        str(args.object_max_scale_delta),
    ]
    step_start = timing_start()
    object_opt_stdout = run_command(object_opt_cmd, cwd=WORKFLOW_ROOT)
    timings["object_pose_optimization"] = timing_elapsed(step_start)
    step_start = timing_start()
    apply_stdout = run_command(
        [
            args.blender_bin,
            "-b",
            "--python",
            str(WORKFLOW_ROOT / "scripts" / "blender_apply_object_pose_optimization.py"),
            "--",
            "--input",
            str(grounded_blend),
            "--object-pose-optimization",
            str(object_report),
            "--output",
            str(optimized_blend),
            "--pack-textures",
        ],
        cwd=WORKFLOW_ROOT,
    )
    timings["apply_object_pose_optimization"] = timing_elapsed(step_start)
    timings["total"] = timing_elapsed(total_start)

    outputs = {
        "blend": str(optimized_blend),
        "manifest": str(manifest_path),
        "camera_mesh": str(camera_mesh),
        "camera_report": str(camera_report),
        "camera_preview": str(camera_preview),
        "object_mesh": str(object_mesh),
        "object_report": str(object_report),
        "object_preview_dir": str(object_preview_dir),
        "apply_report": str(optimized_blend.with_suffix(".object_pose_apply_report.json")),
    }
    report = {
        "status": "ok",
        "stage": {
            "key": "optimized",
            "description": "Differentiable rendering refinement after rotated->grounded and before scene-graph support/collision separation.",
        },
        "input": {"blend": str(grounded_blend), "image": str(image_path), "mask": str(mask_path)},
        "outputs": outputs,
        "camera_pose_optimization": {"mesh": str(camera_mesh), "report": str(camera_report), "preview": str(camera_preview)},
        "object_pose_optimization": {
            "mesh": str(object_mesh),
            "report": str(object_report),
            "preview_dir": str(object_preview_dir),
            "optimized_blend": str(optimized_blend),
            "apply_report": str(optimized_blend.with_suffix(".object_pose_apply_report.json")),
        },
        "optimization_order": ["camera_extrinsics", "object_translation", "object_yaw_z", "object_uniform_scale"],
        "timings": timings,
        "stdout": {
            "camera_export": camera_export_stdout,
            "camera_optimization": camera_opt_stdout,
            "object_export": object_export_stdout,
            "object_optimization": object_opt_stdout,
            "apply_object_optimization": apply_stdout,
        },
    }
    report_path = output_dir / f"{prefix}_report.json"
    outputs["report"] = str(report_path)
    report["outputs"] = outputs
    report_path.write_text(json.dumps(numpy_to_json(report), ensure_ascii=False, indent=2), encoding="utf-8")
    return outputs, report


def run(args: argparse.Namespace) -> dict[str, Any]:
    total_start = timing_start()
    top_timings: dict[str, dict[str, float]] = {}
    image_path = args.image.resolve()
    blend_path = args.blend.resolve()
    mask_path = args.mask.resolve() if args.mask else None
    ground_mask_path = args.ground_mask.resolve() if args.ground_mask else None
    output_dir = args.output_dir.resolve() if args.output_dir else blend_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.moge_cache:
        step_start = timing_start()
        moge = load_moge_cache(args.moge_cache.resolve())
        top_timings["moge_load_cache"] = timing_elapsed(step_start)
    else:
        step_start = timing_start()
        moge = infer_moge(image_path, device=args.device, checkpoint=args.checkpoint, resize_max=args.resize_max, resolution_level=args.resolution_level, num_tokens=args.num_tokens, use_fp16=not args.no_fp16)
        top_timings["moge_infer"] = timing_elapsed(step_start)
    h, w = moge["mask"].shape
    foreground_mask = load_mask(mask_path, (h, w))
    ground_mask = load_mask(ground_mask_path, (h, w))
    coordinate_system = str(moge.get("coordinate_system", "opencv_camera"))
    target_up = axis_vector(args.scene_up_axis)
    scene_to_blender = scene_to_blender_matrix(args.scene_to_blender_transform)
    if coordinate_system.startswith("pytorch3d_scene") or coordinate_system.startswith("sam3d_scene"):
        raw_source_up_scene, ground_report = estimate_ground_normal_scene(moge["points"], moge.get("normal"), moge["mask"], foreground_mask, ground_mask, args)
        ground_normal_cv = None
    else:
        ground_normal_cv, ground_report = estimate_ground_normal_cv(moge["points"], moge.get("normal"), moge["mask"], foreground_mask, ground_mask, args)
        raw_source_up_scene = cv_to_pytorch3d_scene_normal(ground_normal_cv)
    raw_source_up_blender = transform_direction(scene_to_blender, raw_source_up_scene, name="source up in Blender coordinates")
    source_up_flip_for_target = float(np.dot(raw_source_up_blender, target_up)) < 0.0
    source_up_scene = -raw_source_up_scene if source_up_flip_for_target else raw_source_up_scene
    source_up_blender = -raw_source_up_blender if source_up_flip_for_target else raw_source_up_blender
    if ground_normal_cv is not None and source_up_flip_for_target:
        ground_normal_cv = -ground_normal_cv
    ground_debug_dir, ground_debug_report = write_ground_normal_debug(
        output_dir=output_dir,
        image_path=image_path,
        moge=moge,
        foreground_mask=foreground_mask,
        ground_mask=ground_mask,
        ground_mask_path=ground_mask_path,
        coordinate_system=coordinate_system,
        ground_report=ground_report,
        ground_normal_cv=ground_normal_cv,
        raw_source_up_scene=raw_source_up_scene,
        source_up_scene=source_up_scene,
        raw_source_up_blender=raw_source_up_blender,
        source_up_blender=source_up_blender,
        target_up=target_up,
        scene_to_blender=scene_to_blender,
        source_up_flip_for_target=source_up_flip_for_target,
        args=args,
    )
    compact_ground_report = compact_ground_report_for_json(ground_report)
    moge_report = {
        "root": str(MOGE_ROOT),
        "checkpoint": str(args.checkpoint),
        "cache": str(args.moge_cache.resolve()) if args.moge_cache else None,
        "manual_ground_mask": str(ground_mask_path) if ground_mask_path else None,
        "cache_source": moge.get("source"),
        "coordinate_system": coordinate_system,
        "scene_up_axis": args.scene_up_axis,
        "device": args.device,
        "resize_max": int(args.resize_max),
        "resolution_level": int(args.resolution_level),
        "num_tokens": args.num_tokens,
        "resize_scale": float(moge["resize_scale"]),
        "intrinsics": moge["intrinsics"],
        "opencv_camera_to_pytorch3d_scene_matrix": OPENCV_CAMERA_TO_PYTORCH3D_SCENE,
        "scene_to_blender_transform": args.scene_to_blender_transform,
        "scene_to_blender_matrix": scene_to_blender,
        "ground_normal_cv": ground_normal_cv,
        "raw_source_up_scene": raw_source_up_scene,
        "source_up_scene": source_up_scene,
        "raw_source_up_blender": raw_source_up_blender,
        "source_up_blender": source_up_blender,
        "source_up_blender_flipped_to_target": source_up_flip_for_target,
        "ground_normal_debug_dir": str(ground_debug_dir),
        "ground_normal_debug": ground_debug_report,
        **compact_ground_report,
    }
    input_report = {"image": str(image_path), "mask": str(mask_path) if mask_path else None, "blend": str(blend_path)}
    with tempfile.TemporaryDirectory(prefix="sam3d_moge_upright_") as tmp_dir:
        tmp_glb = Path(tmp_dir) / "input.glb"
        step_start = timing_start()
        blend_to_glb(blend_path, tmp_glb, args.blender_bin)
        loaded = trimesh.load(tmp_glb, force="scene", process=False)
        top_timings["blend_to_trimesh_scene"] = timing_elapsed(step_start)
        if isinstance(loaded, trimesh.Trimesh):
            scene = trimesh.Scene(loaded)
        elif isinstance(loaded, trimesh.Scene):
            scene = loaded
        else:
            raise TypeError(f"unsupported scene type: {type(loaded)!r}")
        stage_configs = [
            {
                "key": "rotated",
                "description": "Global upright stage. MoGe/SAM3D scene ground normal is first converted to Blender Z-up coordinates, then globally rotated to Blender +Z.",
                "snap_bbox_to_ground": False,
                "separate_overlaps": False,
                "bbox_snap_mode": "none",
                "support_adjust": False,
                "support_require_scene_graph": False,
                "apply_upright": True,
                "input_stage": None,
            },
            {
                "key": "grounded",
                "description": "Use the upright Blender Z-up scene, then apply one shared Blender +Z translation to place the whole scene on ground_y. This does not independently snap each object to the floor.",
                "snap_bbox_to_ground": True,
                "separate_overlaps": False,
                "bbox_snap_mode": "global_support_aware",
                "support_adjust": False,
                "support_require_scene_graph": False,
                "apply_upright": False,
                "input_stage": "rotated",
            },
            {
                "key": "separated",
                "description": "Use the differentiable-rendering optimized grounded scene, then apply scene-graph support-aware convex vertical correction plus collision-aware separation. BBOX overlap is only a broad-phase candidate filter; convex hull SAT collision is the default narrow phase before any separation.",
                "snap_bbox_to_ground": False,
                "separate_overlaps": True,
                "bbox_snap_mode": "global_support_aware",
                "support_adjust": True,
                "support_require_scene_graph": True,
                "apply_upright": False,
                "input_stage": "optimized",
            },
        ]
        stage_outputs: dict[str, dict[str, str]] = {}
        stage_reports: dict[str, dict[str, Any]] = {}
        stage_report_paths: dict[str, Path] = {}
        for cfg in stage_configs:
            stage_start = timing_start()
            apply_upright = bool(cfg.get("apply_upright", True))
            transformed, transform_report = transform_scene(
                scene,
                source_up_blender if apply_upright else target_up,
                target_up,
                args.ground_y,
                translate_scene_to_ground=False,
                snap_bbox_to_ground=bool(cfg["snap_bbox_to_ground"]),
                separate_overlaps=False,
                overlap_margin=args.bbox_overlap_margin,
                overlap_iters=args.bbox_overlap_iters,
                min_overlap_volume_ratio=args.bbox_min_overlap_volume_ratio,
                overlap_eps=args.bbox_overlap_eps,
                rotation_mode=args.rotation_mode if apply_upright else "none",
            )
            transform_report["bbox_separate_overlaps"] = bool(cfg["separate_overlaps"])
            transform_report["bbox_overlap_reports"] = []
            transform_report["ground_y"] = float(args.ground_y)
            transform_report["source_up_coordinate_system"] = "blender"
            transform_report["source_up_moge_scene"] = source_up_scene
            transform_report["scene_to_blender_transform"] = args.scene_to_blender_transform
            transform_report["scene_to_blender_matrix"] = scene_to_blender
            transform_report["bbox_snap_mode"] = str(cfg["bbox_snap_mode"])
            transform_report["support_adjust"] = bool(cfg["support_adjust"])
            transform_report["support_require_scene_graph"] = bool(cfg.get("support_require_scene_graph", False))
            transform_report["support_gap"] = float(args.support_gap)
            transform_report["support_xy_overlap_ratio"] = float(args.support_xy_overlap_ratio)
            transform_report["support_max_gap"] = float(args.support_max_gap)
            transform_report["support_max_gap_ratio"] = float(args.support_max_gap_ratio)
            transform_report["support_max_penetration"] = float(args.support_max_penetration)
            transform_report["support_max_penetration_ratio"] = float(args.support_max_penetration_ratio)
            transform_report["support_min_lower_area_ratio"] = float(args.support_min_lower_area_ratio)
            transform_report["scene_graph"] = str(args.scene_graph.resolve()) if args.scene_graph else None
            transform_report["overlap_collision_method"] = str(args.overlap_collision_method)
            transform_report["support_collision_method"] = str(args.overlap_collision_method)
            transform_report["convex_hull_max_vertices"] = int(args.convex_hull_max_vertices)
            transform_report["convex_decomposition_method"] = str(args.convex_decomposition_method)
            transform_report["coacd_conda_bin"] = str(args.coacd_conda_bin)
            transform_report["coacd_env"] = str(args.coacd_env)
            transform_report["coacd_timeout"] = int(args.coacd_timeout)
            transform_report["coacd_workers"] = int(args.coacd_workers)
            transform_report["coacd_source_max_faces"] = int(args.coacd_source_max_faces)
            transform_report["coacd_source_simplification_backend"] = str(args.coacd_source_simplification_backend)
            transform_report["coacd_source_simplification_agg"] = float(args.coacd_source_simplification_agg)
            transform_report["coacd_blender_source_decimate"] = bool(args.coacd_blender_source_decimate)
            transform_report["coacd_max_convex_parts"] = int(args.coacd_max_convex_parts)
            transform_report["coacd_threshold"] = float(args.coacd_threshold)
            transform_report["coacd_preprocess_mode"] = str(args.coacd_preprocess_mode)
            transform_report["coacd_preprocess_resolution"] = int(args.coacd_preprocess_resolution)
            transform_report["coacd_resolution"] = int(args.coacd_resolution)
            transform_report["coacd_mcts_nodes"] = int(args.coacd_mcts_nodes)
            transform_report["coacd_mcts_iterations"] = int(args.coacd_mcts_iterations)
            transform_report["coacd_mcts_max_depth"] = int(args.coacd_mcts_max_depth)
            transform_report["coacd_max_ch_vertex"] = int(args.coacd_max_ch_vertex)
            transform_report["coacd_apx_mode"] = str(args.coacd_apx_mode)
            transform_report["coacd_seed"] = int(args.coacd_seed)
            transform_report["coacd_real_metric"] = bool(args.coacd_real_metric)
            transform_report["coacd_merge"] = bool(args.coacd_merge)
            transform_report["coacd_decimate"] = bool(args.coacd_decimate)
            transform_report["convex_collision_eps"] = float(args.convex_collision_eps)
            transform_report["convex_min_horizontal_axis"] = float(args.convex_min_horizontal_axis)
            input_stage = cfg.get("input_stage")
            input_blend_for_stage = blend_path
            if input_stage:
                input_stage_outputs = stage_outputs.get(str(input_stage)) or {}
                input_blend_for_stage = Path(str(input_stage_outputs.get("blend") or blend_path))
            stage_input_report = dict(input_report)
            stage_input_report["blend"] = str(input_blend_for_stage)
            stage_report = {
                "status": "ok",
                "stage": {"key": cfg["key"], "description": cfg["description"]},
                "input": stage_input_report,
                "moge": moge_report,
                "transform": transform_report,
                "note": "This stage is applied directly to the source Blend object transforms to preserve Blender materials and image textures.",
            }
            _ = transformed
            outputs, stage_report = export_stage_preserve_blend(
                input_blend=input_blend_for_stage,
                output_dir=output_dir,
                output_name=f"{args.output_name}_{cfg['key']}",
                stage_report=stage_report,
                blender_bin=args.blender_bin,
            )
            stage_report["timing"] = timing_elapsed(stage_start)
            if outputs.get("report"):
                Path(outputs["report"]).write_text(
                    json.dumps(numpy_to_json(stage_report), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            stage_outputs[str(cfg["key"])] = outputs
            stage_reports[str(cfg["key"])] = stage_report
            if outputs.get("report"):
                stage_report_paths[str(cfg["key"])] = Path(outputs["report"])
            if str(cfg["key"]) == "grounded":
                try:
                    optimized_start = timing_start()
                    optimized_outputs, optimized_report = optimize_grounded_stage(
                        args=args,
                        output_dir=output_dir,
                        grounded_blend=Path(outputs["blend"]),
                        grounded_report=Path(outputs["report"]),
                        rotated_report=stage_report_paths["rotated"],
                        image_path=image_path,
                        mask_path=mask_path,
                    )
                    optimized_report.setdefault("timing", timing_elapsed(optimized_start))
                    stage_outputs["optimized"] = optimized_outputs
                    stage_reports["optimized"] = optimized_report
                    if optimized_outputs.get("report"):
                        Path(optimized_outputs["report"]).write_text(
                            json.dumps(numpy_to_json(optimized_report), ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                    if optimized_outputs.get("report"):
                        stage_report_paths["optimized"] = Path(optimized_outputs["report"])
                except Exception as exc:
                    stage_outputs["optimized"] = {
                        "blend": outputs.get("blend"),
                        "report": outputs.get("report"),
                        "fallback_reason": "differentiable_optimization_failed",
                    }
                    stage_reports["optimized"] = {
                        "status": "error",
                        "stage": {"key": "optimized"},
                        "input": {"blend": outputs.get("blend")},
                        "error": str(exc),
                        "note": "Differentiable optimization failed; separated stage will fall back to grounded input if needed.",
                    }
        rotated_report_path = stage_report_paths.get("rotated")
        if rotated_report_path is not None:
            optimized_camera_report: Path | None = None
            optimized_stage_report = stage_reports.get("optimized") or {}
            optimized_camera_report_value = (((optimized_stage_report.get("camera_pose_optimization") or {}).get("report")))
            if optimized_camera_report_value:
                candidate = Path(str(optimized_camera_report_value))
                if candidate.is_file():
                    optimized_camera_report = candidate
            for stage_name, outputs in stage_outputs.items():
                blend_value = outputs.get("blend")
                stage_report_path = stage_report_paths.get(stage_name)
                if not blend_value or stage_report_path is None:
                    continue
                try:
                    camera_outputs, camera_data, camera_stdout = add_original_input_camera_to_stage(
                        blend_path=Path(blend_value),
                        stage_report_path=stage_report_path,
                        camera_report_path=rotated_report_path,
                        camera_optimization_path=optimized_camera_report if stage_name in {"optimized", "separated"} else None,
                        blender_bin=args.blender_bin,
                    )
                    outputs.update(camera_outputs)
                    stage_report = stage_reports.get(stage_name) or {}
                    stage_report.setdefault("outputs", {}).update(camera_outputs)
                    stage_report["original_input_camera"] = camera_data
                    stage_report["original_input_camera_stdout"] = camera_stdout
                    stage_reports[stage_name] = stage_report
                    stage_report_path.write_text(
                        json.dumps(numpy_to_json(stage_report), ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                except Exception as exc:
                    stage_report = stage_reports.get(stage_name) or {}
                    stage_report["original_input_camera_error"] = str(exc)
                    stage_reports[stage_name] = stage_report
                    stage_report_path.write_text(
                        json.dumps(numpy_to_json(stage_report), ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
    report_json = unique_report_path(output_dir, args.output_name)
    top_timings["total"] = timing_elapsed(total_start)
    report = {
        "status": "ok",
        "input": input_report,
        "outputs": {"report": str(report_json), "stages": stage_outputs},
        "moge": moge_report,
        "stages": stage_reports,
        "timings": top_timings,
        "note": "This aggregate report records SAM3D MoGe postprocess stages in order: rotated, grounded, optimized, separated. Steps 1-3 preserve the original SAM3D/MoGe behavior; from step 4 onward the grounded scene is refined by differentiable rendering before scene-graph support/collision separation.",
    }
    report_json.write_text(json.dumps(numpy_to_json(report), ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Use MoGe monocular geometry to create rotated, grounded, differentiable-rendering optimized, and support/collision-separated SAM3D Blend scenes.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--mask", type=Path, default=None)
    parser.add_argument("--ground-mask", type=Path, default=None, help="Optional binary manual ground mask. Used only for MoGe ground normal estimation, not for object geometry generation.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-name", default="sam3d_moge")
    parser.add_argument("--checkpoint", type=Path, default=MOGE_CKPT)
    parser.add_argument("--moge-cache", type=Path, default=None, help="Reuse SAM3D-exported MoGe pointmap cache instead of loading MoGe again.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resize-max", type=int, default=1280)
    parser.add_argument("--resolution-level", type=int, default=6)
    parser.add_argument("--num-tokens", type=int, default=None)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--bottom-fraction", type=float, default=0.45)
    parser.add_argument("--max-ground-samples", type=int, default=50000)
    parser.add_argument("--normal-agreement-deg", type=float, default=35.0)
    parser.add_argument("--normal-weight", type=float, default=0.55)
    parser.add_argument("--ground-y", type=float, default=0.0)
    parser.add_argument("--scene-up-axis", choices=sorted(AXIS_TO_VECTOR), default="z", help="Target up axis for grounding. Downstream Blender/SAPIEN stages are Z-up.")
    parser.add_argument("--scene-to-blender-transform", choices=sorted(SCENE_TO_BLENDER_TRANSFORMS), default="pytorch3d_y_up_to_blender_z_up", help="Fixed axis transform from MoGe/SAM3D scene coordinates to Blender coordinates before computing the upright rotation.")
    parser.add_argument("--rotation-mode", choices=["none", "upright"], default="upright", help="Use 'upright' to rotate the MoGe-estimated ground normal, after scene-to-Blender axis conversion, to Blender +Z. Use 'none' only for offline debugging.")
    parser.add_argument("--snap-bbox-to-ground", action="store_true")
    parser.add_argument("--separate-overlapping-bboxes", action="store_true")
    parser.add_argument("--bbox-overlap-margin", type=float, default=0.01)
    parser.add_argument("--bbox-overlap-iters", type=int, default=32)
    parser.add_argument("--bbox-min-overlap-volume-ratio", type=float, default=0.0)
    parser.add_argument("--bbox-overlap-eps", type=float, default=1e-8)
    parser.add_argument("--overlap-collision-method", choices=["bbox", "convex_hull_sat"], default="convex_hull_sat", help="Narrow-phase method for separated-stage overlap resolution. BBOX remains the broad-phase candidate filter.")
    parser.add_argument("--convex-hull-max-vertices", type=int, default=8000, help="Maximum sampled world vertices per object before Blender convex-hull generation.")
    parser.add_argument("--convex-decomposition-method", choices=["single_convex_hull", "coacd"], default="coacd", help="Convex geometry source used by convex_hull_sat. CoACD checks multiple convex parts; single_convex_hull preserves the old behavior.")
    parser.add_argument("--coacd-conda-bin", default=os.environ.get("CONDA_BIN", "conda"))
    parser.add_argument("--coacd-env", default=os.environ.get("FYSIVERSE_CONDA_ENV", "fysiverse-scene"))
    parser.add_argument("--coacd-timeout", type=int, default=120)
    parser.add_argument("--coacd-workers", type=int, default=4)
    parser.add_argument("--coacd-source-max-faces", type=int, default=5000)
    parser.add_argument("--coacd-source-simplification-backend", choices=["fast_simplification", "none"], default="fast_simplification")
    parser.add_argument("--coacd-source-simplification-agg", type=float, default=7.0)
    parser.add_argument("--coacd-max-convex-parts", type=int, default=8)
    parser.add_argument("--coacd-threshold", type=float, default=0.05)
    parser.add_argument("--coacd-preprocess-mode", default="auto")
    parser.add_argument("--coacd-preprocess-resolution", type=int, default=50)
    parser.add_argument("--coacd-resolution", type=int, default=2000)
    parser.add_argument("--coacd-mcts-nodes", type=int, default=20)
    parser.add_argument("--coacd-mcts-iterations", type=int, default=80)
    parser.add_argument("--coacd-mcts-max-depth", type=int, default=3)
    parser.add_argument("--coacd-max-ch-vertex", type=int, default=256)
    parser.add_argument("--coacd-apx-mode", default="ch")
    parser.add_argument("--coacd-seed", type=int, default=0)
    parser.add_argument("--coacd-real-metric", action="store_true")
    parser.add_argument("--coacd-merge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--coacd-decimate", action="store_true")
    parser.add_argument("--coacd-blender-source-decimate", action="store_true")
    parser.add_argument("--convex-collision-eps", type=float, default=1e-6, help="SAT tolerance for convex hull collision checks.")
    parser.add_argument("--convex-min-horizontal-axis", type=float, default=0.25, help="Minimum horizontal component of the convex MTV required for horizontal separation; mostly vertical collisions are left to support/gravity.")
    parser.add_argument("--support-gap", type=float, default=0.001, help="Blender Z-up clearance added after support-pair convex SAT vertical separation.")
    parser.add_argument("--support-xy-overlap-ratio", type=float, default=0.15, help="Minimum XY bbox overlap ratio relative to the upper object for support inference.")
    parser.add_argument("--support-max-gap", type=float, default=-1.0, help="Maximum existing vertical gap for a support candidate. Negative means support-max-gap-ratio * scene height.")
    parser.add_argument("--support-max-gap-ratio", type=float, default=0.03)
    parser.add_argument("--support-max-penetration", type=float, default=-1.0, help="Maximum allowed vertical penetration for a support candidate. Negative means support-max-penetration-ratio * scene height.")
    parser.add_argument("--support-max-penetration-ratio", type=float, default=0.01)
    parser.add_argument("--support-min-lower-area-ratio", type=float, default=0.25, help="Lower/supporter XY area must be at least this ratio of upper object XY area.")
    parser.add_argument("--scene-graph", type=Path, default=None, help="Optional mask-scoped VLM scene graph JSON. Support relations in this file override bbox support inference.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--blender-bin", default=BLENDER_BIN)
    parser.add_argument("--conda-bin", default=os.environ.get("CONDA_BIN", "conda"))
    parser.add_argument("--nvdiffrast-env", default=os.environ.get("FYSIVERSE_CONDA_ENV", "fysiverse-scene"))
    parser.add_argument("--camera-opt-steps", type=int, default=300)
    parser.add_argument("--camera-opt-max-side", type=int, default=512)
    parser.add_argument("--camera-opt-rot-lr", type=float, default=0.015)
    parser.add_argument("--camera-opt-trans-lr", type=float, default=0.01)
    parser.add_argument("--object-pose-workers", type=int, default=0, help="<=0 means one worker per object.")
    parser.add_argument("--object-pose-max-side", type=int, default=512)
    parser.add_argument("--object-position-steps", type=int, default=120)
    parser.add_argument("--object-yaw-steps", type=int, default=80)
    parser.add_argument("--object-scale-steps", type=int, default=60)
    parser.add_argument("--object-translation-mode", choices=["camera_plane", "world_xy", "world_xyz"], default="camera_plane")
    parser.add_argument("--object-translation-initial-grid", type=int, default=3)
    parser.add_argument("--object-center-weight", type=float, default=0.15)
    parser.add_argument("--object-area-weight", type=float, default=0.05)
    parser.add_argument("--object-yaw-initial-samples", type=int, default=5)
    parser.add_argument("--object-scale-initial-samples", type=int, default=5)
    parser.add_argument("--object-phase-accept-iou-drop", type=float, default=0.01)
    parser.add_argument("--object-position-lr", type=float, default=0.01)
    parser.add_argument("--object-yaw-lr", type=float, default=0.02)
    parser.add_argument("--object-scale-lr", type=float, default=0.01)
    parser.add_argument("--object-max-translation", type=float, default=0.35)
    parser.add_argument("--object-max-yaw-deg", type=float, default=35.0)
    parser.add_argument("--object-max-scale-delta", type=float, default=0.25)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run(args)
    outputs = report["outputs"]
    print(f"Wrote report: {outputs['report']}")
    for stage_name, stage_outputs in (outputs.get("stages") or {}).items():
        transform = ((report.get("stages") or {}).get(stage_name) or {}).get("transform") or {}
        angle = transform.get("rotation_angle_deg")
        angle_text = f", angle={angle:.3f} deg" if isinstance(angle, (int, float)) else ""
        print(f"Wrote {stage_name} BLEND: {stage_outputs.get('blend')}{angle_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
