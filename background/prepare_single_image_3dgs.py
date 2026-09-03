#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOGE_ROOT = Path(os.environ.get("MOGE_ROOT", WORKFLOW_ROOT / "third_party" / "MoGe"))
DEFAULT_MOGE_CKPT = Path(os.environ.get("MOGE_CKPT", WORKFLOW_ROOT / "checkpoints" / "moge-2-vitl-normal" / "model.pt"))
OPENCV_TO_PYTORCH3D = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
OPENCV_TO_BLENDER_CAMERA = np.diag([1.0, -1.0, -1.0]).astype(np.float32)


def resize_image(image_rgb: np.ndarray, resize_max: int) -> tuple[np.ndarray, float]:
    if resize_max <= 0:
        return image_rgb, 1.0
    height, width = image_rgb.shape[:2]
    scale = min(1.0, float(resize_max) / float(max(height, width)))
    if scale >= 1.0:
        return image_rgb, 1.0
    out = cv2.resize(
        image_rgb,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return out, scale


def read_image_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def write_image_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR))


def load_binary_mask(path: Path | None, size_hw: tuple[int, int]) -> np.ndarray | None:
    if path is None:
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    height, width = size_hw
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def resize_array(array: np.ndarray, size_hw: tuple[int, int], interpolation: int) -> np.ndarray:
    height, width = size_hw
    array = np.asarray(array)
    if array.shape[:2] == (height, width):
        return array.copy()
    return cv2.resize(array, (width, height), interpolation=interpolation)


def dilate_binary_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return np.asarray(mask, dtype=bool)
    radius = int(pixels)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(np.asarray(mask, dtype=np.uint8), kernel, iterations=1) > 0


def depth_to_opencv_points(depth: np.ndarray, intrinsics_px: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    intrinsics_px = np.asarray(intrinsics_px, dtype=np.float32).reshape(3, 3)
    height, width = depth.shape[:2]
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    fx = max(float(intrinsics_px[0, 0]), 1e-8)
    fy = max(float(intrinsics_px[1, 1]), 1e-8)
    cx = float(intrinsics_px[0, 2])
    cy = float(intrinsics_px[1, 2])
    z = depth.astype(np.float32)
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def numpy_to_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return numpy_to_json(value.tolist())
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): numpy_to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [numpy_to_json(item) for item in value]
    return value


def infer_moge(
    image_path: Path,
    *,
    moge_root: Path,
    checkpoint: Path,
    device: str,
    resize_max: int,
    resolution_level: int,
    num_tokens: int | None,
    use_fp16: bool,
) -> dict[str, Any]:
    if str(moge_root) not in sys.path:
        sys.path.insert(0, str(moge_root))

    import torch
    from moge.model.v2 import MoGeModel

    image_rgb = read_image_rgb(image_path)
    image_rgb, resize_scale = resize_image(image_rgb, resize_max)

    model = MoGeModel.from_pretrained(str(checkpoint)).to(torch.device(device)).eval()
    input_image = torch.tensor(image_rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
    with torch.no_grad():
        output = model.infer(
            input_image,
            resolution_level=resolution_level,
            num_tokens=num_tokens,
            use_fp16=use_fp16,
        )

    result = {
        "image": image_rgb,
        "points": output["points"].detach().cpu().numpy().astype(np.float32),
        "depth": output["depth"].detach().cpu().numpy().astype(np.float32),
        "mask": output["mask"].detach().cpu().numpy().astype(bool),
        "intrinsics": output["intrinsics"].detach().cpu().numpy().astype(np.float32),
        "coordinate_system": "opencv_camera",
        "resize_scale": float(resize_scale),
        "source": "moge_v2_infer",
    }
    if "normal" in output:
        result["normal"] = output["normal"].detach().cpu().numpy().astype(np.float32)
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def load_moge_cache(cache_path: Path, fallback_image_path: Path) -> dict[str, Any]:
    data = np.load(cache_path, allow_pickle=False)
    points = data["points"] if "points" in data else data["pointmap"] if "pointmap" in data else data["pcd"]
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"invalid MoGe point map shape: {points.shape}")

    mask = np.asarray(data["mask"], dtype=bool) if "mask" in data else np.isfinite(points).all(axis=-1)
    if mask.shape != points.shape[:2]:
        mask = np.isfinite(points).all(axis=-1)

    if "image" in data:
        image_rgb = np.asarray(data["image"], dtype=np.uint8)[..., :3]
    else:
        image_rgb = read_image_rgb(fallback_image_path)
        if image_rgb.shape[:2] != points.shape[:2]:
            image_rgb = cv2.resize(image_rgb, (points.shape[1], points.shape[0]), interpolation=cv2.INTER_AREA)

    coordinate_system = str(data["coordinate_system"]) if "coordinate_system" in data else "opencv_camera"
    if coordinate_system.startswith(("pytorch3d_scene", "sam3d_scene")):
        points = points @ OPENCV_TO_PYTORCH3D.T
        coordinate_system = f"opencv_camera_from_{coordinate_system}"

    return {
        "image": image_rgb,
        "points": points.astype(np.float32),
        "depth": np.asarray(data["depth"], dtype=np.float32) if "depth" in data else points[..., 2].astype(np.float32),
        "mask": mask,
        "intrinsics": np.asarray(data["intrinsics"], dtype=np.float32) if "intrinsics" in data else None,
        "coordinate_system": coordinate_system,
        "resize_scale": 1.0,
        "source": f"moge_cache:{cache_path}",
    }


def pixel_intrinsics(intrinsics: np.ndarray | None, width: int, height: int) -> np.ndarray:
    if intrinsics is None:
        # Conservative fallback: 60 degree horizontal FOV.
        fx = fy = 0.5 * float(width) / math.tan(math.radians(60.0) * 0.5)
        cx = 0.5 * float(width)
        cy = 0.5 * float(height)
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    intrinsics = np.asarray(intrinsics, dtype=np.float32).reshape(3, 3)
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    if max(abs(fx), abs(fy), abs(cx), abs(cy)) <= 4.0:
        return np.array(
            [
                [fx * float(width), 0.0, cx * float(width)],
                [0.0, fy * float(height), cy * float(height)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    return intrinsics.astype(np.float32)


def pixel_intrinsics_from_camera_json(camera_data: dict[str, Any], width: int, height: int) -> np.ndarray | None:
    normalized = camera_data.get("intrinsics_normalized")
    if normalized is not None:
        return pixel_intrinsics(np.asarray(normalized, dtype=np.float32), width, height)

    pixels = camera_data.get("intrinsics_pixels")
    if pixels is None:
        return None
    intrinsics = np.asarray(pixels, dtype=np.float32).reshape(3, 3)
    image_size = camera_data.get("image_size") or camera_data.get("source_image_size")
    if isinstance(image_size, (list, tuple)) and len(image_size) >= 2:
        src_w, src_h = float(image_size[0]), float(image_size[1])
        if src_w > 0 and src_h > 0:
            sx = float(width) / src_w
            sy = float(height) / src_h
            intrinsics = intrinsics.copy()
            intrinsics[0, 0] *= sx
            intrinsics[0, 2] *= sx
            intrinsics[1, 1] *= sy
            intrinsics[1, 2] *= sy
    return intrinsics.astype(np.float32)


def fit_depth_alignment(
    reference_depth: np.ndarray,
    edited_depth: np.ndarray,
    stable_mask: np.ndarray,
    *,
    mode: str,
    seed: int,
    max_samples: int = 200000,
) -> tuple[np.ndarray, dict[str, Any]]:
    reference_depth = np.asarray(reference_depth, dtype=np.float32)
    edited_depth = np.asarray(edited_depth, dtype=np.float32)
    stable = (
        np.asarray(stable_mask, dtype=bool)
        & np.isfinite(reference_depth)
        & np.isfinite(edited_depth)
        & (reference_depth > 1e-6)
        & (edited_depth > 1e-6)
    )
    ref = reference_depth[stable].astype(np.float64)
    edit = edited_depth[stable].astype(np.float64)
    total = int(ref.size)
    if total > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(total, size=max_samples, replace=False)
        ref = ref[idx]
        edit = edit[idx]

    if ref.size < 100:
        aligned = edited_depth.astype(np.float32)
        return aligned, {
            "mode": mode,
            "status": "insufficient_overlap",
            "overlap_pixels": total,
            "scale": 1.0,
            "shift": 0.0,
        }

    ratios = ref / np.maximum(edit, 1e-8)
    finite_ratio = np.isfinite(ratios) & (ratios > 0.0)
    ratios = ratios[finite_ratio]
    ref = ref[finite_ratio]
    edit = edit[finite_ratio]
    if ratios.size < 100:
        aligned = edited_depth.astype(np.float32)
        return aligned, {
            "mode": mode,
            "status": "insufficient_finite_ratios",
            "overlap_pixels": total,
            "scale": 1.0,
            "shift": 0.0,
        }

    lo, hi = np.percentile(ratios, [5.0, 95.0])
    keep = (ratios >= lo) & (ratios <= hi)
    if int(np.count_nonzero(keep)) >= 100:
        ratios = ratios[keep]
        ref = ref[keep]
        edit = edit[keep]

    scale = float(np.median(ratios))
    shift = 0.0
    status = "scale"
    if mode == "none":
        scale = 1.0
        shift = 0.0
        status = "none"
    elif mode == "scale_shift":
        design = np.stack([edit, np.ones_like(edit)], axis=1)
        solution, *_ = np.linalg.lstsq(design, ref, rcond=None)
        candidate_scale = float(solution[0])
        candidate_shift = float(solution[1])
        median_ref = float(np.median(ref))
        sane = (
            np.isfinite(candidate_scale)
            and np.isfinite(candidate_shift)
            and candidate_scale > 0.0
            and abs(candidate_shift) <= max(0.5 * abs(median_ref), 1e-6)
        )
        if sane:
            scale = candidate_scale
            shift = candidate_shift
            status = "scale_shift"
        else:
            status = "scale_shift_rejected_used_scale"

    aligned = (edited_depth.astype(np.float32) * np.float32(scale) + np.float32(shift)).astype(np.float32)
    residual = ref - (edit * float(scale) + float(shift))
    return aligned, {
        "mode": mode,
        "status": status,
        "overlap_pixels": total,
        "used_samples": int(ref.size),
        "scale": float(scale),
        "shift": float(shift),
        "ratio_percentiles": [float(lo), float(hi)],
        "residual_median_abs": float(np.median(np.abs(residual))),
        "residual_p95_abs": float(np.percentile(np.abs(residual), 95.0)),
    }


def fuse_reference_and_edited_moge(
    *,
    edited_moge: dict[str, Any],
    reference_moge: dict[str, Any],
    foreground_mask: np.ndarray,
    intrinsics_px: np.ndarray,
    depth_align_mode: str,
    foreground_mask_dilate: int,
    seed: int,
) -> dict[str, Any]:
    image_rgb = np.asarray(edited_moge["image"], dtype=np.uint8)[..., :3]
    height, width = image_rgb.shape[:2]
    size_hw = (height, width)

    edited_points = np.asarray(edited_moge["points"], dtype=np.float32)
    edited_valid = np.asarray(edited_moge["mask"], dtype=bool)
    edited_depth = edited_points[..., 2].astype(np.float32)
    if edited_depth.shape != (height, width):
        edited_depth = resize_array(edited_depth, size_hw, cv2.INTER_AREA).astype(np.float32)
        edited_valid = resize_array(edited_valid.astype(np.uint8), size_hw, cv2.INTER_NEAREST) > 0

    reference_depth = np.asarray(reference_moge["points"], dtype=np.float32)[..., 2]
    reference_valid = np.asarray(reference_moge["mask"], dtype=bool)
    reference_depth = resize_array(reference_depth, size_hw, cv2.INTER_AREA).astype(np.float32)
    reference_valid = resize_array(reference_valid.astype(np.uint8), size_hw, cv2.INTER_NEAREST) > 0

    fill_mask = dilate_binary_mask(foreground_mask, foreground_mask_dilate)
    stable_mask = (~fill_mask) & reference_valid & edited_valid
    aligned_edited_depth, alignment = fit_depth_alignment(
        reference_depth,
        edited_depth,
        stable_mask,
        mode=depth_align_mode,
        seed=seed,
    )

    edited_fill_valid = edited_valid & np.isfinite(aligned_edited_depth) & (aligned_edited_depth > 1e-6)
    fused_valid = np.where(fill_mask, edited_fill_valid, reference_valid)
    fused_depth = np.where(fill_mask, aligned_edited_depth, reference_depth).astype(np.float32)
    fused_depth[~fused_valid] = np.nan
    fused_points = depth_to_opencv_points(fused_depth, intrinsics_px)
    fused_points[~fused_valid] = np.nan

    out = dict(edited_moge)
    out.update(
        {
            "image": image_rgb,
            "points": fused_points,
            "depth": fused_depth,
            "mask": fused_valid,
            "intrinsics": np.asarray(intrinsics_px, dtype=np.float32),
            "coordinate_system": "opencv_camera",
            "source": "reference_moge_visible_background_plus_edited_moge_foreground_holes",
            "fusion": {
                "mode": "reference_visible_background_edited_foreground_holes",
                "reference_source": reference_moge.get("source"),
                "edited_source": edited_moge.get("source"),
                "depth_align": alignment,
                "foreground_mask_pixels": int(np.count_nonzero(foreground_mask)),
                "fill_mask_pixels": int(np.count_nonzero(fill_mask)),
                "foreground_mask_dilate": int(foreground_mask_dilate),
                "stable_overlap_pixels": int(np.count_nonzero(stable_mask)),
                "fused_valid_pixels": int(np.count_nonzero(fused_valid)),
                "intrinsics_pixels": np.asarray(intrinsics_px, dtype=np.float32).tolist(),
            },
        }
    )
    return out


def load_camera_alignment(camera_json_path: Path | None, width: int, height: int) -> dict[str, Any] | None:
    if camera_json_path is None:
        return None
    if not camera_json_path.is_file():
        raise FileNotFoundError(camera_json_path)
    data = json.loads(camera_json_path.read_text(encoding="utf-8"))
    matrix_value = data.get("camera_to_world") or data.get("optimized_camera_to_world")
    if matrix_value is None:
        raise ValueError(f"camera JSON has no camera_to_world or optimized_camera_to_world: {camera_json_path}")
    camera_to_world = np.asarray(matrix_value, dtype=np.float32).reshape(4, 4)
    intrinsics = pixel_intrinsics_from_camera_json(data, width, height)
    if intrinsics is None:
        raise ValueError(f"camera JSON has no usable intrinsics: {camera_json_path}")
    return {
        "path": str(camera_json_path),
        "schema": data.get("schema"),
        "camera_source": data.get("camera_source") or ("nvdiffrast_optimized" if "optimized_camera_to_world" in data else "unknown"),
        "camera_to_world": camera_to_world,
        "intrinsics_pixels": intrinsics,
        "source_image_size": data.get("image_size") or data.get("source_image_size"),
        "intrinsics_normalized": data.get("intrinsics_normalized"),
        "coordinate_notes": data.get("coordinate_notes"),
    }


def transform_opencv_points_to_world(points_cv: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    camera_local = np.asarray(points_cv, dtype=np.float32) @ OPENCV_TO_BLENDER_CAMERA.T
    rotation = np.asarray(camera_to_world[:3, :3], dtype=np.float32)
    translation = np.asarray(camera_to_world[:3, 3], dtype=np.float32)
    return (camera_local @ rotation.T + translation).astype(np.float32)


def sample_point_cloud(
    points: np.ndarray,
    image_rgb: np.ndarray,
    valid_mask: np.ndarray,
    *,
    depth_values: np.ndarray | None,
    include_mask: np.ndarray | None,
    max_points: int,
    min_depth: float,
    max_depth: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(points).all(axis=-1)
    depth = np.asarray(depth_values, dtype=np.float32) if depth_values is not None else points[..., 2]
    valid &= np.isfinite(depth)
    valid &= depth > float(min_depth)
    if max_depth > 0:
        valid &= depth < float(max_depth)
    if include_mask is not None:
        valid &= include_mask

    flat_indices = np.flatnonzero(valid.reshape(-1))
    total_valid = int(flat_indices.size)
    if total_valid == 0:
        raise ValueError("MoGe produced no valid points after filtering")

    if max_points > 0 and total_valid > max_points:
        rng = np.random.default_rng(seed)
        flat_indices = np.sort(rng.choice(flat_indices, size=int(max_points), replace=False))

    xyz = points.reshape(-1, 3)[flat_indices].astype(np.float32)
    rgb = image_rgb.reshape(-1, 3)[flat_indices].astype(np.uint8)
    stats = {
        "valid_points_before_sampling": total_valid,
        "points_written": int(xyz.shape[0]),
        "max_points": int(max_points),
        "min_depth": float(min_depth),
        "max_depth": float(max_depth),
        "include_mask_used": include_mask is not None,
        "bbox_min": xyz.min(axis=0).astype(float).tolist(),
        "bbox_max": xyz.max(axis=0).astype(float).tolist(),
        "bbox_center": (0.5 * (xyz.min(axis=0) + xyz.max(axis=0))).astype(float).tolist(),
        "mean": xyz.mean(axis=0).astype(float).tolist(),
    }
    return xyz, rgb, stats


def write_binary_ply_xyzrgb(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    if xyz.shape[0] != rgb.shape[0] or xyz.shape[1] != 3 or rgb.shape[1] != 3:
        raise ValueError(f"invalid xyz/rgb shapes: {xyz.shape}, {rgb.shape}")

    vertices = np.empty(
        xyz.shape[0],
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertices["x"] = xyz[:, 0]
    vertices["y"] = xyz[:, 1]
    vertices["z"] = xyz[:, 2]
    vertices["red"] = rgb[:, 0]
    vertices["green"] = rgb[:, 1]
    vertices["blue"] = rgb[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {xyz.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as fp:
        fp.write(header)
        vertices.tofile(fp)


def write_depth_debug(export_dir: Path, depth: np.ndarray, valid_mask: np.ndarray) -> dict[str, str]:
    debug_dir = export_dir / "moge_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    depth_path = debug_dir / "depth.npy"
    mask_path = debug_dir / "valid_mask.png"
    preview_path = debug_dir / "depth_preview.png"

    np.save(depth_path, np.asarray(depth, dtype=np.float32))
    cv2.imwrite(str(mask_path), valid_mask.astype(np.uint8) * 255)

    finite = np.isfinite(depth) & valid_mask
    if np.any(finite):
        lo, hi = np.percentile(depth[finite], [2.0, 98.0])
        if hi <= lo:
            lo, hi = float(np.min(depth[finite])), float(np.max(depth[finite]))
        normalized = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    else:
        normalized = np.zeros_like(depth, dtype=np.float32)
    preview = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    preview[~valid_mask] = 0
    cv2.imwrite(str(preview_path), preview)

    return {
        "depth_npy": str(depth_path),
        "valid_mask_png": str(mask_path),
        "depth_preview_png": str(preview_path),
    }


def write_export(
    *,
    export_dir: Path,
    image_path: Path,
    moge: dict[str, Any],
    include_mask: np.ndarray | None,
    camera_alignment: dict[str, Any] | None,
    max_points: int,
    min_depth: float,
    max_depth: float,
    seed: int,
) -> dict[str, Any]:
    export_dir.mkdir(parents=True, exist_ok=True)
    view_dir = export_dir / "perspective_views"
    view_dir.mkdir(parents=True, exist_ok=True)

    image_rgb = np.asarray(moge["image"], dtype=np.uint8)[..., :3]
    height, width = image_rgb.shape[:2]
    points = np.asarray(moge["points"], dtype=np.float32)
    valid_mask = np.asarray(moge["mask"], dtype=bool)
    if points.shape[:2] != (height, width) or valid_mask.shape != (height, width):
        raise ValueError(
            f"MoGe image/points/mask shapes do not match: image={image_rgb.shape}, "
            f"points={points.shape}, mask={valid_mask.shape}"
        )

    if camera_alignment is not None:
        intrinsics_px = np.asarray(camera_alignment["intrinsics_pixels"], dtype=np.float32)
        # OOD OriginalInputCamera is a Blender camera local-to-world matrix:
        # columns are [right, up, back]. 3DGRUT stores NeRF/Blender poses and
        # internally negates columns 1:3, yielding an OpenCV [right, down,
        # front] camera-to-world matrix for training rays.
        pose_nerf_c2w = np.asarray(camera_alignment["camera_to_world"], dtype=np.float32)
        export_points = transform_opencv_points_to_world(points, pose_nerf_c2w)
        point_cloud_frame = "foreground_world_blender_z_up"
    else:
        intrinsics_px = pixel_intrinsics(moge.get("intrinsics"), width, height)
        # train_diy writes NeRF JSON, and 3DGRUT's NeRF loader negates columns 1:3.
        # This stored pose therefore becomes identity in 3DGRUT's OpenCV convention.
        pose_nerf_c2w = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
        export_points = points
        point_cloud_frame = "opencv_camera"

    rgb_path = view_dir / "rgb_000000.png"
    write_image_rgb(rgb_path, image_rgb)
    np.save(export_dir / "perspective_poses.npy", pose_nerf_c2w[None])
    np.save(export_dir / "perspective_intrinsics.npy", intrinsics_px[None])
    np.save(view_dir / "absolute_pose_000000.npy", pose_nerf_c2w)
    np.save(view_dir / "transform_matrix_000000.npy", pose_nerf_c2w)
    np.save(view_dir / "intrinsics_000000.npy", intrinsics_px)

    xyz, rgb, point_stats = sample_point_cloud(
        export_points,
        image_rgb,
        valid_mask,
        depth_values=points[..., 2],
        include_mask=include_mask,
        max_points=max_points,
        min_depth=min_depth,
        max_depth=max_depth,
        seed=seed,
    )
    ply_path = export_dir / "initial_point_cloud.ply"
    write_binary_ply_xyzrgb(ply_path, xyz, rgb)

    cache_path = export_dir / "moge_cache_opencv.npz"
    np.savez_compressed(
        cache_path,
        points=points.astype(np.float32),
        depth=np.asarray(moge["depth"], dtype=np.float32),
        mask=valid_mask,
        intrinsics=np.asarray(moge["intrinsics"], dtype=np.float32) if moge.get("intrinsics") is not None else intrinsics_px,
        image=image_rgb,
        coordinate_system=np.array("opencv_camera"),
        source=np.array(str(moge.get("source", "unknown"))),
    )
    debug_outputs = write_depth_debug(export_dir, np.asarray(moge["depth"], dtype=np.float32), valid_mask)

    fx, fy = float(intrinsics_px[0, 0]), float(intrinsics_px[1, 1])
    metadata = {
        "schema": "fysics_single_image_moge_3dgrut_export.v1",
        "source_image": str(image_path),
        "export_dir": str(export_dir),
        "perspective_views": [str(rgb_path)],
        "initial_point_cloud": str(ply_path),
        "perspective_poses": str(export_dir / "perspective_poses.npy"),
        "perspective_intrinsics": str(export_dir / "perspective_intrinsics.npy"),
        "moge_cache": str(cache_path),
        "moge_source": moge.get("source"),
        "moge_fusion": numpy_to_json(moge.get("fusion")) if moge.get("fusion") is not None else None,
        "coordinate_system": {
            "point_cloud": point_cloud_frame,
            "source_moge_points": "OpenCV camera coordinates: +X right, +Y down, +Z forward",
            "stored_pose": "NeRF/Blender camera-to-world [right, up, back]",
            "loaded_3dgrut_pose": "OpenCV camera-to-world after NeRFDataset column flip",
            "alignment": (
                "If camera_alignment is present, MoGe background points were converted to Blender camera "
                "local coordinates [x, -y, -z] and then transformed by OOD OriginalInputCamera camera_to_world. "
                "The trained/exported Gaussian PLY is already in the foreground Blender/SAPIEN Z-up world."
            ),
        },
        "camera_alignment": numpy_to_json(camera_alignment) if camera_alignment is not None else None,
        "image_size": {"width": int(width), "height": int(height)},
        "resize_scale": float(moge.get("resize_scale", 1.0)),
        "intrinsics_pixels": intrinsics_px.tolist(),
        "fov_degrees": {
            "x": math.degrees(2.0 * math.atan(float(width) / (2.0 * max(fx, 1e-8)))),
            "y": math.degrees(2.0 * math.atan(float(height) / (2.0 * max(fy, 1e-8)))),
        },
        "point_cloud": point_stats,
        "debug_outputs": debug_outputs,
    }
    metadata_path = export_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a single MoGe-estimated image as a 3DGRUT train_diy export directory."
    )
    parser.add_argument("--image", required=True, type=Path, help="Input RGB image.")
    parser.add_argument("--export-dir", required=True, type=Path, help="Output 3dgs_train_export directory.")
    parser.add_argument("--moge-cache", type=Path, default=None, help="Optional existing MoGe .npz cache.")
    parser.add_argument(
        "--reference-moge-cache",
        type=Path,
        default=None,
        help="Original-image MoGe cache from OOD_workflow. Visible background geometry is taken from this cache.",
    )
    parser.add_argument(
        "--foreground-mask",
        type=Path,
        default=None,
        help="Session mask_label.png. Pixels >0 are treated as foreground holes filled from edited-image MoGe depth.",
    )
    parser.add_argument("--include-mask", type=Path, default=None, help="Optional binary mask selecting points to keep.")
    parser.add_argument(
        "--camera-json",
        type=Path,
        default=None,
        help="Optional OOD optimized OriginalInputCamera JSON. If set, exports point cloud in foreground world coordinates.",
    )
    parser.add_argument("--moge-root", type=Path, default=DEFAULT_MOGE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_MOGE_CKPT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resize-max", type=int, default=1024)
    parser.add_argument("--resolution-level", type=int, default=5)
    parser.add_argument("--num-tokens", type=int, default=None)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--max-points", type=int, default=300000)
    parser.add_argument("--min-depth", type=float, default=1e-4)
    parser.add_argument("--max-depth", type=float, default=0.0, help="<=0 disables far-depth filtering.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fusion-depth-align",
        choices=["scale", "scale_shift", "none"],
        default="scale",
        help="How edited-image MoGe depth is aligned to the original MoGe depth before filling foreground holes.",
    )
    parser.add_argument(
        "--foreground-mask-dilate",
        type=int,
        default=3,
        help="Dilate mask_label foreground by this many pixels before replacing depth with edited-image MoGe depth.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    image_path = args.image.expanduser().resolve()
    export_dir = args.export_dir.expanduser().resolve()

    if args.moge_cache:
        moge = load_moge_cache(args.moge_cache.expanduser().resolve(), image_path)
    else:
        moge = infer_moge(
            image_path,
            moge_root=args.moge_root.expanduser().resolve(),
            checkpoint=args.checkpoint.expanduser().resolve(),
            device=args.device,
            resize_max=int(args.resize_max),
            resolution_level=int(args.resolution_level),
            num_tokens=args.num_tokens,
            use_fp16=not args.no_fp16,
        )

    image_rgb = np.asarray(moge["image"], dtype=np.uint8)
    camera_alignment = load_camera_alignment(args.camera_json.expanduser().resolve(), image_rgb.shape[1], image_rgb.shape[0]) if args.camera_json else None
    if args.reference_moge_cache:
        if args.foreground_mask is None:
            raise ValueError("--reference-moge-cache requires --foreground-mask")
        reference_moge = load_moge_cache(args.reference_moge_cache.expanduser().resolve(), image_path)
        foreground_mask = load_binary_mask(args.foreground_mask.expanduser().resolve(), image_rgb.shape[:2])
        intrinsics_px = (
            np.asarray(camera_alignment["intrinsics_pixels"], dtype=np.float32)
            if camera_alignment is not None
            else pixel_intrinsics(reference_moge.get("intrinsics"), image_rgb.shape[1], image_rgb.shape[0])
        )
        moge = fuse_reference_and_edited_moge(
            edited_moge=moge,
            reference_moge=reference_moge,
            foreground_mask=foreground_mask,
            intrinsics_px=intrinsics_px,
            depth_align_mode=args.fusion_depth_align,
            foreground_mask_dilate=int(args.foreground_mask_dilate),
            seed=int(args.seed),
        )
    include_mask = load_binary_mask(args.include_mask.expanduser().resolve(), image_rgb.shape[:2]) if args.include_mask else None
    metadata = write_export(
        export_dir=export_dir,
        image_path=image_path,
        moge=moge,
        include_mask=include_mask,
        camera_alignment=camera_alignment,
        max_points=int(args.max_points),
        min_depth=float(args.min_depth),
        max_depth=float(args.max_depth),
        seed=int(args.seed),
    )

    print("[prepare_single_image_3dgs] wrote export")
    print(f"  export_dir: {metadata['export_dir']}")
    print(f"  image_size: {metadata['image_size']['width']}x{metadata['image_size']['height']}")
    print(f"  points: {metadata['point_cloud']['points_written']}")
    print(f"  initial_point_cloud: {metadata['initial_point_cloud']}")
    print(f"  metadata: {Path(metadata['export_dir']) / 'metadata.json'}")


if __name__ == "__main__":
    main()
