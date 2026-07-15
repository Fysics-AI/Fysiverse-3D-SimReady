#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr


BLENDER_LOCAL_TO_OPENCV = np.diag([1.0, -1.0, -1.0]).astype(np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize Blender camera extrinsics with nvdiffrast silhouette alignment."
    )
    parser.add_argument("--mesh", required=True, type=Path, help="NPZ from blender_export_nvdiffrast_mesh.py")
    parser.add_argument("--target-mask", required=True, type=Path, help="2D label/binary mask. Nonzero pixels are foreground.")
    parser.add_argument("--camera-report", required=True, type=Path, help="SAM3D/MoGe rotated report.")
    parser.add_argument("--stage-report", type=Path, default=None, help="Current stage report carrying shared grounding translation.")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--preview-output", type=Path, default=None)
    parser.add_argument("--input-image", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--rot-lr", type=float, default=0.015)
    parser.add_argument("--trans-lr", type=float, default=0.01)
    parser.add_argument("--dt-weight", type=float, default=0.25)
    parser.add_argument("--rot-prior-weight", type=float, default=0.002)
    parser.add_argument("--trans-prior-weight", type=float, default=0.002)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-antialias", action="store_true")
    parser.add_argument("--accept-worse-iou", action="store_true")
    return parser.parse_args()


def read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def matrix3(values: Any, fallback: np.ndarray | None = None) -> np.ndarray:
    if values is None:
        if fallback is None:
            raise ValueError("missing 3x3 matrix")
        return fallback.copy()
    return np.asarray(values, dtype=np.float64).reshape(3, 3)


def matrix4(values: Any, fallback: np.ndarray | None = None) -> np.ndarray:
    if values is None:
        if fallback is None:
            raise ValueError("missing 4x4 matrix")
        return fallback.copy()
    return np.asarray(values, dtype=np.float64).reshape(4, 4)


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError(f"cannot normalize vector {vec}")
    return vec / norm


def raw_camera_to_blender_matrix(report: dict[str, Any]) -> np.ndarray:
    moge = report.get("moge") or {}
    scene_to_blender = matrix3(moge.get("scene_to_blender_matrix"), np.eye(3, dtype=np.float64))
    cv_to_scene = matrix3(
        moge.get("opencv_camera_to_pytorch3d_scene_matrix"),
        np.diag([-1.0, -1.0, 1.0]).astype(np.float64),
    )

    right_scene = cv_to_scene @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
    up_scene = cv_to_scene @ np.array([0.0, -1.0, 0.0], dtype=np.float64)
    back_scene = cv_to_scene @ np.array([0.0, 0.0, -1.0], dtype=np.float64)

    right = normalize(scene_to_blender @ right_scene)
    up = normalize(scene_to_blender @ up_scene)
    back = normalize(scene_to_blender @ back_scene)
    out = np.eye(4, dtype=np.float64)
    out[:3, 0] = right
    out[:3, 1] = up
    out[:3, 2] = back
    return out


def stage_global_translation(stage_report: dict[str, Any] | None) -> np.ndarray:
    if not stage_report:
        return np.zeros(3, dtype=np.float64)
    bbox = ((stage_report.get("transform") or {}).get("bbox_snap_objects"))
    if isinstance(bbox, dict) and bbox.get("global_translation") is not None:
        return np.asarray(bbox["global_translation"], dtype=np.float64).reshape(3)
    return np.zeros(3, dtype=np.float64)


def initial_camera_to_world(camera_report: dict[str, Any], stage_report: dict[str, Any] | None) -> np.ndarray:
    raw_c2w = raw_camera_to_blender_matrix(camera_report)
    upright = matrix4((camera_report.get("transform") or {}).get("matrix_4x4"), np.eye(4, dtype=np.float64))
    extra = np.eye(4, dtype=np.float64)
    extra[:3, 3] = stage_global_translation(stage_report)
    return extra @ upright @ raw_c2w


def normalized_intrinsics(report: dict[str, Any]) -> tuple[float, float, float, float]:
    intrinsics = (report.get("moge") or {}).get("intrinsics")
    if intrinsics is None:
        raise ValueError("camera report does not contain moge.intrinsics")
    mat = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    return float(mat[0, 0]), float(mat[1, 1]), float(mat[0, 2]), float(mat[1, 2])


def image_size_from_report(report: dict[str, Any], target_mask: Path) -> tuple[int, int]:
    sample = ((report.get("moge") or {}).get("sample_info") or {})
    size = sample.get("image_size")
    if isinstance(size, list) and len(size) >= 2:
        return int(size[0]), int(size[1])
    image_path = ((report.get("input") or {}).get("image")) or ((report.get("stage") or {}).get("input") or {}).get("image")
    if image_path and Path(image_path).is_file():
        img = imageio.imread(image_path)
        return int(img.shape[1]), int(img.shape[0])
    mask = cv2.imread(str(target_mask), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(target_mask)
    return int(mask.shape[1]), int(mask.shape[0])


def target_resolution(width: int, height: int, max_side: int) -> tuple[int, int]:
    if max_side <= 0:
        out_w, out_h = width, height
    else:
        scale = min(1.0, float(max_side) / float(max(width, height)))
        out_w = max(2, int(round(width * scale)))
        out_h = max(2, int(round(height * scale)))
    if out_w % 2:
        out_w += 1
    if out_h % 2:
        out_h += 1
    return out_w, out_h


def load_target_mask(path: Path, width: int, height: int) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.ndim == 3:
        if mask.shape[2] == 4:
            mask = mask[..., 3]
        else:
            mask = mask[..., :3].max(axis=2)
    mask = (mask > 0).astype(np.float32)
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask.astype(np.float32)


def load_preview_image(path: Path | None, width: int, height: int) -> np.ndarray | None:
    if path is None or not path.is_file():
        return None
    image = imageio.imread(path)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[2] == 4:
        image = image[..., :3]
    image = cv2.resize(image.astype(np.uint8), (width, height), interpolation=cv2.INTER_AREA)
    return image


def skew(vec: torch.Tensor) -> torch.Tensor:
    x, y, z = vec[0], vec[1], vec[2]
    zero = torch.zeros((), dtype=vec.dtype, device=vec.device)
    return torch.stack(
        [
            torch.stack([zero, -z, y]),
            torch.stack([z, zero, -x]),
            torch.stack([-y, x, zero]),
        ]
    )


def so3_exp(rotvec: torch.Tensor) -> torch.Tensor:
    theta2 = torch.dot(rotvec, rotvec)
    theta = torch.sqrt(theta2 + 1e-12)
    k = skew(rotvec)
    small = theta2 < 1e-8
    safe_theta = torch.clamp(theta, min=1e-6)
    safe_theta2 = torch.clamp(theta2, min=1e-12)
    a_full = torch.sin(theta) / safe_theta
    b_full = (1.0 - torch.cos(theta)) / safe_theta2
    a_taylor = 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0
    b_taylor = 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0
    a = torch.where(small, a_taylor, a_full)
    b = torch.where(small, b_taylor, b_full)
    ident = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    return ident + a * k + b * (k @ k)


def compose_camera(initial_c2w: torch.Tensor, rotvec: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    delta_r = so3_exp(rotvec)
    out = torch.eye(4, dtype=initial_c2w.dtype, device=initial_c2w.device)
    out[:3, :3] = delta_r @ initial_c2w[:3, :3]
    out[:3, 3] = initial_c2w[:3, 3] + translation
    return out


def project_to_clip(
    vertices_world: torch.Tensor,
    camera_to_world: torch.Tensor,
    intrinsics: tuple[float, float, float, float],
    near: float,
    far: float,
) -> torch.Tensor:
    fx, fy, cx, cy = intrinsics
    rot = camera_to_world[:3, :3]
    trans = camera_to_world[:3, 3]
    local = (vertices_world - trans[None, :]) @ rot
    cv = local * torch.tensor([1.0, -1.0, -1.0], dtype=vertices_world.dtype, device=vertices_world.device)[None, :]
    z = cv[:, 2].clamp_min(max(float(near) * 0.25, 1e-5))
    u = float(fx) * cv[:, 0] / z + float(cx)
    v = float(fy) * cv[:, 1] / z + float(cy)

    x_ndc = 2.0 * u - 1.0
    # nvdiffrast's tensor row order maps clip-space +Y to larger row indices.
    # With OpenCV image coordinates (+Y down), this is clip_y = 2*v - 1.
    y_ndc = 2.0 * v - 1.0
    z_ndc = 2.0 * (z - float(near)) / max(float(far - near), 1e-6) - 1.0
    return torch.stack([x_ndc * z, y_ndc * z, z_ndc * z, z], dim=1)


def render_silhouette(
    ctx: Any,
    vertices_world: torch.Tensor,
    faces: torch.Tensor,
    camera_to_world: torch.Tensor,
    intrinsics: tuple[float, float, float, float],
    near: float,
    far: float,
    height: int,
    width: int,
    antialias: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    clip = project_to_clip(vertices_world, camera_to_world, intrinsics, near, far).contiguous()
    rast, _ = dr.rasterize(ctx, clip[None, :, :], faces, resolution=[height, width])
    ones = torch.ones((1, vertices_world.shape[0], 1), dtype=vertices_world.dtype, device=vertices_world.device)
    sil, _ = dr.interpolate(ones, rast, faces)
    if antialias:
        sil = dr.antialias(sil.contiguous(), rast, clip[None, :, :], faces)
    return sil[0, :, :, 0].clamp(0.0, 1.0), clip


def compute_near_far(vertices: torch.Tensor, camera_to_world: torch.Tensor) -> tuple[float, float]:
    rot = camera_to_world[:3, :3]
    trans = camera_to_world[:3, 3]
    local = (vertices - trans[None, :]) @ rot
    cv_z = (-local[:, 2]).detach().cpu().numpy()
    positive = cv_z[np.isfinite(cv_z) & (cv_z > 1e-5)]
    if positive.size == 0:
        return 0.01, 100.0
    near = max(0.001, float(np.percentile(positive, 0.5)) * 0.25)
    far = max(near + 1.0, float(np.percentile(positive, 99.5)) * 4.0)
    return near, far


def mask_metrics(rendered: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    r = rendered > 0.5
    t = target > 0.5
    inter = int(np.logical_and(r, t).sum())
    union = int(np.logical_or(r, t).sum())
    iou = float(inter / union) if union else 0.0

    def bbox(mask: np.ndarray) -> list[int] | None:
        yy, xx = np.where(mask)
        if len(xx) == 0:
            return None
        return [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())]

    rb = bbox(r)
    tb = bbox(t)
    center_error = None
    if rb is not None and tb is not None:
        rc = np.array([(rb[0] + rb[2]) * 0.5, (rb[1] + rb[3]) * 0.5], dtype=np.float64)
        tc = np.array([(tb[0] + tb[2]) * 0.5, (tb[1] + tb[3]) * 0.5], dtype=np.float64)
        center_error = float(np.linalg.norm(rc - tc))
    return {
        "iou": iou,
        "intersection_pixels": inter,
        "union_pixels": union,
        "render_bbox_xyxy": rb,
        "target_bbox_xyxy": tb,
        "bbox_center_error_pixels": center_error,
    }


def save_preview(path: Path, target: np.ndarray, initial: np.ndarray, optimized: np.ndarray, image: np.ndarray | None) -> None:
    h, w = target.shape
    panels = []
    base = image if image is not None else np.full((h, w, 3), 36, dtype=np.uint8)

    def overlay(render: np.ndarray, title_color: tuple[int, int, int]) -> np.ndarray:
        out = base.copy().astype(np.float32)
        target_rgb = np.zeros_like(out)
        target_rgb[..., 1] = 255
        render_rgb = np.zeros_like(out)
        render_rgb[..., 0] = 255
        render_rgb[..., 2] = 255
        out = out * 0.55 + target_rgb * (target[..., None] * 0.22) + render_rgb * (render[..., None] * 0.35)
        out = np.clip(out, 0, 255).astype(np.uint8)
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), title_color, 2)
        return out

    target_panel = np.repeat((target * 255.0).astype(np.uint8)[..., None], 3, axis=2)
    initial_panel = overlay(initial, (255, 180, 0))
    optimized_panel = overlay(optimized, (0, 255, 255))
    diff = np.zeros((h, w, 3), dtype=np.uint8)
    diff[..., 1] = (target * 255).astype(np.uint8)
    diff[..., 0] = (optimized * 255).astype(np.uint8)
    diff[..., 2] = (initial * 255).astype(np.uint8)
    panels.extend([target_panel, initial_panel, optimized_panel, diff])
    out = np.concatenate(panels, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, out)


def as_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return as_json(value.tolist())
    if isinstance(value, torch.Tensor):
        return as_json(value.detach().cpu().numpy())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): as_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_json(v) for v in value]
    return value


def main() -> int:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    camera_report = read_json(args.camera_report)
    assert camera_report is not None
    stage_report = read_json(args.stage_report)
    initial_c2w_np = initial_camera_to_world(camera_report, stage_report)
    intrinsics = normalized_intrinsics(camera_report)
    image_width, image_height = image_size_from_report(camera_report, args.target_mask)
    render_width, render_height = target_resolution(image_width, image_height, int(args.max_side))
    target_np = load_target_mask(args.target_mask, render_width, render_height)
    target = torch.from_numpy(target_np).to(device=device, dtype=torch.float32)

    outside_dist = cv2.distanceTransform((target_np < 0.5).astype(np.uint8), cv2.DIST_L2, 3)
    if float(outside_dist.max()) > 0:
        outside_dist = outside_dist / float(outside_dist.max())
    outside_dt = torch.from_numpy(outside_dist.astype(np.float32)).to(device)

    mesh = np.load(args.mesh, allow_pickle=False)
    vertices_np = np.asarray(mesh["vertices"], dtype=np.float32)
    faces_np = np.asarray(mesh["faces"], dtype=np.int32)
    mesh_meta = json.loads(str(mesh["metadata"])) if "metadata" in mesh else {}
    vertices = torch.from_numpy(vertices_np).to(device=device, dtype=torch.float32)
    faces = torch.from_numpy(faces_np).to(device=device, dtype=torch.int32).contiguous()
    initial_c2w = torch.tensor(initial_c2w_np, dtype=torch.float32, device=device)
    near, far = compute_near_far(vertices, initial_c2w)

    ctx = dr.RasterizeCudaContext() if device.type == "cuda" else dr.RasterizeGLContext()
    antialias = not bool(args.no_antialias)
    with torch.no_grad():
        initial_sil, _ = render_silhouette(
            ctx, vertices, faces, initial_c2w, intrinsics, near, far, render_height, render_width, antialias
        )
        initial_np = initial_sil.detach().cpu().numpy()
        initial_metrics = mask_metrics(initial_np, target_np)

    rotvec = torch.zeros(3, dtype=torch.float32, device=device, requires_grad=True)
    translation = torch.zeros(3, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam(
        [
            {"params": [rotvec], "lr": float(args.rot_lr)},
            {"params": [translation], "lr": float(args.trans_lr)},
        ]
    )

    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_c2w = initial_c2w.detach().clone()
    best_sil = initial_sil.detach().clone()
    for step in range(1, int(args.steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        c2w = compose_camera(initial_c2w, rotvec, translation)
        sil, _ = render_silhouette(ctx, vertices, faces, c2w, intrinsics, near, far, render_height, render_width, antialias)
        l1 = F.l1_loss(sil, target)
        dt_loss = torch.mean(sil * outside_dt)
        prior = float(args.rot_prior_weight) * torch.sum(rotvec * rotvec) + float(args.trans_prior_weight) * torch.sum(translation * translation)
        loss = l1 + float(args.dt_weight) * dt_loss + prior
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu())
        if value < best_loss:
            best_loss = value
            best_c2w = c2w.detach().clone()
            best_sil = sil.detach().clone()
        if step == 1 or step == int(args.steps) or (int(args.log_every) > 0 and step % int(args.log_every) == 0):
            entry = {
                "step": float(step),
                "loss": value,
                "l1": float(l1.detach().cpu()),
                "dt": float(dt_loss.detach().cpu()),
                "rot_norm_deg": float(torch.linalg.norm(rotvec.detach()).cpu()) * 180.0 / math.pi,
                "trans_norm": float(torch.linalg.norm(translation.detach()).cpu()),
            }
            history.append(entry)
            print(
                "step={step:.0f} loss={loss:.6f} l1={l1:.6f} dt={dt:.6f} rot_deg={rot_norm_deg:.4f} trans={trans_norm:.5f}".format(
                    **entry
                ),
                flush=True,
            )

    optimized_np = best_sil.detach().cpu().numpy()
    optimized_metrics = mask_metrics(optimized_np, target_np)
    optimized_c2w_np = best_c2w.detach().cpu().numpy().astype(np.float64)
    accepted = True
    if (not args.accept_worse_iou) and optimized_metrics["iou"] < initial_metrics["iou"]:
        accepted = False
        optimized_np = initial_np
        optimized_metrics = initial_metrics
        optimized_c2w_np = initial_c2w_np.copy()
    delta_np = np.linalg.inv(initial_c2w_np) @ optimized_c2w_np

    preview_path = args.preview_output
    if preview_path is not None:
        image = load_preview_image(args.input_image, render_width, render_height)
        save_preview(preview_path, target_np, initial_np, optimized_np, image)

    fx, fy, cx, cy = intrinsics
    output = {
        "schema": "fysiverse_nvdiffrast_camera_pose_optimization.v1",
        "status": "ok" if accepted else "kept_initial_due_to_iou_guard",
        "mesh": str(args.mesh),
        "target_mask": str(args.target_mask),
        "input_image": str(args.input_image) if args.input_image else None,
        "camera_report": str(args.camera_report),
        "stage_report": str(args.stage_report) if args.stage_report else None,
        "preview": str(preview_path) if preview_path else None,
        "render_resolution": [int(render_width), int(render_height)],
        "source_image_size": [int(image_width), int(image_height)],
        "intrinsics_normalized": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        "initial_camera_to_world": initial_c2w_np,
        "optimized_camera_to_world": optimized_c2w_np,
        "delta_initial_to_optimized": delta_np,
        "metrics": {
            "initial": initial_metrics,
            "optimized": optimized_metrics,
            "iou_delta": float(optimized_metrics["iou"] - initial_metrics["iou"]),
            "best_loss": float(best_loss),
            "optimized_pose_accepted": bool(accepted),
        },
        "optimization": {
            "steps": int(args.steps),
            "rot_lr": float(args.rot_lr),
            "trans_lr": float(args.trans_lr),
            "dt_weight": float(args.dt_weight),
            "rot_prior_weight": float(args.rot_prior_weight),
            "trans_prior_weight": float(args.trans_prior_weight),
            "near": float(near),
            "far": float(far),
            "history": history,
        },
        "mesh_metadata": mesh_meta,
        "coordinate_notes": {
            "optimized_variable": "Blender camera-to-world matrix in Blender world coordinates.",
            "blender_camera_local_axes": "+X image right, +Y image up, -Z forward.",
            "opencv_camera_axes": "+X image right, +Y image down, +Z forward.",
            "world_to_opencv_camera": "p_local = R_c2w^T @ (p_world - t); p_cv = [p_local.x, -p_local.y, -p_local.z].",
            "nvdiffrast_clip_mapping": "u=fx*X/Z+cx, v=fy*Y/Z+cy, clip_x=2u-1, clip_y=2v-1. Empirically, nvdiffrast tensor rows map clip +Y downward.",
            "intrinsics": "Fixed from MoGe normalized intrinsics; only extrinsics are optimized.",
            "supervision": "Union silhouette from target mask nonzero pixels.",
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(as_json(output), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote optimized camera report: {args.output_json}")
    if preview_path is not None:
        print(f"Wrote preview: {preview_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
