#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


def read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def relpath(path: Path, base: Path) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix()


def asset_url(path_value: str, base: Path) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()
    return path.as_posix()


def matrix_to_quaternion_xyzw(matrix: np.ndarray) -> list[float]:
    rot = np.asarray(matrix[:3, :3], dtype=np.float64)
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(rot)))
        if idx == 0:
            s = math.sqrt(max(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2], 1e-12)) * 2.0
            qw = (rot[2, 1] - rot[1, 2]) / s
            qx = 0.25 * s
            qy = (rot[0, 1] + rot[1, 0]) / s
            qz = (rot[0, 2] + rot[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(max(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2], 1e-12)) * 2.0
            qw = (rot[0, 2] - rot[2, 0]) / s
            qx = (rot[0, 1] + rot[1, 0]) / s
            qy = 0.25 * s
            qz = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = math.sqrt(max(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1], 1e-12)) * 2.0
            qw = (rot[1, 0] - rot[0, 1]) / s
            qx = (rot[0, 2] + rot[2, 0]) / s
            qy = (rot[1, 2] + rot[2, 1]) / s
            qz = 0.25 * s
    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    quat /= max(float(np.linalg.norm(quat)), 1e-12)
    return quat.astype(float).tolist()


def decompose_matrix_rows(rows: Any) -> dict[str, list[float]] | None:
    if not isinstance(rows, list) or len(rows) < 4:
        return None
    matrix = np.asarray(rows, dtype=np.float64).reshape(4, 4)
    basis = matrix[:3, :3].copy()
    scale = np.linalg.norm(basis, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    rotation = basis / scale[None, :]
    return {
        "position": matrix[:3, 3].astype(float).tolist(),
        "quaternion_xyzw": matrix_to_quaternion_xyzw(rotation),
        "scale": scale.astype(float).tolist(),
    }


def camera_payload(camera_json: Path | None, fallback_bbox_center: list[float] | None) -> dict[str, Any]:
    data = read_json(camera_json)
    if not data:
        look_at = fallback_bbox_center or [0.0, 0.0, 0.0]
        return {"position": [1.4, -1.8, 1.1], "look_at": look_at}
    matrix_value = data.get("camera_to_world") or data.get("optimized_camera_to_world")
    if matrix_value is None:
        look_at = fallback_bbox_center or [0.0, 0.0, 0.0]
        return {"position": [1.4, -1.8, 1.1], "look_at": look_at}
    c2w = np.asarray(matrix_value, dtype=np.float64).reshape(4, 4)
    position = c2w[:3, 3].astype(float)
    forward = -c2w[:3, 2]
    forward /= max(float(np.linalg.norm(forward)), 1e-12)
    look_at = position + forward
    return {
        "position": position.tolist(),
        "look_at": look_at.astype(float).tolist(),
        "quaternion_xyzw": matrix_to_quaternion_xyzw(c2w),
        "camera_source": data.get("camera_source") or ("nvdiffrast_optimized" if data.get("optimized_camera_to_world") else "unknown"),
        "intrinsics_pixels": data.get("intrinsics_pixels"),
        "intrinsics_normalized": data.get("intrinsics_normalized"),
        "image_size": data.get("image_size") or data.get("source_image_size"),
        "source": str(camera_json) if camera_json else None,
    }


def objects_from_final_manifest(final_manifest: Path | None, bg_dir: Path) -> list[dict[str, Any]]:
    data = read_json(final_manifest)
    if not data:
        return []
    objects: list[dict[str, Any]] = []
    for raw in data.get("objects") or []:
        export = raw.get("sapien_export") or {}
        visual_path = export.get("visual_path")
        if not visual_path:
            continue
        pose = (raw.get("sapien_final_pose") or {}).get("final_pose") or export.get("initial_pose")
        transform = decompose_matrix_rows(pose) if pose is not None else None
        entry: dict[str, Any] = {
            "name": raw.get("final_3d_object_name") or export.get("name") or raw.get("mask_name"),
            "label": raw.get("semantic_label") or raw.get("description"),
            "url": asset_url(str(visual_path), bg_dir),
            "pose_matrix": pose,
            "gltf_visual_to_blender_z_up": True,
            "mask_id": raw.get("mask_id"),
            "source": "final_scene_manifest",
        }
        if transform is not None:
            entry.update(transform)
        objects.append({key: value for key, value in entry.items() if value is not None})
    return objects


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write session background 3DGS manifests.")
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--bg-dir", required=True, type=Path)
    parser.add_argument("--background-image", required=True, type=Path)
    parser.add_argument("--edit-report", type=Path, default=None)
    parser.add_argument("--export-dir", required=True, type=Path)
    parser.add_argument("--train-out-dir", required=True, type=Path)
    parser.add_argument("--ksplat", required=True, type=Path)
    parser.add_argument("--camera-json", type=Path, default=None)
    parser.add_argument("--final-scene-manifest", type=Path, default=None)
    parser.add_argument("--create-ksplat-command", default="")
    parser.add_argument("--compression-level", type=int, default=2)
    parser.add_argument("--alpha-threshold", type=int, default=5)
    parser.add_argument("--sh-degree", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    session = args.session.expanduser().resolve()
    bg_dir = args.bg_dir.expanduser().resolve()
    export_dir = args.export_dir.expanduser().resolve()
    train_out_dir = args.train_out_dir.expanduser().resolve()
    ksplat = args.ksplat.expanduser().resolve()
    bg_dir.mkdir(parents=True, exist_ok=True)

    export_metadata_path = export_dir / "metadata.json"
    export_metadata = read_json(export_metadata_path) or {}
    point_cloud = export_metadata.get("point_cloud") or {}
    bbox_center = point_cloud.get("bbox_center")
    camera = camera_payload(args.camera_json.expanduser().resolve() if args.camera_json else None, bbox_center)
    final_manifest = args.final_scene_manifest.expanduser().resolve() if args.final_scene_manifest else None

    scene_json = {
        "schema": "fysics_hybrid_3dgs_scene.v1",
        "coordinate_system": "foreground_world_blender_z_up",
        "world_up": [0, 0, 1],
        "world_transform": {
            "position": [0, 0, 0],
            "quaternion_xyzw": [0, 0, 0, 1],
            "scale": [1, 1, 1],
            "matrix": [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            "note": "Identity by default. If the web viewer applies an initial scene-level transform to foreground objects, apply the same transform to the splat.",
        },
        "splat": {
            "url": relpath(ksplat, bg_dir),
            "format": "ksplat",
            "position": [0, 0, 0],
            "quaternion_xyzw": [0, 0, 0, 1],
            "scale": [1, 1, 1],
            "alpha_removal_threshold": int(args.alpha_threshold),
            "spherical_harmonics_degree": int(args.sh_degree),
            "coordinate_system": "foreground_world_blender_z_up",
        },
        "camera": {
            key: camera[key]
            for key in ("position", "look_at", "quaternion_xyzw")
            if key in camera
        },
        "objects": objects_from_final_manifest(final_manifest, bg_dir),
        "alignment": {
            "strategy": "background_splats_are_trained_and_exported_in_foreground_world_coordinates",
            "camera_source": camera.get("camera_source"),
            "camera_json": str(args.camera_json) if args.camera_json else None,
        },
    }
    (bg_dir / "scene.json").write_text(json.dumps(scene_json, indent=2), encoding="utf-8")

    manifest = {
        "schema": "fysics_session_background_3dgs.v1",
        "status": "ok",
        "session": str(session),
        "input_image": str(session / "input" / "image.png"),
        "background_image": str(args.background_image.expanduser().resolve()),
        "edit_report": str(args.edit_report.expanduser().resolve()) if args.edit_report else None,
        "output_dir": str(bg_dir),
        "export_dir": str(export_dir),
        "train_out_dir": str(train_out_dir),
        "initial_point_cloud": str(export_dir / "initial_point_cloud.ply"),
        "gaussian_ply": str(train_out_dir / "export_last.ply"),
        "checkpoint": str(train_out_dir / "ckpt_last.pt"),
        "ksplat": str(ksplat),
        "scene_json": str(bg_dir / "scene.json"),
        "camera": camera,
        "final_scene_manifest": str(final_manifest) if final_manifest else None,
        "create_ksplat": {
            "command": args.create_ksplat_command,
            "compression_level": int(args.compression_level),
            "alpha_threshold": int(args.alpha_threshold),
            "spherical_harmonics_degree": int(args.sh_degree),
        },
        "export_metadata": export_metadata,
        "alignment_notes": {
            "preferred_camera": "nvdiffrast optimized OriginalInputCamera from OOD_workflow.",
            "point_transform": "MoGe OpenCV points are converted by [x, -y, -z] into Blender camera local coordinates, then by optimized camera_to_world into foreground Blender/SAPIEN Z-up world.",
            "splat_transform_for_web": "identity; splat centers are already in foreground world coordinates.",
            "initial_web_alignment": "Any web-side initial scene-level transform applied to foreground objects must also be applied to the background splat.",
        },
    }
    (bg_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if final_manifest is not None and final_manifest.is_file():
        final_data = read_json(final_manifest) or {}
        outputs = final_data.setdefault("outputs", {})
        if isinstance(outputs, dict):
            outputs["background_3dgs_ksplat"] = str(ksplat)
            outputs["background_3dgs_manifest"] = str(bg_dir / "manifest.json")
            outputs["background_3dgs_scene"] = str(bg_dir / "scene.json")
            final_manifest.write_text(json.dumps(final_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "ok", "manifest": str(bg_dir / "manifest.json"), "scene": str(bg_dir / "scene.json")}))


if __name__ == "__main__":
    main()
