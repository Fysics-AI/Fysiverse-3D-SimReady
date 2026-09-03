#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


GROUND_PLANE = {
    "type": "fixed_z0",
    "point": [0.0, 0.0, 0.0],
    "normal": [0.0, 0.0, 1.0],
    "z": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact scene-description draft from final_scene_manifest.json.")
    parser.add_argument("--final-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def vec_sub(a: list[float], b: list[float]) -> list[float]:
    return [float(x) - float(y) for x, y in zip(a, b)]


def vec_norm(v: list[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in v))


def extent_from_bounds(bmin: list[float] | None, bmax: list[float] | None) -> list[float] | None:
    if not bmin or not bmax:
        return None
    return [float(mx) - float(mn) for mn, mx in zip(bmin, bmax)]


def object_from_manifest(entry: dict[str, Any]) -> dict[str, Any]:
    export = entry.get("sapien_export") or {}
    pose = entry.get("sapien_final_pose") or {}
    bmin = export.get("bbox_min")
    bmax = export.get("bbox_max")
    center = export.get("bbox_center")
    semantic = entry.get("description")
    return {
        "mask_id": entry.get("mask_id"),
        "mask_name": entry.get("mask_name"),
        "object_name": entry.get("final_3d_object_name"),
        "semantic_label": semantic,
        "visual_description": None,
        "crop_rgb": entry.get("crop_rgb"),
        "sam3d_input_mask": entry.get("sam3d_input_mask"),
        "bbox_2d_xyxy": entry.get("bbox_2d_xyxy"),
        "bbox_2d_xywh": entry.get("bbox_2d_xywh"),
        "bbox_3d_min": bmin,
        "bbox_3d_max": bmax,
        "bbox_3d_center": center,
        "bbox_3d_extent": extent_from_bounds(bmin, bmax),
        "pose_matrix": pose.get("final_pose") or export.get("initial_pose"),
        "sapien_export": {
            "visual_path": export.get("visual_path"),
            "collision_path": export.get("collision_path"),
            "collision_paths": export.get("collision_paths"),
            "collision_part_count": export.get("collision_part_count"),
            "collision_decomposition": export.get("collision_decomposition"),
        },
        "role_candidates": ["active", "target", "passive", "static"],
        "physical_hints": {
            "likely_material": None,
            "rigid": True,
            "hollow_or_light": None,
        },
        "needs_crop_review": semantic is None,
    }


def build_relations(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    for i, a in enumerate(objects):
        ac = a.get("bbox_3d_center")
        if not ac:
            continue
        for b in objects[i + 1 :]:
            bc = b.get("bbox_3d_center")
            if not bc:
                continue
            delta = vec_sub(bc, ac)
            xy = [delta[0], delta[1], 0.0]
            relations.append(
                {
                    "a": a.get("object_name"),
                    "b": b.get("object_name"),
                    "center_delta_b_minus_a": delta,
                    "center_distance": vec_norm(delta),
                    "xy_distance": vec_norm(xy),
                }
            )
    relations.sort(key=lambda item: float(item.get("xy_distance", 1e9)))
    return relations


def build(final_manifest: Path) -> dict[str, Any]:
    manifest = json.loads(final_manifest.read_text(encoding="utf-8"))
    session_dir = Path(manifest.get("inputs", {}).get("session_dir", final_manifest.parents[1]))
    overlay = session_dir / "segmentation" / "overlay.png"
    objects = [object_from_manifest(entry) for entry in manifest.get("objects", [])]
    return {
        "schema": "simulator_scene_description.v1",
        "source_final_scene_manifest": str(final_manifest),
        "session_id": manifest.get("session_id"),
        "coordinate_system": "Blender/SAPIEN Z-up; gravity is negative Z.",
        "ground_plane": GROUND_PLANE,
        "inputs": {
            "image": manifest.get("inputs", {}).get("image"),
            "overlay": str(overlay) if overlay.is_file() else None,
            "final_blend": manifest.get("outputs", {}).get("final_blend"),
            "label_mask": manifest.get("inputs", {}).get("label_mask"),
        },
        "objects": objects,
        "relations": build_relations(objects),
        "missing_or_uncertain": [
            "semantic labels must be inferred from crop images when null",
            "material, mass, and exact metric scale are not provided by upstream",
            "support/contact graph between non-ground objects is not explicit",
        ],
    }


def main() -> int:
    args = parse_args()
    scene = build(args.final_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scene, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote scene description: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
