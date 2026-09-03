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
    parser = argparse.ArgumentParser(description="Build a compact cached scene digest for simulator_assistance.")
    parser.add_argument("--final-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def vec_sub(a: list[float], b: list[float]) -> list[float]:
    return [float(x) - float(y) for x, y in zip(a, b)]


def vec_norm(v: list[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in v))


def extent_from_bounds(bmin: list[float] | None, bmax: list[float] | None) -> list[float] | None:
    if not bmin or not bmax:
        return None
    return [float(mx) - float(mn) for mn, mx in zip(bmin, bmax)]


def translation_from_matrix(matrix: list[list[float]] | None) -> list[float] | None:
    if not matrix or len(matrix) < 3:
        return None
    return [float(matrix[0][3]), float(matrix[1][3]), float(matrix[2][3])]


def load_json_if_exists(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_scene_graph_path(manifest: dict[str, Any], session_dir: Path) -> Path | None:
    candidates: list[Path] = []
    graph_ref = (manifest.get("intermediate") or {}).get("scene_graph")
    if graph_ref:
        graph_path = Path(graph_ref)
        candidates.append(graph_path if graph_path.is_absolute() else session_dir / graph_path)
    candidates.append(session_dir / "results" / "scene_graph.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def scene_graph_objects_by_mask(scene_graph: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for item in scene_graph.get("objects") or []:
        mask_id = item.get("mask_id")
        if mask_id is not None:
            result[int(mask_id)] = item
    return result


def object_card(entry: dict[str, Any], scene_graph_objects: dict[int, dict[str, Any]]) -> dict[str, Any]:
    export = entry.get("sapien_export") or {}
    pose = entry.get("sapien_final_pose") or {}
    pose_matrix = pose.get("final_pose") or export.get("initial_pose")
    bmin = export.get("bbox_min")
    bmax = export.get("bbox_max")
    center = export.get("bbox_center")
    graph_obj = scene_graph_objects.get(int(entry.get("mask_id") or -1), {})
    return {
        "mask_id": entry.get("mask_id"),
        "mask_name": entry.get("mask_name"),
        "name": entry.get("final_3d_object_name"),
        "final_3d_object_name": entry.get("final_3d_object_name"),
        "semantic_label": entry.get("semantic_label") or graph_obj.get("semantic_label"),
        "description": entry.get("description") or graph_obj.get("description"),
        "vlm_confidence": entry.get("vlm_confidence") or graph_obj.get("confidence"),
        "vlm_visibility": entry.get("vlm_visibility") or graph_obj.get("visibility"),
        "vlm_support_status": entry.get("vlm_support_status") or graph_obj.get("support_status"),
        "crop_rgb": entry.get("crop_rgb"),
        "sam3d_input_mask": entry.get("sam3d_input_mask"),
        "bbox_2d_xyxy": entry.get("bbox_2d_xyxy"),
        "bbox_2d_xywh": entry.get("bbox_2d_xywh"),
        "area_pixels": entry.get("area_pixels"),
        "bbox_3d_min": bmin,
        "bbox_3d_max": bmax,
        "bbox_3d_center": center,
        "bbox_3d_extent": extent_from_bounds(bmin, bmax),
        "pose_translation": translation_from_matrix(pose_matrix),
        "visual_path": export.get("visual_path"),
        "collision_path": export.get("collision_path"),
        "collision_paths": export.get("collision_paths"),
        "collision_part_count": export.get("collision_part_count"),
        "collision_decomposition": export.get("collision_decomposition"),
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
            delta_b_minus_a = vec_sub(bc, ac)
            relations.append(
                {
                    "a": a.get("name"),
                    "b": b.get("name"),
                    "delta_b_minus_a": delta_b_minus_a,
                    "center_distance": vec_norm(delta_b_minus_a),
                    "xy_distance": vec_norm([delta_b_minus_a[0], delta_b_minus_a[1], 0.0]),
                }
            )
    relations.sort(key=lambda item: float(item.get("xy_distance", 1e9)))
    return relations


def compact_scene_graph(scene_graph: dict[str, Any], scene_graph_path: Path | None) -> dict[str, Any] | None:
    if not scene_graph:
        return None
    return {
        "path": str(scene_graph_path) if scene_graph_path else None,
        "schema": scene_graph.get("schema"),
        "status": scene_graph.get("status"),
        "scope": scene_graph.get("scope"),
        "scope_image": scene_graph.get("scope_image"),
        "visualization": scene_graph.get("visualization"),
        "objects": [
            {
                "mask_id": item.get("mask_id"),
                "mask_name": item.get("mask_name"),
                "semantic_label": item.get("semantic_label"),
                "description": item.get("description"),
                "confidence": item.get("confidence"),
                "visibility": item.get("visibility"),
                "support_status": item.get("support_status"),
            }
            for item in scene_graph.get("objects") or []
        ],
        "relations": [
            {
                "subject_mask_id": item.get("subject_mask_id"),
                "object_mask_id": item.get("object_mask_id"),
                "relation": item.get("relation"),
                "confidence": item.get("confidence"),
                "evidence": item.get("evidence"),
            }
            for item in scene_graph.get("relations") or []
        ],
        "support_relations": [
            {
                "upper_mask_id": item.get("upper_mask_id"),
                "lower_mask_id": item.get("lower_mask_id"),
                "confidence": item.get("confidence"),
                "reason": item.get("reason"),
                "source": item.get("source"),
            }
            for item in scene_graph.get("support_relations") or []
        ],
        "root_mask_ids": scene_graph.get("root_mask_ids"),
    }


def compact_support(manifest: dict[str, Any]) -> dict[str, Any]:
    support = manifest.get("support") or {}
    return {
        "bbox_snap_mode": support.get("bbox_snap_mode"),
        "support_relation_source": support.get("support_relation_source"),
        "support_gap": support.get("support_gap"),
        "coordinate_system": support.get("coordinate_system"),
        "support_relations": [
            {
                "upper_name": item.get("upper_name"),
                "upper_mask_id": item.get("upper_mask_id"),
                "lower_name": item.get("lower_name"),
                "lower_mask_id": item.get("lower_mask_id"),
                "confidence": item.get("confidence"),
                "reason": item.get("reason"),
                "source": item.get("source"),
                "bottom_to_lower_top_gap": item.get("bottom_to_lower_top_gap"),
            }
            for item in support.get("support_relations") or []
        ],
        "scene_graph_support_relations": [
            {
                "upper_mask_id": item.get("upper_mask_id"),
                "lower_mask_id": item.get("lower_mask_id"),
                "confidence": item.get("confidence"),
                "reason": item.get("reason"),
                "source": item.get("source"),
            }
            for item in support.get("scene_graph_support_relations") or []
        ],
    }


def build(final_manifest: Path) -> dict[str, Any]:
    manifest = json.loads(final_manifest.read_text(encoding="utf-8"))
    session_dir = Path(manifest.get("inputs", {}).get("session_dir", final_manifest.parents[1]))
    scene_graph_path = resolve_scene_graph_path(manifest, session_dir)
    scene_graph = load_json_if_exists(scene_graph_path)
    graph_objects = scene_graph_objects_by_mask(scene_graph)
    objects = [object_card(entry, graph_objects) for entry in manifest.get("objects", [])]
    overlay = session_dir / "segmentation" / "overlay.png"
    return {
        "schema": "simulator_scene_digest.v1",
        "source_final_scene_manifest": str(final_manifest.resolve()),
        "session_id": manifest.get("session_id"),
        "coordinate_system": "Blender/SAPIEN Z-up; gravity is negative Z.",
        "ground_plane": GROUND_PLANE,
        "inputs": {
            "image": manifest.get("inputs", {}).get("image"),
            "overlay": str(overlay) if overlay.is_file() else None,
            "label_mask": manifest.get("inputs", {}).get("label_mask"),
            "final_blend": manifest.get("outputs", {}).get("final_blend"),
            "scene_graph": str(scene_graph_path) if scene_graph_path else None,
        },
        "objects": objects,
        "relations": build_relations(objects),
        "scene_graph": compact_scene_graph(scene_graph, scene_graph_path),
        "support": compact_support(manifest),
        "notes": [
            "ground plane is fixed at z=0 for simulator_assistance.",
            "upstream scene_graph semantic labels and support relations are hints; simulator_assistance still owns task roles and executable physics parameters.",
        ],
    }


def main() -> int:
    args = parse_args()
    if args.reuse_existing and args.output.is_file() and args.output.stat().st_mtime >= args.final_manifest.stat().st_mtime:
        print(f"Reusing scene digest: {args.output}")
        return 0
    digest = build(args.final_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(digest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote scene digest: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
