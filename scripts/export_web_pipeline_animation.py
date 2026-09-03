#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


try:
    import bpy  # type: ignore
    from mathutils import Matrix, Vector  # type: ignore
except Exception:  # pragma: no cover - only available inside Blender.
    bpy = None
    Matrix = None
    Vector = None


DEFAULT_BLENDER = os.environ.get("BLENDER_BIN", "blender")
WORKFLOW_ROOT = Path(__file__).resolve().parents[1]


def running_inside_blender() -> bool:
    if bpy is None:
        return False
    executable = Path(sys.executable).name.lower()
    binary_path = str(getattr(getattr(bpy, "app", None), "binary_path", "") or "").lower()
    return "blender" in executable or "blender" in Path(binary_path).name.lower()


STAGES = [
    {
        "key": "raw",
        "title": "01 SAM3D raw",
        "blend": "sam3d.blend",
        "description": "Direct SAM3D asset before MoGe upright, camera refinement, scene-graph correction, or physics.",
        "diagnostic": "Check object identity, texture, gross geometry, and whether the raw asset already looks wrong.",
    },
    {
        "key": "rotated",
        "title": "02 MoGe upright rotation",
        "blend": "sam3d_moge_rotated.blend",
        "description": "Estimate the ground normal from MoGe plus the optional GSAM ground mask, then rotate the scene to Z-up.",
        "diagnostic": "If this stage tilts, inspect the ground normal estimate and coordinate conversion.",
    },
    {
        "key": "grounded",
        "title": "03 Grounded to z=0",
        "blend": "sam3d_moge_grounded.blend",
        "description": "Translate the upright scene onto the z=0 support plane.",
        "diagnostic": "If all objects float or sink together, inspect global grounding and floor selection.",
    },
    {
        "key": "optimized",
        "title": "04 Differentiable rendering refinement",
        "blend": "sam3d_moge_optimized.blend",
        "description": "Optimize camera extrinsics, then per-object translation, yaw, and scale against 2D masks.",
        "diagnostic": "If masks align but 3D becomes implausible, inspect object pose constraints and accept criteria.",
    },
    {
        "key": "separated",
        "title": "05 Scene-graph + convex separation",
        "blend": "sam3d_moge_separated.blend",
        "description": "Use support relations and convex collision checks to correct support and resolve collisions.",
        "diagnostic": "If supported objects jump apart or support breaks, inspect scene graph relations and convex proxies.",
    },
]

GRAVITY_STAGE = {
    "key": "gravity",
    "title": "06 SAPIEN gravity settling",
    "blend": "sam3d_moge_separated_gravity_animation.blend",
    "description": "Replay the rigid-body gravity settling trajectory.",
    "diagnostic": "If the final state slides or collapses, inspect collision decomposition, friction, damping, and penetrations.",
}


def split_argv() -> list[str]:
    argv = sys.argv
    if "--" in argv:
        return argv[argv.index("--") + 1 :]
    return argv[1:]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a lightweight web-loadable pipeline animation manifest from stage Blend files and compressed web GLBs."
    )
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--final-manifest", type=Path, default=None)
    parser.add_argument("--web-assets-manifest", type=Path, default=None)
    parser.add_argument("--frames-per-static-stage", type=int, default=24)
    parser.add_argument("--transition-frames", type=int, default=16)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--gravity-frame-step", type=int, default=1)
    parser.add_argument("--blender-bin", default=DEFAULT_BLENDER)
    return parser


def run_in_blender(args: argparse.Namespace) -> int:
    script = Path(__file__).resolve()
    cmd = [
        str(args.blender_bin),
        "-b",
        "--python",
        str(script),
        "--",
        "--session",
        str(args.session),
        "--frames-per-static-stage",
        str(args.frames_per_static_stage),
        "--transition-frames",
        str(args.transition_frames),
        "--fps",
        str(args.fps),
        "--gravity-frame-step",
        str(args.gravity_frame_step),
    ]
    if args.output_dir is not None:
        cmd.extend(["--output-dir", str(args.output_dir)])
    if args.output is not None:
        cmd.extend(["--output", str(args.output)])
    if args.final_manifest is not None:
        cmd.extend(["--final-manifest", str(args.final_manifest)])
    if args.web_assets_manifest is not None:
        cmd.extend(["--web-assets-manifest", str(args.web_assets_manifest)])
    return subprocess.run(cmd, check=False).returncode


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def canonical_object_name(name: str) -> str:
    return re.sub(r"\.\d{3}$", "", str(name))


def mask_id_from_name(name: str) -> int | None:
    match = re.search(r"mask[_-](\d+)", str(name))
    if not match:
        return None
    return int(match.group(1))


def matrix_rows(matrix: Any) -> list[list[float]]:
    return [[float(matrix[row][col]) for col in range(4)] for row in range(4)]


def matrix_from_rows(rows: Any) -> Any:
    assert Matrix is not None
    if not isinstance(rows, list) or len(rows) < 4:
        return Matrix.Identity(4)
    return Matrix([[float(rows[row][col]) for col in range(4)] for row in range(4)])


def vector_from_array(values: Any) -> Any:
    assert Vector is not None
    if not isinstance(values, list) or len(values) < 3:
        return Vector((0.0, 0.0, 0.0))
    return Vector((float(values[0]), float(values[1]), float(values[2])))


def bbox_center_from_min_max(bbox_min: Any, bbox_max: Any) -> Any:
    assert Vector is not None
    if not isinstance(bbox_min, list) or not isinstance(bbox_max, list):
        return Vector((0.0, 0.0, 0.0))
    return (vector_from_array(bbox_min) + vector_from_array(bbox_max)) * 0.5


def object_visual_bbox_center(obj: Any) -> list[float]:
    assert Vector is not None
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    center = sum(corners, Vector((0.0, 0.0, 0.0))) / max(1, len(corners))
    return [float(center[0]), float(center[1]), float(center[2])]


def corrected_stage_matrix(stage_matrix_rows: Any, stage_center: Any, base_matrix_rows: Any, base_center: Any) -> list[list[float]]:
    assert Matrix is not None
    stage_matrix = matrix_from_rows(stage_matrix_rows)
    base_matrix = matrix_from_rows(base_matrix_rows)
    try:
        predicted_center = stage_matrix @ base_matrix.inverted() @ base_center
    except Exception:
        predicted_center = base_center.copy()
    delta = stage_center - predicted_center
    correction = Matrix.Translation(delta)
    return matrix_rows(correction @ stage_matrix)


def asset_ref(path_value: str | None, base_dir: Path) -> str | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    if path.is_absolute():
        try:
            return os.path.relpath(path.resolve(), base_dir.resolve()).replace(os.sep, "/")
        except Exception:
            return str(path)
    return str(path).replace(os.sep, "/")


def original_camera_json_path(result_dir: Path) -> Path | None:
    for name in (
        "sam3d_moge_separated_original_input_camera.json",
        "sam3d_moge_optimized_original_input_camera.json",
        "sam3d_moge_differentiable_camera_pose_optimization.json",
    ):
        path = result_dir / name
        if path.is_file():
            return path
    return None


def camera_payload(result_dir: Path, output_dir: Path) -> dict[str, Any] | None:
    path = original_camera_json_path(result_dir)
    if path is None:
        return None
    raw = read_json(path)
    camera_to_world = raw.get("camera_to_world") or raw.get("optimized_camera_to_world")
    if not camera_to_world:
        return None
    payload = {
        "source_json": asset_ref(str(path), output_dir),
        "camera_source": raw.get("camera_source") or ("nvdiffrast_optimized" if raw.get("optimized_camera_to_world") else None),
        "camera_to_world": camera_to_world,
        "intrinsics_normalized": raw.get("intrinsics_normalized"),
        "intrinsics_pixels": raw.get("intrinsics_pixels"),
        "image_size": raw.get("image_size"),
        "fov_degrees": raw.get("fov_degrees"),
        "blender_camera": raw.get("blender_camera"),
    }
    return {key: value for key, value in payload.items() if value is not None}


def link_mask_objects(source: Path, wanted_names: set[str]) -> list[Any]:
    assert bpy is not None
    with bpy.data.libraries.load(str(source), link=True) as (data_from, data_to):
        data_to.objects = [
            name
            for name in data_from.objects
            if canonical_object_name(name) in wanted_names or mask_id_from_name(name) is not None
        ]
    coll = bpy.data.collections.new(f"__web_pipeline_animation_read_{source.stem}")
    bpy.context.scene.collection.children.link(coll)
    objects = []
    for obj in data_to.objects:
        if obj is None:
            continue
        canon = canonical_object_name(obj.name)
        if canon not in wanted_names:
            continue
        try:
            coll.objects.link(obj)
        except RuntimeError:
            pass
        objects.append(obj)
    return objects


def cleanup_linked_objects(objects: list[Any]) -> None:
    assert bpy is not None
    collections = set()
    for obj in objects:
        collections.update(obj.users_collection)
        bpy.data.objects.remove(obj, do_unlink=True)
    for coll in collections:
        if coll.name.startswith("__web_pipeline_animation_read_"):
            bpy.data.collections.remove(coll)


def object_frame_range(objects: list[Any]) -> tuple[int, int]:
    min_frame: float | None = None
    max_frame: float | None = None
    for obj in objects:
        action = getattr(getattr(obj, "animation_data", None), "action", None)
        if action is None:
            continue
        for fcurve in action.fcurves:
            for key in fcurve.keyframe_points:
                frame = float(key.co.x)
                min_frame = frame if min_frame is None else min(min_frame, frame)
                max_frame = frame if max_frame is None else max(max_frame, frame)
    if min_frame is None or max_frame is None:
        return 0, 0
    return int(round(min_frame)), int(round(max_frame))


def extract_static_stage(source: Path, wanted_names: set[str]) -> dict[str, dict[str, Any]]:
    assert bpy is not None
    objects = link_mask_objects(source, wanted_names)
    try:
        bpy.context.scene.frame_set(1)
        bpy.context.view_layer.update()
        return {
            canonical_object_name(obj.name): {
                "matrix_world": matrix_rows(obj.matrix_world),
                "bbox_center": object_visual_bbox_center(obj),
            }
            for obj in objects
        }
    finally:
        cleanup_linked_objects(objects)


def extract_gravity_stage(source: Path, wanted_names: set[str], frame_step: int) -> tuple[int, int, dict[int, dict[str, dict[str, Any]]]]:
    assert bpy is not None
    objects = link_mask_objects(source, wanted_names)
    try:
        start, end = object_frame_range(objects)
        if end <= start:
            start, end = 0, 0
        step = max(1, int(frame_step))
        frames = list(range(start, end + 1, step))
        if frames and frames[-1] != end:
            frames.append(end)
        if not frames:
            frames = [start]
        samples: dict[int, dict[str, list[list[float]]]] = {}
        for frame in frames:
            bpy.context.scene.frame_set(int(frame))
            bpy.context.view_layer.update()
            samples[int(frame)] = {
                canonical_object_name(obj.name): {
                    "matrix_world": matrix_rows(obj.matrix_world),
                    "bbox_center": object_visual_bbox_center(obj),
                }
                for obj in objects
            }
        return start, end, samples
    finally:
        cleanup_linked_objects(objects)


def build_timeline(static_stage_keys: list[str], gravity_start: int, gravity_end: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    frame = 0
    hold = max(1, int(args.frames_per_static_stage))
    transition = max(0, int(args.transition_frames))
    for index, key in enumerate(static_stage_keys):
        timeline.append({"kind": "hold", "stage": key, "frame_start": frame, "frame_end": frame + hold - 1})
        frame += hold
        if index + 1 < len(static_stage_keys) and transition > 0:
            timeline.append(
                {
                    "kind": "transition",
                    "from_stage": key,
                    "to_stage": static_stage_keys[index + 1],
                    "frame_start": frame,
                    "frame_end": frame + transition - 1,
                }
            )
            frame += transition
    gravity_duration = max(1, int(gravity_end) - int(gravity_start))
    timeline.append(
        {
            "kind": "gravity",
            "stage": "gravity",
            "frame_start": frame,
            "frame_end": frame + gravity_duration,
            "source_frame_start": int(gravity_start),
            "source_frame_end": int(gravity_end),
        }
    )
    return timeline


def update_final_manifest(final_manifest_path: Path, output_manifest: Path, output_dir: Path) -> None:
    final_manifest = read_json(final_manifest_path)
    if not final_manifest:
        return
    outputs = final_manifest.setdefault("outputs", {})
    if isinstance(outputs, dict):
        outputs["web_pipeline_animation_manifest"] = str(output_manifest)
        outputs["web_pipeline_animation_dir"] = str(output_dir)
    write_json(final_manifest_path, final_manifest)


def export_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if bpy is None:
        raise RuntimeError("This function must run inside Blender.")
    session_dir = args.session.expanduser().resolve()
    result_dir = session_dir / "results"
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else result_dir / "web_pipeline_animation")
    output_manifest = (args.output.expanduser().resolve() if args.output else output_dir / "animation_manifest.json")
    final_manifest_path = (
        args.final_manifest.expanduser().resolve()
        if args.final_manifest
        else result_dir / "final_scene_manifest.json"
    )
    final_manifest = read_json(final_manifest_path)
    web_manifest_path = (
        args.web_assets_manifest.expanduser().resolve()
        if args.web_assets_manifest
        else Path(final_manifest.get("outputs", {}).get("web_assets_manifest") or result_dir / "web_assets" / "manifest.json").expanduser().resolve()
    )
    web_manifest = read_json(web_manifest_path)
    if not web_manifest:
        raise FileNotFoundError(f"Missing web assets manifest: {web_manifest_path}")

    web_objects = []
    wanted_names: set[str] = set()
    for raw in web_manifest.get("objects") or []:
        name = canonical_object_name(raw.get("name") or raw.get("final_3d_object_name") or "")
        web_glb = raw.get("web_glb") or raw.get("web_asset_glb") or raw.get("visual_path")
        base_matrix = raw.get("matrix_world")
        if not name or not web_glb or not base_matrix:
            continue
        wanted_names.add(name)
        web_objects.append(
            {
                "name": name,
                "mask_id": raw.get("mask_id"),
                "label": raw.get("semantic_label") or raw.get("description") or name,
                "url": asset_ref(str(web_glb), output_dir),
                "base_matrix_world": base_matrix,
                "bbox_min": raw.get("bbox_min"),
                "bbox_max": raw.get("bbox_max"),
            }
        )
    if not web_objects:
        raise RuntimeError(f"No usable web objects found in {web_manifest_path}")

    static_samples: dict[str, dict[str, dict[str, Any]]] = {}
    stage_entries = []
    warnings: list[str] = []
    postprocess_report = read_json(result_dir / "sam3d_moge_report.json")
    report_stages = ((postprocess_report.get("outputs") or {}).get("stages") or {}) if postprocess_report else {}

    def stage_source(spec: dict[str, Any]) -> Path:
        key = str(spec["key"])
        report_stage = report_stages.get(key) or {}
        blend = report_stage.get("blend")
        if blend:
            path = Path(str(blend))
            if path.is_file():
                if report_stage.get("fallback_reason"):
                    warnings.append(f"{key} uses fallback stage: {report_stage.get('fallback_reason')}")
                return path
        default = result_dir / str(spec["blend"])
        if key == "optimized" and not default.is_file():
            fallback = result_dir / "sam3d_moge_grounded.blend"
            if fallback.is_file():
                warnings.append("optimized uses fallback stage: missing optimized blend")
                return fallback
        return default

    for spec in STAGES:
        source = stage_source(spec)
        if not source.is_file():
            warnings.append(f"missing stage blend: {source}")
            continue
        static_samples[spec["key"]] = extract_static_stage(source, wanted_names)
        stage_entries.append({**spec, "source_blend": source.name, "blend": source.name, "kind": "static"})

    gravity_source = result_dir / GRAVITY_STAGE["blend"]
    if gravity_source.is_file():
        gravity_start, gravity_end, gravity_samples = extract_gravity_stage(
            gravity_source,
            wanted_names,
            args.gravity_frame_step,
        )
        stage_entries.append({**GRAVITY_STAGE, "source_blend": gravity_source.name, "blend": GRAVITY_STAGE["blend"], "kind": "animated"})
    else:
        warnings.append(f"missing gravity blend: {gravity_source}")
        gravity_start = 0
        gravity_end = 0
        gravity_samples = {0: static_samples.get("separated", {})}

    for obj in web_objects:
        name = obj["name"]
        base_matrix = obj["base_matrix_world"]
        base_center = bbox_center_from_min_max(obj.get("bbox_min"), obj.get("bbox_max"))
        obj["stage_matrices"] = {
            key: corrected_stage_matrix(
                sample["matrix_world"],
                vector_from_array(sample["bbox_center"]),
                base_matrix,
                base_center,
            )
            for key, matrices in static_samples.items()
            if name in matrices
            for sample in [matrices[name]]
        }
        obj["stage_bbox_centers"] = {
            key: matrices[name]["bbox_center"]
            for key, matrices in static_samples.items()
            if name in matrices
        }
        obj["gravity_keyframes"] = [
            {
                "source_frame": frame,
                "matrix_world": sample["matrix_world"],
                "bbox_center": sample["bbox_center"],
            }
            for frame, matrices in sorted(gravity_samples.items())
            if name in matrices
            for sample in [matrices[name]]
        ]
        if not obj["gravity_keyframes"] and "separated" in obj["stage_matrices"]:
            obj["gravity_keyframes"] = [
                {"source_frame": gravity_start, "matrix_world": obj["stage_matrices"]["separated"]}
            ]

    timeline = build_timeline([stage["key"] for stage in stage_entries if stage["kind"] == "static"], gravity_start, gravity_end, args)
    total_frames = max((segment["frame_end"] for segment in timeline), default=0) + 1
    payload = {
        "schema": "fysiverse_web_pipeline_animation.v1",
        "status": "ok",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "session": session_dir.name,
        "session_id": session_dir.name,
        "final_scene_manifest": asset_ref(str(final_manifest_path), output_dir),
        "web_assets_manifest": asset_ref(str(web_manifest_path), output_dir),
        "visual_frame": "world",
        "fps": int(args.fps),
        "total_frames": total_frames,
        "camera": camera_payload(result_dir, output_dir),
        "timeline": timeline,
        "stages": stage_entries,
        "objects": web_objects,
        "warnings": warnings,
        "notes": {
            "asset_strategy": "Each object reuses the compressed final web GLB. Static stage matrices are bbox-center corrected so final web assets can approximate stages whose offsets were baked into mesh vertices. Gravity keyframes are sampled directly from the SAPIEN gravity animation Blend with no correction or interpolation.",
            "matrix_convention": "Blender row-major 4x4 matrix_world values.",
        },
    }
    write_json(output_manifest, payload)
    update_final_manifest(final_manifest_path, output_manifest, output_dir)
    return payload


def main() -> int:
    parser = build_parser()
    args = parser.parse_args(split_argv())
    if not running_inside_blender():
        return run_in_blender(args)
    payload = export_manifest(args)
    print(json.dumps({
        "status": payload.get("status"),
        "session_id": payload.get("session_id"),
        "objects": len(payload.get("objects") or []),
        "total_frames": payload.get("total_frames"),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
