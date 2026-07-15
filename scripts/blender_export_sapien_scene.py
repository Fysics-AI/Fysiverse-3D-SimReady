#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import bpy
from mathutils import Vector


SKIP_PREFIXES = (
    "__gravity_ground__",
    "__rigidbody_preview_ground__",
    "__sapien_ground__",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export each mesh object from a Blend scene for SAPIEN simulation.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--collision-decomposition-method", choices=["none", "coacd"], default="none")
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
    parser.add_argument("--coacd-no-merge", action="store_true")
    parser.add_argument("--coacd-decimate", action="store_true")
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    return parser.parse_args(argv)


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return cleaned or "object"


def should_skip(obj: bpy.types.Object) -> bool:
    if obj.type != "MESH":
        return True
    if any(obj.name.startswith(prefix) for prefix in SKIP_PREFIXES):
        return True
    return False


def world_bbox_from_points(points: list[Vector]) -> tuple[Vector, Vector]:
    return (
        Vector((min(v.x for v in points), min(v.y for v in points), min(v.z for v in points))),
        Vector((max(v.x for v in points), max(v.y for v in points), max(v.z for v in points))),
    )


def matrix_translation(center: Vector) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, float(center.x)],
        [0.0, 1.0, 0.0, float(center.y)],
        [0.0, 0.0, 1.0, float(center.z)],
        [0.0, 0.0, 0.0, 1.0],
    ]


def write_collision_obj(mesh: bpy.types.Mesh, path: Path) -> None:
    mesh.calc_loop_triangles()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# SAPIEN convex collision mesh\n")
        for vertex in mesh.vertices:
            co = vertex.co
            f.write(f"v {co.x:.9g} {co.y:.9g} {co.z:.9g}\n")
        for tri in mesh.loop_triangles:
            indices = [idx + 1 for idx in tri.vertices]
            f.write(f"f {indices[0]} {indices[1]} {indices[2]}\n")


def run_coacd_decomposition(source_obj: Path, object_dir: Path, base: str, args: argparse.Namespace) -> dict[str, Any]:
    if str(args.collision_decomposition_method) != "coacd":
        return {
            "status": "disabled",
            "method": "none",
            "collision_paths": [str(source_obj)],
        }
    parts_dir = object_dir / f"{base}_coacd"
    manifest_path = parts_dir / "coacd_manifest.json"
    cmd = [
        str(args.coacd_conda_bin),
        "run",
        "--no-capture-output",
        "-n",
        str(args.coacd_env),
        "python",
        str(Path(__file__).resolve().parent / "coacd_decompose_mesh.py"),
        "--input",
        str(source_obj),
        "--output-dir",
        str(parts_dir),
        "--manifest",
        str(manifest_path),
        "--part-prefix",
        f"{base}_coacd",
        "--save-simplified-source",
        str(parts_dir / f"{base}_coacd_simplified_source.obj"),
        "--max-source-faces",
        str(args.coacd_source_max_faces),
        "--simplification-backend",
        str(args.coacd_source_simplification_backend),
        "--simplification-agg",
        str(args.coacd_source_simplification_agg),
        "--max-convex-parts",
        str(args.coacd_max_convex_parts),
        "--threshold",
        str(args.coacd_threshold),
        "--preprocess-mode",
        str(args.coacd_preprocess_mode),
        "--preprocess-resolution",
        str(args.coacd_preprocess_resolution),
        "--resolution",
        str(args.coacd_resolution),
        "--mcts-nodes",
        str(args.coacd_mcts_nodes),
        "--mcts-iterations",
        str(args.coacd_mcts_iterations),
        "--mcts-max-depth",
        str(args.coacd_mcts_max_depth),
        "--max-ch-vertex",
        str(args.coacd_max_ch_vertex),
        "--apx-mode",
        str(args.coacd_apx_mode),
        "--seed",
        str(args.coacd_seed),
    ]
    if args.coacd_real_metric:
        cmd.append("--real-metric")
    if args.coacd_no_merge:
        cmd.append("--no-merge")
    if args.coacd_decimate:
        cmd.append("--decimate")
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=max(1, int(args.coacd_timeout)),
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        collision_paths = [str(Path(part["path"])) for part in manifest.get("parts") or [] if part.get("path")]
        if not collision_paths:
            raise RuntimeError("CoACD manifest did not contain any part paths")
        return {
            "status": "ok",
            "method": "coacd",
            "source_path": str(source_obj),
            "simplified_source_path": ((manifest.get("simplified_source_mesh") or {}).get("path")),
            "simplified_source_mesh": manifest.get("simplified_source_mesh"),
            "manifest_path": str(manifest_path),
            "collision_paths": collision_paths,
            "part_count": len(collision_paths),
            "stdout": proc.stdout[-4000:],
            "config": manifest.get("config") or {},
        }
    except Exception as exc:
        return {
            "status": "fallback_single_mesh",
            "method": "coacd",
            "source_path": str(source_obj),
            "collision_paths": [str(source_obj)],
            "part_count": 1,
            "error": str(exc),
        }


def apply_collision_decompositions(objects: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if str(args.collision_decomposition_method) != "coacd":
        for entry in objects:
            collision_path = str(entry["collision_path"])
            entry["collision_paths"] = [collision_path]
            entry["collision_decomposition"] = {"status": "disabled", "method": "none", "collision_paths": [collision_path]}
            entry["collision_part_count"] = 1
        return

    workers = max(1, int(args.coacd_workers))
    start = time.monotonic()
    print(f"[sapien-export] CoACD start objects={len(objects)} workers={workers}", flush=True)

    def run_one(entry: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        decomposition = run_coacd_decomposition(Path(entry["collision_path"]), Path(entry["_object_dir"]), str(entry["_base"]), args)
        return int(entry["index"]), decomposition

    if workers == 1 or len(objects) <= 1:
        results = [run_one(entry) for entry in objects]
    else:
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(objects))) as executor:
            future_to_entry = {executor.submit(run_one, entry): entry for entry in objects}
            for future in concurrent.futures.as_completed(future_to_entry):
                entry = future_to_entry[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append(
                        (
                            int(entry["index"]),
                            {
                                "status": "fallback_single_mesh",
                                "method": "coacd",
                                "source_path": str(entry["collision_path"]),
                                "collision_paths": [str(entry["collision_path"])],
                                "part_count": 1,
                                "error": str(exc),
                            },
                        )
                    )

    by_index = {idx: decomposition for idx, decomposition in results}
    for entry in objects:
        decomposition = by_index[int(entry["index"])]
        entry["collision_paths"] = decomposition.get("collision_paths") or [str(entry["collision_path"])]
        entry["collision_decomposition"] = {key: value for key, value in decomposition.items() if key != "collision_paths"}
        entry["collision_part_count"] = len(entry["collision_paths"])
        entry.pop("_object_dir", None)
        entry.pop("_base", None)
    print(f"[sapien-export] CoACD done elapsed={time.monotonic() - start:.2f}s", flush=True)


def export_visual_glb(obj: bpy.types.Object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    kwargs: dict[str, Any] = {
        "filepath": str(path),
        "export_format": "GLB",
        "use_selection": True,
        "export_texcoords": True,
        "export_normals": True,
        "export_materials": "EXPORT",
    }
    try:
        bpy.ops.export_scene.gltf(**kwargs)
    except TypeError:
        kwargs.pop("export_materials", None)
        bpy.ops.export_scene.gltf(**kwargs)


def make_actor_frame_object(obj: bpy.types.Object, index: int, out_dir: Path, args: argparse.Namespace) -> dict[str, Any] | None:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = bpy.data.meshes.new_from_object(eval_obj, depsgraph=depsgraph)
    if len(mesh.vertices) == 0:
        bpy.data.meshes.remove(mesh)
        return None

    world_points = [obj.matrix_world @ vertex.co for vertex in mesh.vertices]
    bbox_min, bbox_max = world_bbox_from_points(world_points)
    center = (bbox_min + bbox_max) * 0.5

    for vertex, world_co in zip(mesh.vertices, world_points):
        vertex.co = world_co - center
    mesh.update()

    temp_obj = bpy.data.objects.new(f"__sapien_export_{index:04d}_{safe_name(obj.name)}", mesh)
    bpy.context.collection.objects.link(temp_obj)
    temp_obj.matrix_world.identity()
    for mat in obj.data.materials:
        temp_obj.data.materials.append(mat)
    bpy.context.view_layer.update()

    object_dir = out_dir / "objects"
    base = f"{index:04d}_{safe_name(obj.name)}"
    visual_path = object_dir / f"{base}.glb"
    collision_path = object_dir / f"{base}_collision.obj"
    export_visual_glb(temp_obj, visual_path)
    write_collision_obj(mesh, collision_path)

    bpy.data.objects.remove(temp_obj, do_unlink=True)
    bpy.data.meshes.remove(mesh)

    return {
        "index": index,
        "name": obj.name,
        "visual_path": str(visual_path),
        "collision_path": str(collision_path),
        "collision_paths": [str(collision_path)],
        "collision_decomposition": {"status": "pending", "method": str(args.collision_decomposition_method)},
        "collision_part_count": 1,
        "_object_dir": str(object_dir),
        "_base": base,
        "initial_pose": matrix_translation(center),
        "bbox_min": [float(bbox_min.x), float(bbox_min.y), float(bbox_min.z)],
        "bbox_max": [float(bbox_max.x), float(bbox_max.y), float(bbox_max.z)],
        "bbox_center": [float(center.x), float(center.y), float(center.z)],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    bpy.ops.wm.open_mainfile(filepath=str(args.input))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    objects: list[dict[str, Any]] = []
    bounds_min = Vector((float("inf"), float("inf"), float("inf")))
    bounds_max = Vector((float("-inf"), float("-inf"), float("-inf")))
    for obj in list(bpy.context.scene.objects):
        if should_skip(obj):
            continue
        entry = make_actor_frame_object(obj, len(objects), args.output_dir, args)
        if entry is None:
            continue
        objects.append(entry)
        bmin = Vector(entry["bbox_min"])
        bmax = Vector(entry["bbox_max"])
        bounds_min = Vector((min(bounds_min.x, bmin.x), min(bounds_min.y, bmin.y), min(bounds_min.z, bmin.z)))
        bounds_max = Vector((max(bounds_max.x, bmax.x), max(bounds_max.y, bmax.y), max(bounds_max.z, bmax.z)))

    apply_collision_decompositions(objects, args)

    if not objects:
        raise RuntimeError(f"No mesh objects exported from {args.input}")

    manifest = {
        "input_blend": str(args.input),
        "output_dir": str(args.output_dir),
        "objects": objects,
        "bounds_min": [float(bounds_min.x), float(bounds_min.y), float(bounds_min.z)],
        "bounds_max": [float(bounds_max.x), float(bounds_max.y), float(bounds_max.z)],
        "coordinate_system": "Blender/SAPIEN Z-up; each exported mesh is in actor frame centered at its world-space bbox center.",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    args = parse_args()
    manifest = run(args)
    print(f"Exported {len(manifest['objects'])} mesh objects for SAPIEN")
    print(f"Wrote manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
