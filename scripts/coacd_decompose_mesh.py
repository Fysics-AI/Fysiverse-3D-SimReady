#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

import coacd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decompose a triangle mesh into convex OBJ parts with CoACD.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--part-prefix", default="part")
    parser.add_argument("--save-simplified-source", type=Path, default=None, help="Optional OBJ path for the exact simplified geometry passed to CoACD.")
    parser.add_argument("--max-source-faces", type=int, default=0)
    parser.add_argument("--simplification-backend", choices=["fast_simplification", "none"], default="fast_simplification")
    parser.add_argument("--simplification-agg", type=float, default=7.0)
    parser.add_argument("--max-convex-parts", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--preprocess-mode", default="auto")
    parser.add_argument("--preprocess-resolution", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=2000)
    parser.add_argument("--mcts-nodes", type=int, default=20)
    parser.add_argument("--mcts-iterations", type=int, default=80)
    parser.add_argument("--mcts-max-depth", type=int, default=3)
    parser.add_argument("--max-ch-vertex", type=int, default=256)
    parser.add_argument("--apx-mode", default="ch")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--real-metric", action="store_true")
    parser.add_argument("--no-merge", action="store_true")
    parser.add_argument("--decimate", action="store_true")
    return parser.parse_args()


def validate_simplified_mesh(vertices: np.ndarray, faces: np.ndarray, backend: str) -> trimesh.Trimesh:
    out = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=np.float64), faces=np.asarray(faces, dtype=np.int32), process=False)
    if len(out.faces) < 4 or len(out.vertices) < 4:
        raise RuntimeError(f"{backend} produced a degenerate mesh: vertices={len(out.vertices)} faces={len(out.faces)}")
    return out


def simplify_with_fast_simplification(mesh: trimesh.Trimesh, max_faces: int, agg: float) -> trimesh.Trimesh:
    import fast_simplification

    vertices, faces = fast_simplification.simplify(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int32),
        target_count=int(max_faces),
        agg=float(agg),
    )
    return validate_simplified_mesh(vertices, faces, "fast_simplification")


def simplify_mesh_for_coacd(mesh: trimesh.Trimesh, max_faces: int, *, backend: str, agg: float) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    source_faces = int(len(mesh.faces))
    source_vertices = int(len(mesh.vertices))
    info: dict[str, Any] = {
        "source_vertex_count": source_vertices,
        "source_face_count": source_faces,
        "max_source_faces": int(max_faces),
        "requested_backend": str(backend),
        "simplification_agg": float(agg),
        "simplified": False,
        "backend": "none",
        "attempts": [],
    }
    if int(max_faces) <= 0 or source_faces <= int(max_faces):
        info["vertex_count"] = source_vertices
        info["face_count"] = source_faces
        return mesh, info
    if str(backend) == "none":
        info["vertex_count"] = source_vertices
        info["face_count"] = source_faces
        return mesh, info
    if str(backend) != "fast_simplification":
        raise ValueError(f"unsupported simplification backend: {backend}")
    start = time.monotonic()
    try:
        out = simplify_with_fast_simplification(mesh, max_faces, agg)
        elapsed = float(time.monotonic() - start)
        info["attempts"].append({"backend": "fast_simplification", "status": "ok", "elapsed_sec": elapsed})
        info.update(
            {
                "simplified": True,
                "backend": "fast_simplification",
                "vertex_count": int(len(out.vertices)),
                "face_count": int(len(out.faces)),
                "elapsed_sec": elapsed,
            }
        )
        return out, info
    except Exception as exc:
        info["attempts"].append({"backend": "fast_simplification", "status": "failed", "elapsed_sec": float(time.monotonic() - start), "error": str(exc)})
    info.update(
        {
            "simplified": False,
            "simplify_error": "fast_simplification failed",
            "vertex_count": source_vertices,
            "face_count": source_faces,
        }
    )
    return mesh, info


def load_mesh(path: Path, *, max_source_faces: int, simplification_backend: str, simplification_agg: float) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    elif isinstance(loaded, trimesh.Scene):
        geometries = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not geometries:
            raise RuntimeError(f"No triangle mesh geometry in {path}")
        mesh = trimesh.util.concatenate(geometries)
    else:
        raise TypeError(f"Unsupported mesh type from {path}: {type(loaded)!r}")
    if mesh.vertices is None or mesh.faces is None or len(mesh.vertices) < 4 or len(mesh.faces) < 4:
        raise RuntimeError(f"Mesh is too small for CoACD: vertices={len(mesh.vertices)} faces={len(mesh.faces)}")
    mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices, dtype=np.float64), faces=np.asarray(mesh.faces, dtype=np.int32), process=False)
    return simplify_mesh_for_coacd(mesh, max_source_faces, backend=simplification_backend, agg=simplification_agg)


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray, *, header: str = "CoACD convex collision part") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {header}\n")
        for x, y, z in np.asarray(vertices, dtype=np.float64):
            f.write(f"v {float(x):.9g} {float(y):.9g} {float(z):.9g}\n")
        for face in np.asarray(faces, dtype=np.int64):
            if len(face) < 3:
                continue
            indices = [int(idx) + 1 for idx in face[:3]]
            f.write(f"f {indices[0]} {indices[1]} {indices[2]}\n")


def write_source_obj(path: Path, mesh: trimesh.Trimesh, mesh_info: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_obj(
        path,
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int32),
        header="CoACD simplified source mesh; geometry only, no UV/material/texture",
    )
    return {
        "path": str(path),
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "is_simplified": bool(mesh_info.get("simplified", False)),
        "backend": str(mesh_info.get("backend", "none")),
        "texture_support": "none",
        "note": "This OBJ stores only geometry vertices/faces passed to CoACD; UVs/materials/textures are not preserved by this simplification path.",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    mesh, mesh_info = load_mesh(
        args.input,
        max_source_faces=int(args.max_source_faces),
        simplification_backend=str(args.simplification_backend),
        simplification_agg=float(args.simplification_agg),
    )
    simplified_source: dict[str, Any] | None = None
    if args.save_simplified_source is not None:
        simplified_source = write_source_obj(args.save_simplified_source, mesh, mesh_info)
    coacd_mesh = coacd.Mesh(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int32),
    )
    max_convex_hull = int(args.max_convex_parts)
    if max_convex_hull <= 0:
        max_convex_hull = -1
    parts = coacd.run_coacd(
        coacd_mesh,
        threshold=float(args.threshold),
        max_convex_hull=max_convex_hull,
        preprocess_mode=str(args.preprocess_mode),
        preprocess_resolution=int(args.preprocess_resolution),
        resolution=int(args.resolution),
        mcts_nodes=int(args.mcts_nodes),
        mcts_iterations=int(args.mcts_iterations),
        mcts_max_depth=int(args.mcts_max_depth),
        merge=not bool(args.no_merge),
        decimate=bool(args.decimate),
        max_ch_vertex=int(args.max_ch_vertex),
        apx_mode=str(args.apx_mode),
        seed=int(args.seed),
        real_metric=bool(args.real_metric),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    part_entries: list[dict[str, Any]] = []
    for idx, part in enumerate(parts):
        vertices, faces = part
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int32)
        if len(vertices) < 4 or len(faces) < 4:
            continue
        part_path = args.output_dir / f"{args.part_prefix}_{idx:03d}.obj"
        write_obj(part_path, vertices, faces)
        part_entries.append(
            {
                "index": idx,
                "path": str(part_path),
                "vertex_count": int(len(vertices)),
                "face_count": int(len(faces)),
            }
        )
    if not part_entries:
        raise RuntimeError("CoACD returned no usable convex parts")
    report = {
        "status": "ok",
        "method": "coacd",
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "part_count": len(part_entries),
        "parts": part_entries,
        "source_vertex_count": int(mesh_info.get("source_vertex_count", len(mesh.vertices))),
        "source_face_count": int(mesh_info.get("source_face_count", len(mesh.faces))),
        "coacd_input_vertex_count": int(len(mesh.vertices)),
        "coacd_input_face_count": int(len(mesh.faces)),
        "simplified_source_mesh": simplified_source,
        "source_simplification": mesh_info,
        "config": {
            "max_source_faces": int(args.max_source_faces),
            "simplification_backend": str(args.simplification_backend),
            "simplification_agg": float(args.simplification_agg),
            "max_convex_parts": int(args.max_convex_parts),
            "threshold": float(args.threshold),
            "preprocess_mode": str(args.preprocess_mode),
            "preprocess_resolution": int(args.preprocess_resolution),
            "resolution": int(args.resolution),
            "mcts_nodes": int(args.mcts_nodes),
            "mcts_iterations": int(args.mcts_iterations),
            "mcts_max_depth": int(args.mcts_max_depth),
            "max_ch_vertex": int(args.max_ch_vertex),
            "apx_mode": str(args.apx_mode),
            "seed": int(args.seed),
            "real_metric": bool(args.real_metric),
            "merge": not bool(args.no_merge),
            "decimate": bool(args.decimate),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    report = run(parse_args())
    print(json.dumps({"status": report["status"], "part_count": report["part_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
