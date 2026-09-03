#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SKIP_PREFIXES = (
    "__gravity_ground__",
    "__rigidbody_preview_ground__",
    "__sapien_ground__",
)


def split_argv() -> list[str]:
    argv = sys.argv
    if "--" in argv:
        return argv[argv.index("--") + 1 :]
    return argv[1:]


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("._")
    return cleaned or "object"


def read_json(path: Path | None) -> dict[str, Any]:
    if not path or not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def collect_existing(paths: list[Path | str | None]) -> list[str]:
    out: list[str] = []
    for item in paths:
        if not item:
            continue
        path = Path(str(item))
        if path.is_file():
            out.append(str(path))
    return out


def command_available(command: str) -> str | None:
    if not command:
        return None
    if "/" in command:
        path = Path(command).expanduser()
        return str(path) if path.is_file() and os.access(path, os.X_OK) else None
    found = shutil.which(command)
    if found:
        return found
    for root in (Path.cwd(), Path(__file__).resolve().parents[1]):
        candidate = root / "node_modules" / ".bin" / command
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def run_command(cmd: list[str], timeout: int) -> str:
    proc = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=max(1, int(timeout)),
    )
    return proc.stdout


def file_size(path: Path) -> int | None:
    try:
        return int(path.stat().st_size)
    except Exception:
        return None


def copy_uncompressed(src: Path, dst: Path, reason: str) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "status": "copied_uncompressed",
        "method": "none",
        "input": str(src),
        "output": str(dst),
        "input_bytes": file_size(src),
        "output_bytes": file_size(dst),
        "warning": reason,
    }


def failed_compression_result(src: Path, dst: Path | None, method: str, warning: str) -> dict[str, Any]:
    return {
        "status": "failed",
        "method": method,
        "input": str(src),
        "output": str(dst) if dst is not None else None,
        "input_bytes": file_size(src),
        "output_bytes": file_size(dst) if dst is not None and dst.is_file() else None,
        "warning": warning,
    }


def gltfpack_command(args: argparse.Namespace) -> list[str] | None:
    gltfpack = command_available(str(args.gltfpack_bin))
    if gltfpack:
        return [gltfpack]
    if args.allow_npx_gltfpack:
        npx = command_available("npx")
        if npx:
            return [npx, "--yes", "gltfpack"]
    return None


def compress_gltfpack(src: Path, dst: Path, args: argparse.Namespace, *, texture_compress: bool) -> dict[str, Any]:
    cmd_prefix = gltfpack_command(args)
    if cmd_prefix is None:
        raise FileNotFoundError(f"gltfpack not found: {args.gltfpack_bin}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    compression_flag = str(getattr(args, "gltfpack_compression", "cc") or "cc")
    if compression_flag not in {"c", "cc", "cz"}:
        compression_flag = "cc"
    base_cmd = cmd_prefix + ["-i", str(src), "-o", str(dst), f"-{compression_flag}", "-ce", str(args.gltfpack_compression_extension), "-kn", "-ke"]
    texture_args: list[str] = []
    if getattr(args, "gltfpack_texture_quality", None) is not None:
        texture_args.extend(["-tq", str(args.gltfpack_texture_quality)])
    if getattr(args, "gltfpack_texture_limit", None) is not None:
        texture_args.extend(["-tl", str(args.gltfpack_texture_limit)])
    if getattr(args, "gltfpack_texture_scale", None) is not None:
        texture_args.extend(["-ts", str(args.gltfpack_texture_scale)])
    simplify_args: list[str] = []
    if getattr(args, "gltfpack_simplify_ratio", None) is not None:
        simplify_args.extend(["-si", str(args.gltfpack_simplify_ratio)])
    if getattr(args, "gltfpack_simplify_error", None) is not None:
        simplify_args.extend(["-se", str(args.gltfpack_simplify_error)])
    if getattr(args, "gltfpack_simplify_aggressive", False):
        simplify_args.append("-sa")
    if getattr(args, "gltfpack_simplify_permissive", False):
        simplify_args.append("-sp")
    if getattr(args, "gltfpack_position_bits", None) is not None:
        simplify_args.extend(["-vp", str(args.gltfpack_position_bits)])
    if getattr(args, "gltfpack_texcoord_bits", None) is not None:
        simplify_args.extend(["-vt", str(args.gltfpack_texcoord_bits)])
    if getattr(args, "gltfpack_normal_bits", None) is not None:
        simplify_args.extend(["-vn", str(args.gltfpack_normal_bits)])
    base_cmd.extend(texture_args)
    base_cmd.extend(simplify_args)
    attempts = [base_cmd + ["-tc"]] if texture_compress else []
    attempts.append(base_cmd)
    last_error: Exception | None = None
    for cmd in attempts:
        try:
            started = time.monotonic()
            stdout = run_command(cmd, args.compressor_timeout)
            duration_s = round(max(0.0, time.monotonic() - started), 6)
            if not dst.is_file():
                raise RuntimeError(f"gltfpack did not create {dst}")
            return {
                "status": "ok",
                "method": "gltfpack",
                "input": str(src),
                "output": str(dst),
                "input_bytes": file_size(src),
                "output_bytes": file_size(dst),
                "duration_s": duration_s,
                "command": cmd,
                "texture_compress": "-tc" in cmd,
                "gltfpack_compression": compression_flag,
                "gltfpack_compression_extension": str(args.gltfpack_compression_extension),
                "stdout_tail": stdout[-4000:],
            }
        except Exception as exc:
            last_error = exc
    raise RuntimeError(str(last_error))


def gltf_transform_command(args: argparse.Namespace) -> list[str] | None:
    direct = command_available(str(args.gltf_transform_bin))
    if direct:
        return [direct]
    if args.allow_npx_gltf_transform:
        npx = command_available("npx")
        if npx:
            return [npx, "--yes", "@gltf-transform/cli"]
    return None


def compress_gltf_transform(src: Path, dst: Path, args: argparse.Namespace, *, texture_compress: bool) -> dict[str, Any]:
    prefix = gltf_transform_command(args)
    if not prefix:
        raise FileNotFoundError(f"gltf-transform not found: {args.gltf_transform_bin}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = prefix + ["optimize", str(src), str(dst), "--compress", "meshopt"]
    if texture_compress:
        cmd.extend(["--texture-compress", "webp"])
    started = time.monotonic()
    stdout = run_command(cmd, args.compressor_timeout)
    duration_s = round(max(0.0, time.monotonic() - started), 6)
    if not dst.is_file():
        raise RuntimeError(f"gltf-transform did not create {dst}")
    return {
        "status": "ok",
        "method": "gltf-transform",
        "input": str(src),
        "output": str(dst),
        "input_bytes": file_size(src),
        "output_bytes": file_size(dst),
        "duration_s": duration_s,
        "command": cmd,
        "texture_compress": texture_compress,
        "stdout_tail": stdout[-4000:],
    }


def compress_asset(
    src: Path,
    dst: Path,
    args: argparse.Namespace,
    *,
    mode: str,
    texture_compress: bool,
) -> dict[str, Any]:
    if mode == "none":
        return copy_uncompressed(src, dst, "compression disabled")

    methods: list[str]
    if mode == "auto_preview":
        methods = ["gltf-transform", "gltfpack"]
    elif mode == "auto_web":
        methods = ["gltfpack", "gltf-transform"]
    else:
        methods = [mode]

    errors: list[str] = []
    for method in methods:
        try:
            if method == "gltfpack":
                return compress_gltfpack(src, dst, args, texture_compress=texture_compress)
            if method == "gltf-transform":
                return compress_gltf_transform(src, dst, args, texture_compress=texture_compress)
        except Exception as exc:
            errors.append(f"{method}: {exc}")
    raise RuntimeError("compression unavailable or failed; " + " | ".join(errors))


def find_final_blend(manifest: dict[str, Any], result_dir: Path) -> Path:
    outputs = manifest.get("outputs") or {}
    candidates = [
        outputs.get("final_blend"),
        outputs.get("gravity_settled_blend"),
        outputs.get("fallback_blend"),
        result_dir / "sam3d_moge_separated_gravity_settled.blend",
        result_dir / "sam3d_moge_separated.blend",
        result_dir / "sam3d_moge_optimized.blend",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(str(candidate))
        if path.is_file():
            return path
    raise FileNotFoundError("Could not find final blend from final_scene_manifest.json")


def matrix_rows(matrix: Any) -> list[list[float]]:
    return [[float(matrix[row][col]) for col in range(4)] for row in range(4)]


def run_blender_export(args: argparse.Namespace) -> int:
    import bpy
    from mathutils import Vector

    input_blend = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_data = read_json(Path(args.manifest).expanduser().resolve() if args.manifest else None)

    bpy.ops.wm.open_mainfile(filepath=str(input_blend))

    def should_skip(obj: Any) -> bool:
        if obj.type != "MESH":
            return True
        if any(obj.name.startswith(prefix) for prefix in SKIP_PREFIXES):
            return True
        if getattr(obj, "hide_render", False):
            return True
        return False

    def export_glb(objects: list[Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.object.select_all(action="DESELECT")
        for item in objects:
            item.select_set(True)
        bpy.context.view_layer.objects.active = objects[0]
        kwargs: dict[str, Any] = {
            "filepath": str(path),
            "export_format": "GLB",
            "use_selection": True,
            "export_texcoords": True,
            "export_normals": True,
            "export_materials": "EXPORT",
            "export_extras": True,
            "export_yup": True,
        }
        fallback_keys = ["export_yup", "export_extras", "export_materials", "export_normals", "export_texcoords"]
        while True:
            try:
                bpy.ops.export_scene.gltf(**kwargs)
                return
            except TypeError:
                if not fallback_keys:
                    raise
                kwargs.pop(fallback_keys.pop(0), None)

    def world_bbox(obj: Any) -> tuple[list[float], list[float]]:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = bpy.data.meshes.new_from_object(eval_obj, depsgraph=depsgraph)
        try:
            if len(mesh.vertices) == 0:
                origin = obj.matrix_world.translation
                return [float(origin.x), float(origin.y), float(origin.z)], [float(origin.x), float(origin.y), float(origin.z)]
            points = [obj.matrix_world @ vertex.co for vertex in mesh.vertices]
            bbox_min = Vector((min(v.x for v in points), min(v.y for v in points), min(v.z for v in points)))
            bbox_max = Vector((max(v.x for v in points), max(v.y for v in points), max(v.z for v in points)))
            return [float(v) for v in bbox_min], [float(v) for v in bbox_max]
        finally:
            bpy.data.meshes.remove(mesh)

    final_objects = manifest_data.get("objects") or []
    manifest_by_name = {
        str(item.get("final_3d_object_name")): item
        for item in final_objects
        if item.get("final_3d_object_name")
    }
    ordered: list[Any] = []
    seen: set[str] = set()
    for item in final_objects:
        name = item.get("final_3d_object_name")
        obj = bpy.data.objects.get(str(name)) if name else None
        if obj is not None and not should_skip(obj) and obj.name not in seen:
            ordered.append(obj)
            seen.add(obj.name)
    for obj in bpy.context.scene.objects:
        if not should_skip(obj) and obj.name not in seen:
            ordered.append(obj)
            seen.add(obj.name)

    if not ordered:
        raise RuntimeError(f"No visible mesh objects exported from {input_blend}")

    scene_raw = output_dir / "scene_raw.glb"
    export_glb(ordered, scene_raw)

    objects_dir = output_dir / "objects_raw"
    objects: list[dict[str, Any]] = []
    for index, obj in enumerate(ordered):
        raw = manifest_by_name.get(obj.name) or {}
        mask_id = raw.get("mask_id")
        label = raw.get("semantic_label") or raw.get("description")
        base_parts = []
        if mask_id is not None:
            base_parts.append(f"mask_{int(mask_id):03d}")
        base_parts.append(safe_name(label or obj.name))
        base = safe_name("_".join(base_parts))
        object_raw = objects_dir / f"{index:04d}_{base}.glb"
        if mask_id is not None:
            obj["mask_id"] = int(mask_id)
        if label:
            obj["semantic_label"] = str(label)
        obj["final_3d_object_name"] = obj.name
        export_glb([obj], object_raw)
        bbox_min, bbox_max = world_bbox(obj)
        objects.append(
            {
                "index": index,
                "name": obj.name,
                "mask_id": mask_id,
                "semantic_label": raw.get("semantic_label"),
                "description": raw.get("description"),
                "base_name": base,
                "raw_glb": str(object_raw),
                "matrix_world": matrix_rows(obj.matrix_world),
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
            }
        )

    export_manifest = {
        "schema": "fysiverse_web_asset_raw_export.v1",
        "status": "ok",
        "input_blend": str(input_blend),
        "output_dir": str(output_dir),
        "scene_raw_glb": str(scene_raw),
        "objects_raw_dir": str(objects_dir),
        "objects": objects,
        "coordinate_system": "Standard glTF export from Blender. Object node transforms preserve the final optimized scene poses.",
    }
    write_json(Path(args.export_manifest), export_manifest)
    print(f"[web-assets-blender] exported scene={scene_raw} objects={len(objects)}", flush=True)
    return 0


def update_final_manifest(final_manifest_path: Path, web_manifest: dict[str, Any]) -> None:
    final_manifest = read_json(final_manifest_path)
    if not final_manifest:
        return
    outputs = final_manifest.setdefault("outputs", {})
    outputs["web_assets_manifest"] = web_manifest.get("manifest_path")
    outputs["web_preview_glb"] = web_manifest.get("outputs", {}).get("scene_preview_glb")
    outputs["web_objects_dir"] = web_manifest.get("outputs", {}).get("web_objects_dir")

    by_mask: dict[int, dict[str, Any]] = {}
    by_name: dict[str, dict[str, Any]] = {}
    for item in web_manifest.get("objects") or []:
        if item.get("mask_id") is not None:
            by_mask[int(item["mask_id"])] = item
        if item.get("name"):
            by_name[str(item["name"])] = item
    for obj in final_manifest.get("objects") or []:
        web_obj = None
        if obj.get("mask_id") is not None:
            web_obj = by_mask.get(int(obj["mask_id"]))
        if web_obj is None and obj.get("final_3d_object_name"):
            web_obj = by_name.get(str(obj["final_3d_object_name"]))
        if web_obj is not None:
            obj["web_asset_glb"] = web_obj.get("web_glb")
            obj["web_asset_raw_glb"] = web_obj.get("raw_glb")
    write_json(final_manifest_path, final_manifest)


def run_export(args: argparse.Namespace) -> dict[str, Any]:
    session = Path(args.session).expanduser().resolve() if args.session else None
    result_dir = Path(args.result_dir).expanduser().resolve() if args.result_dir else None
    if session is not None and result_dir is None:
        result_dir = session / "results"
    if result_dir is None:
        raise ValueError("--session or --result-dir is required")

    final_manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else result_dir / "final_scene_manifest.json"
    final_manifest = read_json(final_manifest_path)
    input_blend = Path(args.blend).expanduser().resolve() if args.blend else find_final_blend(final_manifest, result_dir)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else result_dir / "web_assets"
    raw_dir = output_dir / "raw"
    web_objects_dir = output_dir / "web_objects"
    scene_preview_glb = output_dir / "scene_preview.glb"
    raw_manifest_path = raw_dir / "export_manifest.json"
    manifest_path = output_dir / "manifest.json"

    blender_cmd = [
        str(args.blender_bin),
        "-b",
        str(input_blend),
        "--python",
        str(Path(__file__).resolve()),
        "--",
        "--blender-export",
        "--input",
        str(input_blend),
        "--output-dir",
        str(raw_dir),
        "--manifest",
        str(final_manifest_path),
        "--export-manifest",
        str(raw_manifest_path),
    ]
    blender_stdout = run_command(blender_cmd, args.blender_timeout)
    raw_manifest = read_json(raw_manifest_path)
    scene_raw = Path(str(raw_manifest.get("scene_raw_glb")))
    if not scene_raw.is_file():
        raise FileNotFoundError(scene_raw)

    texture_compress = not bool(args.no_texture_compress)
    preview_mode = "auto_preview" if args.preview_compressor == "auto" else args.preview_compressor
    web_mode = "auto_web" if args.web_compressor == "auto" else args.web_compressor

    scene_preview_value: str | None = None
    if scene_preview_glb.exists():
        scene_preview_glb.unlink()
    try:
        scene_result = compress_asset(scene_raw, scene_preview_glb, args, mode=preview_mode, texture_compress=texture_compress)
        scene_preview_value = str(scene_preview_glb)
    except Exception as exc:
        if scene_preview_glb.exists():
            scene_preview_glb.unlink()
        scene_result = failed_compression_result(
            scene_raw,
            None,
            preview_mode,
            f"scene preview compression failed; omitted scene_preview.glb: {exc}",
        )

    objects: list[dict[str, Any]] = []
    compression_results: list[dict[str, Any]] = [scene_result]
    for item in raw_manifest.get("objects") or []:
        raw_glb = Path(str(item.get("raw_glb")))
        web_glb = web_objects_dir / raw_glb.name
        if web_glb.exists():
            web_glb.unlink()
        try:
            result = compress_asset(raw_glb, web_glb, args, mode=web_mode, texture_compress=texture_compress)
        except Exception as exc:
            if web_glb.exists():
                web_glb.unlink()
            result = copy_uncompressed(
                raw_glb,
                web_glb,
                f"object compression failed; copied uncompressed fallback: {exc}",
            )
        compression_results.append(result)
        objects.append(
            {
                **item,
                "web_glb": str(web_glb),
                "compression": result,
            }
        )

    raw_removed = False
    if not args.keep_raw and raw_dir.is_dir():
        shutil.rmtree(raw_dir)
        raw_removed = True
        scene_raw_value: str | None = None
    else:
        scene_raw_value = str(scene_raw)

    warnings = [
        str(item.get("warning"))
        for item in compression_results
        if item.get("warning")
    ]
    web_manifest = {
        "schema": "fysiverse_web_assets.v1",
        "status": "ok" if not warnings else "partial",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "session": str(session) if session else None,
        "result_dir": str(result_dir),
        "final_scene_manifest": str(final_manifest_path),
        "input_blend": str(input_blend),
        "manifest_path": str(manifest_path),
        "outputs": {
            "scene_preview_glb": scene_preview_value,
            "raw_scene_glb": scene_raw_value,
            "raw_removed": raw_removed,
            "web_objects_dir": str(web_objects_dir),
            "object_count": len(objects),
        },
        "compression": {
            "preview_compressor": args.preview_compressor,
            "web_compressor": args.web_compressor,
            "texture_compress": texture_compress,
            "results": compression_results,
        },
        "objects": [
            {**item, "raw_glb": item.get("raw_glb") if not raw_removed else None}
            for item in objects
        ],
        "warnings": warnings,
        "blender_stdout_tail": blender_stdout[-4000:],
    }
    write_json(manifest_path, web_manifest)
    update_final_manifest(final_manifest_path, web_manifest)
    return web_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export compressed GLB assets for fast web/preview usage.")
    parser.add_argument("--session", type=Path, default=None, help="Session directory. Defaults output to sessions/<id>/results/web_assets.")
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None, help="final_scene_manifest.json")
    parser.add_argument("--blend", type=Path, default=None, help="Final optimized Blend file. Defaults to final_scene_manifest outputs.final_blend.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--blender-bin", default=os.environ.get("BLENDER_BIN", "blender"))
    parser.add_argument("--preview-compressor", choices=["auto", "gltf-transform", "gltfpack", "none"], default="auto")
    parser.add_argument("--web-compressor", choices=["auto", "gltfpack", "gltf-transform", "none"], default="auto")
    parser.add_argument("--gltfpack-bin", default=os.environ.get("GLTFPACK_BIN", "gltfpack"))
    parser.add_argument("--gltf-transform-bin", default=os.environ.get("GLTF_TRANSFORM_BIN", "gltf-transform"))
    parser.add_argument("--gltfpack-compression", choices=["c", "cc", "cz"], default="cc", help="gltfpack geometry compression flag. cz is smaller and slower.")
    parser.add_argument("--gltfpack-compression-extension", choices=["ext", "khr"], default="ext", help="gltfpack -ce meshopt compression extension flavor. Use ext for broader web loader compatibility.")
    parser.add_argument("--gltfpack-texture-quality", type=int, default=None, help="gltfpack -tq texture quality, 1-10.")
    parser.add_argument("--gltfpack-texture-limit", type=int, default=None, help="gltfpack -tl maximum texture dimension.")
    parser.add_argument("--gltfpack-texture-scale", type=float, default=None, help="gltfpack -ts texture dimension scale, 0-1.")
    parser.add_argument("--gltfpack-simplify-ratio", type=float, default=None, help="gltfpack -si target triangle/point ratio, 0-1.")
    parser.add_argument("--gltfpack-simplify-error", type=float, default=None, help="gltfpack -se simplification error tolerance.")
    parser.add_argument("--gltfpack-simplify-aggressive", action="store_true", help="Pass gltfpack -sa.")
    parser.add_argument("--gltfpack-simplify-permissive", action="store_true", help="Pass gltfpack -sp.")
    parser.add_argument("--gltfpack-position-bits", type=int, default=None, help="gltfpack -vp position quantization bits.")
    parser.add_argument("--gltfpack-texcoord-bits", type=int, default=None, help="gltfpack -vt texcoord quantization bits.")
    parser.add_argument("--gltfpack-normal-bits", type=int, default=None, help="gltfpack -vn normal quantization bits.")
    parser.add_argument("--allow-npx-gltfpack", action="store_true")
    parser.add_argument("--allow-npx-gltf-transform", action="store_true")
    parser.add_argument("--no-texture-compress", action="store_true")
    parser.add_argument("--keep-raw", action="store_true", help="Keep raw uncompressed GLB intermediates under output_dir/raw.")
    parser.add_argument("--blender-timeout", type=int, default=600)
    parser.add_argument("--compressor-timeout", type=int, default=300)
    parser.add_argument("--strict", action="store_true", help="Return nonzero if compression falls back to uncompressed copies.")

    parser.add_argument("--blender-export", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--input", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--export-manifest", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args(split_argv())
    if args.blender_export:
        return run_blender_export(args)
    manifest = run_export(args)
    print(json.dumps({
        "status": manifest.get("status"),
        "scene_preview_glb": (manifest.get("outputs") or {}).get("scene_preview_glb"),
        "web_objects_dir": (manifest.get("outputs") or {}).get("web_objects_dir"),
        "object_count": (manifest.get("outputs") or {}).get("object_count"),
        "manifest": manifest.get("manifest_path"),
        "warnings": manifest.get("warnings") or [],
    }, ensure_ascii=False, indent=2))
    if args.strict and manifest.get("warnings"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
