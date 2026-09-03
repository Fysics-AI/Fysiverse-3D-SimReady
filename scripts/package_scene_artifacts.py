#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def copy_file(src: Path | str | None, dst: Path) -> str | None:
    if not src:
        return None
    src_path = Path(str(src))
    if not src_path.is_file():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst)
    return str(dst)


def rel(path: Path, root: Path) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def is_absolute_local_path(value: str) -> bool:
    if value.startswith(("http://", "https://", "data:")):
        return False
    return value.startswith("/")


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
    raise FileNotFoundError("Could not locate final blend")


def run_command(cmd: list[str], timeout: int) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(WORKFLOW_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=max(1, int(timeout)),
        check=False,
    )
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout)
    return proc.stdout or ""


def ensure_web_assets(args: argparse.Namespace, session_dir: Path, final_manifest: Path, result_dir: Path) -> dict[str, Any]:
    web_dir = result_dir / "web_assets"
    manifest_path = web_dir / "manifest.json"
    raw_manifest_path = web_dir / "raw" / "export_manifest.json"
    raw_manifest = read_json(raw_manifest_path)
    web_manifest = read_json(manifest_path)
    if raw_manifest.get("scene_raw_glb") and web_manifest.get("objects"):
        return web_manifest

    cmd = [
        sys.executable,
        str(WORKFLOW_ROOT / "scripts" / "export_compressed_glb_assets.py"),
        "--session",
        str(session_dir),
        "--manifest",
        str(final_manifest),
        "--output-dir",
        str(web_dir),
        "--blender-bin",
        args.blender_bin,
        "--preview-compressor",
        "gltfpack",
        "--web-compressor",
        "gltfpack",
        "--gltfpack-bin",
        args.gltfpack_bin,
        "--gltf-transform-bin",
        args.gltf_transform_bin,
        "--gltfpack-compression",
        args.gltfpack_compression,
        "--gltfpack-compression-extension",
        args.gltfpack_compression_extension,
        "--keep-raw",
    ]
    if args.no_texture_compress:
        cmd.append("--no-texture-compress")
    run_command(cmd, args.web_assets_timeout)
    web_manifest = read_json(manifest_path)
    if not web_manifest:
        raise FileNotFoundError(manifest_path)
    return web_manifest


def copied_object_maps(web_manifest: dict[str, Any], package_root: Path) -> tuple[dict[int, dict[str, str]], dict[str, dict[str, str]]]:
    by_mask: dict[int, dict[str, str]] = {}
    by_name: dict[str, dict[str, str]] = {}
    raw_dir = package_root / "glb" / "raw_objects"
    web_dir = package_root / "glb" / "web_objects"
    for item in web_manifest.get("objects") or []:
        entry: dict[str, str] = {}
        raw_glb = item.get("raw_glb")
        web_glb = item.get("web_glb")
        if raw_glb and Path(str(raw_glb)).is_file():
            dst = raw_dir / Path(str(raw_glb)).name
            copy_file(raw_glb, dst)
            entry["raw_object_glb"] = rel(dst, package_root)
        if web_glb and Path(str(web_glb)).is_file():
            dst = web_dir / Path(str(web_glb)).name
            copy_file(web_glb, dst)
            entry["web_object_glb"] = rel(dst, package_root)
        if not entry:
            continue
        if item.get("mask_id") is not None:
            by_mask[int(item["mask_id"])] = entry
        if item.get("name"):
            by_name[str(item["name"])] = entry
    return by_mask, by_name


def sanitize_pose_entry(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    keep = {}
    for key in ("name", "initial_pose", "final_pose", "matrix", "position", "quaternion_xyzw", "scale"):
        if key in value:
            keep[key] = value[key]
    return keep


def sanitize_manifest(
    *,
    source_manifest: dict[str, Any],
    package_root: Path,
    final_blend_rel: str | None,
    final_glb_rel: str | None,
    pipeline_blend_rel: str | None,
    web_animation_rel: str | None,
    background_3dgs_rel: dict[str, str | None],
    by_mask: dict[int, dict[str, str]],
    by_name: dict[str, dict[str, str]],
) -> dict[str, Any]:
    objects = []
    for raw in source_manifest.get("objects") or []:
        mask_id = raw.get("mask_id")
        name = raw.get("final_3d_object_name")
        asset_entry = {}
        if mask_id is not None:
            asset_entry = by_mask.get(int(mask_id), {})
        if not asset_entry and name:
            asset_entry = by_name.get(str(name), {})
        objects.append(
            {
                "mask_id": int(mask_id) if mask_id is not None else None,
                "mask_name": raw.get("mask_name"),
                "semantic_label": raw.get("semantic_label"),
                "description": raw.get("description"),
                "bbox_2d_xyxy": raw.get("bbox_2d_xyxy"),
                "bbox_2d_xywh": raw.get("bbox_2d_xywh"),
                "area_pixels": raw.get("area_pixels"),
                "sam3d_status": raw.get("sam3d_status"),
                "final_3d_object_name": name,
                "sapien_final_pose": sanitize_pose_entry(raw.get("sapien_final_pose")),
                **asset_entry,
            }
        )

    return {
        "schema": "fysiverse_open_scene_package.v1",
        "status": "ok",
        "session_id": source_manifest.get("session_id"),
        "packaged_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "coordinate_system": "Blender/SAPIEN Z-up. glTF exports use Blender's standard glTF coordinate conversion while preserving independent object nodes.",
        "outputs": {
            "final_blend": final_blend_rel,
            "pipeline_animation_blend": pipeline_blend_rel,
            "final_state_glb": final_glb_rel,
            "raw_objects_dir": "glb/raw_objects",
            "web_objects_dir": "glb/web_objects",
            "web_pipeline_animation_manifest": web_animation_rel,
            "background_3dgs_ksplat": background_3dgs_rel.get("background_3dgs_ksplat"),
            "background_3dgs_scene": background_3dgs_rel.get("background_3dgs_scene"),
            "background_3dgs_manifest": background_3dgs_rel.get("background_3dgs_manifest"),
            "background_image": background_3dgs_rel.get("background_image"),
        },
        "support": {
            "scene_graph_support_relations": (source_manifest.get("support") or {}).get("scene_graph_support_relations"),
        },
        "objects": objects,
        "notes": [
            "Intermediate reports, previews, and SAPIEN collision assets are intentionally omitted.",
            "Raw per-object GLBs are uncompressed. Web object GLBs are compressed for browser loading.",
        ],
    }


def rewrite_web_animation_manifest(src: Path, dst: Path, package_root: Path) -> str | None:
    data = read_json(src)
    if not data:
        return None
    for obj in data.get("objects") or []:
        url = obj.get("url")
        if not url:
            continue
        name = Path(str(url)).name
        candidate = package_root / "glb" / "web_objects" / name
        if candidate.is_file():
            obj["url"] = rel(candidate, dst.parent)
    data["final_scene_manifest"] = rel(package_root / "final_scene_manifest.json", dst.parent)
    data.pop("web_assets_manifest", None)
    write_json(dst, data)
    return rel(dst, package_root)


def copy_background_3dgs(
    source_manifest: dict[str, Any],
    package_root: Path,
    by_mask: dict[int, dict[str, str]],
    by_name: dict[str, dict[str, str]],
) -> dict[str, str | None]:
    outputs = source_manifest.get("outputs") or {}
    scene_src = Path(str(outputs.get("background_3dgs_scene") or ""))
    ksplat_src = Path(str(outputs.get("background_3dgs_ksplat") or ""))
    manifest_src = Path(str(outputs.get("background_3dgs_manifest") or ""))
    if not scene_src.is_file() or not ksplat_src.is_file():
        return {}

    bg_root = package_root / "background_3dgs"
    bg_root.mkdir(parents=True, exist_ok=True)
    ksplat_dst = bg_root / "background.ksplat"
    shutil.copy2(ksplat_src, ksplat_dst)

    scene = read_json(scene_src)
    scene["final_scene_manifest"] = rel(package_root / "final_scene_manifest.json", bg_root)
    scene.setdefault("splat", {})["url"] = rel(ksplat_dst, bg_root)
    scene.get("alignment", {}).pop("camera_json", None)
    for obj in scene.get("objects") or []:
        asset_entry: dict[str, str] = {}
        if obj.get("mask_id") is not None:
            asset_entry = by_mask.get(int(obj["mask_id"]), {})
        if not asset_entry and obj.get("name"):
            asset_entry = by_name.get(str(obj["name"]), {})
        packaged_url = asset_entry.get("web_object_glb") or asset_entry.get("raw_object_glb")
        if packaged_url:
            obj["url"] = rel(package_root / packaged_url, bg_root)
        obj.pop("source", None)

    scene_dst = bg_root / "scene.json"
    write_json(scene_dst, scene)

    bg_image_rel = None
    if manifest_src.is_file():
        bg_manifest = read_json(manifest_src)
        bg_image = Path(str(bg_manifest.get("background_image") or ""))
        if bg_image.is_file():
            bg_image_dst = bg_root / "background.png"
            shutil.copy2(bg_image, bg_image_dst)
            bg_image_rel = rel(bg_image_dst, package_root)

    manifest_dst = bg_root / "manifest.json"
    write_json(
        manifest_dst,
        {
            "schema": "fysiverse_open_background_3dgs.v1",
            "status": "ok",
            "ksplat": rel(ksplat_dst, bg_root),
            "scene_json": rel(scene_dst, bg_root),
            "background_image": rel(package_root / bg_image_rel, bg_root) if bg_image_rel else None,
        },
    )
    return {
        "background_3dgs_ksplat": rel(ksplat_dst, package_root),
        "background_3dgs_scene": rel(scene_dst, package_root),
        "background_3dgs_manifest": rel(manifest_dst, package_root),
        "background_image": bg_image_rel,
    }


def validate_no_absolute_paths(root: Path) -> list[str]:
    bad: list[str] = []
    for path in root.rglob("*"):
        if path.name == "package_validation.json":
            continue
        if path.is_dir() or path.suffix.lower() not in {".json", ".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        try:
            data = json.loads(text)
        except Exception:
            continue
        for item in walk_strings(data):
            if is_absolute_local_path(item):
                bad.append(f"{path}: absolute path {item}")
                break
    return bad


def walk_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_strings(item)


def run(args: argparse.Namespace) -> dict[str, Any]:
    session_dir = args.session.expanduser().resolve()
    result_dir = session_dir / "results"
    final_manifest_path = args.final_manifest.expanduser().resolve() if args.final_manifest else result_dir / "final_scene_manifest.json"
    source_manifest = read_json(final_manifest_path)
    if not source_manifest:
        raise FileNotFoundError(final_manifest_path)

    package_root = args.output_dir.expanduser().resolve()
    package_root.mkdir(parents=True, exist_ok=True)

    final_blend = find_final_blend(source_manifest, result_dir)
    final_blend_rel = copy_file(final_blend, package_root / "blends" / "final_state.blend")

    pipeline_blend = source_manifest.get("outputs", {}).get("pipeline_stage_diagnostic_blend") or result_dir / "pipeline_stage_diagnostic" / "pipeline_stage_diagnostic.blend"
    pipeline_blend_rel = copy_file(pipeline_blend, package_root / "blends" / "pipeline_animation.blend")

    web_manifest = ensure_web_assets(args, session_dir, final_manifest_path, result_dir)
    raw_scene = ((web_manifest.get("outputs") or {}).get("raw_scene_glb")) or result_dir / "web_assets" / "raw" / "scene_raw.glb"
    final_glb_rel = copy_file(raw_scene, package_root / "glb" / "final_state.glb")
    by_mask, by_name = copied_object_maps(web_manifest, package_root)
    background_3dgs_rel = copy_background_3dgs(source_manifest, package_root, by_mask, by_name)

    web_animation_src = source_manifest.get("outputs", {}).get("web_pipeline_animation_manifest") or result_dir / "web_pipeline_animation" / "animation_manifest.json"
    web_animation_rel = None
    if Path(str(web_animation_src)).is_file():
        web_animation_rel = rewrite_web_animation_manifest(Path(str(web_animation_src)), package_root / "web" / "animation_manifest.json", package_root)

    release_manifest = sanitize_manifest(
        source_manifest=source_manifest,
        package_root=package_root,
        final_blend_rel=rel(package_root / final_blend_rel, package_root) if final_blend_rel else None,
        final_glb_rel=rel(package_root / final_glb_rel, package_root) if final_glb_rel else None,
        pipeline_blend_rel=rel(package_root / pipeline_blend_rel, package_root) if pipeline_blend_rel else None,
        web_animation_rel=web_animation_rel,
        background_3dgs_rel=background_3dgs_rel,
        by_mask=by_mask,
        by_name=by_name,
    )
    write_json(package_root / "final_scene_manifest.json", release_manifest)

    errors = validate_no_absolute_paths(package_root)
    if errors:
        write_json(package_root / "package_validation.json", {"status": "failed", "errors": errors})
        if args.strict:
            raise RuntimeError("package validation failed:\n" + "\n".join(errors[:20]))
    else:
        write_json(package_root / "package_validation.json", {"status": "ok", "errors": []})

    return {
        "status": "ok" if not errors else "partial",
        "package_dir": str(package_root),
        "manifest": str(package_root / "final_scene_manifest.json"),
        "validation_errors": errors,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect the open-source release artifacts for one generated scene.")
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--final-manifest", type=Path, default=None)
    parser.add_argument("--blender-bin", default=os.environ.get("BLENDER_BIN", "blender"))
    parser.add_argument("--gltfpack-bin", default=os.environ.get("GLTFPACK_BIN", "gltfpack"))
    parser.add_argument("--gltf-transform-bin", default=os.environ.get("GLTF_TRANSFORM_BIN", "gltf-transform"))
    parser.add_argument("--gltfpack-compression", choices=["c", "cc", "cz"], default="cz")
    parser.add_argument("--gltfpack-compression-extension", choices=["ext", "khr"], default="ext")
    parser.add_argument("--no-texture-compress", action="store_true")
    parser.add_argument("--web-assets-timeout", type=int, default=1800)
    parser.add_argument("--strict", action="store_true")
    return parser


def main() -> int:
    result = run(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
