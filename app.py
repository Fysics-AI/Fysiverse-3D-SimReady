#!/usr/bin/env python3
"""CLI orchestrator for the open-source scene-generation pipeline.

This file keeps the main flow visible: prepare inputs, run each generation
stage, write the final manifest, then export the publishable artifacts. Heavy
stage logic lives in `system_infer/` and `scripts/` so individual components can
be inspected or replaced without rewriting the whole pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from PIL import Image

WORKFLOW_ROOT = Path(__file__).resolve().parent
if str(WORKFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_ROOT))

from pipeline_utils import (  # noqa: E402
    collect_existing,
    ensure_dir,
    glb_to_blend,
    mask_to_label_array,
    prepare_sam3d_input,
    save_color_mask,
    save_label_mask,
    save_rgb_image,
    write_mask_crops_and_manifest,
)
from system_infer.gsam2_infer import GSAM2Infer  # noqa: E402
from system_infer.sam3d_infer import SAM3DInfer  # noqa: E402
from scripts.load_pipeline_config import load_exports_from_config  # noqa: E402


DEFAULT_CONFIG = WORKFLOW_ROOT / "config" / "main.yaml"


# ---------------------------------------------------------------------------
# Small I/O and environment helpers
# ---------------------------------------------------------------------------


def now_session_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]


def read_json(path: Path | str | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value in (None, "") else float(value)


def apply_pipeline_config(config_path: Path | None) -> None:
    if config_path is None:
        return
    path = config_path.expanduser()
    if not path.is_absolute():
        cwd_path = path.resolve()
        path = cwd_path if cwd_path.is_file() else WORKFLOW_ROOT / path
    if not path.is_file():
        return
    for key, value in load_exports_from_config(path).items():
        if value != "" or key not in os.environ:
            os.environ[key] = value
    os.environ["PIPELINE_CONFIG"] = str(path)


def command_for_log(cmd: list[str]) -> str:
    redacted: list[str] = []
    hide_next = False
    for item in cmd:
        text = str(item)
        if hide_next:
            redacted.append("<redacted>")
            hide_next = False
            continue
        redacted.append(text)
        if text in {"--api-key"}:
            hide_next = True
    return " ".join(redacted)


def run_command(cmd: list[str], *, cwd: Path = WORKFLOW_ROOT, timeout: int = 3600, env: dict[str, str] | None = None) -> str:
    print("[run] " + command_for_log(cmd), flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=max(1, int(timeout)),
        check=False,
    )
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout)
    return proc.stdout or ""


# ---------------------------------------------------------------------------
# CLI value parsing
# ---------------------------------------------------------------------------


def parse_box(value: str) -> list[float]:
    parts = [part.strip() for part in value.replace(",", " ").split() if part.strip()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("box must be x1,y1,x2,y2")
    return [float(part) for part in parts]


def parse_point(value: str) -> list[float | int]:
    parts = [part.strip() for part in value.replace(",", " ").split() if part.strip()]
    if len(parts) not in {2, 3}:
        raise argparse.ArgumentTypeError("point must be x,y or x,y,label")
    label = int(parts[2]) if len(parts) == 3 else 1
    return [float(parts[0]), float(parts[1]), 0 if label == 0 else 1]


def normalize_box(box: list[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = box
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    left, right = sorted((int(round(x1)), int(round(x2))))
    top, bottom = sorted((int(round(y1)), int(round(y2))))
    return [
        max(0, min(width - 1, left)),
        max(0, min(height - 1, top)),
        max(0, min(width - 1, right)),
        max(0, min(height - 1, bottom)),
    ]


def normalize_point(point: list[float | int], width: int, height: int) -> list[int]:
    x, y, label = point
    x = float(x)
    y = float(y)
    if max(abs(x), abs(y)) <= 1.0:
        x, y = x * width, y * height
    return [
        max(0, min(width - 1, int(round(x)))),
        max(0, min(height - 1, int(round(y)))),
        0 if int(label) == 0 else 1,
    ]


# ---------------------------------------------------------------------------
# Runtime and session setup
# ---------------------------------------------------------------------------


def setup_runtime_env(args: argparse.Namespace) -> None:
    os.environ.setdefault("FYSIVERSE_CONDA_ENV", args.conda_env)
    os.environ.setdefault("CONDA_BIN", args.conda_bin)
    os.environ.setdefault("BLENDER_BIN", args.blender_bin)
    os.environ.setdefault("GSAM2_ROOT", str(args.gsam2_root))
    os.environ.setdefault("SAM3D_ROOT", str(args.sam3d_root))
    os.environ.setdefault("SAM3D_ENV_DIR", str(args.sam3d_env_dir))
    os.environ.setdefault("SAM3D_CONFIG", str(args.sam3d_config))
    os.environ.setdefault("MOGE_ROOT", str(args.moge_root))
    os.environ.setdefault("MOGE_CKPT", str(args.moge_checkpoint))


def prepare_session(args: argparse.Namespace) -> dict[str, Any]:
    session_id = args.session_id or now_session_id()
    session_dir = Path(args.session_dir).expanduser().resolve() if args.session_dir else (Path(args.output_root).expanduser().resolve() / session_id)
    image_path = save_rgb_image(args.image, session_dir / "input" / "image.png")
    result_dir = ensure_dir(session_dir / "results")
    return {
        "session_id": session_id,
        "session_dir": session_dir,
        "result_dir": result_dir,
        "image_path": image_path,
        "manifest_path": result_dir / "final_scene_manifest.json",
    }


# ---------------------------------------------------------------------------
# Mask preparation
# ---------------------------------------------------------------------------


def ensure_object_mask(args: argparse.Namespace, session: dict[str, Any]) -> Path:
    image = Image.open(session["image_path"]).convert("RGB")
    width, height = image.size
    label_path = Path(session["session_dir"]) / "input" / "mask_label.png"
    color_path = Path(session["session_dir"]) / "input" / "mask.png"

    if args.mask:
        label = mask_to_label_array(args.mask)
        if label.shape != (height, width):
            raise ValueError(f"mask/image size mismatch: mask={label.shape[::-1]} image={(width, height)}")
        save_label_mask(label, label_path)
        save_color_mask(label, color_path)
    else:
        boxes = [normalize_box(item, width, height) for item in (args.box or [])]
        points = [normalize_point(item, width, height) for item in (args.point or [])]
        text_prompt = (args.text_prompt or "").strip()
        if not boxes and not points and not text_prompt:
            raise ValueError("Provide --mask, --box, --point, or --text-prompt for object segmentation.")
        gsam = GSAM2Infer(gpu=args.gsam_gpu)
        resp = gsam.segment(
            {
                "image_path": str(session["image_path"]),
                "out_dir": str(Path(session["session_dir"]) / "segmentation"),
                "points": points,
                "manual_boxes": boxes,
                "text_prompt": text_prompt,
                "use_grounding": bool(text_prompt and not boxes and not points),
                "multi_object": True,
                "box_index": 0,
            }
        )
        if resp.get("status") != "ok" or not resp.get("mask_path"):
            raise RuntimeError("GSAM2 object segmentation produced no mask: " + json.dumps(resp, ensure_ascii=False))
        label = mask_to_label_array(resp["mask_path"])
        save_label_mask(label, label_path)
        save_color_mask(label, color_path)

    write_mask_crops_and_manifest(
        session_dir=Path(session["session_dir"]),
        scene_id=str(session["session_id"]),
        image_path=Path(session["image_path"]),
        label_path=label_path,
        manifest_path=Path(session["manifest_path"]),
    )
    session["label_mask_path"] = label_path
    session["color_mask_path"] = color_path
    return label_path


def ensure_ground_mask(args: argparse.Namespace, session: dict[str, Any]) -> Path | None:
    if args.ground_mask:
        ground = (mask_to_label_array(args.ground_mask) > 0).astype("uint8")
        path = Path(session["session_dir"]) / "input" / "ground_mask.png"
        save_label_mask(ground, path)
        return path

    if not args.ground_point:
        return None

    image = Image.open(session["image_path"]).convert("RGB")
    width, height = image.size
    points = [normalize_point(item, width, height) for item in args.ground_point]
    gsam = GSAM2Infer(gpu=args.gsam_gpu)
    resp = gsam.segment(
        {
            "image_path": str(session["image_path"]),
            "out_dir": str(Path(session["session_dir"]) / "segmentation" / "ground"),
            "points": points,
            "manual_boxes": [],
            "text_prompt": "",
            "use_grounding": False,
            "multi_object": False,
            "box_index": 0,
        }
    )
    if resp.get("status") != "ok" or not resp.get("mask_path"):
        raise RuntimeError("GSAM2 ground segmentation produced no mask: " + json.dumps(resp, ensure_ascii=False))
    ground = (mask_to_label_array(resp["mask_path"]) > 0).astype("uint8")
    path = Path(session["session_dir"]) / "input" / "ground_mask.png"
    save_label_mask(ground, path)
    return path


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def build_scene_graph(args: argparse.Namespace, session: dict[str, Any]) -> Path | None:
    if args.scene_graph:
        scene_graph = Path(args.scene_graph).expanduser()
        if not scene_graph.is_file():
            raise FileNotFoundError(f"scene graph not found: {scene_graph}")
        return scene_graph.resolve()
    if args.skip_scene_graph:
        return None
    output = Path(session["result_dir"]) / "scene_graph.json"
    log = Path(session["result_dir"]) / "scene_graph_stdout.log"
    visualization = Path(session["result_dir"]) / "scene_graph.png"
    cmd = [
        sys.executable,
        str(WORKFLOW_ROOT / "scripts" / "build_scene_graph_codex.py"),
        "--backend",
        args.scene_graph_backend,
        "--manifest",
        str(session["manifest_path"]),
        "--output",
        str(output),
        "--log",
        str(log),
        "--visualization",
        str(visualization),
        "--timeout",
        str(args.scene_graph_timeout),
        "--max-crop-images",
        str(args.scene_graph_max_crop_images),
        "--api-base-url",
        args.scene_graph_api_base_url,
        "--api-model",
        args.scene_graph_api_model,
        "--reasoning-effort",
        args.scene_graph_reasoning_effort,
        "--codex-bin",
        args.codex_bin,
    ]
    if args.scene_graph_model:
        cmd.extend(["--model", args.scene_graph_model])
    if args.scene_graph_retry_without_crops:
        cmd.append("--api-retry-without-crops")
    env = os.environ.copy()
    if args.scene_graph_api_key:
        env["SCENE_GRAPH_API_KEY"] = args.scene_graph_api_key
    run_command(cmd, timeout=args.scene_graph_timeout + 30, env=env)
    graph = read_json(output)
    if graph.get("status") != "ok":
        raise RuntimeError(f"scene graph failed or invalid: {output}")
    return output


def run_sam3d(args: argparse.Namespace, session: dict[str, Any]) -> dict[str, Any]:
    scene_dir = prepare_sam3d_input(
        Path(session["session_dir"]),
        str(session["session_id"]),
        Path(session["image_path"]),
        Path(session["label_mask_path"]),
        Path(session["manifest_path"]),
    )
    out_dir = ensure_dir(Path(session["result_dir"]) / "sam3d_raw")
    infer = SAM3DInfer(
        config=args.sam3d_config,
        gpu=args.sam3d_gpu,
        root=args.sam3d_root,
        env_dir=args.sam3d_env_dir,
        timeout_seconds=args.sam3d_timeout,
    )
    resp = infer.generate({"scene_dir": str(scene_dir), "output_dir": str(out_dir), "scene_id": str(session["session_id"]), "seed": args.seed})
    target_glb = Path(session["result_dir"]) / "sam3d.glb"
    shutil.copy2(resp["glb"], target_glb)
    if resp.get("pose_metadata"):
        target_meta = Path(session["result_dir"]) / "sam3d_pose_metadata.json"
        shutil.copy2(resp["pose_metadata"], target_meta)
        resp["pose_metadata"] = str(target_meta)
    if resp.get("moge_cache") and Path(str(resp["moge_cache"])).is_file():
        target_cache = Path(session["result_dir"]) / "sam3d_moge_cache.npz"
        shutil.copy2(resp["moge_cache"], target_cache)
        resp["moge_cache"] = str(target_cache)
        session["sam3d_moge_cache"] = target_cache
    raw_blend = target_glb.with_suffix(".blend")
    glb_to_blend(target_glb, raw_blend, args.blender_bin)
    target_glb.unlink(missing_ok=True)
    resp["raw_blend"] = str(raw_blend)
    return resp


def run_postprocess(args: argparse.Namespace, session: dict[str, Any], raw_blend: Path, scene_graph: Path | None, ground_mask: Path | None) -> dict[str, Any]:
    cmd = [
        args.nvdiffrast_conda_bin,
        "run",
        "--no-capture-output",
        "-n",
        args.nvdiffrast_env,
        "python",
        str(WORKFLOW_ROOT / "scripts" / "postprocess_sam3d_moge_upright.py"),
        "--image",
        str(session["image_path"]),
        "--blend",
        str(raw_blend),
        "--mask",
        str(session["label_mask_path"]),
        "--output-dir",
        str(session["result_dir"]),
        "--output-name",
        "sam3d_moge",
        "--scene-up-axis",
        "z",
        "--rotation-mode",
        "upright",
        "--scene-to-blender-transform",
        "pytorch3d_y_up_to_blender_z_up",
        "--resize-max",
        str(args.moge_resize_max),
        "--resolution-level",
        str(args.moge_resolution_level),
        "--device",
        args.moge_device,
        "--checkpoint",
        str(args.moge_checkpoint),
        "--blender-bin",
        args.blender_bin,
        "--bbox-overlap-margin",
        str(args.bbox_overlap_margin),
        "--bbox-overlap-iters",
        str(args.bbox_overlap_iters),
        "--bbox-min-overlap-volume-ratio",
        "0",
        "--overlap-collision-method",
        args.overlap_collision_method,
        "--convex-hull-max-vertices",
        str(args.convex_hull_max_vertices),
        "--convex-decomposition-method",
        args.separated_convex_decomposition_method,
        "--conda-bin",
        args.nvdiffrast_conda_bin,
        "--nvdiffrast-env",
        args.nvdiffrast_env,
        "--coacd-conda-bin",
        args.coacd_conda_bin,
        "--coacd-env",
        args.coacd_env,
        "--coacd-workers",
        str(args.coacd_workers),
        "--coacd-timeout",
        str(args.coacd_timeout),
        "--coacd-source-max-faces",
        str(args.coacd_source_max_faces),
        "--coacd-source-simplification-backend",
        args.coacd_source_simplification_backend,
        "--coacd-source-simplification-agg",
        str(args.coacd_source_simplification_agg),
        "--coacd-max-convex-parts",
        str(args.coacd_max_convex_parts),
        "--coacd-threshold",
        str(args.coacd_threshold),
        "--coacd-preprocess-mode",
        args.coacd_preprocess_mode,
        "--coacd-preprocess-resolution",
        str(args.coacd_preprocess_resolution),
        "--coacd-resolution",
        str(args.coacd_resolution),
        "--coacd-mcts-nodes",
        str(args.coacd_mcts_nodes),
        "--coacd-mcts-iterations",
        str(args.coacd_mcts_iterations),
        "--coacd-mcts-max-depth",
        str(args.coacd_mcts_max_depth),
        "--coacd-max-ch-vertex",
        str(args.coacd_max_ch_vertex),
        "--coacd-apx-mode",
        args.coacd_apx_mode,
        "--coacd-seed",
        str(args.coacd_seed),
        "--convex-collision-eps",
        str(args.convex_collision_eps),
        "--convex-min-horizontal-axis",
        str(args.convex_min_horizontal_axis),
        "--support-gap",
        "0.001",
        "--support-xy-overlap-ratio",
        "0.15",
        "--support-max-gap-ratio",
        "0.03",
        "--support-max-penetration-ratio",
        "0.01",
        "--support-min-lower-area-ratio",
        "0.25",
        "--camera-opt-steps",
        str(args.camera_opt_steps),
        "--camera-opt-max-side",
        str(args.camera_opt_max_side),
        "--object-pose-workers",
        str(args.object_pose_workers),
        "--object-pose-max-side",
        str(args.object_pose_max_side),
        "--object-position-steps",
        str(args.object_position_steps),
        "--object-yaw-steps",
        str(args.object_yaw_steps),
        "--object-scale-steps",
        str(args.object_scale_steps),
        "--object-translation-mode",
        args.object_translation_mode,
        "--object-translation-initial-grid",
        str(args.object_translation_initial_grid),
        "--object-center-weight",
        str(args.object_center_weight),
        "--object-area-weight",
        str(args.object_area_weight),
        "--object-yaw-initial-samples",
        str(args.object_yaw_initial_samples),
        "--object-scale-initial-samples",
        str(args.object_scale_initial_samples),
        "--object-phase-accept-iou-drop",
        str(args.object_phase_accept_iou_drop),
        "--object-max-translation",
        str(args.object_max_translation),
        "--object-max-yaw-deg",
        str(args.object_max_yaw_deg),
        "--object-max-scale-delta",
        str(args.object_max_scale_delta),
    ]
    if args.coacd_real_metric:
        cmd.append("--coacd-real-metric")
    if not args.coacd_merge:
        cmd.append("--no-coacd-merge")
    if args.coacd_decimate:
        cmd.append("--coacd-decimate")
    if args.coacd_blender_source_decimate:
        cmd.append("--coacd-blender-source-decimate")
    if scene_graph is not None:
        cmd.extend(["--scene-graph", str(scene_graph)])
    if ground_mask is not None:
        cmd.extend(["--ground-mask", str(ground_mask)])
    if session.get("sam3d_moge_cache"):
        cmd.extend(["--moge-cache", str(session["sam3d_moge_cache"])])
    stdout = run_command(cmd, timeout=args.moge_timeout)
    reports = sorted(Path(session["result_dir"]).glob("sam3d_moge*_report.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    aggregate = next((read_json(path) for path in reports if {"rotated", "grounded", "optimized", "separated"} <= set(((read_json(path).get("outputs") or {}).get("stages") or {}).keys())), {})
    if not aggregate:
        raise RuntimeError("MoGe postprocess finished without an aggregate stage report.\n" + stdout[-4000:])
    return aggregate


def choose_postprocess_blend(session: dict[str, Any]) -> Path:
    result_dir = Path(session["result_dir"])
    for name in ("sam3d_moge_separated.blend", "sam3d_moge_optimized.blend", "sam3d_moge_grounded.blend", "sam3d.blend"):
        path = result_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError("No postprocess blend found in " + str(result_dir))


def run_gravity(args: argparse.Namespace, session: dict[str, Any], input_blend: Path) -> dict[str, Any]:
    if args.skip_gravity:
        return {"output_blend": str(input_blend), "animation_blend": None}
    result_dir = Path(session["result_dir"])
    output_blend = result_dir / f"{input_blend.stem}_gravity_settled.blend"
    animation_blend = result_dir / f"{input_blend.stem}_gravity_animation.blend"
    work_dir = ensure_dir(result_dir / f"{input_blend.stem}_sapien_gravity")
    manifest_path = work_dir / "manifest.json"
    pose_path = work_dir / "final_poses.json"
    trajectory_path = work_dir / "pose_trajectory.json"
    report_path = work_dir / "sapien_gravity_report.json"
    frames_dir = work_dir / "frames"
    video_path = work_dir / "gravity.mp4"

    export_cmd = [
        args.blender_bin,
        "-b",
        "--python",
        str(WORKFLOW_ROOT / "scripts" / "blender_export_sapien_scene.py"),
        "--",
        "--input",
        str(input_blend),
        "--output-dir",
        str(work_dir),
        "--manifest",
        str(manifest_path),
        "--collision-decomposition-method",
        "coacd" if args.sapien_collision_decomposition_method == "coacd" else "none",
        "--coacd-conda-bin",
        args.coacd_conda_bin,
        "--coacd-env",
        args.coacd_env,
        "--coacd-timeout",
        str(args.coacd_timeout),
        "--coacd-workers",
        str(args.coacd_workers),
        "--coacd-source-max-faces",
        str(args.coacd_source_max_faces),
        "--coacd-source-simplification-backend",
        args.coacd_source_simplification_backend,
        "--coacd-source-simplification-agg",
        str(args.coacd_source_simplification_agg),
        "--coacd-max-convex-parts",
        str(args.coacd_max_convex_parts),
        "--coacd-threshold",
        str(args.coacd_threshold),
        "--coacd-preprocess-mode",
        args.coacd_preprocess_mode,
        "--coacd-preprocess-resolution",
        str(args.coacd_preprocess_resolution),
        "--coacd-resolution",
        str(args.coacd_resolution),
        "--coacd-mcts-nodes",
        str(args.coacd_mcts_nodes),
        "--coacd-mcts-iterations",
        str(args.coacd_mcts_iterations),
        "--coacd-mcts-max-depth",
        str(args.coacd_mcts_max_depth),
        "--coacd-max-ch-vertex",
        str(args.coacd_max_ch_vertex),
        "--coacd-apx-mode",
        args.coacd_apx_mode,
        "--coacd-seed",
        str(args.coacd_seed),
    ]
    if args.coacd_real_metric:
        export_cmd.append("--coacd-real-metric")
    if not args.coacd_merge:
        export_cmd.append("--coacd-no-merge")
    if args.coacd_decimate:
        export_cmd.append("--coacd-decimate")
    run_command(export_cmd, timeout=args.gravity_timeout)

    simulate_cmd = [
        args.conda_bin,
        "run",
        "--no-capture-output",
        "-n",
        args.sapien_env,
        "python",
        str(WORKFLOW_ROOT / "scripts" / "sapien_gravity_simulate.py"),
        "--manifest",
        str(manifest_path),
        "--pose-output",
        str(pose_path),
        "--trajectory-output",
        str(trajectory_path),
        "--frames-dir",
        str(frames_dir),
        "--video-path",
        str(video_path),
        "--report",
        str(report_path),
        "--steps",
        str(args.sapien_steps),
        "--timestep",
        str(args.sapien_timestep),
        "--gravity-z",
        str(args.gravity_z),
        "--ground-z",
        "-0.001",
        "--render-every",
        str(args.sapien_render_every),
        "--settle-window",
        str(args.sapien_settle_window),
        "--settle-translation-threshold",
        str(args.sapien_settle_translation_threshold),
        "--video-fps",
        str(args.sapien_video_fps),
        "--width",
        str(args.sapien_video_width),
        "--height",
        str(args.sapien_video_height),
        "--static-friction",
        str(args.sapien_static_friction),
        "--dynamic-friction",
        str(args.sapien_dynamic_friction),
        "--linear-damping",
        str(args.sapien_linear_damping),
        "--angular-damping",
        str(args.sapien_angular_damping),
    ]
    if not args.sapien_render_video:
        simulate_cmd.append("--no-render-video")
    if args.sapien_default_scene:
        simulate_cmd.append("--default-scene")
    if args.sapien_physx_gpu:
        simulate_cmd.append("--physx-gpu")
    run_command(simulate_cmd, timeout=args.gravity_timeout)

    apply_cmd = [
        args.blender_bin,
        "-b",
        "--python",
        str(WORKFLOW_ROOT / "scripts" / "blender_apply_sapien_poses.py"),
        "--",
        "--input",
        str(input_blend),
        "--poses",
        str(pose_path),
        "--trajectory",
        str(trajectory_path),
        "--output",
        str(output_blend),
        "--animation-output",
        str(animation_blend),
        "--pack-textures",
    ]
    run_command(apply_cmd, timeout=args.gravity_timeout)
    return {
        "output_blend": str(output_blend),
        "animation_blend": str(animation_blend),
        "sapien_export_manifest": str(manifest_path),
        "sapien_final_poses": str(pose_path),
        "sapien_pose_trajectory": str(trajectory_path),
        "sapien_report": str(report_path),
    }


def write_final_manifest(session: dict[str, Any], sam3d_resp: dict[str, Any], raw_blend: Path, postprocess_blend: Path, gravity: dict[str, Any], scene_graph_path: Path | None) -> Path:
    manifest_path = Path(session["manifest_path"])
    existing = read_json(manifest_path)
    objects_by_mask: dict[int, dict[str, Any]] = {}
    for obj in existing.get("objects") or []:
        if obj.get("mask_id") is not None:
            objects_by_mask[int(obj["mask_id"])] = dict(obj)
    for obj in sam3d_resp.get("objects") or []:
        if obj.get("mask_id") is not None:
            objects_by_mask.setdefault(int(obj["mask_id"]), {}).update({k: v for k, v in obj.items() if v is not None})

    scene_graph = read_json(scene_graph_path)
    graph_by_mask = {int(obj["mask_id"]): obj for obj in scene_graph.get("objects", []) if obj.get("mask_id") is not None}
    export_manifest = read_json(gravity.get("sapien_export_manifest"))
    final_poses = read_json(gravity.get("sapien_final_poses"))
    exported = list(export_manifest.get("objects") or [])
    pose_by_name = {item.get("name"): item for item in final_poses.get("objects", [])}
    ok_mask_ids = [mask_id for mask_id, obj in sorted(objects_by_mask.items(), key=lambda item: int(item[1].get("sam3d_input_index", item[0]))) if obj.get("status") == "ok"]
    name_by_mask = {mask_id: item.get("name") for mask_id, item in zip(ok_mask_ids, exported) if item.get("name")}
    export_by_name = {item.get("name"): item for item in exported}

    objects = []
    for mask_id in sorted(objects_by_mask):
        obj = objects_by_mask[mask_id]
        graph_obj = graph_by_mask.get(mask_id, {})
        name = name_by_mask.get(mask_id) or obj.get("object_name")
        objects.append(
            {
                "mask_id": mask_id,
                "mask_name": obj.get("mask_name") or f"mask_{mask_id:03d}",
                "semantic_label": graph_obj.get("semantic_label") or obj.get("semantic_label") or obj.get("label"),
                "description": graph_obj.get("description") or obj.get("description") or graph_obj.get("semantic_label") or obj.get("label"),
                "bbox_2d_xyxy": obj.get("bbox_2d_xyxy"),
                "bbox_2d_xywh": obj.get("bbox_2d_xywh"),
                "area_pixels": obj.get("area_pixels"),
                "crop_rgb": obj.get("crop_rgb"),
                "sam3d_input_index": obj.get("sam3d_input_index"),
                "sam3d_input_mask": obj.get("sam3d_input_mask"),
                "sam3d_status": obj.get("status"),
                "final_3d_object_name": name,
                "sapien_export": export_by_name.get(name),
                "sapien_final_pose": pose_by_name.get(name),
            }
        )

    payload = {
        "schema": "fysiverse_final_scene_manifest.v1",
        "status": "ok",
        "session_id": session["session_id"],
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "inputs": {
            "image": str(session["image_path"]),
            "label_mask": str(session["label_mask_path"]),
            "color_mask": str(session.get("color_mask_path")),
            "ground_mask": str(session.get("ground_mask_path")) if session.get("ground_mask_path") else None,
            "session_dir": str(session["session_dir"]),
            "result_dir": str(session["result_dir"]),
        },
        "outputs": {
            "final_blend": gravity.get("output_blend") or str(postprocess_blend),
            "gravity_settled_blend": gravity.get("output_blend"),
            "gravity_animation_blend": gravity.get("animation_blend"),
            "final_manifest": str(manifest_path),
        },
        "intermediate": {
            "sam3d_raw_blend": str(raw_blend),
            "sam3d_postprocess_blend": str(postprocess_blend),
            "sam3d_pose_metadata": sam3d_resp.get("pose_metadata"),
            "sam3d_moge_cache": sam3d_resp.get("moge_cache"),
            "scene_graph": str(scene_graph_path) if scene_graph_path else None,
            "sapien_export_manifest": gravity.get("sapien_export_manifest"),
            "sapien_final_poses": gravity.get("sapien_final_poses"),
            "sapien_pose_trajectory": gravity.get("sapien_pose_trajectory"),
            "sapien_report": gravity.get("sapien_report"),
        },
        "support": {
            "scene_graph_support_relations": scene_graph.get("support_relations") if scene_graph else None,
            "coordinate_system": "Blender/SAPIEN Z-up.",
        },
        "objects": objects,
    }
    write_json(manifest_path, payload)
    return manifest_path


def record_optional_export_warning(manifest_path: Path, label: str, exc: BaseException) -> None:
    data = read_json(manifest_path)
    warning: dict[str, Any] = {
        "stage": label,
        "error_type": type(exc).__name__,
        "message": str(exc),
    }
    if isinstance(exc, subprocess.CalledProcessError):
        warning["returncode"] = exc.returncode
        warning["cmd"] = " ".join(str(part) for part in exc.cmd)
        output = exc.output or ""
        warning["output_tail"] = output[-4000:]
    data.setdefault("optional_export_warnings", []).append(warning)
    write_json(manifest_path, data)
    print(f"[warn] optional export failed: {label}: {exc}", flush=True)


def clear_optional_export_warning(manifest_path: Path, label: str) -> None:
    data = read_json(manifest_path)
    warnings = data.get("optional_export_warnings")
    if not isinstance(warnings, list):
        return
    kept = [item for item in warnings if not isinstance(item, dict) or item.get("stage") != label]
    if len(kept) == len(warnings):
        return
    if kept:
        data["optional_export_warnings"] = kept
    else:
        data.pop("optional_export_warnings", None)
    write_json(manifest_path, data)


def run_optional_export_command(
    label: str,
    cmd: list[str],
    timeout: int,
    manifest_path: Path,
    *,
    env: dict[str, str] | None = None,
) -> bool:
    try:
        run_command(cmd, timeout=timeout, env=env)
        clear_optional_export_warning(manifest_path, label)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, RuntimeError) as exc:
        record_optional_export_warning(manifest_path, label, exc)
        return False


def run_background_3dgs_if_enabled(args: argparse.Namespace, session: dict[str, Any], manifest_path: Path) -> dict[str, str | None]:
    if not args.background_3dgs_enabled:
        return {}
    bg_dir = Path(session["result_dir"]) / "3dgs_bg"
    ksplat_path = bg_dir / "background.ksplat"
    manifest_out = bg_dir / "manifest.json"
    scene_out = bg_dir / "scene.json"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.background_3dgs_gpu)
    env["ITERATIONS"] = str(args.background_3dgs_iterations)
    env["KSPLAT_COMPRESSION"] = str(args.background_3dgs_ksplat_compression)
    env["KSPLAT_ALPHA_THRESHOLD"] = str(args.background_3dgs_ksplat_alpha_threshold)
    env["KSPLAT_SH_DEGREE"] = str(args.background_3dgs_ksplat_sh_degree)
    env.setdefault("MOGE_RESIZE_MAX", str(args.moge_resize_max))
    env.setdefault("MOGE_RESOLUTION_LEVEL", str(args.moge_resolution_level))
    env.setdefault("MOGE_DEVICE", str(args.moge_device))
    cmd = ["bash", str(WORKFLOW_ROOT / "background" / "run_session_background_3dgs.sh"), str(session["session_dir"])]
    ok = run_optional_export_command("background_3dgs", cmd, args.background_3dgs_timeout, manifest_path, env=env)
    if not ok:
        return {"background_3dgs_ksplat": None, "background_3dgs_manifest": None, "background_3dgs_scene": None}
    if not ksplat_path.is_file():
        record_optional_export_warning(manifest_path, "background_3dgs", FileNotFoundError(ksplat_path))
        return {"background_3dgs_ksplat": None, "background_3dgs_manifest": None, "background_3dgs_scene": None}
    data = read_json(manifest_path)
    outputs = data.setdefault("outputs", {})
    outputs["background_3dgs_ksplat"] = str(ksplat_path)
    outputs["background_3dgs_manifest"] = str(manifest_out) if manifest_out.is_file() else None
    outputs["background_3dgs_scene"] = str(scene_out) if scene_out.is_file() else None
    write_json(manifest_path, data)
    return {
        "background_3dgs_ksplat": str(ksplat_path),
        "background_3dgs_manifest": str(manifest_out) if manifest_out.is_file() else None,
        "background_3dgs_scene": str(scene_out) if scene_out.is_file() else None,
    }


def run_optional_exports(args: argparse.Namespace, session: dict[str, Any], manifest_path: Path) -> None:
    if not args.skip_diagnostic:
        cmd = [
            sys.executable,
            str(WORKFLOW_ROOT / "scripts" / "create_pipeline_stage_diagnostic_blend.py"),
            "--session",
            str(session["session_dir"]),
            "--blender-bin",
            args.blender_bin,
        ]
        ok = run_optional_export_command("pipeline_stage_diagnostic", cmd, args.diagnostic_timeout, manifest_path)
        diag = Path(session["result_dir"]) / "pipeline_stage_diagnostic" / "pipeline_stage_diagnostic.blend"
        data = read_json(manifest_path)
        data.setdefault("outputs", {})["pipeline_stage_diagnostic_blend"] = str(diag) if ok and diag.is_file() else None
        write_json(manifest_path, data)

    if not args.skip_web_assets:
        cmd = [
            sys.executable,
            str(WORKFLOW_ROOT / "scripts" / "export_compressed_glb_assets.py"),
            "--session",
            str(session["session_dir"]),
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(Path(session["result_dir"]) / "web_assets"),
            "--blender-bin",
            args.blender_bin,
            "--preview-compressor",
            args.web_preview_compressor,
            "--web-compressor",
            args.web_object_compressor,
            "--gltfpack-bin",
            args.gltfpack_bin,
            "--gltf-transform-bin",
            args.gltf_transform_bin,
            "--gltfpack-compression",
            args.gltfpack_compression,
            "--gltfpack-compression-extension",
            args.gltfpack_compression_extension,
            "--gltfpack-texture-quality",
            str(args.gltfpack_texture_quality),
            "--gltfpack-texture-limit",
            str(args.gltfpack_texture_limit),
            "--gltfpack-simplify-ratio",
            str(args.gltfpack_simplify_ratio),
            "--gltfpack-simplify-error",
            str(args.gltfpack_simplify_error),
            "--gltfpack-position-bits",
            str(args.gltfpack_position_bits),
            "--gltfpack-texcoord-bits",
            str(args.gltfpack_texcoord_bits),
            "--gltfpack-normal-bits",
            str(args.gltfpack_normal_bits),
            "--keep-raw",
        ]
        if args.gltfpack_simplify_aggressive:
            cmd.append("--gltfpack-simplify-aggressive")
        if args.gltfpack_simplify_permissive:
            cmd.append("--gltfpack-simplify-permissive")
        if args.no_texture_compress:
            cmd.append("--no-texture-compress")
        run_optional_export_command("web_assets", cmd, args.web_assets_timeout, manifest_path)

    if not args.skip_web_animation:
        cmd = [
            sys.executable,
            str(WORKFLOW_ROOT / "scripts" / "export_web_pipeline_animation.py"),
            "--session",
            str(session["session_dir"]),
            "--final-manifest",
            str(manifest_path),
            "--output-dir",
            str(Path(session["result_dir"]) / "web_pipeline_animation"),
            "--blender-bin",
            args.blender_bin,
        ]
        run_optional_export_command("web_pipeline_animation", cmd, args.web_animation_timeout, manifest_path)


def package_outputs(args: argparse.Namespace, session: dict[str, Any], manifest_path: Path) -> Path | None:
    if args.skip_package:
        return None
    output_dir = Path(args.package_dir).expanduser().resolve() if args.package_dir else Path(session["result_dir"]) / "release_package"
    cmd = [
        sys.executable,
        str(WORKFLOW_ROOT / "scripts" / "package_scene_artifacts.py"),
        "--session",
        str(session["session_dir"]),
        "--output-dir",
        str(output_dir),
        "--final-manifest",
        str(manifest_path),
        "--blender-bin",
        args.blender_bin,
        "--gltfpack-bin",
        args.gltfpack_bin,
        "--gltf-transform-bin",
        args.gltf_transform_bin,
        "--gltfpack-compression",
        args.gltfpack_compression,
        "--gltfpack-compression-extension",
        args.gltfpack_compression_extension,
    ]
    if args.no_texture_compress:
        cmd.append("--no-texture-compress")
    run_command(cmd, timeout=args.package_timeout)
    return output_dir


# ---------------------------------------------------------------------------
# Top-level orchestration and CLI
# ---------------------------------------------------------------------------


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    setup_runtime_env(args)
    session = prepare_session(args)
    ensure_object_mask(args, session)
    ground_mask = ensure_ground_mask(args, session)
    if ground_mask:
        session["ground_mask_path"] = ground_mask
    scene_graph = build_scene_graph(args, session)
    sam3d_resp = run_sam3d(args, session)
    raw_blend = Path(sam3d_resp["raw_blend"])
    run_postprocess(args, session, raw_blend, scene_graph, ground_mask)
    postprocess_blend = choose_postprocess_blend(session)
    gravity = run_gravity(args, session, postprocess_blend)
    manifest_path = write_final_manifest(session, sam3d_resp, raw_blend, postprocess_blend, gravity, scene_graph)
    background_3dgs = run_background_3dgs_if_enabled(args, session, manifest_path)
    run_optional_exports(args, session, manifest_path)
    package_dir = package_outputs(args, session, manifest_path)
    outputs = read_json(manifest_path).get("outputs") or {}
    return {
        "status": "ok",
        "session": str(session["session_dir"]),
        "final_manifest": str(manifest_path),
        "final_blend": outputs.get("final_blend"),
        "background_3dgs": background_3dgs,
        "release_package": str(package_dir) if package_dir else None,
        "files": collect_existing(
            [
                manifest_path,
                outputs.get("final_blend"),
                background_3dgs.get("background_3dgs_ksplat"),
                background_3dgs.get("background_3dgs_manifest"),
                background_3dgs.get("background_3dgs_scene"),
            ]
        ),
    }


def env_str(name: str, default: str = "", *fallback_names: str) -> str:
    for key in (name, *fallback_names):
        value = os.environ.get(key)
        if value not in (None, ""):
            return value
    return default


def env_path(name: str, default: Path | str, *fallback_names: str) -> Path:
    return Path(env_str(name, str(default), *fallback_names))


def fill_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Populate internal runtime knobs from YAML-loaded environment exports."""
    args.conda_bin = env_str("CONDA_BIN", "conda")
    args.conda_env = env_str("FYSIVERSE_CONDA_ENV", "fysiverse-scene")
    args.blender_bin = env_str("BLENDER_BIN", "blender")
    args.gsam2_root = env_path("GSAM2_ROOT", WORKFLOW_ROOT / "third_party" / "Grounded-SAM-2")
    args.sam3d_root = env_path("SAM3D_ROOT", WORKFLOW_ROOT / "third_party" / "SAM3D")
    args.sam3d_env_dir = env_path("SAM3D_ENV_DIR", os.environ.get("CONDA_PREFIX", sys.prefix))
    args.sam3d_config = env_path("SAM3D_CONFIG", WORKFLOW_ROOT / "third_party" / "SAM3D" / "runtime_configs" / "pipeline_hf_cache_moge_modelpt.yaml")
    args.moge_root = env_path("MOGE_ROOT", WORKFLOW_ROOT / "third_party" / "MoGe")
    args.moge_checkpoint = env_path("MOGE_CKPT", WORKFLOW_ROOT / "checkpoints" / "moge-2-vitl-normal" / "model.pt")

    args.gsam_gpu = env_str("GSAM_GPU", "0")
    args.sam3d_gpu = env_str("SAM3D_GPU", "0")
    args.sam3d_timeout = env_int("SAM3D_TIMEOUT_SECONDS", 14400)
    args.moge_device = env_str("MOGE_DEVICE", "cuda")
    args.moge_resize_max = env_int("MOGE_RESIZE_MAX", 1024)
    args.moge_resolution_level = env_int("MOGE_RESOLUTION_LEVEL", 5)
    args.moge_timeout = env_int("MOGE_TIMEOUT", 1800)

    args.skip_scene_graph = bool(args.skip_scene_graph or not env_bool("ENABLE_SCENE_GRAPH", True))
    args.scene_graph_backend = env_str("SCENE_GRAPH_BACKEND", "openai_compatible")
    args.scene_graph_api_key = os.environ.get("SCENE_GRAPH_API_KEY", "")
    args.scene_graph_api_base_url = os.environ.get("SCENE_GRAPH_API_BASE_URL", "")
    args.scene_graph_api_model = os.environ.get("SCENE_GRAPH_API_MODEL", "")
    args.scene_graph_model = os.environ.get("SCENE_GRAPH_MODEL", "")
    args.scene_graph_timeout = env_int("SCENE_GRAPH_TIMEOUT", 300)
    args.scene_graph_max_crop_images = env_int("SCENE_GRAPH_MAX_CROP_IMAGES", 40)
    args.scene_graph_reasoning_effort = env_str("SCENE_GRAPH_REASONING_EFFORT", "inherit")
    args.scene_graph_retry_without_crops = env_bool("SCENE_GRAPH_API_RETRY_WITHOUT_CROPS", True)
    args.codex_bin = env_str("CODEX_BIN", "codex")

    args.background_3dgs_enabled = bool(args.run_background_3dgs or env_bool("BACKGROUND_3DGS_ENABLED", False))
    if args.skip_background_3dgs:
        args.background_3dgs_enabled = False
    args.background_3dgs_gpu = env_str("BACKGROUND_3DGS_GPU", "0")
    args.background_3dgs_iterations = env_int("BACKGROUND_3DGS_ITERATIONS", 300)
    args.background_3dgs_timeout = env_int("BACKGROUND_3DGS_TIMEOUT", 7200)
    args.background_3dgs_ksplat_compression = env_int("BACKGROUND_3DGS_KSPLAT_COMPRESSION", 2)
    args.background_3dgs_ksplat_alpha_threshold = env_int("BACKGROUND_3DGS_KSPLAT_ALPHA_THRESHOLD", 5)
    args.background_3dgs_ksplat_sh_degree = env_int("BACKGROUND_3DGS_KSPLAT_SH_DEGREE", 0)

    args.camera_opt_steps = env_int("MOGE_CAMERA_OPT_STEPS", 300)
    args.camera_opt_max_side = env_int("MOGE_CAMERA_OPT_MAX_SIDE", 512)
    args.object_pose_workers = env_int("MOGE_OBJECT_POSE_WORKERS", 0)
    args.object_pose_max_side = env_int("MOGE_OBJECT_POSE_MAX_SIDE", 512)
    args.object_position_steps = env_int("MOGE_OBJECT_POSITION_STEPS", 120)
    args.object_yaw_steps = env_int("MOGE_OBJECT_YAW_STEPS", 80)
    args.object_scale_steps = env_int("MOGE_OBJECT_SCALE_STEPS", 60)
    args.object_translation_mode = env_str("MOGE_OBJECT_TRANSLATION_MODE", "camera_plane")
    args.object_translation_initial_grid = env_int("MOGE_OBJECT_TRANSLATION_INITIAL_GRID", 3)
    args.object_center_weight = env_float("MOGE_OBJECT_CENTER_WEIGHT", 0.15)
    args.object_area_weight = env_float("MOGE_OBJECT_AREA_WEIGHT", 0.05)
    args.object_yaw_initial_samples = env_int("MOGE_OBJECT_YAW_INITIAL_SAMPLES", 5)
    args.object_scale_initial_samples = env_int("MOGE_OBJECT_SCALE_INITIAL_SAMPLES", 5)
    args.object_phase_accept_iou_drop = env_float("MOGE_OBJECT_PHASE_ACCEPT_IOU_DROP", 0.01)
    args.object_max_translation = env_float("MOGE_OBJECT_MAX_TRANSLATION", 0.35)
    args.object_max_yaw_deg = env_float("MOGE_OBJECT_MAX_YAW_DEG", 35.0)
    args.object_max_scale_delta = env_float("MOGE_OBJECT_MAX_SCALE_DELTA", 0.25)
    args.nvdiffrast_conda_bin = env_str("MOGE_NVDIFFRAST_CONDA_BIN", args.conda_bin)
    args.nvdiffrast_env = env_str("MOGE_NVDIFFRAST_ENV", args.conda_env)

    args.coacd_workers = env_int("COACD_WORKERS", 4)
    args.coacd_conda_bin = env_str("COACD_CONDA_BIN", args.conda_bin)
    args.coacd_env = env_str("COACD_ENV", args.conda_env)
    args.coacd_timeout = env_int("COACD_TIMEOUT", 120)
    args.coacd_source_max_faces = env_int("COACD_SOURCE_MAX_FACES", 5000)
    args.coacd_source_simplification_backend = env_str("COACD_SOURCE_SIMPLIFICATION_BACKEND", "fast_simplification")
    args.coacd_source_simplification_agg = env_float("COACD_SOURCE_SIMPLIFICATION_AGG", 7.0)
    args.coacd_blender_source_decimate = env_bool("COACD_BLENDER_SOURCE_DECIMATE", False)
    args.coacd_max_convex_parts = env_int("COACD_MAX_CONVEX_PARTS", 8)
    args.coacd_threshold = env_float("COACD_THRESHOLD", 0.05)
    args.coacd_preprocess_mode = env_str("COACD_PREPROCESS_MODE", "auto")
    args.coacd_preprocess_resolution = env_int("COACD_PREPROCESS_RESOLUTION", 50)
    args.coacd_resolution = env_int("COACD_RESOLUTION", 2000)
    args.coacd_mcts_nodes = env_int("COACD_MCTS_NODES", 20)
    args.coacd_mcts_iterations = env_int("COACD_MCTS_ITERATIONS", 80)
    args.coacd_mcts_max_depth = env_int("COACD_MCTS_MAX_DEPTH", 3)
    args.coacd_max_ch_vertex = env_int("COACD_MAX_CH_VERTEX", 256)
    args.coacd_apx_mode = env_str("COACD_APX_MODE", "ch")
    args.coacd_seed = env_int("COACD_SEED", 0)
    args.coacd_real_metric = env_bool("COACD_REAL_METRIC", False)
    args.coacd_merge = env_bool("COACD_MERGE", True)
    args.coacd_decimate = env_bool("COACD_DECIMATE", False)
    args.convex_collision_eps = env_float("CONVEX_COLLISION_EPS", 1e-6)
    args.convex_min_horizontal_axis = env_float("CONVEX_MIN_HORIZONTAL_AXIS", 0.25)
    args.bbox_overlap_margin = env_float("BBOX_OVERLAP_MARGIN", 0.01)
    args.bbox_overlap_iters = env_int("BBOX_OVERLAP_ITERS", 32)
    args.overlap_collision_method = env_str("OVERLAP_COLLISION_METHOD", "convex_hull_sat")
    args.convex_hull_max_vertices = env_int("CONVEX_HULL_MAX_VERTICES", 8000)
    args.separated_convex_decomposition_method = env_str("SEPARATED_CONVEX_DECOMPOSITION_METHOD", "coacd")
    args.sapien_collision_decomposition_method = env_str("SAPIEN_COLLISION_DECOMPOSITION_METHOD", "coacd")

    args.gravity_timeout = env_int("GRAVITY_TIMEOUT", 1800)
    args.gravity_z = env_float("GRAVITY_Z", -9.81)
    args.sapien_env = env_str("SAPIEN_ENV", args.conda_env)
    args.sapien_steps = env_int("SAPIEN_STEPS", 240)
    args.sapien_timestep = env_float("SAPIEN_TIMESTEP", 0.01)
    args.sapien_render_video = env_bool("SAPIEN_RENDER_VIDEO", False)
    args.sapien_render_every = env_int("SAPIEN_RENDER_EVERY", 1)
    args.sapien_settle_window = env_int("SAPIEN_SETTLE_WINDOW", 20)
    args.sapien_settle_translation_threshold = env_float("SAPIEN_SETTLE_TRANSLATION_THRESHOLD", 5e-4)
    args.sapien_video_fps = env_int("SAPIEN_VIDEO_FPS", 30)
    args.sapien_video_width = env_int("SAPIEN_VIDEO_WIDTH", 960)
    args.sapien_video_height = env_int("SAPIEN_VIDEO_HEIGHT", 720)
    args.sapien_static_friction = env_float("SAPIEN_STATIC_FRICTION", 4.0)
    args.sapien_dynamic_friction = env_float("SAPIEN_DYNAMIC_FRICTION", 3.0)
    args.sapien_linear_damping = env_float("SAPIEN_LINEAR_DAMPING", 3.0)
    args.sapien_angular_damping = env_float("SAPIEN_ANGULAR_DAMPING", 30.0)
    args.sapien_default_scene = env_bool("SAPIEN_DEFAULT_SCENE", False)
    args.sapien_physx_gpu = env_bool("SAPIEN_PHYSX_GPU", False)

    args.skip_diagnostic = bool(args.skip_diagnostic or not env_bool("PIPELINE_STAGE_DIAGNOSTIC", True))
    args.diagnostic_timeout = env_int("PIPELINE_STAGE_DIAGNOSTIC_TIMEOUT", 900)
    args.skip_web_assets = bool(args.skip_web_assets or not env_bool("WEB_ASSETS_EXPORT", True))
    args.web_assets_timeout = env_int("WEB_ASSETS_TIMEOUT", 1800)
    args.web_preview_compressor = env_str("WEB_ASSETS_PREVIEW_COMPRESSOR", "gltfpack")
    args.web_object_compressor = env_str("WEB_ASSETS_WEB_COMPRESSOR", "gltfpack")
    args.gltfpack_bin = env_str("WEB_ASSETS_GLTFPACK_BIN", "gltfpack", "GLTFPACK_BIN")
    args.gltf_transform_bin = env_str("WEB_ASSETS_GLTF_TRANSFORM_BIN", "gltf-transform", "GLTF_TRANSFORM_BIN")
    args.gltfpack_compression = env_str("WEB_ASSETS_GLTFPACK_COMPRESSION", "cz")
    args.gltfpack_compression_extension = env_str("WEB_ASSETS_GLTFPACK_COMPRESSION_EXTENSION", "ext")
    args.gltfpack_texture_quality = env_int("WEB_ASSETS_GLTFPACK_TEXTURE_QUALITY", 3)
    args.gltfpack_texture_limit = env_int("WEB_ASSETS_GLTFPACK_TEXTURE_LIMIT", 128)
    args.gltfpack_simplify_ratio = env_float("WEB_ASSETS_GLTFPACK_SIMPLIFY_RATIO", 0.02)
    args.gltfpack_simplify_error = env_float("WEB_ASSETS_GLTFPACK_SIMPLIFY_ERROR", 0.10)
    args.gltfpack_simplify_aggressive = env_bool("WEB_ASSETS_GLTFPACK_SIMPLIFY_AGGRESSIVE", True)
    args.gltfpack_simplify_permissive = env_bool("WEB_ASSETS_GLTFPACK_SIMPLIFY_PERMISSIVE", True)
    args.gltfpack_position_bits = env_int("WEB_ASSETS_GLTFPACK_POSITION_BITS", 9)
    args.gltfpack_texcoord_bits = env_int("WEB_ASSETS_GLTFPACK_TEXCOORD_BITS", 7)
    args.gltfpack_normal_bits = env_int("WEB_ASSETS_GLTFPACK_NORMAL_BITS", 5)
    args.no_texture_compress = env_bool("WEB_ASSETS_NO_TEXTURE_COMPRESS", False)
    args.skip_web_animation = bool(args.skip_web_animation or not env_bool("WEB_PIPELINE_ANIMATION", True))
    args.web_animation_timeout = env_int("WEB_PIPELINE_ANIMATION_TIMEOUT", 900)
    args.package_timeout = env_int("PACKAGE_TIMEOUT", 2400)
    return args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open-source image-to-3D scene generation pipeline.")
    parser.add_argument("--config", type=Path, default=Path(os.environ.get("PIPELINE_CONFIG", str(DEFAULT_CONFIG))), help="Pipeline YAML config. Runtime parameters are read from this file.")
    parser.add_argument("--image", required=True, type=Path, help="Input RGB image.")
    parser.add_argument("--mask", type=Path, default=None, help="Optional precomputed label mask. Nonzero instance ids become objects.")
    parser.add_argument("--box", type=parse_box, action="append", default=[], help="Object box x1,y1,x2,y2. Repeatable. Pixel or normalized coords.")
    parser.add_argument("--point", type=parse_point, action="append", default=[], help="Object point x,y,label. Repeatable. label 1=fg, 0=bg.")
    parser.add_argument("--text-prompt", default="", help="Used by GroundingDINO only when no object boxes/points are supplied.")
    parser.add_argument("--ground-mask", type=Path, default=None, help="Optional precomputed binary ground mask.")
    parser.add_argument("--ground-point", type=parse_point, action="append", default=[], help="Optional ground point x,y,label for MoGe ground normal.")
    parser.add_argument("--scene-graph", type=Path, default=None, help="Optional precomputed mask-scoped scene graph JSON.")
    parser.add_argument("--output-root", type=Path, default=WORKFLOW_ROOT / "sessions", help="Parent directory for generated sessions.")
    parser.add_argument("--session-dir", type=Path, default=None, help="Use an explicit session directory instead of output_root/session_id.")
    parser.add_argument("--session-id", default="", help="Optional stable session id. Defaults to a timestamp id.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-scene-graph", action="store_true", help="Skip VLM scene graph even if enabled in config.")
    parser.add_argument("--skip-gravity", action="store_true", help="Skip SAPIEN gravity settling.")
    parser.add_argument("--run-background-3dgs", action="store_true", help="Run optional 3DGS background reconstruction after final scene reconstruction.")
    parser.add_argument("--skip-background-3dgs", action="store_true", help="Disable 3DGS background reconstruction even if enabled in config.")
    parser.add_argument("--skip-diagnostic", action="store_true", help="Skip diagnostic Blend export.")
    parser.add_argument("--skip-web-assets", action="store_true", help="Skip compressed web GLB export.")
    parser.add_argument("--skip-web-animation", action="store_true", help="Skip browser animation manifest export.")
    parser.add_argument("--skip-package", action="store_true", help="Skip release_package creation.")
    parser.add_argument("--package-dir", type=Path, default=None, help="Optional release package output directory.")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=Path(os.environ.get("PIPELINE_CONFIG", str(DEFAULT_CONFIG))))
    known, _ = pre.parse_known_args(argv)
    apply_pipeline_config(known.config)
    return fill_config_defaults(build_parser().parse_args(argv))


def main() -> int:
    args = parse_args()
    result = run_pipeline(args)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
