#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


WORKFLOW_ROOT = Path(__file__).resolve().parents[3]
SKILL_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BLENDER_BIN = os.environ.get("BLENDER_BIN", "blender")
DEFAULT_CONDA_BIN = os.environ.get("SIMULATOR_ASSISTANCE_RUNNER_CONDA_BIN") or os.environ.get("CONDA_BIN", "conda")
DEFAULT_CONDA_ENV = os.environ.get("SIMULATOR_ASSISTANCE_RUNNER_CONDA_ENV") or os.environ.get("FYSIVERSE_CONDA_ENV", "fysiverse-scene")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a scene, run a SAPIEN plan, and create a Blender animation.")
    parser.add_argument("--final-manifest", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--blender-bin", default=DEFAULT_BLENDER_BIN)
    parser.add_argument("--conda-bin", default=DEFAULT_CONDA_BIN)
    parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENV)
    parser.add_argument("--reuse-upstream-sapien-export", choices=["always", "auto", "never"], default="always")
    parser.add_argument("--convex-decomposition-method", choices=["single_convex_hull", "coacd"], default="coacd")
    parser.add_argument("--coacd-timeout", type=int, default=120)
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
    parser.add_argument("--coacd-merge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--coacd-decimate", action="store_true")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--keep-export", action="store_true", help="Reserved; export is currently kept for audit in all cases.")
    parser.add_argument("--skip-original-camera-render", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-camera-pose-optimization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-object-pose-optimization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-width", type=int, default=480)
    parser.add_argument("--render-height", type=int, default=480, help="<=0 preserves the original input image aspect ratio.")
    parser.add_argument("--render-fps", type=int, default=12)
    parser.add_argument("--render-samples", type=int, default=8)
    parser.add_argument("--render-max-frames", type=int, default=50, help="0 renders the full animation frame range.")
    parser.add_argument("--render-compute-backend", default="CUDA")
    parser.add_argument("--camera-opt-steps", type=int, default=300)
    parser.add_argument("--camera-opt-max-side", type=int, default=512)
    parser.add_argument("--camera-opt-rot-lr", type=float, default=0.015)
    parser.add_argument("--camera-opt-trans-lr", type=float, default=0.01)
    parser.add_argument("--object-pose-workers", type=int, default=0, help="<=0 means one worker per object.")
    parser.add_argument("--object-pose-max-side", type=int, default=512)
    parser.add_argument("--object-position-steps", type=int, default=120)
    parser.add_argument("--object-yaw-steps", type=int, default=80)
    parser.add_argument("--object-scale-steps", type=int, default=60)
    parser.add_argument("--object-translation-mode", choices=["camera_plane", "world_xy", "world_xyz"], default="camera_plane")
    parser.add_argument("--object-translation-initial-grid", type=int, default=3)
    parser.add_argument("--object-center-weight", type=float, default=0.15)
    parser.add_argument("--object-area-weight", type=float, default=0.05)
    parser.add_argument("--object-yaw-initial-samples", type=int, default=5)
    parser.add_argument("--object-scale-initial-samples", type=int, default=5)
    parser.add_argument("--object-phase-accept-iou-drop", type=float, default=0.01)
    parser.add_argument("--object-position-lr", type=float, default=0.01)
    parser.add_argument("--object-yaw-lr", type=float, default=0.02)
    parser.add_argument("--object-scale-lr", type=float, default=0.01)
    parser.add_argument("--object-max-translation", type=float, default=0.35)
    parser.add_argument("--object-max-yaw-deg", type=float, default=35.0)
    parser.add_argument("--object-max-scale-delta", type=float, default=0.25)
    return parser.parse_args()


def run_command(cmd: list[str], *, cwd: Path) -> str:
    print("+ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        print(proc.stdout, file=sys.stderr)
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout)
    if proc.stdout.strip():
        print(proc.stdout)
    return proc.stdout


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def existing_path(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_file() else None


def validate_collision_paths(entry: dict[str, Any]) -> list[str]:
    paths = [str(path) for path in (entry.get("collision_paths") or []) if path]
    if not paths and entry.get("collision_path"):
        paths = [str(entry["collision_path"])]
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        name = entry.get("name") or entry.get("index")
        raise FileNotFoundError(f"Missing reused collision asset(s) for {name}: {missing[:5]}")
    if not paths:
        name = entry.get("name") or entry.get("index")
        raise RuntimeError(f"Cannot reuse SAPIEN export for {name}: no collision_path/collision_paths")
    return paths


def build_reused_sapien_manifest(
    *,
    final_manifest_path: Path,
    final_manifest: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    upstream_manifest = existing_path((final_manifest.get("intermediate") or {}).get("sapien_export_manifest"))
    if upstream_manifest is None:
        raise FileNotFoundError(
            "final_scene_manifest.json has no existing intermediate.sapien_export_manifest; "
            "run the main 3D pipeline first or rerun with --reuse-upstream-sapien-export auto/never."
        )

    source_payload = load_json(upstream_manifest)
    source_by_name = {str(entry.get("name")): entry for entry in (source_payload.get("objects") or []) if entry.get("name")}
    reused_objects: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for obj in final_manifest.get("objects") or []:
        export = obj.get("sapien_export")
        name = obj.get("final_3d_object_name") or (export or {}).get("name")
        if not name:
            skipped.append({"mask_id": obj.get("mask_id"), "reason": "no final_3d_object_name"})
            continue
        if not isinstance(export, dict):
            export = source_by_name.get(str(name))
        if not isinstance(export, dict):
            raise RuntimeError(f"Cannot reuse SAPIEN export for {name}: missing sapien_export entry")

        entry = dict(export)
        entry["name"] = str(name)
        entry["index"] = len(reused_objects)

        final_pose = ((obj.get("sapien_final_pose") or {}).get("final_pose"))
        if final_pose is not None:
            entry["initial_pose"] = final_pose
            entry["reused_initial_pose_source"] = "final_scene_manifest.objects[].sapien_final_pose.final_pose"
        elif entry.get("initial_pose") is not None:
            entry["reused_initial_pose_source"] = "upstream_sapien_export.initial_pose"
        else:
            raise RuntimeError(f"Cannot reuse SAPIEN export for {name}: no final_pose or initial_pose")

        paths = validate_collision_paths(entry)
        entry["collision_paths"] = paths
        if not entry.get("collision_path"):
            entry["collision_path"] = paths[0]
        reused_objects.append(entry)

    if not reused_objects:
        raise RuntimeError(f"No reusable SAPIEN objects found in {final_manifest_path}")

    payload = dict(source_payload)
    payload.update(
        {
            "schema": "simulator_assistance_reused_sapien_manifest.v1",
            "source_final_manifest": str(final_manifest_path),
            "source_sapien_export_manifest": str(upstream_manifest),
            "reuse_policy": "collision assets reused; initial_pose replaced by upstream final settled pose when available",
            "objects": reused_objects,
            "skipped_final_manifest_objects": skipped,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "manifest": str(output_path),
        "source_sapien_export_manifest": str(upstream_manifest),
        "object_count": len(reused_objects),
        "skipped": skipped,
    }


def find_camera_reports(final_manifest_path: Path, final_manifest: dict[str, Any]) -> tuple[Path, Path | None]:
    result_dir_value = (final_manifest.get("inputs") or {}).get("result_dir")
    result_dir = Path(result_dir_value) if result_dir_value else final_manifest_path.parent
    rotated_report = result_dir / "sam3d_moge_rotated_report.json"
    grounded_report = result_dir / "sam3d_moge_grounded_report.json"
    postprocess_report = (final_manifest.get("intermediate") or {}).get("sam3d_postprocess_report")

    if not rotated_report.is_file():
        candidates = sorted(result_dir.glob("*moge*rotated_report.json"))
        if candidates:
            rotated_report = candidates[0]
    stage_report: Path | None = grounded_report if grounded_report.is_file() else None
    if stage_report is None and postprocess_report:
        p = Path(postprocess_report)
        if p.is_file():
            stage_report = p
    if not rotated_report.is_file():
        raise FileNotFoundError(f"Cannot find SAM3D/MoGe rotated camera report near {result_dir}")
    return rotated_report, stage_report


def find_upstream_camera_optimization(final_manifest: dict[str, Any]) -> Path | None:
    path = existing_path((final_manifest.get("intermediate") or {}).get("moge_camera_pose_optimization"))
    if path is not None:
        return path
    result_dir_value = (final_manifest.get("inputs") or {}).get("result_dir")
    if not result_dir_value:
        return None
    result_dir = Path(result_dir_value)
    candidates = sorted(result_dir.glob("*camera_pose_optimization*.json"))
    return candidates[0] if candidates else None


def main() -> int:
    args = parse_args()
    final_manifest = load_json(args.final_manifest)
    plan = load_json(args.plan)
    final_blend = final_manifest.get("outputs", {}).get("final_blend")
    if not final_blend:
        raise RuntimeError(f"No outputs.final_blend in {args.final_manifest}")
    final_blend_path = Path(final_blend)
    if not final_blend_path.is_file():
        raise FileNotFoundError(final_blend_path)
    input_image = (final_manifest.get("inputs") or {}).get("image")
    label_mask = (final_manifest.get("inputs") or {}).get("label_mask")
    if label_mask:
        label_mask_path = Path(label_mask)
    else:
        label_mask_path = None

    args.work_dir.mkdir(parents=True, exist_ok=True)
    export_dir = args.work_dir / "export"
    export_manifest = export_dir / "manifest.json"
    sequence_output = args.work_dir / "animation_sequence.json"
    pose_output = args.work_dir / "final_poses.json"
    report_output = args.work_dir / "sapien_sequence_report.json"
    animation_blend = args.work_dir / "simulation_animation.blend"
    runner_report = args.work_dir / "run_report.json"
    original_view_video = args.work_dir / "simulation_animation_original_view.mp4"
    original_view_report = args.work_dir / "simulation_animation_original_view_render.json"
    original_view_blend = args.work_dir / "simulation_animation_original_view_camera.blend"
    camera_opt_mesh = args.work_dir / "camera_pose_optimization_mesh.npz"
    camera_opt_report = args.work_dir / "camera_pose_optimization.json"
    camera_opt_preview = args.work_dir / "camera_pose_optimization_preview.png"
    object_pose_mesh = args.work_dir / "object_pose_optimization_mesh.npz"
    object_pose_report = args.work_dir / "object_pose_optimization.json"
    object_pose_preview_dir = args.work_dir / "object_pose_optimization_previews"
    object_pose_blend = args.work_dir / "object_pose_optimized.blend"

    camera_report: Path | None = None
    stage_report: Path | None = None
    camera_pose_optimization: dict[str, Any] | None = None
    upstream_camera_pose_optimization = find_upstream_camera_optimization(final_manifest)
    object_pose_optimization: dict[str, Any] | None = None
    simulation_source_blend_path = final_blend_path

    if not args.skip_original_camera_render or not args.skip_object_pose_optimization:
        camera_report, stage_report = find_camera_reports(args.final_manifest, final_manifest)
        if not args.skip_camera_pose_optimization:
            if label_mask_path is None or not label_mask_path.is_file():
                raise FileNotFoundError(f"Cannot optimize camera pose without inputs.label_mask in {args.final_manifest}")
            export_mesh_cmd = [
                args.blender_bin,
                "-b",
                "--python",
                str(WORKFLOW_ROOT / "scripts" / "blender_export_nvdiffrast_mesh.py"),
                "--",
                "--input",
                str(final_blend_path),
                "--output",
                str(camera_opt_mesh),
                "--frame",
                "1",
            ]
            run_command(export_mesh_cmd, cwd=WORKFLOW_ROOT)
            optimize_cmd = [
                args.conda_bin,
                "run",
                "--no-capture-output",
                "-n",
                args.conda_env,
                "python",
                str(WORKFLOW_ROOT / "scripts" / "optimize_camera_pose_nvdiffrast.py"),
                "--mesh",
                str(camera_opt_mesh),
                "--target-mask",
                str(label_mask_path),
                "--camera-report",
                str(camera_report),
                "--output-json",
                str(camera_opt_report),
                "--preview-output",
                str(camera_opt_preview),
                "--steps",
                str(args.camera_opt_steps),
                "--max-side",
                str(args.camera_opt_max_side),
                "--rot-lr",
                str(args.camera_opt_rot_lr),
                "--trans-lr",
                str(args.camera_opt_trans_lr),
            ]
            if stage_report is not None:
                optimize_cmd.extend(["--stage-report", str(stage_report)])
            if input_image and Path(input_image).is_file():
                optimize_cmd.extend(["--input-image", str(input_image)])
            run_command(optimize_cmd, cwd=WORKFLOW_ROOT)
            camera_pose_optimization = {
                "mesh": str(camera_opt_mesh),
                "report": str(camera_opt_report),
                "preview": str(camera_opt_preview),
            }

    if not args.skip_object_pose_optimization:
        if label_mask_path is None or not label_mask_path.is_file():
            raise FileNotFoundError(f"Cannot optimize object poses without inputs.label_mask in {args.final_manifest}")
        run_command(
            [
                args.blender_bin,
                "-b",
                "--python",
                str(WORKFLOW_ROOT / "scripts" / "blender_export_nvdiffrast_mesh.py"),
                "--",
                "--input",
                str(final_blend_path),
                "--output",
                str(object_pose_mesh),
                "--frame",
                "1",
            ],
            cwd=WORKFLOW_ROOT,
        )
        object_opt_cmd = [
            args.conda_bin,
            "run",
            "--no-capture-output",
            "-n",
            args.conda_env,
            "python",
            str(WORKFLOW_ROOT / "scripts" / "optimize_object_poses_nvdiffrast.py"),
            "--mesh",
            str(object_pose_mesh),
            "--target-mask",
            str(label_mask_path),
            "--final-manifest",
            str(args.final_manifest),
            "--output-json",
            str(object_pose_report),
            "--preview-dir",
            str(object_pose_preview_dir),
            "--workers",
            str(args.object_pose_workers),
            "--max-side",
            str(args.object_pose_max_side),
            "--position-steps",
            str(args.object_position_steps),
            "--yaw-steps",
            str(args.object_yaw_steps),
            "--scale-steps",
            str(args.object_scale_steps),
            "--translation-mode",
            str(args.object_translation_mode),
            "--translation-initial-grid",
            str(args.object_translation_initial_grid),
            "--center-weight",
            str(args.object_center_weight),
            "--area-weight",
            str(args.object_area_weight),
            "--yaw-initial-samples",
            str(args.object_yaw_initial_samples),
            "--scale-initial-samples",
            str(args.object_scale_initial_samples),
            "--phase-accept-iou-drop",
            str(args.object_phase_accept_iou_drop),
            "--position-lr",
            str(args.object_position_lr),
            "--yaw-lr",
            str(args.object_yaw_lr),
            "--scale-lr",
            str(args.object_scale_lr),
            "--max-translation",
            str(args.object_max_translation),
            "--max-yaw-deg",
            str(args.object_max_yaw_deg),
            "--max-scale-delta",
            str(args.object_max_scale_delta),
        ]
        if camera_pose_optimization is not None:
            object_opt_cmd.extend(["--camera-optimization", str(camera_opt_report)])
        elif camera_report is not None:
            object_opt_cmd.extend(["--camera-report", str(camera_report)])
            if stage_report is not None:
                object_opt_cmd.extend(["--stage-report", str(stage_report)])
        else:
            raise RuntimeError("Object pose optimization requires camera optimization or camera report")
        if input_image and Path(input_image).is_file():
            object_opt_cmd.extend(["--input-image", str(input_image)])
        run_command(object_opt_cmd, cwd=WORKFLOW_ROOT)
        run_command(
            [
                args.blender_bin,
                "-b",
                "--python",
                str(WORKFLOW_ROOT / "scripts" / "blender_apply_object_pose_optimization.py"),
                "--",
                "--input",
                str(final_blend_path),
                "--object-pose-optimization",
                str(object_pose_report),
                "--output",
                str(object_pose_blend),
                "--pack-textures",
            ],
            cwd=WORKFLOW_ROOT,
        )
        simulation_source_blend_path = object_pose_blend
        object_pose_optimization = {
            "mesh": str(object_pose_mesh),
            "report": str(object_pose_report),
            "preview_dir": str(object_pose_preview_dir),
            "optimized_blend": str(object_pose_blend),
            "apply_report": str(object_pose_blend.with_suffix(".object_pose_apply_report.json")),
        }

    sapien_export_reuse: dict[str, Any] | None = None
    if args.reuse_upstream_sapien_export != "never":
        if object_pose_optimization is not None:
            message = (
                "Cannot reuse upstream SAPIEN export after local object pose optimization, "
                "because object transforms/scale may no longer match reused collision assets."
            )
            if args.reuse_upstream_sapien_export == "always":
                raise RuntimeError(message)
            print(f"{message} Falling back to fresh export.", flush=True)
        else:
            try:
                sapien_export_reuse = build_reused_sapien_manifest(
                    final_manifest_path=args.final_manifest,
                    final_manifest=final_manifest,
                    output_path=export_manifest,
                )
                print(
                    "Reused upstream SAPIEN collision assets: "
                    f"{sapien_export_reuse['source_sapien_export_manifest']} -> {export_manifest}",
                    flush=True,
                )
            except Exception as exc:
                if args.reuse_upstream_sapien_export == "always":
                    raise
                print(f"Could not reuse upstream SAPIEN export ({exc}); falling back to fresh export.", flush=True)

    if sapien_export_reuse is None:
        export_cmd = [
            args.blender_bin,
            "-b",
            "--python",
            str(WORKFLOW_ROOT / "scripts" / "blender_export_sapien_scene.py"),
            "--",
            "--input",
            str(simulation_source_blend_path),
            "--output-dir",
            str(export_dir),
            "--manifest",
            str(export_manifest),
            "--collision-decomposition-method",
            "coacd" if args.convex_decomposition_method == "coacd" else "none",
            "--coacd-conda-bin",
            args.conda_bin,
            "--coacd-env",
            args.conda_env,
            "--coacd-timeout",
            str(args.coacd_timeout),
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
        run_command(export_cmd, cwd=WORKFLOW_ROOT)

    run_command(
        [
            args.conda_bin,
            "run",
            "--no-capture-output",
            "-n",
            args.conda_env,
            "python",
            str(SKILL_DIR / "scripts" / "sapien_plan_simulate.py"),
            "--manifest",
            str(export_manifest),
            "--plan",
            str(args.plan),
            "--sequence-output",
            str(sequence_output),
            "--pose-output",
            str(pose_output),
            "--report",
            str(report_output),
        ],
        cwd=WORKFLOW_ROOT,
    )

    fps = int(plan.get("output", {}).get("fps", args.fps)) if isinstance(plan.get("output"), dict) else int(args.fps)
    run_command(
        [
            args.blender_bin,
            "-b",
            "--python",
            str(SKILL_DIR / "scripts" / "blender_apply_animation_sequence.py"),
            "--",
            "--input",
            str(simulation_source_blend_path),
            "--sequence",
            str(sequence_output),
            "--output",
            str(animation_blend),
            "--fps",
            str(fps),
            "--pack-textures",
        ],
        cwd=WORKFLOW_ROOT,
    )

    original_camera_render: dict[str, Any] | None = None
    if not args.skip_original_camera_render:
        if camera_report is None and upstream_camera_pose_optimization is None:
            camera_report, stage_report = find_camera_reports(args.final_manifest, final_manifest)
        render_cmd = [
            args.blender_bin,
            "-b",
            "--python",
            str(WORKFLOW_ROOT / "scripts" / "render_original_camera_animation.py"),
            "--",
            "--input",
            str(animation_blend),
            "--output",
            str(original_view_video),
            "--output-json",
            str(original_view_report),
            "--camera-blend-output",
            str(original_view_blend),
            "--width",
            str(args.render_width),
            "--height",
            str(args.render_height),
            "--fps",
            str(args.render_fps),
            "--samples",
            str(args.render_samples),
            "--compute-backend",
            args.render_compute_backend,
        ]
        if camera_pose_optimization is not None:
            render_cmd.extend(["--camera-optimization", str(camera_opt_report)])
        elif upstream_camera_pose_optimization is not None:
            render_cmd.extend(["--camera-optimization", str(upstream_camera_pose_optimization)])
        else:
            render_cmd.extend(["--camera-report", str(camera_report)])
            if stage_report is not None:
                render_cmd.extend(["--stage-report", str(stage_report)])
        if args.render_max_frames > 0:
            render_cmd.extend(["--max-frames", str(args.render_max_frames)])
        run_command(render_cmd, cwd=WORKFLOW_ROOT)
        original_camera_render = {
            "video": str(original_view_video),
            "report": str(original_view_report),
            "camera_blend": str(original_view_blend),
            "camera_report": str(camera_report) if camera_report is not None else None,
            "stage_report": str(stage_report) if stage_report is not None else None,
            "camera_pose_optimization": camera_pose_optimization,
            "upstream_camera_pose_optimization": str(upstream_camera_pose_optimization) if upstream_camera_pose_optimization is not None else None,
        }

    report = {
        "schema": "simulator_plan_run_report.v1",
        "status": "ok",
        "source_final_manifest": str(args.final_manifest),
        "source_plan": str(args.plan),
        "source_blend": str(final_blend_path),
        "simulation_source_blend": str(simulation_source_blend_path),
        "export_manifest": str(export_manifest),
        "sapien_export_reuse": sapien_export_reuse,
        "animation_sequence": str(sequence_output),
        "final_poses": str(pose_output),
        "sapien_report": str(report_output),
        "animation_blend": str(animation_blend),
        "camera_pose_optimization": camera_pose_optimization,
        "object_pose_optimization": object_pose_optimization,
        "original_camera_render": original_camera_render,
        "goal": plan.get("goal"),
    }
    runner_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote run report: {runner_report}")
    print(f"Wrote Blender animation: {animation_blend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
