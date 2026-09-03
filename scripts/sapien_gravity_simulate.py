#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import sapien


RUN_STARTED_AT = time.time()


def log(message: str) -> None:
    elapsed = time.time() - RUN_STARTED_AT
    print(f"[sapien {elapsed:8.3f}s] {message}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAPIEN rigid-body gravity simulation for exported Blend mesh objects.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--pose-output", required=True, type=Path)
    parser.add_argument("--trajectory-output", type=Path, default=None)
    parser.add_argument("--frames-dir", required=True, type=Path)
    parser.add_argument("--video-path", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--no-render-video", action="store_true", help="Compatibility flag. Rendering is disabled in this pipeline.")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--timestep", type=float, default=1.0 / 100.0)
    parser.add_argument("--gravity-z", type=float, default=-9.81)
    parser.add_argument("--ground-z", type=float, default=-0.001)
    parser.add_argument("--render-every", type=int, default=2)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-fov", type=float, default=50.0)
    parser.add_argument("--static-friction", type=float, default=0.8)
    parser.add_argument("--dynamic-friction", type=float, default=0.6)
    parser.add_argument("--restitution", type=float, default=0.0)
    parser.add_argument("--linear-damping", type=float, default=0.05)
    parser.add_argument("--angular-damping", type=float, default=0.05)
    parser.add_argument("--density", type=float, default=1000.0)
    parser.add_argument("--disable-object-collisions", action="store_true", help="Reserved. Dynamic object collisions are enabled by default for stability auditing.")
    parser.add_argument("--settle-window", type=int, default=20, help="Stop after this many consecutive stable physics frames. Use 0 to disable early stop.")
    parser.add_argument("--settle-translation-threshold", type=float, default=5e-4, help="Maximum per-frame object translation delta for a frame to count as stable.")
    parser.add_argument("--default-scene", action="store_true", help="Compatibility flag. Ignored because rendering is disabled.")
    parser.add_argument("--physx-gpu", action="store_true", help="Use PhysxGpuSystem for physics. This is unrelated to video rendering.")
    parser.add_argument("--log-render-stages", action="store_true", default=True, help="Compatibility flag. Rendering is disabled.")
    parser.add_argument("--render-primitive-box", action="store_true", help="Compatibility flag. Rendering is disabled.")
    parser.add_argument("--mounted-camera", action="store_true", help="Compatibility flag. Rendering is disabled.")
    return parser.parse_args()


def log_sapien_runtime() -> None:
    version = getattr(sapien, "__version__", "unknown")
    log(f"SAPIEN version: {version}")


def mat_to_pose(matrix: list[list[float]]) -> sapien.Pose:
    return sapien.Pose(np.asarray(matrix, dtype=np.float64))


def pose_to_matrix(pose: sapien.Pose) -> list[list[float]]:
    return np.asarray(pose.to_transformation_matrix(), dtype=np.float64).tolist()


def collect_actor_poses(actors: list[tuple[dict[str, Any], Any]], frame: int) -> dict[str, Any]:
    return {
        "frame": int(frame),
        "objects": [
            {
                "index": int(entry["index"]),
                "name": entry["name"],
                "pose": pose_to_matrix(actor.get_pose()),
            }
            for entry, actor in actors
        ],
    }


def collect_actor_translations(actors: list[tuple[dict[str, Any], Any]]) -> dict[str, np.ndarray]:
    return {
        str(entry["name"]): np.asarray(actor.get_pose().p, dtype=np.float64).copy()
        for entry, actor in actors
    }


def max_translation_delta(previous: dict[str, np.ndarray], current: dict[str, np.ndarray]) -> float:
    deltas = [
        float(np.linalg.norm(current[name] - previous[name]))
        for name in current
        if name in previous
    ]
    return max(deltas, default=0.0)


def make_scene(args: argparse.Namespace) -> sapien.Scene:
    log(f"creating physics-only scene physx_gpu={bool(args.physx_gpu)}")
    if args.default_scene:
        log("--default-scene ignored: rendering is disabled, so no RenderSystem is created")
    scene_config = sapien.physx.PhysxSceneConfig()
    scene_config.gravity = np.array([0.0, 0.0, float(args.gravity_z)], dtype=np.float32)
    sapien.physx.set_scene_config(scene_config)
    sapien.physx.set_default_material(
        static_friction=float(args.static_friction),
        dynamic_friction=float(args.dynamic_friction),
        restitution=float(args.restitution),
    )
    physx_system = sapien.physx.PhysxGpuSystem() if args.physx_gpu else sapien.physx.PhysxCpuSystem()
    scene = sapien.Scene([physx_system])
    log("scene object created")
    scene.set_timestep(float(args.timestep))
    log(f"scene timestep set to {float(args.timestep)}")
    scene.add_ground(float(args.ground_z), render=False)
    log(f"ground added at z={float(args.ground_z)}")
    return scene


def entry_collision_paths(entry: dict[str, Any]) -> list[str]:
    paths = [str(path) for path in (entry.get("collision_paths") or []) if path]
    if paths:
        return paths
    path = entry.get("collision_path")
    return [str(path)] if path else []


def add_actor(scene: sapien.Scene, entry: dict[str, Any], args: argparse.Namespace):
    log(f"actor[{entry.get('index')}:{entry.get('name')}] create builder")
    builder = scene.create_actor_builder()
    log(f"actor[{entry.get('index')}:{entry.get('name')}] visual loading skipped")
    collision_paths = entry_collision_paths(entry)
    if not collision_paths:
        raise RuntimeError(f"actor[{entry.get('index')}:{entry.get('name')}] has no collision_path/collision_paths")
    for part_index, collision_path in enumerate(collision_paths):
        try:
            log(
                f"actor[{entry.get('index')}:{entry.get('name')}] add convex collision part "
                f"{part_index + 1}/{len(collision_paths)}: {collision_path} density={float(args.density)}"
            )
            builder.add_convex_collision_from_file(collision_path, density=float(args.density))
        except TypeError:
            log(f"actor[{entry.get('index')}:{entry.get('name')}] collision density argument unsupported; retrying without density")
            builder.add_convex_collision_from_file(collision_path)
    log(f"actor[{entry.get('index')}:{entry.get('name')}] collision added parts={len(collision_paths)}")
    log(f"actor[{entry.get('index')}:{entry.get('name')}] build dynamic actor")
    actor = builder.build(name=entry["name"])
    log(f"actor[{entry.get('index')}:{entry.get('name')}] built")
    actor.set_pose(mat_to_pose(entry["initial_pose"]))
    log(f"actor[{entry.get('index')}:{entry.get('name')}] pose set")
    physx = getattr(actor, "find_component_by_type", lambda *_: None)(sapien.physx.PhysxRigidDynamicComponent)
    if physx is not None:
        try:
            physx.set_linear_damping(float(args.linear_damping))
            physx.set_angular_damping(float(args.angular_damping))
            log(f"actor[{entry.get('index')}:{entry.get('name')}] damping set")
        except Exception as exc:
            log(f"actor[{entry.get('index')}:{entry.get('name')}] failed to set damping: {exc}")
    else:
        log(f"actor[{entry.get('index')}:{entry.get('name')}] no PhysxRigidDynamicComponent found")
    return actor


def run(args: argparse.Namespace) -> dict[str, Any]:
    log("run start")
    log_sapien_runtime()
    log(f"loading manifest: {args.manifest}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    log(f"manifest loaded objects={len(manifest.get('objects', []))}")
    scene = make_scene(args)
    actors = []
    for entry in manifest["objects"]:
        actor = add_actor(scene, entry, args)
        actors.append((entry, actor))
    log(f"actors loaded: {len(actors)}")

    rendered = 0
    if not args.no_render_video:
        log("video rendering requested but disabled in this pipeline; running physics only")
    else:
        log("video rendering disabled; running physics only")
    trajectory_frames: list[dict[str, Any]] = [collect_actor_poses(actors, 0)]
    record_every = max(1, int(args.render_every))
    requested_steps = int(args.steps)
    settle_window = max(0, int(args.settle_window))
    settle_threshold = max(0.0, float(args.settle_translation_threshold))
    stability_enabled = settle_window > 0 and settle_threshold > 0.0
    stable_count = 0
    stopped_early = False
    final_step = requested_steps
    last_max_delta = None
    previous_translations = collect_actor_translations(actors)
    delta_tail: list[dict[str, Any]] = []
    if stability_enabled:
        log(f"early stop enabled: settle_window={settle_window} translation_threshold={settle_threshold:g}")
    else:
        log("early stop disabled")
    for step in range(1, requested_steps + 1):
        if step == 1 or step == requested_steps or step % max(1, requested_steps // 10) == 0:
            log(f"physics step {step}/{requested_steps} start")
        scene.step()
        current_translations = collect_actor_translations(actors)
        last_max_delta = max_translation_delta(previous_translations, current_translations)
        if stability_enabled and last_max_delta < settle_threshold:
            stable_count += 1
        else:
            stable_count = 0
        delta_tail.append(
            {
                "frame": int(step),
                "max_translation_delta": float(last_max_delta),
                "stable_count": int(stable_count),
            }
        )
        if len(delta_tail) > max(20, settle_window):
            delta_tail = delta_tail[-max(20, settle_window):]
        if step % record_every == 0 or step == requested_steps:
            if trajectory_frames[-1]["frame"] != int(step):
                trajectory_frames.append(collect_actor_poses(actors, step))
        if step == 1 or step == requested_steps or step % max(1, requested_steps // 10) == 0:
            log(f"physics step {step}/{requested_steps} done max_translation_delta={last_max_delta:.6g} stable_count={stable_count}")
        if stability_enabled and stable_count >= settle_window:
            stopped_early = True
            final_step = int(step)
            if trajectory_frames[-1]["frame"] != int(step):
                trajectory_frames.append(collect_actor_poses(actors, step))
            log(
                f"stable early stop at step {final_step}/{requested_steps}: "
                f"{stable_count} consecutive frames below {settle_threshold:g}"
            )
            break
        previous_translations = current_translations

    log("collecting final poses")
    final_objects = []
    for entry, actor in actors:
        final_objects.append(
            {
                "index": int(entry["index"]),
                "name": entry["name"],
                "initial_pose": entry["initial_pose"],
                "final_pose": pose_to_matrix(actor.get_pose()),
            }
        )
    pose_data = {
        "status": "ok",
        "source_manifest": str(args.manifest),
        "steps": int(final_step),
        "requested_steps": int(requested_steps),
        "timestep": float(args.timestep),
        "gravity_z": float(args.gravity_z),
        "ground_z": float(args.ground_z),
        "object_collisions": not bool(args.disable_object_collisions),
        "early_stop": {
            "enabled": bool(stability_enabled),
            "stopped_early": bool(stopped_early),
            "settle_frame": int(final_step),
            "requested_steps": int(requested_steps),
            "settle_window": int(settle_window),
            "translation_threshold": float(settle_threshold),
            "last_max_translation_delta": None if last_max_delta is None else float(last_max_delta),
        },
        "objects": final_objects,
    }
    args.pose_output.parent.mkdir(parents=True, exist_ok=True)
    log(f"writing final poses: {args.pose_output}")
    args.pose_output.write_text(json.dumps(pose_data, ensure_ascii=False, indent=2), encoding="utf-8")

    trajectory_path = args.trajectory_output or args.pose_output.with_name("pose_trajectory.json")
    trajectory_data = {
        "status": "ok",
        "source_manifest": str(args.manifest),
        "steps": int(final_step),
        "requested_steps": int(requested_steps),
        "timestep": float(args.timestep),
        "gravity_z": float(args.gravity_z),
        "ground_z": float(args.ground_z),
        "record_every": int(record_every),
        "frame_count": len(trajectory_frames),
        "early_stop": {
            "enabled": bool(stability_enabled),
            "stopped_early": bool(stopped_early),
            "settle_frame": int(final_step),
            "requested_steps": int(requested_steps),
            "settle_window": int(settle_window),
            "translation_threshold": float(settle_threshold),
            "last_max_translation_delta": None if last_max_delta is None else float(last_max_delta),
            "delta_tail": delta_tail,
        },
        "objects": [
            {
                "index": int(entry["index"]),
                "name": entry["name"],
                "initial_pose": entry["initial_pose"],
            }
            for entry, _actor in actors
        ],
        "frames": trajectory_frames,
    }
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"writing pose trajectory: {trajectory_path}")
    trajectory_path.write_text(json.dumps(trajectory_data, ensure_ascii=False, indent=2), encoding="utf-8")

    video_ok = False
    report = {
        **pose_data,
        "trajectory_path": str(trajectory_path),
        "trajectory_frame_count": len(trajectory_frames),
        "frames_dir": str(args.frames_dir),
        "rendered_frames": rendered,
        "video_path": str(args.video_path),
        "video_ok": bool(video_ok),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    log(f"writing report: {args.report}")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log("run done")
    return report


def main() -> int:
    args = parse_args()
    report = run(args)
    log(f"SAPIEN objects simulated: {len(report['objects'])}")
    log(f"Wrote final poses: {args.pose_output}")
    log(f"Wrote pose trajectory: {report.get('trajectory_path')}")
    log("Video rendering disabled; no frames or mp4 were written")
    log(f"Wrote report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
