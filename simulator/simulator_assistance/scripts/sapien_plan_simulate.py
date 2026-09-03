#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import sapien


RUN_STARTED_AT = time.time()


def log(message: str) -> None:
    elapsed = time.time() - RUN_STARTED_AT
    print(f"[sapien-plan {elapsed:8.3f}s] {message}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a SAPIEN rigid-body simulation from a simulator_assistance plan.")
    parser.add_argument("--manifest", required=True, type=Path, help="SAPIEN export manifest from blender_export_sapien_scene.py.")
    parser.add_argument("--plan", required=True, type=Path, help="simulation_plan.json.")
    parser.add_argument("--sequence-output", required=True, type=Path)
    parser.add_argument("--pose-output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def mat_to_pose(matrix: list[list[float]]) -> sapien.Pose:
    return sapien.Pose(np.asarray(matrix, dtype=np.float64))


def pose_to_matrix(pose: sapien.Pose) -> list[list[float]]:
    return np.asarray(pose.to_transformation_matrix(), dtype=np.float64).tolist()


def euler_xyz_matrix(deg: list[float]) -> np.ndarray:
    rx, ry, rz = [math.radians(float(v)) for v in deg]
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    mx = np.array([[1, 0, 0, 0], [0, cx, -sx, 0], [0, sx, cx, 0], [0, 0, 0, 1]], dtype=np.float64)
    my = np.array([[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1]], dtype=np.float64)
    mz = np.array([[cz, -sz, 0, 0], [sz, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    return mz @ my @ mx


def apply_initial_offset(matrix: list[list[float]], offset: dict[str, Any] | None) -> list[list[float]]:
    base = np.asarray(matrix, dtype=np.float64)
    if not offset:
        return base.tolist()
    result = base.copy()
    rotation = offset.get("rotation_euler_deg")
    if rotation:
        order = str(offset.get("rotation_order", "xyz")).lower()
        if order != "xyz":
            raise ValueError(f"Unsupported rotation_order={order!r}; only xyz is implemented.")
        result = result @ euler_xyz_matrix(rotation)
    translation = offset.get("translation")
    if translation:
        result[:3, 3] += np.asarray(translation, dtype=np.float64)
    return result.tolist()


def get_object_plan(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for obj in plan.get("objects") or []:
        if obj.get("name"):
            by_name[str(obj["name"])] = obj
    return by_name


def plan_body_type(obj_plan: dict[str, Any], default_body_type: str) -> str:
    body_type = obj_plan.get("body_type")
    if body_type is None and str(obj_plan.get("type", "")).lower() in {"dynamic", "kinematic", "static"}:
        body_type = obj_plan.get("type")
    return str(body_type or default_body_type).lower()


def apply_top_level_actions(
    actions: list[dict[str, Any]],
    plan_by_name: dict[str, dict[str, Any]],
    actors_by_name: dict[str, Any],
    components_by_name: dict[str, Any],
) -> list[dict[str, Any]]:
    applied: list[dict[str, Any]] = []
    for action in actions:
        action_type = str(action.get("type", "")).lower()
        name = action.get("object") or action.get("name") or action.get("target")
        if action_type in {"wait", "sleep"}:
            continue
        if not name:
            continue
        name = str(name)
        component = components_by_name.get(name)
        obj_plan = plan_by_name.get(name)
        if component is None or obj_plan is None:
            continue
        if action_type in {"set_initial_velocity", "set_velocity", "initial_velocity"}:
            linear_velocity = action.get("linear_velocity")
            angular_velocity = action.get("angular_velocity")
            if linear_velocity is not None:
                obj_plan["initial_linear_velocity"] = linear_velocity
                set_rigid_property(actors_by_name[name], component, "linear_velocity", np.asarray(linear_velocity, dtype=np.float32))
            if angular_velocity is not None:
                obj_plan["initial_angular_velocity"] = angular_velocity
                set_rigid_property(actors_by_name[name], component, "angular_velocity", np.asarray(angular_velocity, dtype=np.float32))
            applied.append({"step": 0, "object": name, "action": action})
    return applied


def make_scene(engine: dict[str, Any], defaults: dict[str, Any]) -> sapien.Scene:
    gravity_z = float(engine.get("gravity_z", -9.81))
    timestep = float(engine.get("timestep", 0.005))
    scene_config = sapien.physx.PhysxSceneConfig()
    scene_config.gravity = np.array([0.0, 0.0, gravity_z], dtype=np.float32)
    sapien.physx.set_scene_config(scene_config)
    sapien.physx.set_default_material(
        static_friction=float(defaults.get("static_friction", 0.8)),
        dynamic_friction=float(defaults.get("dynamic_friction", 0.6)),
        restitution=float(defaults.get("restitution", 0.05)),
    )
    physx_system = sapien.physx.PhysxGpuSystem() if bool(engine.get("physx_gpu", False)) else sapien.physx.PhysxCpuSystem()
    scene = sapien.Scene([physx_system])
    scene.set_timestep(timestep)
    scene.add_ground(0.0, render=False)
    return scene


def set_rigid_property(actor: Any, component: Any, name: str, value: Any) -> None:
    setter = getattr(component, f"set_{name}", None)
    if callable(setter):
        setter(value)
        return
    if hasattr(component, name):
        setattr(component, name, value)
        return
    setter = getattr(actor, f"set_{name}", None)
    if callable(setter):
        setter(value)


def entry_collision_paths(entry: dict[str, Any]) -> list[str]:
    paths = [str(path) for path in (entry.get("collision_paths") or []) if path]
    if paths:
        return paths
    path = entry.get("collision_path")
    return [str(path)] if path else []


def add_actor(scene: sapien.Scene, entry: dict[str, Any], obj_plan: dict[str, Any], defaults: dict[str, Any]) -> Any:
    body_type = plan_body_type(obj_plan, str(defaults.get("unmentioned_body_type", "static")))
    if body_type not in {"dynamic", "kinematic", "static"}:
        raise ValueError(f"Unsupported body_type={body_type!r} for {entry.get('name')}")

    builder = scene.create_actor_builder()
    density = float(obj_plan.get("density", defaults.get("density", 700.0)))
    collision_paths = entry_collision_paths(entry)
    if not collision_paths:
        raise RuntimeError(f"actor[{entry.get('index')}:{entry.get('name')}] has no collision_path/collision_paths")
    log(f"actor[{entry.get('index')}:{entry.get('name')}] collision_parts={len(collision_paths)} type={body_type} density={density}")
    for part_index, collision_path in enumerate(collision_paths):
        log(f"actor[{entry.get('index')}:{entry.get('name')}] collision_part[{part_index}]={collision_path}")
        try:
            builder.add_convex_collision_from_file(collision_path, density=density)
        except TypeError:
            builder.add_convex_collision_from_file(collision_path)

    if body_type == "static":
        actor = builder.build_static(name=entry["name"])
    elif body_type == "kinematic":
        actor = builder.build_kinematic(name=entry["name"])
    else:
        actor = builder.build(name=entry["name"])

    pose_matrix = apply_initial_offset(entry["initial_pose"], obj_plan.get("initial_pose_offset"))
    actor.set_pose(mat_to_pose(pose_matrix))

    component = getattr(actor, "find_component_by_type", lambda *_: None)(sapien.physx.PhysxRigidDynamicComponent)
    if component is not None:
        linear_damping = float(obj_plan.get("linear_damping", defaults.get("linear_damping", 0.02)))
        angular_damping = float(obj_plan.get("angular_damping", defaults.get("angular_damping", 0.02)))
        set_rigid_property(actor, component, "linear_damping", linear_damping)
        set_rigid_property(actor, component, "angular_damping", angular_damping)
        if obj_plan.get("initial_linear_velocity") is not None:
            set_rigid_property(actor, component, "linear_velocity", np.asarray(obj_plan["initial_linear_velocity"], dtype=np.float32))
        if obj_plan.get("initial_angular_velocity") is not None:
            set_rigid_property(actor, component, "angular_velocity", np.asarray(obj_plan["initial_angular_velocity"], dtype=np.float32))
    return actor


def apply_events(step: int, actors_by_name: dict[str, Any], components_by_name: dict[str, Any], plan_objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    applied = []
    for obj in plan_objects:
        name = obj.get("name")
        if not name:
            continue
        actor = actors_by_name.get(name)
        component = components_by_name.get(name)
        if actor is None:
            continue
        for event in obj.get("events") or []:
            if int(event.get("step", -1)) != step:
                continue
            if component is not None and event.get("linear_velocity") is not None:
                set_rigid_property(actor, component, "linear_velocity", np.asarray(event["linear_velocity"], dtype=np.float32))
            if component is not None and event.get("angular_velocity") is not None:
                set_rigid_property(actor, component, "angular_velocity", np.asarray(event["angular_velocity"], dtype=np.float32))
            if component is not None and (event.get("force") is not None or event.get("torque") is not None):
                force = np.asarray(event.get("force", [0, 0, 0]), dtype=np.float32)
                torque = np.asarray(event.get("torque", [0, 0, 0]), dtype=np.float32)
                add_force_torque = getattr(component, "add_force_torque", None)
                if callable(add_force_torque):
                    try:
                        add_force_torque(force=force, torque=torque)
                    except TypeError:
                        add_force_torque(force, torque)
            applied.append({"step": step, "object": name, "event": event})
    return applied


def run(args: argparse.Namespace) -> dict[str, Any]:
    log(f"SAPIEN version: {getattr(sapien, '__version__', 'unknown')}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    engine = plan.get("engine") or {}
    defaults = plan.get("defaults") or {}
    steps = int(engine.get("steps", 360))
    timestep = float(engine.get("timestep", 0.005))
    for action in plan.get("actions") or []:
        if str(action.get("type", "")).lower() in {"wait", "sleep"} and action.get("duration") is not None and "steps" not in engine:
            steps = max(steps, int(math.ceil(float(action["duration"]) / max(timestep, 1e-8))))
    sample_every = max(1, int(engine.get("sample_every", 2)))

    scene = make_scene(engine, defaults)
    plan_by_name = get_object_plan(plan)
    default_body_type = str(defaults.get("unmentioned_body_type", "static")).lower()

    actors: list[tuple[dict[str, Any], Any, dict[str, Any]]] = []
    actors_by_name: dict[str, Any] = {}
    components_by_name: dict[str, Any] = {}
    for entry in manifest.get("objects") or []:
        name = str(entry.get("name"))
        obj_plan = plan_by_name.get(name, {"name": name, "body_type": default_body_type})
        actor = add_actor(scene, entry, obj_plan, defaults)
        actors.append((entry, actor, obj_plan))
        actors_by_name[name] = actor
        component = getattr(actor, "find_component_by_type", lambda *_: None)(sapien.physx.PhysxRigidDynamicComponent)
        if component is not None:
            components_by_name[name] = component

    frames = []
    applied_events: list[dict[str, Any]] = apply_top_level_actions(
        plan.get("actions") or [],
        plan_by_name,
        actors_by_name,
        components_by_name,
    )

    def sample(step: int) -> None:
        frames.append(
            {
                "frame": len(frames),
                "step": step,
                "time": float(step) * timestep,
                "objects": [
                    {
                        "index": int(entry["index"]),
                        "name": entry["name"],
                        "pose": pose_to_matrix(actor.get_pose()),
                    }
                    for entry, actor, _ in actors
                ],
            }
        )

    sample(0)
    for step in range(1, steps + 1):
        applied_events.extend(apply_events(step, actors_by_name, components_by_name, plan.get("objects") or []))
        scene.step()
        if step == steps or step % sample_every == 0:
            sample(step)
        if step == 1 or step == steps or step % max(1, steps // 10) == 0:
            log(f"physics step {step}/{steps}")

    final_objects = [
        {
            "index": int(entry["index"]),
            "name": entry["name"],
            "initial_pose": entry["initial_pose"],
            "final_pose": pose_to_matrix(actor.get_pose()),
            "body_type": plan_body_type(obj_plan, default_body_type),
        }
        for entry, actor, obj_plan in actors
    ]
    sequence = {
        "schema": "simulator_animation_sequence.v1",
        "source_manifest": str(args.manifest),
        "source_plan": str(args.plan),
        "goal": plan.get("goal"),
        "steps": steps,
        "timestep": timestep,
        "sample_every": sample_every,
        "gravity_z": float(engine.get("gravity_z", -9.81)),
        "ground_z": 0.0,
        "coordinate_system": manifest.get("coordinate_system", "Blender/SAPIEN Z-up"),
        "objects": [
            {
                "index": int(entry["index"]),
                "name": entry["name"],
                "initial_pose": entry["initial_pose"],
                "body_type": plan_body_type(obj_plan, default_body_type),
            }
            for entry, _, obj_plan in actors
        ],
        "frames": frames,
    }
    poses = {
        "schema": "simulator_final_poses.v1",
        "status": "ok",
        "source_manifest": str(args.manifest),
        "source_plan": str(args.plan),
        "steps": steps,
        "timestep": timestep,
        "objects": final_objects,
    }
    report = {
        "status": "ok",
        "source_manifest": str(args.manifest),
        "source_plan": str(args.plan),
        "sequence_output": str(args.sequence_output),
        "pose_output": str(args.pose_output),
        "objects": final_objects,
        "sampled_frames": len(frames),
        "applied_events": applied_events,
        "assumptions": plan.get("assumptions") or [],
        "success_conditions": plan.get("success_conditions") or [],
    }
    args.sequence_output.parent.mkdir(parents=True, exist_ok=True)
    args.pose_output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.sequence_output.write_text(json.dumps(sequence, ensure_ascii=False, indent=2), encoding="utf-8")
    args.pose_output.write_text(json.dumps(poses, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    report = run(parse_args())
    log(f"Wrote sequence: {report['sequence_output']}")
    log(f"Wrote final poses: {report['pose_output']}")
    log(f"Sampled frames: {report['sampled_frames']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
