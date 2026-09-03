#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ENGINE = {
    "name": "sapien",
    "steps": 360,
    "timestep": 0.005,
    "sample_every": 2,
    "gravity_z": -9.81,
    "ground_z": 0.0,
    "physx_gpu": False,
}

DEFAULTS = {
    "unmentioned_body_type": "static",
    "static_friction": 0.8,
    "dynamic_friction": 0.6,
    "restitution": 0.05,
    "density": 700.0,
    "linear_damping": 0.02,
    "angular_damping": 0.02,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge simulator_assistance sub-agent JSON outputs.")
    parser.add_argument("--goal", required=True)
    parser.add_argument("--scene-digest", required=True, type=Path)
    parser.add_argument("--task-spec", required=True, type=Path)
    parser.add_argument("--object-semantics", required=True, type=Path)
    parser.add_argument("--physics-params", required=True, type=Path)
    parser.add_argument("--initial-conditions", required=True, type=Path)
    parser.add_argument("--scene-description-output", required=True, type=Path)
    parser.add_argument("--plan-output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--no-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def by_name(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in data.get("objects") or []:
        name = item.get("name") or item.get("object_name") or item.get("final_3d_object_name")
        if name:
            result[str(name)] = item
    return result


def merge_dicts(*items: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for item in items:
        for key, value in item.items():
            if value is not None:
                merged[key] = value
    return merged


def object_identity(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": card.get("name"),
        "mask_id": card.get("mask_id"),
        "mask_name": card.get("mask_name"),
        "final_3d_object_name": card.get("final_3d_object_name") or card.get("name"),
    }


def build_scene_description(digest: dict[str, Any], task_spec: dict[str, Any], semantics_by_name: dict[str, dict[str, Any]]) -> dict[str, Any]:
    objects = []
    for card in digest.get("objects") or []:
        name = card.get("name")
        sem = semantics_by_name.get(str(name), {})
        role_candidates = ["passive", "static"]
        if name == task_spec.get("active_object"):
            role_candidates = ["active"]
        elif name == task_spec.get("target_object"):
            role_candidates = ["target", "passive"]
        elif name in set(task_spec.get("passive_static_objects") or []):
            role_candidates = ["static"]
        objects.append(
            {
                "mask_id": card.get("mask_id"),
                "mask_name": card.get("mask_name"),
                "object_name": name,
                "final_3d_object_name": card.get("final_3d_object_name") or name,
                "semantic_label": sem.get("semantic_label") or card.get("semantic_label") or card.get("description"),
                "visual_description": sem.get("visual_description") or card.get("description"),
                "upstream_vlm": {
                    "semantic_label": card.get("semantic_label"),
                    "description": card.get("description"),
                    "confidence": card.get("vlm_confidence"),
                    "visibility": card.get("vlm_visibility"),
                    "support_status": card.get("vlm_support_status"),
                },
                "crop_rgb": card.get("crop_rgb"),
                "sam3d_input_mask": card.get("sam3d_input_mask"),
                "bbox_2d_xyxy": card.get("bbox_2d_xyxy"),
                "bbox_2d_xywh": card.get("bbox_2d_xywh"),
                "bbox_3d_min": card.get("bbox_3d_min"),
                "bbox_3d_max": card.get("bbox_3d_max"),
                "bbox_3d_center": card.get("bbox_3d_center"),
                "bbox_3d_extent": card.get("bbox_3d_extent"),
                "role_candidates": role_candidates,
                "physical_hints": {
                    "likely_material": sem.get("material_hint"),
                    "rigid": sem.get("rigid", True),
                    "hollow_or_light": sem.get("hollow_or_light"),
                },
                "needs_crop_review": False,
            }
        )
    return {
        "schema": "simulator_scene_description.v1",
        "source_final_scene_manifest": digest.get("source_final_scene_manifest"),
        "session_id": digest.get("session_id"),
        "coordinate_system": digest.get("coordinate_system"),
        "ground_plane": digest.get("ground_plane"),
        "inputs": digest.get("inputs") or {},
        "task_roles": {
            "active_object": task_spec.get("active_object"),
            "target_object": task_spec.get("target_object"),
            "passive_static_objects": task_spec.get("passive_static_objects") or [],
            "goal_interpretation": task_spec.get("goal_interpretation") or task_spec.get("action"),
        },
        "objects": objects,
        "relations": digest.get("relations") or [],
        "scene_graph": digest.get("scene_graph"),
        "support": digest.get("support"),
        "missing_or_uncertain": [
            "semantic labels and support relations may come from upstream scene_graph, but task roles and executable physics parameters are inferred by simulator_assistance.",
            "real metric scale and mass are approximate unless upstream provides calibration.",
        ],
    }


def build_plan(
    goal: str,
    scene_description_path: Path,
    digest: dict[str, Any],
    task_spec: dict[str, Any],
    semantics_by_name: dict[str, dict[str, Any]],
    physics_by_name: dict[str, dict[str, Any]],
    initial_by_name: dict[str, dict[str, Any]],
    physics_params: dict[str, Any],
    initial_conditions: dict[str, Any],
    no_run: bool,
) -> dict[str, Any]:
    engine = merge_dicts(DEFAULT_ENGINE, physics_params.get("engine") or {})
    engine["ground_z"] = 0.0
    defaults = merge_dicts(DEFAULTS, physics_params.get("defaults") or {})
    active = task_spec.get("active_object")
    target = task_spec.get("target_object")
    static_objects = set(task_spec.get("passive_static_objects") or [])

    objects = []
    for card in digest.get("objects") or []:
        name = str(card.get("name"))
        sem = semantics_by_name.get(name, {})
        role = "passive_static_obstacle"
        body_type = "static"
        if name == active:
            role = "active"
            body_type = "dynamic"
        elif name == target:
            role = "target"
            body_type = "dynamic"
        elif name in static_objects:
            role = "passive_static_obstacle"
            body_type = "static"
        base = {
            **object_identity(card),
            "semantic_label": sem.get("semantic_label") or card.get("semantic_label") or card.get("description"),
            "role": role,
            "body_type": body_type,
            "density": defaults.get("density"),
            "linear_damping": defaults.get("linear_damping"),
            "angular_damping": defaults.get("angular_damping"),
        }
        merged = merge_dicts(base, physics_by_name.get(name, {}), initial_by_name.get(name, {}))
        objects.append(merged)

    assumptions = []
    for source in (task_spec, physics_params, initial_conditions):
        assumptions.extend(source.get("assumptions") or [])
    if not assumptions:
        assumptions.append("Object semantics and physical parameters were inferred by simulator_assistance.")

    plan = {
        "schema": "simulator_simulation_plan.v1",
        "goal": goal,
        "scene_description": str(scene_description_path),
        "source_final_scene_manifest": digest.get("source_final_scene_manifest"),
        "ground_plane": digest.get("ground_plane"),
        "execution_constraints": {
            "do_not_run": bool(no_run),
            "sapien_rendering_enabled": False,
            "sapien_command_prefix": "${CONDA_BIN:-conda} run --no-capture-output -n ${FYSIVERSE_CONDA_ENV:-fysiverse-scene}",
            "blender_binary": "${BLENDER_BIN:-blender}",
        },
        "assumptions": assumptions,
        "success_conditions": task_spec.get("success_conditions") or [],
        "engine": engine,
        "defaults": defaults,
        "objects": objects,
        "planning_notes": {
            "mode": "parallel_agent_merge",
            "renderer_policy": "Do not render in SAPIEN. Save pose sequence and Blender keyframes.",
            "ground_policy": "Ground plane is fixed z=0 in Blender/SAPIEN Z-up coordinates.",
        },
    }
    return plan


def write_summary(path: Path, goal: str, scene: dict[str, Any], plan: dict[str, Any]) -> None:
    roles = scene.get("task_roles") or {}
    lines = [
        "# Codex Simulator Assistance Summary",
        "",
        f"Goal: `{goal}`",
        "",
        f"Active object: `{roles.get('active_object')}`",
        f"Target object: `{roles.get('target_object')}`",
        f"Static/passive objects: `{', '.join(roles.get('passive_static_objects') or [])}`",
        "",
        "Ground plane: fixed `z=0`, Z-up, gravity along negative Z.",
        "",
        f"Plan status: {'not run' if plan.get('execution_constraints', {}).get('do_not_run') else 'ready to run'}",
        "",
        "Files written:",
        "- `scene_description.json`",
        "- `simulation_plan.json`",
        "- `codex_summary.md`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    digest = load_json(args.scene_digest)
    task_spec = load_json(args.task_spec)
    semantics = load_json(args.object_semantics)
    physics_params = load_json(args.physics_params)
    initial_conditions = load_json(args.initial_conditions)

    semantics_by_name = by_name(semantics)
    physics_by_name = by_name(physics_params)
    initial_by_name = by_name(initial_conditions)

    scene = build_scene_description(digest, task_spec, semantics_by_name)
    plan = build_plan(
        args.goal,
        args.scene_description_output,
        digest,
        task_spec,
        semantics_by_name,
        physics_by_name,
        initial_by_name,
        physics_params,
        initial_conditions,
        args.no_run,
    )

    args.scene_description_output.parent.mkdir(parents=True, exist_ok=True)
    args.plan_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.scene_description_output.write_text(json.dumps(scene, ensure_ascii=False, indent=2), encoding="utf-8")
    args.plan_output.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(args.summary_output, args.goal, scene, plan)
    print(f"Wrote scene description: {args.scene_description_output}")
    print(f"Wrote simulation plan: {args.plan_output}")
    print(f"Wrote summary: {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
