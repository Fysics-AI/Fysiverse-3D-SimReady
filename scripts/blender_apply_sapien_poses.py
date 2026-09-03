#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Matrix


GROUND_PREFIXES = ("__gravity_ground__", "__rigidbody_preview_ground__", "__sapien_ground__")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply SAPIEN final actor poses back to the original Blend scene.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--poses", required=True, type=Path)
    parser.add_argument("--trajectory", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--animation-output", type=Path, default=None)
    parser.add_argument("--pack-textures", action="store_true")
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    return parser.parse_args(argv)


def matrix_from_list(values: list[list[float]]) -> Matrix:
    return Matrix([[float(v) for v in row] for row in values])


def image_is_packed(image: bpy.types.Image) -> bool:
    if getattr(image, "packed_file", None) is not None:
        return True
    packed_files = getattr(image, "packed_files", None)
    return bool(packed_files)


def pack_external_images() -> list[str]:
    packed = []
    for image in bpy.data.images:
        if image.source != "FILE" or not image.filepath or image_is_packed(image):
            continue
        try:
            image.pack()
            packed.append(image.name)
        except Exception as exc:
            print(f"Warning: failed to pack image texture {image.name}: {exc}")
    return packed


def clear_rigid_bodies_and_preview_grounds() -> None:
    for obj in list(bpy.context.scene.objects):
        if obj.rigid_body is not None:
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.rigidbody.object_remove()
    if bpy.context.scene.rigidbody_world is not None:
        bpy.ops.rigidbody.world_remove()
    grounds = [obj for obj in bpy.context.scene.objects if obj.name.startswith(GROUND_PREFIXES)]
    if grounds:
        bpy.ops.object.select_all(action="DESELECT")
        for obj in grounds:
            obj.select_set(True)
        bpy.ops.object.delete()


def apply_pose_delta(obj: bpy.types.Object, initial_pose: Matrix, target_pose: Matrix, base_matrix: Matrix | None = None) -> None:
    delta = target_pose @ initial_pose.inverted()
    obj.matrix_world = delta @ (base_matrix if base_matrix is not None else obj.matrix_world)


def iter_scene_mesh_objects() -> list[bpy.types.Object]:
    return [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and not obj.name.startswith(GROUND_PREFIXES)
    ]


def scene_xy_bounds() -> tuple[float, float, float, float] | None:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    xs: list[float] = []
    ys: list[float] = []
    for obj in iter_scene_mesh_objects():
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = getattr(eval_obj, "data", None)
        if mesh is None:
            continue
        for vertex in mesh.vertices:
            world_co = obj.matrix_world @ vertex.co
            xs.append(float(world_co.x))
            ys.append(float(world_co.y))
    if not xs or not ys:
        return None
    return min(xs), max(xs), min(ys), max(ys)


def make_ground_material() -> bpy.types.Material:
    material = bpy.data.materials.get("__sapien_ground_material__")
    if material is None:
        material = bpy.data.materials.new("__sapien_ground_material__")
        material.diffuse_color = (0.42, 0.42, 0.42, 1.0)
    return material


def create_visible_sapien_ground(ground_z: float = -0.001) -> str | None:
    bounds = scene_xy_bounds()
    if bounds is None:
        return None
    min_x, max_x, min_y, max_y = bounds
    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5
    extent = max(max_x - min_x, max_y - min_y, 1.0)
    half = extent * 0.75 + 0.25

    mesh = bpy.data.meshes.new("__sapien_ground_mesh__")
    mesh.from_pydata(
        [
            (center_x - half, center_y - half, float(ground_z)),
            (center_x + half, center_y - half, float(ground_z)),
            (center_x + half, center_y + half, float(ground_z)),
            (center_x - half, center_y + half, float(ground_z)),
        ],
        [],
        [(0, 1, 2, 3)],
    )
    mesh.update()
    ground = bpy.data.objects.new("__sapien_ground__", mesh)
    bpy.context.collection.objects.link(ground)
    ground.data.materials.append(make_ground_material())
    ground.hide_select = True
    bpy.context.view_layer.update()
    return ground.name


def clear_target_animation(object_names: set[str]) -> None:
    for name in object_names:
        obj = bpy.data.objects.get(name)
        if obj is not None and obj.animation_data is not None:
            obj.animation_data_clear()


def set_linear_keyframe_interpolation(object_names: set[str]) -> None:
    for name in object_names:
        obj = bpy.data.objects.get(name)
        action = getattr(getattr(obj, "animation_data", None), "action", None) if obj is not None else None
        if action is None:
            continue
        for fcurve in action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = "LINEAR"


def save_animation_blend(input_path: Path, trajectory_path: Path, output_path: Path, pack_textures: bool) -> dict:
    bpy.ops.wm.open_mainfile(filepath=str(input_path))
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    clear_rigid_bodies_and_preview_grounds()

    initial_by_name = {
        entry["name"]: matrix_from_list(entry["initial_pose"])
        for entry in trajectory.get("objects", [])
        if entry.get("name") and entry.get("initial_pose")
    }
    original_matrix_by_name = {
        name: obj.matrix_world.copy()
        for name in initial_by_name
        if (obj := bpy.data.objects.get(name)) is not None
    }
    clear_target_animation(set(original_matrix_by_name))
    applied: set[str] = set()
    missing: set[str] = set()
    frames = list(trajectory.get("frames") or [])
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = max([int(frame.get("frame", 0)) for frame in frames] or [0])
    fps = round(1.0 / float(trajectory.get("timestep", 1.0 / 24.0)))
    if fps > 0:
        bpy.context.scene.render.fps = fps

    for frame in frames:
        frame_index = int(frame.get("frame", 0))
        bpy.context.scene.frame_set(frame_index)
        for entry in frame.get("objects", []):
            name = entry.get("name")
            obj = bpy.data.objects.get(name)
            initial_pose = initial_by_name.get(name)
            if obj is None or initial_pose is None:
                if name:
                    missing.add(str(name))
                continue
            target_pose = matrix_from_list(entry["pose"])
            obj.rotation_mode = "QUATERNION"
            base_matrix = original_matrix_by_name.get(name)
            apply_pose_delta(obj, initial_pose, target_pose, base_matrix=base_matrix)
            obj.keyframe_insert(data_path="location", frame=frame_index)
            obj.keyframe_insert(data_path="rotation_quaternion", frame=frame_index)
            obj.keyframe_insert(data_path="scale", frame=frame_index)
            applied.add(str(name))

    set_linear_keyframe_interpolation(applied)
    first_frame = int(bpy.context.scene.frame_start)
    bpy.context.scene.frame_set(first_frame)
    bpy.context.view_layer.update()
    ground_name = create_visible_sapien_ground(float(trajectory.get("ground_z", -0.001)))
    bpy.context.scene.frame_set(first_frame)
    bpy.context.view_layer.update()
    if pack_textures:
        bpy.data.use_autopack = True
        pack_external_images()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path))
    return {
        "animation_output": str(output_path),
        "trajectory": str(trajectory_path),
        "frame_start": int(bpy.context.scene.frame_start),
        "frame_end": int(bpy.context.scene.frame_end),
        "saved_current_frame": int(bpy.context.scene.frame_current),
        "fps": int(bpy.context.scene.render.fps),
        "applied": sorted(applied),
        "missing": sorted(missing),
        "ground": ground_name,
    }


def run(args: argparse.Namespace) -> dict:
    bpy.ops.wm.open_mainfile(filepath=str(args.input))
    pose_data = json.loads(args.poses.read_text(encoding="utf-8"))
    clear_rigid_bodies_and_preview_grounds()

    applied = []
    missing = []
    target_names = {str(entry.get("name")) for entry in pose_data.get("objects", []) if entry.get("name")}
    clear_target_animation(target_names)
    for entry in pose_data.get("objects", []):
        name = entry["name"]
        obj = bpy.data.objects.get(name)
        if obj is None:
            missing.append(name)
            continue
        initial_pose = matrix_from_list(entry["initial_pose"])
        final_pose = matrix_from_list(entry["final_pose"])
        apply_pose_delta(obj, initial_pose, final_pose)
        applied.append(name)

    bpy.context.view_layer.update()
    if args.pack_textures:
        bpy.data.use_autopack = True
        pack_external_images()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output))
    animation_report = None
    if args.trajectory is not None and args.animation_output is not None:
        animation_report = save_animation_blend(args.input, args.trajectory, args.animation_output, args.pack_textures)
    report = {
        "status": "ok",
        "input": str(args.input),
        "poses": str(args.poses),
        "trajectory": str(args.trajectory) if args.trajectory else None,
        "output": str(args.output),
        "animation_output": str(args.animation_output) if args.animation_output else None,
        "applied": applied,
        "missing": missing,
        "ground": None,
        "animation": animation_report,
    }
    return report


def main() -> int:
    report = run(parse_args())
    print(f"Applied SAPIEN poses to {len(report['applied'])} objects")
    if report["missing"]:
        print("Missing objects:", ", ".join(report["missing"]))
    print(f"Wrote Blend: {report['output']}")
    if report.get("animation_output"):
        print(f"Wrote animation Blend: {report['animation_output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
