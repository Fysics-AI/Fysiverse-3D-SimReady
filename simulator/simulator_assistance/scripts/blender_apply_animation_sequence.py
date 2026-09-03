#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a simulator animation sequence to an original Blend scene as keyframes.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--sequence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=24)
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


def clear_rigid_bodies() -> None:
    for obj in list(bpy.context.scene.objects):
        if obj.rigid_body is None:
            continue
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.rigidbody.object_remove()
    if bpy.context.scene.rigidbody_world is not None:
        bpy.ops.rigidbody.world_remove()


def run(args: argparse.Namespace) -> dict:
    bpy.ops.wm.open_mainfile(filepath=str(args.input))
    sequence = json.loads(args.sequence.read_text(encoding="utf-8"))
    clear_rigid_bodies()
    bpy.context.scene.render.fps = int(args.fps)

    initial_pose_by_name = {
        entry["name"]: matrix_from_list(entry["initial_pose"])
        for entry in sequence.get("objects", [])
        if entry.get("name") and entry.get("initial_pose")
    }
    original_world_by_name = {}
    for name in initial_pose_by_name:
        obj = bpy.data.objects.get(name)
        if obj is not None:
            original_world_by_name[name] = obj.matrix_world.copy()

    missing = set()
    inserted = 0
    last_frame_number = 1
    for frame_entry in sequence.get("frames", []):
        frame_number = int(frame_entry.get("frame", 0)) + 1
        last_frame_number = max(last_frame_number, frame_number)
        bpy.context.scene.frame_set(frame_number)
        for object_pose in frame_entry.get("objects", []):
            name = object_pose.get("name")
            obj = bpy.data.objects.get(name)
            initial_pose = initial_pose_by_name.get(name)
            original_world = original_world_by_name.get(name)
            if obj is None or initial_pose is None or original_world is None:
                if name:
                    missing.add(str(name))
                continue
            frame_pose = matrix_from_list(object_pose["pose"])
            delta = frame_pose @ initial_pose.inverted()
            obj.matrix_world = delta @ original_world
            obj.keyframe_insert(data_path="location", frame=frame_number)
            if obj.rotation_mode != "QUATERNION":
                obj.rotation_mode = "QUATERNION"
            obj.keyframe_insert(data_path="rotation_quaternion", frame=frame_number)
            obj.keyframe_insert(data_path="scale", frame=frame_number)
            inserted += 1

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = last_frame_number
    bpy.context.view_layer.update()
    if args.pack_textures:
        bpy.data.use_autopack = True
        pack_external_images()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output))
    return {
        "status": "ok",
        "input": str(args.input),
        "sequence": str(args.sequence),
        "output": str(args.output),
        "inserted_keyframes": inserted,
        "missing": sorted(missing),
        "frame_start": 1,
        "frame_end": last_frame_number,
    }


def main() -> int:
    report = run(parse_args())
    print(f"Inserted keyframes: {report['inserted_keyframes']}")
    if report["missing"]:
        print("Missing objects:", ", ".join(report["missing"]))
    print(f"Wrote Blend animation: {report['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
