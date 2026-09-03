#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import bpy
from mathutils import Matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply an optimized camera_to_world matrix to a Blend camera.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--camera-optimization", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--camera-name", default="OriginalInputCamera")
    parser.add_argument("--set-active", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--set-render-resolution", action=argparse.BooleanOptionalAction, default=True)
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    return parser.parse_args(argv)


def matrix4(values: Any) -> Matrix:
    return Matrix([[float(v) for v in row] for row in values]).to_4x4()


def configure_camera_data(cam: bpy.types.Object, report: dict[str, Any], set_render_resolution: bool) -> None:
    intrinsics = report["intrinsics_normalized"]
    image_size = report["source_image_size"]
    fx = float(intrinsics[0][0])
    fy = float(intrinsics[1][1])
    cx = float(intrinsics[0][2])
    cy = float(intrinsics[1][2])
    width, height = int(image_size[0]), int(image_size[1])

    cam.data.type = "PERSP"
    cam.data.sensor_fit = "HORIZONTAL"
    cam.data.sensor_width = 36.0
    cam.data.lens = fx * cam.data.sensor_width
    cam.data.clip_start = 0.001
    cam.data.clip_end = 10000.0
    cam.data.shift_x = 0.5 - cx
    cam.data.shift_y = cy - 0.5
    cam.data["fx_normalized"] = fx
    cam.data["fy_normalized"] = fy
    cam.data["cx_normalized"] = cx
    cam.data["cy_normalized"] = cy
    cam.data["fov_x_degrees"] = math.degrees(2.0 * math.atan(0.5 / max(fx, 1e-12)))
    cam.data["fov_y_degrees"] = math.degrees(2.0 * math.atan(0.5 / max(fy, 1e-12)))

    if set_render_resolution and bpy.context.scene is not None:
        bpy.context.scene.render.resolution_x = width
        bpy.context.scene.render.resolution_y = height
        bpy.context.scene.render.resolution_percentage = 100


def main() -> int:
    args = parse_args()
    output = args.output or args.input
    report = json.loads(args.camera_optimization.read_text(encoding="utf-8"))
    c2w = matrix4(report["optimized_camera_to_world"])

    bpy.ops.wm.open_mainfile(filepath=str(args.input))
    cam_data = bpy.data.cameras.get(args.camera_name)
    if cam_data is None:
        cam_data = bpy.data.cameras.new(args.camera_name)
    cam = bpy.data.objects.get(args.camera_name)
    if cam is None:
        cam = bpy.data.objects.new(args.camera_name, cam_data)
        bpy.context.scene.collection.objects.link(cam)
    else:
        cam.data = cam_data
    cam.matrix_world = c2w
    configure_camera_data(cam, report, bool(args.set_render_resolution))
    cam["camera_optimization_report"] = str(args.camera_optimization)
    cam["camera_optimization_schema"] = str(report.get("schema"))
    if args.set_active:
        bpy.context.scene.camera = cam

    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    print(f"Applied optimized camera {args.camera_name} to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
