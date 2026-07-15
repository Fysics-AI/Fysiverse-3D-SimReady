#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any


try:
    import bpy  # type: ignore
    from mathutils import Matrix, Vector  # type: ignore
except Exception:  # pragma: no cover - exercised only outside Blender.
    bpy = None
    Matrix = None
    Vector = None


DEFAULT_BLENDER = os.environ.get("BLENDER_BIN", "blender")


@dataclass
class StageSpec:
    key: str
    title: str
    source: Path
    description: str
    diagnostic: str
    use_local_animation: bool = False


def split_argv() -> list[str]:
    argv = sys.argv
    if "--" in argv:
        return argv[argv.index("--") + 1 :]
    return argv[1:]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a stage-by-stage diagnostic Blend animation for the SAM3D -> MoGe -> SAPIEN pipeline."
    )
    parser.add_argument("--session", type=Path, required=True, help="Session directory, e.g. sessions/<session_id>.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <session>/results/pipeline_stage_diagnostic.")
    parser.add_argument("--output-blend", type=Path, default=None, help="Default: <output-dir>/pipeline_stage_diagnostic.blend.")
    parser.add_argument("--frames-per-static-stage", type=int, default=48)
    parser.add_argument("--transition-frames", type=int, default=24)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--render-width", type=int, default=1600)
    parser.add_argument("--render-height", type=int, default=1000)
    parser.add_argument("--blender-bin", default=DEFAULT_BLENDER)
    parser.add_argument("--force-linked-gravity", action="store_true", help="Do not append gravity animation locally; use a static linked gravity stage.")
    return parser.parse_args(argv)


def run_in_blender(args: argparse.Namespace) -> int:
    script = Path(__file__).resolve()
    cmd = [
        str(args.blender_bin),
        "-b",
        "--python",
        str(script),
        "--",
        "--session",
        str(args.session),
        "--frames-per-static-stage",
        str(args.frames_per_static_stage),
        "--transition-frames",
        str(args.transition_frames),
        "--fps",
        str(args.fps),
        "--render-width",
        str(args.render_width),
        "--render-height",
        str(args.render_height),
    ]
    if args.output_dir is not None:
        cmd.extend(["--output-dir", str(args.output_dir)])
    if args.output_blend is not None:
        cmd.extend(["--output-blend", str(args.output_blend)])
    if args.force_linked_gravity:
        cmd.append("--force-linked-gravity")
    return subprocess.run(cmd, check=False).returncode


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


def make_stage_specs(session_dir: Path, force_linked_gravity: bool) -> list[StageSpec]:
    result_dir = session_dir / "results"
    return [
        StageSpec(
            "raw",
            "01 SAM3D raw",
            result_dir / "sam3d.blend",
            "Direct SAM3D asset before MoGe upright, camera refinement, scene-graph correction, or physics.",
            "Check object identity, texture, gross geometry, and whether the raw asset already looks wrong.",
        ),
        StageSpec(
            "rotated",
            "02 MoGe upright rotation",
            result_dir / "sam3d_moge_rotated.blend",
            "Estimate the ground normal from MoGe plus the optional GSAM ground mask, then rotate the whole scene to Blender Z-up.",
            "If this stage tilts, the ground normal / coordinate conversion is the likely source.",
        ),
        StageSpec(
            "grounded",
            "03 Grounded to z=0",
            result_dir / "sam3d_moge_grounded.blend",
            "Translate the upright scene onto the z=0 support plane without per-object visual refinement.",
            "If all objects float or sink together, inspect global grounding and support-aware floor selection.",
        ),
        StageSpec(
            "optimized",
            "04 Differentiable rendering refinement",
            result_dir / "sam3d_moge_optimized.blend",
            "Optimize original-input camera extrinsics, then per object translation, yaw around gravity axis, and uniform scale against 2D masks.",
            "If masks align but 3D becomes implausible, inspect object pose optimization constraints and accept criteria.",
        ),
        StageSpec(
            "separated",
            "05 Scene-graph + convex separation",
            result_dir / "sam3d_moge_separated.blend",
            "Use scene-graph support relations plus convex collision checks to correct vertical support and resolve collisions.",
            "If supported objects jump apart or support breaks, inspect scene-graph relations and convex collision proxies.",
        ),
        StageSpec(
            "gravity",
            "06 SAPIEN gravity settling",
            result_dir / "sam3d_moge_separated_gravity_animation.blend",
            "Run rigid-body gravity simulation from the separated scene and keyframe the settling trajectory.",
            "If the final state is unstable or slides away, inspect collision decomposition, friction, damping, and initial penetrations.",
            use_local_animation=not force_linked_gravity,
        ),
    ]


def canonical_object_name(name: str) -> str:
    return re.sub(r"\.\d{3}$", "", str(name))


def is_stage_mesh_object(obj: Any) -> bool:
    return (
        obj is not None
        and obj.type == "MESH"
        and canonical_object_name(obj.name).startswith("mask_")
    )


def clear_scene() -> None:
    assert bpy is not None
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for datablock_group in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.cameras,
        bpy.data.lights,
        bpy.data.curves,
    ):
        for item in list(datablock_group):
            if getattr(item, "users", 0) == 0:
                datablock_group.remove(item)


def make_material(name: str, color: tuple[float, float, float, float], emission: bool = False, strength: float = 1.0):
    assert bpy is not None
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    mat.blend_method = "BLEND"
    mat.show_transparent_back = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    if emission:
        shader = nodes.new(type="ShaderNodeEmission")
        shader.inputs["Color"].default_value = color
        shader.inputs["Strength"].default_value = float(strength)
        output_name = "Emission"
    else:
        shader = nodes.new(type="ShaderNodeBsdfPrincipled")
        shader.inputs["Base Color"].default_value = color
        shader.inputs["Alpha"].default_value = color[3]
        shader.inputs["Roughness"].default_value = 0.82
        output_name = "BSDF"
    output = nodes.new(type="ShaderNodeOutputMaterial")
    mat.node_tree.links.new(shader.outputs[output_name], output.inputs["Surface"])
    return mat


def load_linked_object_stage(stage: StageSpec) -> list[Any]:
    assert bpy is not None
    with bpy.data.libraries.load(str(stage.source), link=True) as (data_from, data_to):
        data_to.objects = [
            name
            for name in data_from.objects
            if not str(name).startswith(("Camera", "Light"))
        ]
    objects = [obj for obj in data_to.objects if obj is not None and obj.type not in {"CAMERA", "LIGHT"}]
    if not objects:
        raise RuntimeError(f"No visible objects found in {stage.source}")
    coll = bpy.data.collections.new(f"stage_{stage.key}_linked_objects")
    bpy.context.scene.collection.children.link(coll)
    for obj in objects:
        try:
            coll.objects.link(obj)
        except RuntimeError:
            pass
    return objects


def link_local_objects_from_blend(stage: StageSpec) -> list[Any]:
    assert bpy is not None
    with bpy.data.libraries.load(str(stage.source), link=False) as (data_from, data_to):
        data_to.objects = list(data_from.objects)
    coll = bpy.data.collections.new(f"stage_{stage.key}_local_animation")
    bpy.context.scene.collection.children.link(coll)
    objects = []
    for obj in data_to.objects:
        if obj is None or obj.type in {"CAMERA", "LIGHT"}:
            continue
        try:
            coll.objects.link(obj)
        except RuntimeError:
            pass
        objects.append(obj)
    if not objects:
        raise RuntimeError(f"No local objects loaded from {stage.source}")
    return objects


def append_base_mesh_objects(source: Path) -> dict[str, Any]:
    assert bpy is not None
    with bpy.data.libraries.load(str(source), link=False) as (data_from, data_to):
        data_to.objects = list(data_from.objects)
    coll = bpy.data.collections.new("diagnostic_single_mesh_set")
    bpy.context.scene.collection.children.link(coll)
    objects: dict[str, Any] = {}
    for obj in data_to.objects:
        if not is_stage_mesh_object(obj):
            if obj is not None:
                bpy.data.objects.remove(obj, do_unlink=True)
            continue
        canon = canonical_object_name(obj.name)
        obj.name = canon
        obj["diagnostic_canonical_name"] = canon
        try:
            coll.objects.link(obj)
        except RuntimeError:
            pass
        objects[canon] = obj
    if not objects:
        raise RuntimeError(f"No mask mesh objects found in base stage: {source}")
    return objects


def cleanup_loaded_objects(objects: list[Any]) -> None:
    assert bpy is not None
    for obj in objects:
        if obj is None:
            continue
        data = getattr(obj, "data", None)
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except Exception:
            pass
        if data is not None and getattr(data, "users", 0) == 0:
            try:
                bpy.data.meshes.remove(data)
            except Exception:
                pass


def load_stage_matrices(source: Path) -> dict[str, Any]:
    assert bpy is not None
    with bpy.data.libraries.load(str(source), link=False) as (data_from, data_to):
        data_to.objects = list(data_from.objects)
    matrices: dict[str, Any] = {}
    loaded = [obj for obj in data_to.objects if obj is not None]
    coll = bpy.data.collections.new(f"__diagnostic_matrix_read_{source.stem}")
    bpy.context.scene.collection.children.link(coll)
    for obj in loaded:
        try:
            coll.objects.link(obj)
        except RuntimeError:
            pass
    bpy.context.view_layer.update()
    for obj in loaded:
        if is_stage_mesh_object(obj):
            matrices[canonical_object_name(obj.name)] = obj.matrix_world.copy()
    cleanup_loaded_objects(loaded)
    try:
        bpy.data.collections.remove(coll)
    except Exception:
        pass
    if not matrices:
        raise RuntimeError(f"No mask object matrices found in {source}")
    return matrices


def apply_matrix_key(obj: Any, matrix: Any, frame: int) -> None:
    loc, rot, scale = matrix.decompose()
    obj.location = loc
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = rot
    obj.scale = scale
    obj.keyframe_insert(data_path="location", frame=int(frame))
    obj.keyframe_insert(data_path="rotation_quaternion", frame=int(frame))
    obj.keyframe_insert(data_path="scale", frame=int(frame))


def key_state(
    base_objects: dict[str, Any],
    matrices: dict[str, Any],
    frame: int,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    applied: dict[str, Any] = {}
    for canon, obj in base_objects.items():
        matrix = matrices.get(canon) or (fallback or {}).get(canon)
        if matrix is None:
            continue
        apply_matrix_key(obj, matrix, frame)
        applied[canon] = matrix
    return applied


def set_transform_interpolation(objects: list[Any], interpolation: str = "LINEAR") -> None:
    for obj in objects:
        action = getattr(getattr(obj, "animation_data", None), "action", None)
        if action is None:
            continue
        for fcurve in action.fcurves:
            if fcurve.data_path not in {"location", "rotation_quaternion", "scale"}:
                continue
            for key in fcurve.keyframe_points:
                key.interpolation = interpolation


def animation_bounds(base_objects: dict[str, Any], state_matrices: list[dict[str, Any]]) -> tuple[Any, Any]:
    assert Vector is not None
    points = []
    for canon, obj in base_objects.items():
        if obj.type != "MESH":
            continue
        corners = [Vector(corner) for corner in obj.bound_box]
        for matrices in state_matrices:
            matrix = matrices.get(canon)
            if matrix is None:
                continue
            for corner in corners:
                points.append(matrix @ corner)
    if not points:
        return Vector((-1.0, -1.0, -1.0)), Vector((1.0, 1.0, 1.0))
    return (
        Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points))),
        Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points))),
    )


def matrix4(values: Any) -> Any:
    assert Matrix is not None
    return Matrix([[float(v) for v in row] for row in values]).to_4x4()


def translation_matrix(values: Any) -> Any:
    assert Matrix is not None and Vector is not None
    vals = values or [0.0, 0.0, 0.0]
    return Matrix.Translation(Vector((float(vals[0]), float(vals[1]), float(vals[2]))))


def stage_report_path(stage_source: Path) -> Path:
    return stage_source.with_name(f"{stage_source.stem}_report.json")


def read_report(path: Path) -> dict[str, Any]:
    data = read_json(path)
    if not data:
        raise FileNotFoundError(f"Missing or invalid report: {path}")
    return data


def raw_object_matrices(base_objects: dict[str, Any]) -> dict[str, Any]:
    return {name: obj.matrix_world.copy() for name, obj in base_objects.items()}


def compose_global(base: dict[str, Any], global_matrix: Any) -> dict[str, Any]:
    return {name: global_matrix @ matrix for name, matrix in base.items()}


def compose_object_deltas(base: dict[str, Any], report_path: Path) -> dict[str, Any]:
    report = read_report(report_path)
    out = {name: matrix.copy() for name, matrix in base.items()}
    for item in report.get("objects") or []:
        name = canonical_object_name(str(item.get("name") or ""))
        if name not in out:
            continue
        if item.get("accepted") is False:
            continue
        if item.get("delta_transform_world") is None:
            continue
        out[name] = matrix4(item["delta_transform_world"]) @ out[name]
    return out


def apply_translation_to_names(matrices: dict[str, Any], names: list[str], translation: Any) -> None:
    if not names:
        return
    trans = translation_matrix(translation)
    for raw_name in names:
        name = canonical_object_name(str(raw_name))
        if name in matrices:
            matrices[name] = trans @ matrices[name]


def compose_separation_deltas(base: dict[str, Any], separated_report_path: Path) -> dict[str, Any]:
    report = read_report(separated_report_path)
    transform = report.get("transform") or {}
    out = {name: matrix.copy() for name, matrix in base.items()}

    support = transform.get("support_adjust_objects") or {}
    for action in support.get("support_actions") or []:
        names = [str(name) for name in action.get("subtree_names") or []]
        if not names and action.get("child_name"):
            names = [str(action["child_name"])]
        apply_translation_to_names(out, names, action.get("translation") or [0.0, 0.0, 0.0])

    for overlap_report in transform.get("bbox_overlap_reports") or []:
        for pair_action in overlap_report.get("pairs") or []:
            pair = pair_action.get("pair") or []
            if len(pair) >= 2:
                apply_translation_to_names(out, [str(pair[0])], pair_action.get("translation_i") or [0.0, 0.0, 0.0])
                apply_translation_to_names(out, [str(pair[1])], pair_action.get("translation_j") or [0.0, 0.0, 0.0])
        for fallback in overlap_report.get("fallback_actions") or []:
            names = [str(name) for name in fallback.get("names") or fallback.get("object_names") or []]
            if not names and fallback.get("name"):
                names = [str(fallback["name"])]
            apply_translation_to_names(out, names, fallback.get("translation") or [0.0, 0.0, 0.0])
    return out


def load_gravity_trajectory(path: Path, separated_matrices: dict[str, Any]) -> tuple[list[tuple[int, dict[str, Any]]], tuple[int, int] | None]:
    report = read_json(path)
    frames = report.get("frames") or []
    objects = report.get("objects") or []
    if not frames or not objects:
        return [], None
    initial_by_name = {
        canonical_object_name(str(item.get("name"))): matrix4(item.get("initial_pose"))
        for item in objects
        if item.get("name") and item.get("initial_pose")
    }
    samples: list[tuple[int, dict[str, Any]]] = []
    for frame in frames:
        frame_index = int(frame.get("frame", 0))
        matrices = {name: matrix.copy() for name, matrix in separated_matrices.items()}
        for entry in frame.get("objects") or []:
            name = canonical_object_name(str(entry.get("name") or ""))
            initial = initial_by_name.get(name)
            base = separated_matrices.get(name)
            if initial is None or base is None or entry.get("pose") is None:
                continue
            target = matrix4(entry["pose"])
            matrices[name] = target @ initial.inverted() @ base
        samples.append((frame_index, matrices))
    return samples, (samples[0][0], samples[-1][0])


def action_frame_range(objects: list[Any]) -> tuple[float, float] | None:
    frames: list[float] = []
    for obj in objects:
        action = getattr(getattr(obj, "animation_data", None), "action", None)
        if action is None:
            continue
        for fcurve in action.fcurves:
            for key in fcurve.keyframe_points:
                frames.append(float(key.co.x))
    if not frames:
        return None
    return min(frames), max(frames)


def object_bounds(objects: list[Any]) -> tuple[Any, Any]:
    assert bpy is not None and Vector is not None
    bpy.context.view_layer.update()
    points = []
    for obj in objects:
        if obj is None or obj.type != "MESH" or not getattr(obj, "bound_box", None):
            continue
        for corner in obj.bound_box:
            points.append(obj.matrix_world @ Vector(corner))
    if not points:
        return Vector((-0.5, -0.5, 0.0)), Vector((0.5, 0.5, 1.0))
    return (
        Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points))),
        Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points))),
    )


def layout_stage_roots(stage_records: list[dict[str, Any]]) -> None:
    assert bpy is not None and Matrix is not None and Vector is not None
    bounds = []
    max_width = 1.0
    for stage in stage_records:
        bmin, bmax = object_bounds(stage["targets"])
        width = max(float(bmax.x - bmin.x), 0.5)
        max_width = max(max_width, width)
        bounds.append((bmin, bmax))
    spacing = max(1.25, max_width * 1.85)
    origin_shift = (len(stage_records) - 1) * spacing * 0.5
    for index, stage in enumerate(stage_records):
        bmin, bmax = bounds[index]
        center = (bmin + bmax) * 0.5
        root = bpy.data.objects.new(f"stage_{stage['key']}_root", None)
        root.empty_display_type = "PLAIN_AXES"
        root.empty_display_size = 0.12
        bpy.context.scene.collection.objects.link(root)
        root.location = Vector((index * spacing - origin_shift - center.x, -center.y, 0.0))
        for obj in stage["targets"]:
            if obj is None:
                continue
            try:
                obj.parent = root
                obj.matrix_parent_inverse = Matrix.Identity(4)
            except Exception:
                pass
        bpy.context.view_layer.update()
        lbmin, lbmax = object_bounds(stage["targets"])
        stage["root"] = root
        stage["layout_bounds_min"] = [float(lbmin.x), float(lbmin.y), float(lbmin.z)]
        stage["layout_bounds_max"] = [float(lbmax.x), float(lbmax.y), float(lbmax.z)]
        stage["column_x"] = float((lbmin.x + lbmax.x) * 0.5)


def offset_object_actions(objects: list[Any], offset: float) -> None:
    for obj in objects:
        action = getattr(getattr(obj, "animation_data", None), "action", None)
        if action is None:
            continue
        action = action.copy()
        obj.animation_data.action = action
        for fcurve in action.fcurves:
            for key in fcurve.keyframe_points:
                key.co.x += offset
                key.handle_left.x += offset
                key.handle_right.x += offset
            fcurve.update()


def set_visibility_keyframes(targets: list[Any], visible_start: int, visible_end: int, scene_start: int, scene_end: int) -> None:
    frames = sorted(
        {
            scene_start,
            max(scene_start, visible_start - 1),
            visible_start,
            visible_end,
            min(scene_end, visible_end + 1),
            scene_end,
        }
    )
    for target in targets:
        for frame in frames:
            visible = visible_start <= frame <= visible_end
            target.hide_viewport = not visible
            target.hide_render = not visible
            target.keyframe_insert(data_path="hide_viewport", frame=frame)
            target.keyframe_insert(data_path="hide_render", frame=frame)


def set_constant_interpolation() -> None:
    assert bpy is not None
    for obj in bpy.data.objects:
        action = getattr(getattr(obj, "animation_data", None), "action", None)
        if action is None:
            continue
        for fcurve in action.fcurves:
            if fcurve.data_path in {"hide_viewport", "hide_render"}:
                for key in fcurve.keyframe_points:
                    key.interpolation = "CONSTANT"


def scene_bounds() -> tuple[Any, Any]:
    assert bpy is not None and Vector is not None
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    points = []
    for inst in depsgraph.object_instances:
        obj = inst.object
        if obj is None or obj.type != "MESH":
            continue
        matrix = inst.matrix_world
        for corner in obj.bound_box:
            points.append(matrix @ Vector(corner))
    if not points:
        return Vector((-1.0, -1.0, -1.0)), Vector((1.0, 1.0, 1.0))
    return (
        Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points))),
        Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points))),
    )


def look_at(obj: Any, target: Any) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_reference_plane(center: Any, extent: float) -> None:
    assert bpy is not None
    mat = make_material("__diagnostic_z0_plane__", (0.18, 0.18, 0.18, 0.24), emission=False)
    size = max(1.0, extent * 1.35)
    bpy.ops.mesh.primitive_plane_add(size=size, location=(float(center.x), float(center.y), 0.0))
    plane = bpy.context.object
    plane.name = "__diagnostic_z0_reference_plane"
    plane.data.materials.append(mat)
    plane.show_transparent = True


def add_stage_texts(stages: list[dict[str, Any]], center: Any, extent: float, camera_rotation: Any) -> list[Any]:
    assert bpy is not None
    text_mat = make_material("__diagnostic_stage_text__", (1.0, 0.96, 0.82, 1.0), emission=True, strength=1.7)
    text_objects = []
    for stage in stages:
        description = textwrap.fill(str(stage["description"]), width=78)
        diagnostic = textwrap.fill("Diagnostic: " + str(stage["diagnostic"]), width=78)
        body = f"{stage['title']}\n{description}\n{diagnostic}"
        bmin = stage.get("layout_bounds_min") or [float(center.x), float(center.y), float(center.z)]
        bmax = stage.get("layout_bounds_max") or [float(center.x), float(center.y), float(center.z)]
        column_x = float(stage.get("column_x", center.x))
        stage_height = max(float(bmax[2]) - float(bmin[2]), 0.5)
        curve = bpy.data.curves.new(f"stage_text_{stage['key']}", type="FONT")
        curve.body = body
        curve.align_x = "CENTER"
        curve.align_y = "CENTER"
        curve.size = max(0.035, min(0.085, stage_height * 0.085))
        curve.materials.append(text_mat)
        obj = bpy.data.objects.new(f"stage_text_{stage['key']}", curve)
        obj.location = (
            column_x,
            float(bmin[1] - max(0.32, extent * 0.035)),
            float(bmax[2] + max(0.18, stage_height * 0.24)),
        )
        obj.rotation_euler = camera_rotation
        bpy.context.scene.collection.objects.link(obj)
        stage["text_targets"] = [obj]
        text_objects.append(obj)
    return text_objects


def add_timeline_stage_texts(stages: list[dict[str, Any]], center: Any, extent: float, camera_rotation: Any) -> list[Any]:
    assert bpy is not None
    assert Vector is not None
    text_mat = make_material("__diagnostic_stage_text__", (1.0, 0.96, 0.82, 1.0), emission=True, strength=1.7)
    cam = bpy.context.scene.camera
    view_height = max(0.5, extent)
    text_location = center.copy()
    if cam is not None:
        center_local = cam.matrix_world.inverted() @ center
        if cam.data.type == "ORTHO":
            frame = cam.data.view_frame(scene=bpy.context.scene)
            xs = [float(corner.x) for corner in frame]
            ys = [float(corner.y) for corner in frame]
            z = float(center_local.z)
        else:
            frame = cam.data.view_frame(scene=bpy.context.scene)
            depth = max(0.1, -float(center_local.z))
            z = -depth * 0.82
            scale = abs(z / float(frame[0].z)) if abs(float(frame[0].z)) > 1e-8 else 1.0
            xs = [float(corner.x) * scale for corner in frame]
            ys = [float(corner.y) * scale for corner in frame]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        view_height = max(max_y - min_y, 0.5)
        local = Vector(((min_x + max_x) * 0.5, min_y + view_height * 0.14, z))
        text_location = cam.matrix_world @ local
    text_objects = []
    for stage in stages:
        description = textwrap.fill(str(stage["description"]), width=78)
        diagnostic = textwrap.fill("Diagnostic: " + str(stage["diagnostic"]), width=78)
        body = f"{stage['title']}\n{description}\n{diagnostic}"
        curve = bpy.data.curves.new(f"stage_text_{stage['key']}", type="FONT")
        curve.body = body
        curve.align_x = "CENTER"
        curve.align_y = "CENTER"
        curve.size = max(0.035, min(0.09, view_height * 0.04))
        curve.materials.append(text_mat)
        obj = bpy.data.objects.new(f"stage_text_{stage['key']}", curve)
        obj.location = text_location
        obj.rotation_euler = camera_rotation
        bpy.context.scene.collection.objects.link(obj)
        stage["text_targets"] = [obj]
        set_visibility_keyframes([obj], int(stage["frame_start"]), int(stage["frame_end"]), int(stages[0]["frame_start"]), int(stages[-1]["frame_end"]))
        text_objects.append(obj)
    return text_objects


def configure_scene(output_blend: Path, fps: int, width: int, height: int, frame_start: int, frame_end: int) -> None:
    assert bpy is not None
    scene = bpy.context.scene
    scene.frame_start = int(frame_start)
    scene.frame_end = int(frame_end)
    scene.frame_set(int(frame_start))
    scene.render.fps = int(fps)
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.eevee.taa_render_samples = 64
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.render.filepath = str(output_blend.with_suffix(".mp4"))


def add_camera_and_light() -> tuple[Any, Any, float]:
    assert bpy is not None and Vector is not None
    bmin, bmax = scene_bounds()
    center = (bmin + bmax) * 0.5
    extent = max(float(bmax.x - bmin.x), float(bmax.y - bmin.y), float(bmax.z - bmin.z), 1.0)
    cam_data = bpy.data.cameras.new("DiagnosticCamera")
    cam = bpy.data.objects.new("DiagnosticCamera", cam_data)
    cam.location = center + Vector((0.0, -extent * 2.25, extent * 0.95))
    look_at(cam, center + Vector((0.0, 0.0, extent * 0.08)))
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = extent * 1.45
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    light_data = bpy.data.lights.new("DiagnosticKeyLight", type="AREA")
    light_data.energy = 550.0
    light_data.size = extent * 1.25
    light = bpy.data.objects.new("DiagnosticKeyLight", light_data)
    light.location = center + Vector((-extent * 0.7, -extent * 1.0, extent * 1.7))
    bpy.context.scene.collection.objects.link(light)
    return center, cam.rotation_euler.copy(), extent


def add_camera_and_light_for_bounds(bmin: Any, bmax: Any) -> tuple[Any, Any, float]:
    assert bpy is not None and Vector is not None
    center = (bmin + bmax) * 0.5
    extent = max(float(bmax.x - bmin.x), float(bmax.y - bmin.y), float(bmax.z - bmin.z), 1.0)
    cam_data = bpy.data.cameras.new("DiagnosticCamera")
    cam = bpy.data.objects.new("DiagnosticCamera", cam_data)
    cam.location = center + Vector((0.0, -extent * 2.25, extent * 0.95))
    look_at(cam, center + Vector((0.0, 0.0, extent * 0.08)))
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = extent * 1.55
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    light_data = bpy.data.lights.new("DiagnosticKeyLight", type="AREA")
    light_data.energy = 650.0
    light_data.size = extent * 1.25
    light = bpy.data.objects.new("DiagnosticKeyLight", light_data)
    light.location = center + Vector((-extent * 0.7, -extent * 1.0, extent * 1.7))
    bpy.context.scene.collection.objects.link(light)
    return center, cam.rotation_euler.copy(), extent


def original_camera_json_path(result_dir: Path) -> Path | None:
    for name in (
        "sam3d_moge_separated_original_input_camera.json",
        "sam3d_moge_optimized_original_input_camera.json",
        "sam3d_moge_grounded_original_input_camera.json",
        "sam3d_moge_rotated_original_input_camera.json",
    ):
        path = result_dir / name
        if path.is_file():
            return path
    return None


def configure_camera_data_from_payload(cam: Any, payload: dict[str, Any]) -> None:
    settings = payload.get("blender_camera") or {}
    cam.data.type = "PERSP"
    cam.data.sensor_fit = str(settings.get("sensor_fit") or "HORIZONTAL")
    cam.data.sensor_width = float(settings.get("sensor_width") or 36.0)
    if settings.get("lens") is not None:
        cam.data.lens = float(settings["lens"])
    else:
        intr = payload.get("intrinsics_normalized") or []
        if intr and intr[0][0] is not None:
            cam.data.lens = float(intr[0][0]) * cam.data.sensor_width
    cam.data.shift_x = float(settings.get("shift_x") or 0.0)
    cam.data.shift_y = float(settings.get("shift_y") or 0.0)
    cam.data.clip_start = float(settings.get("clip_start") or 0.001)
    cam.data.clip_end = max(float(settings.get("clip_end") or 10000.0), 10000.0)


def add_original_view_camera_and_light_for_bounds(
    *,
    camera_json: Path,
    bmin: Any,
    bmax: Any,
) -> tuple[Any, Any, float, dict[str, Any]]:
    assert bpy is not None and Vector is not None
    payload = read_report(camera_json)
    camera_to_world = matrix4(payload.get("camera_to_world"))
    center = (bmin + bmax) * 0.5
    extent = max(float(bmax.x - bmin.x), float(bmax.y - bmin.y), float(bmax.z - bmin.z), 1.0)

    cam_data = bpy.data.cameras.new("DiagnosticCamera")
    cam = bpy.data.objects.new("DiagnosticCamera", cam_data)
    cam.matrix_world = camera_to_world
    configure_camera_data_from_payload(cam, payload)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    image_size = payload.get("image_size") or []
    if len(image_size) >= 2:
        bpy.context.scene.render.resolution_x = int(image_size[0])
        bpy.context.scene.render.resolution_y = int(image_size[1])

    cam_quat = cam.rotation_euler.to_quaternion()
    cam_right = cam_quat @ Vector((1.0, 0.0, 0.0))
    cam_up = cam_quat @ Vector((0.0, 1.0, 0.0))
    cam_back = cam_quat @ Vector((0.0, 0.0, 1.0))

    key_data = bpy.data.lights.new("DiagnosticKeyLight", type="AREA")
    key_data.energy = 650.0
    key_data.size = max(1.0, extent * 1.2)
    key = bpy.data.objects.new("DiagnosticKeyLight", key_data)
    key.location = center + cam_back * (0.8 * extent) + cam_right * (0.55 * extent) + cam_up * (0.75 * extent)
    look_at(key, center)
    bpy.context.scene.collection.objects.link(key)

    fill_data = bpy.data.lights.new("DiagnosticFillLight", type="AREA")
    fill_data.energy = 220.0
    fill_data.size = max(1.0, extent * 1.8)
    fill = bpy.data.objects.new("DiagnosticFillLight", fill_data)
    fill.location = center + cam_back * (0.6 * extent) - cam_right * (0.75 * extent) + cam_up * (0.25 * extent)
    look_at(fill, center)
    bpy.context.scene.collection.objects.link(fill)
    return center, cam.rotation_euler.copy(), extent, payload


def sample_gravity_animation(source: Path) -> tuple[list[tuple[int, dict[str, Any]]], tuple[int, int] | None]:
    assert bpy is not None
    stage = StageSpec(
        "gravity_sample",
        "gravity sample",
        source,
        "",
        "",
        use_local_animation=True,
    )
    objects = link_local_objects_from_blend(stage)
    frame_range = action_frame_range(objects)
    if frame_range is None:
        cleanup_loaded_objects(objects)
        return [], None
    source_start = int(round(frame_range[0]))
    source_end = int(round(frame_range[1]))
    samples: list[tuple[int, dict[str, Any]]] = []
    for frame in range(source_start, source_end + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        matrices = {
            canonical_object_name(obj.name): obj.matrix_world.copy()
            for obj in objects
            if is_stage_mesh_object(obj)
        }
        samples.append((frame, matrices))
    cleanup_loaded_objects(objects)
    return samples, (source_start, source_end)


def write_notes_md(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Pipeline Stage Diagnostic",
        "",
        f"Session: `{manifest['session_id']}`",
        f"Blend: `{manifest['output_blend']}`",
        "",
        "这个诊断 Blend 用于从 SAM3D 原始生成结果一路检查到重力稳定结果。打开后直接播放时间轴：同一套物体会按各阶段保存的 transform 做线性插值，最后接上 SAPIEN 重力稳定关键帧动画。",
        "",
    ]
    for stage in manifest["stages"]:
        lines.extend(
            [
                f"## {stage['title']}",
                "",
                f"- Frames: `{stage['frame_start']}-{stage['frame_end']}`",
                f"- Source: `{stage['source_blend']}`",
                f"- What it does: {stage['description']}",
                f"- What to inspect: {stage['diagnostic']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_diagnostic_blend(args: argparse.Namespace) -> dict[str, Any]:
    assert bpy is not None
    session_dir = args.session.resolve()
    result_dir = session_dir / "results"
    output_dir = (args.output_dir.resolve() if args.output_dir else result_dir / "pipeline_stage_diagnostic")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_blend = args.output_blend.resolve() if args.output_blend else output_dir / "pipeline_stage_diagnostic.blend"

    stages = [stage for stage in make_stage_specs(session_dir, args.force_linked_gravity) if stage.source.is_file()]
    if not stages:
        raise FileNotFoundError(f"No stage Blend files found under {result_dir}")

    clear_scene()
    stage_by_key = {stage.key: stage for stage in stages}
    required_keys = ["raw", "rotated", "grounded", "optimized", "separated"]
    missing_keys = [key for key in required_keys if key not in stage_by_key]
    if missing_keys:
        raise FileNotFoundError(f"Missing required stage Blend files: {missing_keys}")

    base_objects = append_base_mesh_objects(stage_by_key["raw"].source)

    matrices_by_key = {
        key: load_stage_matrices(stage_by_key[key].source)
        for key in required_keys
    }

    scene_start = 1
    segment = max(2, int(args.frames_per_static_stage))
    frame_cursor = scene_start
    stage_records: list[dict[str, Any]] = []
    key_state(base_objects, matrices_by_key["raw"], frame_cursor)
    raw_end = frame_cursor + segment - 1
    key_state(base_objects, matrices_by_key["raw"], raw_end)
    raw_stage = stage_by_key["raw"]
    stage_records.append(
        {
            "key": "raw",
            "title": raw_stage.title,
            "description": raw_stage.description,
            "diagnostic": raw_stage.diagnostic,
            "source_blend": str(raw_stage.source),
            "frame_start": int(frame_cursor),
            "frame_end": int(raw_end),
            "transition": "hold raw SAM3D result",
            "local_animation": False,
        }
    )

    transitions = [
        ("rotated", "raw", "rotated"),
        ("grounded", "rotated", "grounded"),
        ("optimized", "grounded", "optimized"),
        ("separated", "optimized", "separated"),
    ]
    last_state = matrices_by_key["raw"]
    frame_cursor = raw_end + 1
    for stage_key, start_key, end_key in transitions:
        stage = stage_by_key[stage_key]
        frame_start = frame_cursor
        frame_end = frame_start + segment - 1
        key_state(base_objects, matrices_by_key[start_key], frame_start, fallback=last_state)
        last_state = key_state(base_objects, matrices_by_key[end_key], frame_end, fallback=last_state)
        stage_records.append(
            {
                "key": stage.key,
                "title": stage.title,
                "description": stage.description,
                "diagnostic": stage.diagnostic,
                "source_blend": str(stage.source),
                "frame_start": int(frame_start),
                "frame_end": int(frame_end),
                "transition": f"{start_key} -> {end_key}",
                "local_animation": False,
            }
        )
        frame_cursor = frame_end + 1

    gravity_stage = stage_by_key.get("gravity")
    gravity_samples: list[tuple[int, dict[str, Any]]] = []
    gravity_range: tuple[int, int] | None = None
    if gravity_stage is not None and gravity_stage.source.is_file() and not args.force_linked_gravity:
        gravity_samples, gravity_range = sample_gravity_animation(gravity_stage.source)
        if not gravity_samples or gravity_range is None:
            gravity_samples, gravity_range = load_gravity_trajectory(
                result_dir / "sam3d_moge_separated_sapien_gravity" / "pose_trajectory.json",
                matrices_by_key["separated"],
            )

    if gravity_stage is not None:
        gravity_start = frame_cursor
        if gravity_samples and gravity_range is not None:
            source_start, source_end = gravity_range
            for source_frame, matrices in gravity_samples:
                target_frame = gravity_start + int(source_frame - source_start)
                key_state(base_objects, matrices, target_frame, fallback=last_state)
                last_state = matrices or last_state
            gravity_end = gravity_start + int(source_end - source_start)
            gravity_transition = f"SAPIEN keyframes {source_start}->{source_end}"
        else:
            gravity_end = gravity_start + segment - 1
            key_state(base_objects, matrices_by_key["separated"], gravity_start, fallback=last_state)
            key_state(base_objects, matrices_by_key["separated"], gravity_end, fallback=last_state)
            gravity_transition = "static separated fallback"
        stage_records.append(
            {
                "key": gravity_stage.key,
                "title": gravity_stage.title,
                "description": gravity_stage.description,
                "diagnostic": gravity_stage.diagnostic,
                "source_blend": str(gravity_stage.source),
                "frame_start": int(gravity_start),
                "frame_end": int(gravity_end),
                "transition": gravity_transition,
                "local_animation": bool(gravity_samples),
            }
        )
        scene_end = int(gravity_end)
    else:
        scene_end = int(stage_records[-1]["frame_end"])

    set_transform_interpolation(list(base_objects.values()), interpolation="LINEAR")
    bmin, bmax = animation_bounds(
        base_objects,
        list(matrices_by_key.values()) + [matrices for _, matrices in gravity_samples[:: max(1, len(gravity_samples) // 20 or 1)]],
    )
    camera_source = "bounds_fallback"
    camera_json = original_camera_json_path(result_dir)
    camera_payload: dict[str, Any] | None = None
    camera_error: str | None = None
    if camera_json is not None:
        try:
            center, camera_rotation, extent, camera_payload = add_original_view_camera_and_light_for_bounds(
                camera_json=camera_json,
                bmin=bmin,
                bmax=bmax,
            )
            camera_source = "original_input_camera"
        except Exception as exc:
            camera_error = str(exc)
            center, camera_rotation, extent = add_camera_and_light_for_bounds(bmin, bmax)
    else:
        center, camera_rotation, extent = add_camera_and_light_for_bounds(bmin, bmax)

    add_reference_plane(center, extent)
    add_timeline_stage_texts(stage_records, center, extent, camera_rotation)
    set_constant_interpolation()
    render_width = int(args.render_width)
    render_height = int(args.render_height)
    if camera_payload is not None:
        image_size = camera_payload.get("image_size") or []
        if len(image_size) >= 2:
            render_width = int(image_size[0])
            render_height = int(image_size[1])
    configure_scene(output_blend, args.fps, render_width, render_height, scene_start, scene_end)

    always_visible_names = {
        "DiagnosticCamera",
        "DiagnosticKeyLight",
        "DiagnosticFillLight",
        "__diagnostic_z0_reference_plane",
        *sorted(base_objects),
    }

    bpy.context.scene.frame_set(scene_start)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_blend))

    manifest = {
        "schema": "fysiverse_pipeline_stage_diagnostic.v1",
        "session_id": session_dir.name,
        "session_dir": str(session_dir),
        "output_blend": str(output_blend),
        "frame_start": int(scene_start),
        "frame_end": int(scene_end),
        "fps": int(args.fps),
        "notes": str(output_dir / "stage_notes.md"),
        "stages": [
            {k: v for k, v in stage.items() if k not in {"targets", "text_targets", "root"}}
            for stage in stage_records
        ],
        "implementation": {
            "layout": "A single shared mesh/object set is animated through the pipeline. Static stage changes are shown by linear interpolation of object transforms between saved stage states.",
            "state_source": "The animation uses one shared raw mesh/object set. Per-stage transforms are sampled from each saved pipeline Blend output so the diagnostic animation matches the actual artifacts.",
            "gravity_stage": "The final SAPIEN gravity stage samples sam3d_moge_separated_gravity_animation.blend directly, with pose_trajectory.json as a fallback.",
            "always_visible": sorted(always_visible_names),
            "camera": {
                "source": camera_source,
                "camera_json": str(camera_json) if camera_json is not None else None,
                "camera_source": camera_payload.get("camera_source") if camera_payload else None,
                "camera_name": "DiagnosticCamera",
                "render_resolution": [render_width, render_height],
                "fallback_error": camera_error,
            },
        },
    }
    manifest_path = output_dir / "stage_diagnostic_manifest.json"
    write_json(manifest_path, manifest)
    write_notes_md(output_dir / "stage_notes.md", manifest)
    print(json.dumps({"output_blend": str(output_blend), "manifest": str(manifest_path)}, ensure_ascii=False, indent=2), flush=True)
    return manifest


def main() -> int:
    args = parse_args(split_argv())
    if bpy is None:
        return run_in_blender(args)
    build_diagnostic_blend(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
