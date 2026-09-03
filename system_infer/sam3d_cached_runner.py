#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import trimesh
from omegaconf import OmegaConf


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "SAM3D"
DEFAULT_CONFIG = DEFAULT_ROOT / "runtime_configs" / "pipeline_hf_cache_moge_modelpt.yaml"
DEFAULT_CONDA_PREFIX = Path(os.environ.get("CONDA_PREFIX", sys.prefix))
DEFAULT_MPLCONFIGDIR = Path("/tmp/matplotlib-sam3d")

ROOT = Path(os.environ.get("SAM3D_ROOT", str(DEFAULT_ROOT)))
SAM3D_OBJECTS_ROOT = ROOT / "sam-3d-objects"
NOTEBOOK_DIR = SAM3D_OBJECTS_ROOT / "notebook"

if "SAM3D_GPU" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SAM3D_GPU"]
if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = str(DEFAULT_CONDA_PREFIX)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPLCONFIGDIR))
os.environ.setdefault("LIDRA_SKIP_INIT", "true")

if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))

from inference import Inference, load_image, load_masks  # noqa: E402
from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform  # noqa: E402


@dataclass(frozen=True)
class SceneResult:
    scene_id: str
    status: str
    output_path: str | None = None
    pointmap_cache: str | None = None
    manifest_path: str | None = None
    num_masks: int = 0
    num_meshes: int = 0
    objects: list[dict[str, Any]] | None = None
    error: str | None = None
    seconds: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM3D with a single reusable MoGe pointmap cache.")
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--scene_ids", nargs="*", default=None)
    parser.add_argument("--rank", type=int, default=int(os.environ.get("SAM3D_RANK", "0")))
    parser.add_argument("--world_size", type=int, default=int(os.environ.get("SAM3D_WORLD_SIZE", "1")))
    parser.add_argument("--gpu", default=os.environ.get("SAM3D_GPU", None))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--save_report", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail_fast", action="store_true")
    return parser.parse_args()


def discover_scenes(args: argparse.Namespace) -> list[Path]:
    input_root = Path(args.input_root)
    if args.scene_ids:
        scenes = [Path(s) if Path(s).is_absolute() else input_root / s for s in args.scene_ids]
    else:
        scenes = sorted(p for p in input_root.iterdir() if p.is_dir())
    scenes = [p for p in scenes if p.is_dir()]
    return sorted(scenes, key=lambda p: p.name)


def shard_scenes(scenes: list[Path], rank: int, world_size: int) -> list[Path]:
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    return [scene for i, scene in enumerate(scenes) if i % world_size == rank]


def make_runtime_config(config_path: Path, output_dir: Path) -> Path:
    cfg = OmegaConf.load(config_path)
    workspace_dir = config_path.parent
    for key, value in list(cfg.items()):
        if key.endswith("_path") and isinstance(value, str) and value and not os.path.isabs(value):
            cfg[key] = str(workspace_dir / value)
    runtime_dir = output_dir / "_runtime_configs"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = runtime_dir / f"{config_path.stem}_cached.yaml"
    OmegaConf.save(cfg, runtime_path)
    return runtime_path


def build_full_alpha_rgba(image_rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image_rgb, dtype=np.uint8)
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    alpha = np.full((*rgb.shape[:2], 1), 255, dtype=np.uint8)
    return np.concatenate([rgb, alpha], axis=-1)


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert real-first quaternions to rotation matrices.

    SAM3D returns rotations in the same scalar-first convention used by
    PyTorch3D. Keeping the small conversion local avoids making PyTorch3D a
    hard dependency of the open-source wrapper.
    """
    if quaternions.shape[-1] != 4:
        raise ValueError(f"Expected quaternion shape (..., 4), got {tuple(quaternions.shape)}")
    r, i, j, k = torch.unbind(quaternions, dim=-1)
    eps = torch.finfo(quaternions.dtype).eps if quaternions.is_floating_point() else 1e-12
    two_s = 2.0 / torch.clamp((quaternions * quaternions).sum(dim=-1), min=eps)
    matrix = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        dim=-1,
    )
    return matrix.reshape(quaternions.shape[:-1] + (3, 3))


def compute_full_scene_pointmap(inference: Inference, image_rgb: np.ndarray) -> dict[str, torch.Tensor]:
    rgba = build_full_alpha_rgba(image_rgb)
    pipeline = inference._pipeline
    pointmap_dict = pipeline.compute_pointmap(rgba, pointmap=None)
    pointmap_chw = pointmap_dict["pointmap"].detach()
    if pointmap_chw.ndim != 3 or pointmap_chw.shape[0] != 3:
        raise ValueError(f"unexpected pointmap shape: {tuple(pointmap_chw.shape)}")
    pointmap_hwc = pointmap_chw.permute(1, 2, 0).contiguous()
    return {
        "pointmap": pointmap_hwc,
        "intrinsics": pointmap_dict["intrinsics"].detach(),
        "pts_color": pointmap_dict["pts_color"].detach(),
    }


def save_moge_cache(cache: dict[str, torch.Tensor], image_rgb: np.ndarray, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pointmap = cache["pointmap"].detach().cpu().float().numpy()
    intrinsics = cache["intrinsics"].detach().cpu().float().numpy()
    pts_color = cache["pts_color"].detach().cpu().float().numpy()
    if pts_color.ndim == 3 and pts_color.shape[0] == 3:
        pts_color = pts_color.transpose(1, 2, 0)
    valid = np.isfinite(pointmap).all(axis=-1)
    depth = pointmap[..., 2]
    np.savez_compressed(
        output_path,
        points=pointmap.astype(np.float32),
        pointmap=pointmap.astype(np.float32),
        pcd=pointmap.astype(np.float32),
        depth=depth.astype(np.float32),
        mask=valid.astype(np.uint8),
        intrinsics=intrinsics.astype(np.float32),
        image=np.asarray(image_rgb, dtype=np.uint8),
        pts_color=np.asarray(pts_color, dtype=np.float32),
        coordinate_system=np.array("pytorch3d_scene_y_up"),
        source=np.array("sam3d_cached_runner.compute_pointmap"),
    )
    return output_path


def build_mesh_from_output(output: dict) -> trimesh.Trimesh | None:
    mesh_list = output.get("mesh", None)
    if mesh_list is None or len(mesh_list) == 0:
        return None
    raw_mesh = mesh_list[0]
    if not getattr(raw_mesh, "success", True):
        return None
    vertices = raw_mesh.vertices.detach().cpu().numpy()
    faces = raw_mesh.faces.detach().cpu().numpy()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    vertex_attrs = getattr(raw_mesh, "vertex_attrs", None)
    if vertex_attrs is not None:
        vertex_colors = vertex_attrs[:, :3].detach().cpu().numpy()
        if vertex_colors.dtype != np.uint8:
            vertex_colors = np.clip(vertex_colors, 0.0, 1.0)
            vertex_colors = (vertex_colors * 255).astype(np.uint8)
        mesh.visual.vertex_colors = vertex_colors
    return mesh


def transform_mesh_to_scene(mesh: trimesh.Trimesh, output: dict) -> trimesh.Trimesh:
    vertices = torch.from_numpy(np.asarray(mesh.vertices)).float().to(output["rotation"].device)
    points_local = vertices.unsqueeze(0)
    rotation_l2c = quaternion_to_matrix(output["rotation"])
    l2c_transform = compose_transform(
        scale=output["scale"],
        rotation=rotation_l2c,
        translation=output["translation"],
    )
    points_scene = l2c_transform.transform_points(points_local)[0].detach().cpu().numpy()
    transformed = mesh.copy()
    transformed.vertices = points_scene
    return transformed


def safe_object_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value)).strip("._")
    return cleaned or "object"


def load_mask_manifest(scene_dir: Path, num_masks: int) -> tuple[Path | None, list[dict[str, Any]]]:
    candidates = [scene_dir / "mask_manifest.json", scene_dir / "final_scene_manifest.json"]
    manifest_path = next((path for path in candidates if path.is_file()), None)
    if manifest_path is None:
        objects = [
            {
                "object_index": idx,
                "sam3d_input_index": idx,
                "mask_id": idx + 1,
                "mask_name": f"mask_{idx + 1:03d}",
            }
            for idx in range(num_masks)
        ]
        return None, objects

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_objects = list(manifest.get("objects") or [])
    by_index = {int(obj.get("sam3d_input_index", obj.get("object_index", -1))): obj for obj in raw_objects}
    objects: list[dict[str, Any]] = []
    for idx in range(num_masks):
        obj = dict(by_index.get(idx) or {})
        obj.setdefault("object_index", idx)
        obj.setdefault("sam3d_input_index", idx)
        obj.setdefault("mask_id", idx + 1)
        obj.setdefault("mask_name", f"mask_{int(obj['mask_id']):03d}")
        objects.append(obj)
    return manifest_path, objects


def export_scene_glb(entries: Iterable[tuple[trimesh.Trimesh, dict[str, Any]]], output_path: Path) -> int:
    entries = list(entries)
    if not entries:
        return 0
    scene = trimesh.Scene()
    used_names: set[str] = set()
    for mesh_idx, (mesh, meta) in enumerate(entries):
        mask_id = int(meta.get("mask_id", mesh_idx + 1))
        base_name = safe_object_name(str(meta.get("object_name") or f"mask_{mask_id:03d}_object"))
        name = base_name
        suffix = 1
        while name in used_names:
            name = f"{base_name}_{suffix:02d}"
            suffix += 1
        used_names.add(name)
        meta["object_name"] = name
        scene.add_geometry(mesh, geom_name=name, node_name=name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    scene.export(tmp_path, file_type="glb")
    tmp_path.replace(output_path)
    return len(entries)


def process_scene(
    scene_dir: Path,
    output_dir: Path,
    inference: Inference,
    seed: int,
    overwrite: bool,
) -> SceneResult:
    start = time.time()
    scene_id = scene_dir.name
    output_path = output_dir / f"{scene_id}.glb"
    cache_path = output_dir / f"{scene_id}_sam3d_moge_cache.npz"
    image_path = scene_dir / "image.png"
    if not image_path.is_file():
        return SceneResult(scene_id=scene_id, status="failed", error="missing image.png", seconds=time.time() - start)
    image = load_image(str(image_path))
    masks = load_masks(str(scene_dir), extension=".png")
    if not masks:
        return SceneResult(scene_id=scene_id, status="failed", error="no masks", seconds=time.time() - start)
    manifest_path, manifest_objects = load_mask_manifest(scene_dir, len(masks))

    if output_path.is_file() and cache_path.is_file() and not overwrite:
        return SceneResult(
            scene_id=scene_id,
            status="skipped",
            output_path=str(output_path),
            pointmap_cache=str(cache_path),
            manifest_path=str(manifest_path) if manifest_path else None,
            num_masks=len(masks),
            num_meshes=0,
            objects=manifest_objects,
            seconds=time.time() - start,
        )

    cache = compute_full_scene_pointmap(inference, image)
    save_moge_cache(cache, image, cache_path)
    cached_pointmap = cache["pointmap"].detach().cpu()

    scene_entries: list[tuple[trimesh.Trimesh, dict[str, Any]]] = []
    object_reports: list[dict[str, Any]] = []
    for mask_idx, mask in enumerate(masks):
        object_meta = dict(manifest_objects[mask_idx] if mask_idx < len(manifest_objects) else {})
        object_meta.setdefault("object_index", mask_idx)
        object_meta.setdefault("sam3d_input_index", mask_idx)
        object_meta.setdefault("mask_id", mask_idx + 1)
        object_meta.setdefault("mask_name", f"mask_{int(object_meta['mask_id']):03d}")
        object_meta["status"] = "failed"
        output = inference(image, mask, seed=seed, pointmap=cached_pointmap)
        mesh = build_mesh_from_output(output)
        if mesh is None:
            print(f"[WARN][{scene_id}] mask={mask_idx}: no mesh generated", flush=True)
            object_meta["error"] = "no mesh generated"
            object_reports.append(object_meta)
            continue
        object_meta["status"] = "ok"
        object_meta["object_name"] = f"mask_{int(object_meta['mask_id']):03d}_object"
        scene_entries.append((transform_mesh_to_scene(mesh, output), object_meta))
        object_reports.append(object_meta)

    num_meshes = export_scene_glb(scene_entries, output_path)
    if num_meshes == 0:
        return SceneResult(
            scene_id=scene_id,
            status="failed",
            pointmap_cache=str(cache_path),
            manifest_path=str(manifest_path) if manifest_path else None,
            num_masks=len(masks),
            num_meshes=0,
            objects=object_reports,
            error="no meshes generated",
            seconds=time.time() - start,
        )
    return SceneResult(
        scene_id=scene_id,
        status="ok",
        output_path=str(output_path),
        pointmap_cache=str(cache_path),
        manifest_path=str(manifest_path) if manifest_path else None,
        num_masks=len(masks),
        num_meshes=num_meshes,
        objects=object_reports,
        seconds=time.time() - start,
    )


def result_to_json(result: SceneResult) -> str:
    return json.dumps(asdict(result), ensure_ascii=False, sort_keys=True)


def main() -> int:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        print("[FAIL] CUDA is not available in this process.", flush=True)
        return 1
    scenes = shard_scenes(discover_scenes(args), args.rank, args.world_size)
    print(f"[INFO] cached runner rank={args.rank}/{args.world_size} gpu={os.environ.get('CUDA_VISIBLE_DEVICES')} assigned={len(scenes)} output={output_dir}", flush=True)
    runtime_config = make_runtime_config(Path(args.config), output_dir)
    print(f"[INFO] loading SAM3D config={runtime_config}", flush=True)
    inference = Inference(str(runtime_config), compile=bool(args.compile))
    print("[INFO] model loaded", flush=True)

    report_f = None
    if args.save_report:
        log_dir = output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        report_path = log_dir / f"worker_{args.rank:02d}.jsonl"
        report_f = report_path.open("a", encoding="utf-8")
        print(f"[INFO] report={report_path}", flush=True)

    ok = skipped = failed = 0
    try:
        for local_i, scene_dir in enumerate(scenes, start=1):
            print(f"[INFO] rank={args.rank} [{local_i}/{len(scenes)}] scene={scene_dir.name}", flush=True)
            try:
                result = process_scene(scene_dir, output_dir, inference, args.seed, args.overwrite)
            except Exception as exc:
                result = SceneResult(scene_id=scene_dir.name, status="failed", error=repr(exc))
            if result.status == "ok":
                ok += 1
            elif result.status == "skipped":
                skipped += 1
            else:
                failed += 1
            print(f"[RESULT] {result_to_json(result)}", flush=True)
            if report_f is not None:
                report_f.write(result_to_json(result) + "\n")
                report_f.flush()
            if args.fail_fast and result.status == "failed":
                return 1
    finally:
        if report_f is not None:
            report_f.close()
    print(f"[SUMMARY] rank={args.rank} ok={ok} skipped={skipped} failed={failed}", flush=True)
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
