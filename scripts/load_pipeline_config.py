#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Any

import yaml


PathKey = tuple[str, ...]


MAIN_MAPPINGS: list[tuple[PathKey, str]] = [
    (("pipeline", "workflow_root"), "WORKFLOW_ROOT"),
    (("pipeline", "conda_bin"), "CONDA_BIN"),
    (("pipeline", "conda_env"), "FYSIVERSE_CONDA_ENV"),
    (("pipeline", "blender_bin"), "BLENDER_BIN"),
    (("pipeline", "package_timeout"), "PACKAGE_TIMEOUT"),
    (("paths", "gsam2_root"), "GSAM2_ROOT"),
    (("paths", "sam3d_root"), "SAM3D_ROOT"),
    (("paths", "sam3d_env_dir"), "SAM3D_ENV_DIR"),
    (("paths", "sam3d_config"), "SAM3D_CONFIG"),
    (("paths", "moge_root"), "MOGE_ROOT"),
    (("paths", "moge_ckpt"), "MOGE_CKPT"),
    (("server", "name"), "SERVER_NAME"),
    (("server", "port"), "SERVER_PORT"),
    (("server", "port_search_count"), "PORT_SEARCH_COUNT"),
    (("runtime", "start_workers"), "START_WORKERS"),
    (("runtime", "gradio_analytics_enabled"), "GRADIO_ANALYTICS_ENABLED"),
    (("features", "scene_graph"), "ENABLE_SCENE_GRAPH"),
]


MODULE_MAPPINGS: dict[str, list[tuple[PathKey, str]]] = {
    "gsam": [
        (("gpu",), "GSAM_GPU"),
    ],
    "sam3d": [
        (("gpu",), "SAM3D_GPU"),
        (("timeout",), "SAM3D_TIMEOUT_SECONDS"),
    ],
    "moge": [
        (("device",), "MOGE_DEVICE"),
        (("resize_max",), "MOGE_RESIZE_MAX"),
        (("resolution_level",), "MOGE_RESOLUTION_LEVEL"),
        (("timeout",), "MOGE_TIMEOUT"),
        (("nvdiffrast", "conda_bin"), "MOGE_NVDIFFRAST_CONDA_BIN"),
        (("nvdiffrast", "env"), "MOGE_NVDIFFRAST_ENV"),
        (("camera_optimization", "steps"), "MOGE_CAMERA_OPT_STEPS"),
        (("camera_optimization", "max_side"), "MOGE_CAMERA_OPT_MAX_SIDE"),
        (("object_pose_optimization", "workers"), "MOGE_OBJECT_POSE_WORKERS"),
        (("object_pose_optimization", "max_side"), "MOGE_OBJECT_POSE_MAX_SIDE"),
        (("object_pose_optimization", "position_steps"), "MOGE_OBJECT_POSITION_STEPS"),
        (("object_pose_optimization", "yaw_steps"), "MOGE_OBJECT_YAW_STEPS"),
        (("object_pose_optimization", "scale_steps"), "MOGE_OBJECT_SCALE_STEPS"),
        (("object_pose_optimization", "translation_mode"), "MOGE_OBJECT_TRANSLATION_MODE"),
        (("object_pose_optimization", "translation_initial_grid"), "MOGE_OBJECT_TRANSLATION_INITIAL_GRID"),
        (("object_pose_optimization", "center_weight"), "MOGE_OBJECT_CENTER_WEIGHT"),
        (("object_pose_optimization", "area_weight"), "MOGE_OBJECT_AREA_WEIGHT"),
        (("object_pose_optimization", "yaw_initial_samples"), "MOGE_OBJECT_YAW_INITIAL_SAMPLES"),
        (("object_pose_optimization", "scale_initial_samples"), "MOGE_OBJECT_SCALE_INITIAL_SAMPLES"),
        (("object_pose_optimization", "phase_accept_iou_drop"), "MOGE_OBJECT_PHASE_ACCEPT_IOU_DROP"),
        (("object_pose_optimization", "max_translation"), "MOGE_OBJECT_MAX_TRANSLATION"),
        (("object_pose_optimization", "max_yaw_deg"), "MOGE_OBJECT_MAX_YAW_DEG"),
        (("object_pose_optimization", "max_scale_delta"), "MOGE_OBJECT_MAX_SCALE_DELTA"),
    ],
    "sapien": [
        (("gravity", "timeout"), "GRAVITY_TIMEOUT"),
        (("gravity", "z"), "GRAVITY_Z"),
        (("env",), "SAPIEN_ENV"),
        (("steps",), "SAPIEN_STEPS"),
        (("timestep",), "SAPIEN_TIMESTEP"),
        (("render_video",), "SAPIEN_RENDER_VIDEO"),
        (("render_every",), "SAPIEN_RENDER_EVERY"),
        (("settle", "window"), "SAPIEN_SETTLE_WINDOW"),
        (("settle", "translation_threshold"), "SAPIEN_SETTLE_TRANSLATION_THRESHOLD"),
        (("video", "fps"), "SAPIEN_VIDEO_FPS"),
        (("video", "width"), "SAPIEN_VIDEO_WIDTH"),
        (("video", "height"), "SAPIEN_VIDEO_HEIGHT"),
        (("friction", "static"), "SAPIEN_STATIC_FRICTION"),
        (("friction", "dynamic"), "SAPIEN_DYNAMIC_FRICTION"),
        (("damping", "linear"), "SAPIEN_LINEAR_DAMPING"),
        (("damping", "angular"), "SAPIEN_ANGULAR_DAMPING"),
        (("default_scene",), "SAPIEN_DEFAULT_SCENE"),
        (("physx_gpu",), "SAPIEN_PHYSX_GPU"),
    ],
    "collision": [
        (("bbox_overlap_margin",), "BBOX_OVERLAP_MARGIN"),
        (("bbox_overlap_iters",), "BBOX_OVERLAP_ITERS"),
        (("overlap_collision_method",), "OVERLAP_COLLISION_METHOD"),
        (("convex_hull_max_vertices",), "CONVEX_HULL_MAX_VERTICES"),
        (("convex_decomposition_method",), "CONVEX_DECOMPOSITION_METHOD"),
        (("separated_convex_decomposition_method",), "SEPARATED_CONVEX_DECOMPOSITION_METHOD"),
        (("sapien_collision_decomposition_method",), "SAPIEN_COLLISION_DECOMPOSITION_METHOD"),
        (("coacd", "conda_bin"), "COACD_CONDA_BIN"),
        (("coacd", "env"), "COACD_ENV"),
        (("coacd", "timeout"), "COACD_TIMEOUT"),
        (("coacd", "workers"), "COACD_WORKERS"),
        (("coacd", "source_max_faces"), "COACD_SOURCE_MAX_FACES"),
        (("coacd", "source_simplification_backend"), "COACD_SOURCE_SIMPLIFICATION_BACKEND"),
        (("coacd", "source_simplification_agg"), "COACD_SOURCE_SIMPLIFICATION_AGG"),
        (("coacd", "blender_source_decimate"), "COACD_BLENDER_SOURCE_DECIMATE"),
        (("coacd", "max_convex_parts"), "COACD_MAX_CONVEX_PARTS"),
        (("coacd", "threshold"), "COACD_THRESHOLD"),
        (("coacd", "preprocess_mode"), "COACD_PREPROCESS_MODE"),
        (("coacd", "preprocess_resolution"), "COACD_PREPROCESS_RESOLUTION"),
        (("coacd", "resolution"), "COACD_RESOLUTION"),
        (("coacd", "mcts_nodes"), "COACD_MCTS_NODES"),
        (("coacd", "mcts_iterations"), "COACD_MCTS_ITERATIONS"),
        (("coacd", "mcts_max_depth"), "COACD_MCTS_MAX_DEPTH"),
        (("coacd", "max_ch_vertex"), "COACD_MAX_CH_VERTEX"),
        (("coacd", "apx_mode"), "COACD_APX_MODE"),
        (("coacd", "seed"), "COACD_SEED"),
        (("coacd", "real_metric"), "COACD_REAL_METRIC"),
        (("coacd", "merge"), "COACD_MERGE"),
        (("coacd", "decimate"), "COACD_DECIMATE"),
        (("convex_collision_eps",), "CONVEX_COLLISION_EPS"),
        (("convex_min_horizontal_axis",), "CONVEX_MIN_HORIZONTAL_AXIS"),
    ],
    "scene_graph": [
        (("codex_bin",), "CODEX_BIN"),
        (("backend",), "SCENE_GRAPH_BACKEND"),
        (("model",), "SCENE_GRAPH_MODEL"),
        (("timeout",), "SCENE_GRAPH_TIMEOUT"),
        (("wait_before_3d",), "SCENE_GRAPH_WAIT_BEFORE_3D"),
        (("max_crop_images",), "SCENE_GRAPH_MAX_CROP_IMAGES"),
        (("reasoning_effort",), "SCENE_GRAPH_REASONING_EFFORT"),
        (("api", "base_url"), "SCENE_GRAPH_API_BASE_URL"),
        (("api", "model"), "SCENE_GRAPH_API_MODEL"),
        (("api", "key"), "SCENE_GRAPH_API_KEY"),
        (("api", "retry_without_crops"), "SCENE_GRAPH_API_RETRY_WITHOUT_CROPS"),
    ],
    "background_3dgs": [
        (("enabled",), "BACKGROUND_3DGS_ENABLED"),
        (("background_image", "backend"), "BACKGROUND_IMAGE_BACKEND"),
        (("background_image", "skip"), "SKIP_BACKGROUND_EDIT"),
        (("background_image", "force"), "FORCE_EDIT"),
        (("background_image", "timeout"), "BACKGROUND_IMAGE_TIMEOUT"),
        (("background_image", "local_fallback"), "BACKGROUND_LOCAL_FALLBACK"),
        (("dependencies", "qwen_env"), "QWEN_ENV"),
        (("dependencies", "moge_env"), "MOGE_ENV"),
        (("dependencies", "threedgrut_root"), "THREEDGRUT_ROOT"),
        (("dependencies", "gaussian_splats_root"), "GAUSSIAN_SPLATS_ROOT"),
        (("gpu",), "BACKGROUND_3DGS_GPU"),
        (("iterations",), "BACKGROUND_3DGS_ITERATIONS"),
        (("timeout",), "BACKGROUND_3DGS_TIMEOUT"),
        (("max_points",), "MAX_POINTS"),
        (("use_reference_moge_fusion",), "USE_REFERENCE_MOGE_FUSION"),
        (("fusion_depth_align",), "FUSION_DEPTH_ALIGN"),
        (("foreground_mask_dilate",), "FOREGROUND_MASK_DILATE"),
        (("ksplat_compression",), "BACKGROUND_3DGS_KSPLAT_COMPRESSION"),
        (("ksplat_alpha_threshold",), "BACKGROUND_3DGS_KSPLAT_ALPHA_THRESHOLD"),
        (("ksplat_scene_center",), "KSPLAT_SCENE_CENTER"),
        (("ksplat_block_size",), "KSPLAT_BLOCK_SIZE"),
        (("ksplat_bucket_size",), "KSPLAT_BUCKET_SIZE"),
        (("ksplat_sh_degree",), "BACKGROUND_3DGS_KSPLAT_SH_DEGREE"),
    ],
    "simulator_assistance": [
        (("llm_backend",), "SIMULATOR_ASSISTANCE_LLM_BACKEND"),
        (("codex_bin",), "SIMULATOR_ASSISTANCE_CODEX_BIN"),
        (("model",), "SIMULATOR_ASSISTANCE_CODEX_MODEL"),
        (("timeout",), "SIMULATOR_ASSISTANCE_LLM_TIMEOUT"),
        (("max_crop_images",), "SIMULATOR_ASSISTANCE_MAX_CROP_IMAGES"),
        (("api", "base_url"), "SIMULATOR_ASSISTANCE_API_BASE_URL"),
        (("api", "model"), "SIMULATOR_ASSISTANCE_API_MODEL"),
        (("deterministic_runner", "blender_bin"), "BLENDER_BIN"),
        (("deterministic_runner", "conda_bin"), "SIMULATOR_ASSISTANCE_RUNNER_CONDA_BIN"),
        (("deterministic_runner", "conda_env"), "SIMULATOR_ASSISTANCE_RUNNER_CONDA_ENV"),
        (("deterministic_runner", "reuse_upstream_sapien_export"), "SIMULATOR_ASSISTANCE_REUSE_UPSTREAM_SAPIEN_EXPORT"),
        (("deterministic_runner", "skip_camera_pose_optimization"), "SIMULATOR_ASSISTANCE_SKIP_CAMERA_POSE_OPTIMIZATION"),
        (("deterministic_runner", "skip_object_pose_optimization"), "SIMULATOR_ASSISTANCE_SKIP_OBJECT_POSE_OPTIMIZATION"),
        (("deterministic_runner", "skip_original_camera_render"), "SIMULATOR_ASSISTANCE_SKIP_ORIGINAL_CAMERA_RENDER"),
        (("deterministic_runner", "convex_decomposition_method"), "SIMULATOR_ASSISTANCE_CONVEX_DECOMPOSITION_METHOD"),
        (("deterministic_runner", "coacd_timeout"), "SIMULATOR_ASSISTANCE_COACD_TIMEOUT"),
        (("deterministic_runner", "coacd_max_convex_parts"), "SIMULATOR_ASSISTANCE_COACD_MAX_CONVEX_PARTS"),
        (("deterministic_runner", "coacd_threshold"), "SIMULATOR_ASSISTANCE_COACD_THRESHOLD"),
        (("deterministic_runner", "coacd_preprocess_mode"), "SIMULATOR_ASSISTANCE_COACD_PREPROCESS_MODE"),
        (("deterministic_runner", "coacd_preprocess_resolution"), "SIMULATOR_ASSISTANCE_COACD_PREPROCESS_RESOLUTION"),
        (("deterministic_runner", "coacd_resolution"), "SIMULATOR_ASSISTANCE_COACD_RESOLUTION"),
        (("deterministic_runner", "coacd_mcts_nodes"), "SIMULATOR_ASSISTANCE_COACD_MCTS_NODES"),
        (("deterministic_runner", "coacd_mcts_iterations"), "SIMULATOR_ASSISTANCE_COACD_MCTS_ITERATIONS"),
        (("deterministic_runner", "coacd_mcts_max_depth"), "SIMULATOR_ASSISTANCE_COACD_MCTS_MAX_DEPTH"),
        (("deterministic_runner", "coacd_max_ch_vertex"), "SIMULATOR_ASSISTANCE_COACD_MAX_CH_VERTEX"),
        (("deterministic_runner", "coacd_apx_mode"), "SIMULATOR_ASSISTANCE_COACD_APX_MODE"),
        (("deterministic_runner", "coacd_seed"), "SIMULATOR_ASSISTANCE_COACD_SEED"),
        (("deterministic_runner", "coacd_real_metric"), "SIMULATOR_ASSISTANCE_COACD_REAL_METRIC"),
        (("deterministic_runner", "coacd_merge"), "SIMULATOR_ASSISTANCE_COACD_MERGE"),
        (("deterministic_runner", "coacd_decimate"), "SIMULATOR_ASSISTANCE_COACD_DECIMATE"),
    ],
    "web_assets": [
        (("export",), "WEB_ASSETS_EXPORT"),
        (("preview_compressor",), "WEB_ASSETS_PREVIEW_COMPRESSOR"),
        (("web_compressor",), "WEB_ASSETS_WEB_COMPRESSOR"),
        (("gltfpack_bin",), "WEB_ASSETS_GLTFPACK_BIN"),
        (("gltf_transform_bin",), "WEB_ASSETS_GLTF_TRANSFORM_BIN"),
        (("allow_npx_gltfpack",), "WEB_ASSETS_ALLOW_NPX_GLTFPACK"),
        (("allow_npx_gltf_transform",), "WEB_ASSETS_ALLOW_NPX_GLTF_TRANSFORM"),
        (("no_texture_compress",), "WEB_ASSETS_NO_TEXTURE_COMPRESS"),
        (("keep_raw",), "WEB_ASSETS_KEEP_RAW"),
        (("timeout",), "WEB_ASSETS_TIMEOUT"),
        (("blender_timeout",), "WEB_ASSETS_BLENDER_TIMEOUT"),
        (("compressor_timeout",), "WEB_ASSETS_COMPRESSOR_TIMEOUT"),
        (("gltfpack_compression",), "WEB_ASSETS_GLTFPACK_COMPRESSION"),
        (("gltfpack_compression_extension",), "WEB_ASSETS_GLTFPACK_COMPRESSION_EXTENSION"),
        (("gltfpack_texture_quality",), "WEB_ASSETS_GLTFPACK_TEXTURE_QUALITY"),
        (("gltfpack_texture_limit",), "WEB_ASSETS_GLTFPACK_TEXTURE_LIMIT"),
        (("gltfpack_texture_scale",), "WEB_ASSETS_GLTFPACK_TEXTURE_SCALE"),
        (("gltfpack_simplify_ratio",), "WEB_ASSETS_GLTFPACK_SIMPLIFY_RATIO"),
        (("gltfpack_simplify_error",), "WEB_ASSETS_GLTFPACK_SIMPLIFY_ERROR"),
        (("gltfpack_simplify_aggressive",), "WEB_ASSETS_GLTFPACK_SIMPLIFY_AGGRESSIVE"),
        (("gltfpack_simplify_permissive",), "WEB_ASSETS_GLTFPACK_SIMPLIFY_PERMISSIVE"),
        (("gltfpack_position_bits",), "WEB_ASSETS_GLTFPACK_POSITION_BITS"),
        (("gltfpack_texcoord_bits",), "WEB_ASSETS_GLTFPACK_TEXCOORD_BITS"),
        (("gltfpack_normal_bits",), "WEB_ASSETS_GLTFPACK_NORMAL_BITS"),
    ],
    "diagnostic": [
        (("pipeline_stage_diagnostic", "enabled"), "PIPELINE_STAGE_DIAGNOSTIC"),
        (("pipeline_stage_diagnostic", "frames_per_static_stage"), "PIPELINE_STAGE_DIAGNOSTIC_FRAMES_PER_STATIC_STAGE"),
        (("pipeline_stage_diagnostic", "transition_frames"), "PIPELINE_STAGE_DIAGNOSTIC_TRANSITION_FRAMES"),
        (("pipeline_stage_diagnostic", "fps"), "PIPELINE_STAGE_DIAGNOSTIC_FPS"),
        (("pipeline_stage_diagnostic", "timeout"), "PIPELINE_STAGE_DIAGNOSTIC_TIMEOUT"),
        (("web_pipeline_animation", "enabled"), "WEB_PIPELINE_ANIMATION"),
        (("web_pipeline_animation", "frames_per_static_stage"), "WEB_PIPELINE_ANIMATION_FRAMES_PER_STATIC_STAGE"),
        (("web_pipeline_animation", "transition_frames"), "WEB_PIPELINE_ANIMATION_TRANSITION_FRAMES"),
        (("web_pipeline_animation", "fps"), "WEB_PIPELINE_ANIMATION_FPS"),
        (("web_pipeline_animation", "gravity_frame_step"), "WEB_PIPELINE_ANIMATION_GRAVITY_FRAME_STEP"),
        (("web_pipeline_animation", "timeout"), "WEB_PIPELINE_ANIMATION_TIMEOUT"),
    ],
}


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"config must be a mapping: {path}")
    return data


def get_nested(data: dict[str, Any], keys: PathKey) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def stringify(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if value is None:
        return ""
    return str(value)


def resolve_config_path(base_dir: Path, value: Any) -> Path | None:
    if value in (None, False, ""):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path


def collect_exports(data: dict[str, Any], mappings: list[tuple[PathKey, str]], exports: dict[str, str]) -> None:
    for keys, env_name in mappings:
        value = get_nested(data, keys)
        if value is None or value == "":
            continue
        exports[env_name] = stringify(value)


def load_exports_from_config(config: Path) -> dict[str, str]:
    main_path = config.expanduser().resolve()
    main_data = load_yaml(main_path)
    exports: dict[str, str] = {}
    collect_exports(main_data, MAIN_MAPPINGS, exports)

    module_configs = main_data.get("module_configs") or {}
    if not isinstance(module_configs, dict):
        raise TypeError("main config field 'module_configs' must be a mapping")

    for module_name, mappings in MODULE_MAPPINGS.items():
        module_path = resolve_config_path(main_path.parent, module_configs.get(module_name))
        if module_path is None:
            continue
        module_data = load_yaml(module_path)
        collect_exports(module_data, mappings, exports)
    return exports


def main() -> int:
    parser = argparse.ArgumentParser(description="Load pipeline YAML config and emit shell exports.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    exports = load_exports_from_config(args.config)

    for key in sorted(exports):
        print(f"export {key}={shlex.quote(exports[key])}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"failed to load pipeline config: {exc}", file=sys.stderr)
        raise SystemExit(1)
