#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import math
import mimetypes
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WORKFLOW_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = Path(__file__).resolve().parent
DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_DASHSCOPE_MODEL = "qwen3.7-plus"
OPENAI_COMPATIBLE_BACKENDS = {"openai_compatible", "dashscope_qwen"}


@dataclass
class CodexJob:
    name: str
    prompt: str
    output_path: Path
    images: list[Path]
    effort: str


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def bool_config(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def simulator_llm_defaults() -> dict[str, Any]:
    simulator_cfg = load_yaml(WORKFLOW_ROOT / "config" / "simulator_assistance.yaml")
    api_cfg = simulator_cfg.get("api") if isinstance(simulator_cfg.get("api"), dict) else {}
    runner_cfg = simulator_cfg.get("deterministic_runner") if isinstance(simulator_cfg.get("deterministic_runner"), dict) else {}
    llm_backend = os.environ.get("SIMULATOR_ASSISTANCE_LLM_BACKEND") or simulator_cfg.get("llm_backend") or "openai_compatible"
    default_api_base_url = DEFAULT_DASHSCOPE_BASE_URL if llm_backend == "dashscope_qwen" else ""
    default_api_model = DEFAULT_DASHSCOPE_MODEL if llm_backend == "dashscope_qwen" else ""
    return {
        "llm_backend": llm_backend,
        "codex_bin": os.environ.get("SIMULATOR_ASSISTANCE_CODEX_BIN") or simulator_cfg.get("codex_bin") or "codex",
        "codex_model": os.environ.get("SIMULATOR_ASSISTANCE_CODEX_MODEL") or simulator_cfg.get("model") or "",
        "api_base_url": os.environ.get("SIMULATOR_ASSISTANCE_API_BASE_URL") or api_cfg.get("base_url") or default_api_base_url,
        "api_model": os.environ.get("SIMULATOR_ASSISTANCE_API_MODEL") or api_cfg.get("model") or default_api_model,
        "api_key": os.environ.get("SIMULATOR_ASSISTANCE_API_KEY") or api_cfg.get("key") or "",
        "llm_timeout": int(os.environ.get("SIMULATOR_ASSISTANCE_LLM_TIMEOUT") or simulator_cfg.get("timeout") or 600),
        "max_crop_images": int(os.environ.get("SIMULATOR_ASSISTANCE_MAX_CROP_IMAGES") or simulator_cfg.get("max_crop_images") or 200),
        "blender_bin": os.environ.get("BLENDER_BIN") or runner_cfg.get("blender_bin") or "blender",
        "sim_conda_bin": os.environ.get("SIMULATOR_ASSISTANCE_RUNNER_CONDA_BIN") or runner_cfg.get("conda_bin") or os.environ.get("CONDA_BIN") or "conda",
        "sim_conda_env": os.environ.get("SIMULATOR_ASSISTANCE_RUNNER_CONDA_ENV") or runner_cfg.get("conda_env") or os.environ.get("FYSIVERSE_CONDA_ENV") or "fysiverse-scene",
        "reuse_upstream_sapien_export": os.environ.get("SIMULATOR_ASSISTANCE_REUSE_UPSTREAM_SAPIEN_EXPORT") or runner_cfg.get("reuse_upstream_sapien_export") or "always",
        "skip_original_camera_render": bool_config(os.environ.get("SIMULATOR_ASSISTANCE_SKIP_ORIGINAL_CAMERA_RENDER"), bool_config(runner_cfg.get("skip_original_camera_render"), False)),
        "skip_camera_pose_optimization": bool_config(os.environ.get("SIMULATOR_ASSISTANCE_SKIP_CAMERA_POSE_OPTIMIZATION"), bool_config(runner_cfg.get("skip_camera_pose_optimization"), True)),
        "skip_object_pose_optimization": bool_config(os.environ.get("SIMULATOR_ASSISTANCE_SKIP_OBJECT_POSE_OPTIMIZATION"), bool_config(runner_cfg.get("skip_object_pose_optimization"), True)),
        "convex_decomposition_method": os.environ.get("SIMULATOR_ASSISTANCE_CONVEX_DECOMPOSITION_METHOD") or runner_cfg.get("convex_decomposition_method") or "coacd",
        "coacd_timeout": int(os.environ.get("SIMULATOR_ASSISTANCE_COACD_TIMEOUT") or runner_cfg.get("coacd_timeout", 120)),
        "coacd_max_convex_parts": int(os.environ.get("SIMULATOR_ASSISTANCE_COACD_MAX_CONVEX_PARTS") or runner_cfg.get("coacd_max_convex_parts", 8)),
        "coacd_threshold": float(os.environ.get("SIMULATOR_ASSISTANCE_COACD_THRESHOLD") or runner_cfg.get("coacd_threshold", 0.05)),
        "coacd_preprocess_mode": os.environ.get("SIMULATOR_ASSISTANCE_COACD_PREPROCESS_MODE") or runner_cfg.get("coacd_preprocess_mode") or "auto",
        "coacd_preprocess_resolution": int(os.environ.get("SIMULATOR_ASSISTANCE_COACD_PREPROCESS_RESOLUTION") or runner_cfg.get("coacd_preprocess_resolution", 50)),
        "coacd_resolution": int(os.environ.get("SIMULATOR_ASSISTANCE_COACD_RESOLUTION") or runner_cfg.get("coacd_resolution", 2000)),
        "coacd_mcts_nodes": int(os.environ.get("SIMULATOR_ASSISTANCE_COACD_MCTS_NODES") or runner_cfg.get("coacd_mcts_nodes", 20)),
        "coacd_mcts_iterations": int(os.environ.get("SIMULATOR_ASSISTANCE_COACD_MCTS_ITERATIONS") or runner_cfg.get("coacd_mcts_iterations", 80)),
        "coacd_mcts_max_depth": int(os.environ.get("SIMULATOR_ASSISTANCE_COACD_MCTS_MAX_DEPTH") or runner_cfg.get("coacd_mcts_max_depth", 3)),
        "coacd_max_ch_vertex": int(os.environ.get("SIMULATOR_ASSISTANCE_COACD_MAX_CH_VERTEX") or runner_cfg.get("coacd_max_ch_vertex", 256)),
        "coacd_apx_mode": os.environ.get("SIMULATOR_ASSISTANCE_COACD_APX_MODE") or runner_cfg.get("coacd_apx_mode") or "ch",
        "coacd_seed": int(os.environ.get("SIMULATOR_ASSISTANCE_COACD_SEED") or runner_cfg.get("coacd_seed", 0)),
        "coacd_real_metric": bool_config(os.environ.get("SIMULATOR_ASSISTANCE_COACD_REAL_METRIC"), bool_config(runner_cfg.get("coacd_real_metric"), False)),
        "coacd_merge": bool_config(os.environ.get("SIMULATOR_ASSISTANCE_COACD_MERGE"), bool_config(runner_cfg.get("coacd_merge"), True)),
        "coacd_decimate": bool_config(os.environ.get("SIMULATOR_ASSISTANCE_COACD_DECIMATE"), bool_config(runner_cfg.get("coacd_decimate"), False)),
        "render_width": int(os.environ.get("SIMULATOR_ASSISTANCE_RENDER_WIDTH") or runner_cfg.get("render_width", 480)),
        "render_height": int(os.environ.get("SIMULATOR_ASSISTANCE_RENDER_HEIGHT") or runner_cfg.get("render_height", 480)),
        "render_fps": int(os.environ.get("SIMULATOR_ASSISTANCE_RENDER_FPS") or runner_cfg.get("render_fps", 12)),
        "render_samples": int(os.environ.get("SIMULATOR_ASSISTANCE_RENDER_SAMPLES") or runner_cfg.get("render_samples", 8)),
        "render_max_frames": int(os.environ.get("SIMULATOR_ASSISTANCE_RENDER_MAX_FRAMES") or runner_cfg.get("render_max_frames", 50)),
        "render_compute_backend": os.environ.get("SIMULATOR_ASSISTANCE_RENDER_COMPUTE_BACKEND") or runner_cfg.get("render_compute_backend") or "CUDA",
    }


def parse_args() -> argparse.Namespace:
    llm_defaults = simulator_llm_defaults()
    parser = argparse.ArgumentParser(description="Plan and optionally run agentic physical simulation for a reconstructed Fysiverse scene.")
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--llm-backend", choices=["openai_compatible", "codex", "dashscope_qwen"], default=str(llm_defaults["llm_backend"]))
    parser.add_argument("--codex-bin", default=str(llm_defaults["codex_bin"]))
    parser.add_argument("--model", default=str(llm_defaults["codex_model"]) or None, help="Codex model override. Ignored by OpenAI-compatible API backends.")
    parser.add_argument("--api-key", default=str(llm_defaults["api_key"]), help="OpenAI-compatible API key. Prefer SIMULATOR_ASSISTANCE_API_KEY.")
    parser.add_argument("--api-base-url", default=str(llm_defaults["api_base_url"]))
    parser.add_argument("--api-model", default=str(llm_defaults["api_model"]))
    parser.add_argument("--llm-timeout", type=int, default=int(llm_defaults["llm_timeout"]))
    parser.add_argument("--max-crop-images", type=int, default=int(llm_defaults["max_crop_images"]), help="Maximum object crop images attached to LLM calls. Overlay image is always included when available.")
    parser.add_argument("--codex-home", type=Path, default=None, help="Optional CODEX_HOME for the child Codex process.")
    parser.add_argument("--mode", choices=["single", "parallel", "auto"], default="auto")
    parser.add_argument("--planner-effort", default="medium")
    parser.add_argument("--agent-effort", default="medium")
    parser.add_argument("--force-scene-digest", action="store_true", help="Rebuild scene_digest.json even if cache is fresh.")
    parser.add_argument("--no-run", action="store_true", help="Write plan artifacts only; do not run SAPIEN/Blender.")
    parser.add_argument("--blender-bin", default=str(llm_defaults["blender_bin"]))
    parser.add_argument("--sim-conda-bin", default=str(llm_defaults["sim_conda_bin"]))
    parser.add_argument("--sim-conda-env", default=str(llm_defaults["sim_conda_env"]))
    parser.add_argument("--reuse-upstream-sapien-export", choices=["always", "auto", "never"], default=str(llm_defaults["reuse_upstream_sapien_export"]))
    parser.add_argument("--convex-decomposition-method", choices=["single_convex_hull", "coacd"], default=str(llm_defaults["convex_decomposition_method"]))
    parser.add_argument("--coacd-timeout", type=int, default=int(llm_defaults["coacd_timeout"]))
    parser.add_argument("--coacd-max-convex-parts", type=int, default=int(llm_defaults["coacd_max_convex_parts"]))
    parser.add_argument("--coacd-threshold", type=float, default=float(llm_defaults["coacd_threshold"]))
    parser.add_argument("--coacd-preprocess-mode", default=str(llm_defaults["coacd_preprocess_mode"]))
    parser.add_argument("--coacd-preprocess-resolution", type=int, default=int(llm_defaults["coacd_preprocess_resolution"]))
    parser.add_argument("--coacd-resolution", type=int, default=int(llm_defaults["coacd_resolution"]))
    parser.add_argument("--coacd-mcts-nodes", type=int, default=int(llm_defaults["coacd_mcts_nodes"]))
    parser.add_argument("--coacd-mcts-iterations", type=int, default=int(llm_defaults["coacd_mcts_iterations"]))
    parser.add_argument("--coacd-mcts-max-depth", type=int, default=int(llm_defaults["coacd_mcts_max_depth"]))
    parser.add_argument("--coacd-max-ch-vertex", type=int, default=int(llm_defaults["coacd_max_ch_vertex"]))
    parser.add_argument("--coacd-apx-mode", default=str(llm_defaults["coacd_apx_mode"]))
    parser.add_argument("--coacd-seed", type=int, default=int(llm_defaults["coacd_seed"]))
    parser.add_argument("--coacd-real-metric", action=argparse.BooleanOptionalAction, default=bool(llm_defaults["coacd_real_metric"]))
    parser.add_argument("--coacd-merge", action=argparse.BooleanOptionalAction, default=bool(llm_defaults["coacd_merge"]))
    parser.add_argument("--coacd-decimate", action=argparse.BooleanOptionalAction, default=bool(llm_defaults["coacd_decimate"]))
    parser.add_argument("--skip-original-camera-render", action=argparse.BooleanOptionalAction, default=bool(llm_defaults["skip_original_camera_render"]), help="Run simulation but skip original input-view video rendering.")
    parser.add_argument("--skip-camera-pose-optimization", action=argparse.BooleanOptionalAction, default=bool(llm_defaults["skip_camera_pose_optimization"]), help="Reuse the upstream camera result instead of running local nvdiffrast extrinsic optimization.")
    parser.add_argument("--skip-object-pose-optimization", action=argparse.BooleanOptionalAction, default=bool(llm_defaults["skip_object_pose_optimization"]), help="Reuse upstream object poses instead of running local nvdiffrast object alignment.")
    parser.add_argument("--render-width", type=int, default=int(llm_defaults["render_width"]))
    parser.add_argument("--render-height", type=int, default=int(llm_defaults["render_height"]), help="<=0 preserves the original input image aspect ratio.")
    parser.add_argument("--render-fps", type=int, default=int(llm_defaults["render_fps"]))
    parser.add_argument("--render-samples", type=int, default=int(llm_defaults["render_samples"]))
    parser.add_argument("--render-max-frames", type=int, default=int(llm_defaults["render_max_frames"]), help="0 renders the full animation frame range.")
    parser.add_argument("--render-compute-backend", default=str(llm_defaults["render_compute_backend"]))
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
    parser.add_argument("--output-last-message", type=Path, default=None)
    return parser.parse_args()


def resolve_session(session: Path) -> Path:
    if session.is_absolute():
        return session
    direct = (WORKFLOW_ROOT / session).resolve()
    if (direct / "results" / "final_scene_manifest.json").is_file():
        return direct
    under_sessions = (WORKFLOW_ROOT / "sessions" / session).resolve()
    if (under_sessions / "results" / "final_scene_manifest.json").is_file():
        return under_sessions
    if session.parts and session.parts[0] == "sessions":
        return direct
    return under_sessions


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def optional_existing(path: Path) -> str | None:
    return str(path) if path.is_file() else None


def optional_existing_path(path: Path) -> str | None:
    return str(path) if path.exists() else None


def resolve_work_dir(args: argparse.Namespace, session_dir: Path) -> Path:
    if args.work_dir is None:
        run_id = time.strftime("%Y%m%d_%H%M%S")
        return session_dir / "results" / "simulator_assistance" / run_id
    if args.work_dir.is_absolute():
        return args.work_dir
    return WORKFLOW_ROOT / args.work_dir


def codex_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.codex_home is not None:
        codex_home = args.codex_home if args.codex_home.is_absolute() else (WORKFLOW_ROOT / args.codex_home)
        codex_home.mkdir(parents=True, exist_ok=True)
        env["CODEX_HOME"] = str(codex_home)
    return env


def codex_base_cmd(args: argparse.Namespace, last_message: Path, images: list[Path], effort: str) -> list[str]:
    cmd = [
        args.codex_bin,
        "-a",
        "never",
        "exec",
        "-C",
        str(WORKFLOW_ROOT),
        "--skip-git-repo-check",
        "--ephemeral",
        "-s",
        "workspace-write",
        "-c",
        f'model_reasoning_effort="{effort}"',
        "--output-last-message",
        str(last_message),
    ]
    for image in images:
        if image.is_file():
            cmd.extend(["-i", str(image)])
    if args.model:
        cmd.extend(["-m", args.model])
    cmd.append("-")
    return cmd


def mime_type_for_image(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    if guessed in {"image/png", "image/jpeg", "image/webp"}:
        return guessed
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"


def image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type_for_image(path)};base64,{encoded}"


def chat_completions_url(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith("/chat/completions"):
        return stripped
    return f"{stripped}/chat/completions"


def call_openai_compatible_api(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    timeout: int | float,
) -> tuple[str, dict[str, Any]]:
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        OpenAI = None  # type: ignore

    if OpenAI is not None:
        client = OpenAI(api_key=api_key, base_url=base_url.rstrip("/"), timeout=float(timeout))
        completion = client.chat.completions.create(model=model, messages=messages)
        text = completion.choices[0].message.content or ""
        usage = getattr(completion, "usage", None)
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump(exclude_none=True)
        elif usage is not None and not isinstance(usage, dict):
            usage = None
        return text, {
            "client": "openai",
            "id": getattr(completion, "id", None),
            "model": getattr(completion, "model", model),
            "usage": usage,
        }

    payload = {"model": model, "messages": messages}
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        chat_completions_url(base_url),
        data=data,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            response_text = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI-compatible API HTTP {exc.code}: {body[:2000]}") from exc
    response_json = json.loads(response_text)
    text = ((response_json.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    usage = response_json.get("usage")
    return text, {
        "client": "urllib",
        "id": response_json.get("id"),
        "model": response_json.get("model") or model,
        "usage": usage if isinstance(usage, dict) else None,
    }


def uses_openai_compatible_backend(backend: str) -> bool:
    return backend in OPENAI_COMPATIBLE_BACKENDS


def extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fence_start = stripped.find("```")
    if fence_start >= 0:
        stripped = stripped.replace("```json", "```")
        parts = stripped.split("```")
        if len(parts) >= 3:
            stripped = parts[1].strip()
    else:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            stripped = stripped[start : end + 1]
    return json.loads(stripped)


def collect_image_inputs(digest: dict[str, Any], max_crop_images: int | None = None) -> list[Path]:
    images: list[Path] = []
    overlay = (digest.get("inputs") or {}).get("overlay")
    if overlay and Path(overlay).is_file():
        images.append(Path(overlay))
    crop_count = 0
    for obj in digest.get("objects") or []:
        if max_crop_images is not None and crop_count >= max(0, int(max_crop_images)):
            break
        crop = obj.get("crop_rgb")
        if crop and Path(crop).is_file():
            images.append(Path(crop))
            crop_count += 1
    return images


def run_command(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None, input_text: str | None = None) -> str:
    print("+ " + " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.stdout.strip():
        print(proc.stdout)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout)
    return proc.stdout


def json_for_prompt(data: Any, max_chars: int = 80000) -> str:
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... <truncated>"


def json_file_for_prompt(path: Path, max_chars: int = 80000) -> str:
    if not path.is_file():
        return f"<missing: {path}>"
    try:
        return json_for_prompt(load_json(path), max_chars=max_chars)
    except Exception:
        text = path.read_text(encoding="utf-8", errors="replace")
        return text[:max_chars] + ("\n... <truncated>" if len(text) > max_chars else "")


def build_scene_digest(final_manifest: Path, work_dir: Path, reuse: bool) -> Path:
    digest_path = work_dir / "scene_digest.json"
    cmd = [
        "python3",
        str(SKILL_DIR / "scripts" / "build_scene_digest.py"),
        "--final-manifest",
        str(final_manifest),
        "--output",
        str(digest_path),
    ]
    if reuse:
        cmd.append("--reuse-existing")
    run_command(cmd, cwd=WORKFLOW_ROOT)
    return digest_path


def simulation_runner_cmd(args: argparse.Namespace, final_manifest: Path, work_dir: Path) -> list[str]:
    cmd = [
        "python3",
        str(SKILL_DIR / "scripts" / "run_simulation_plan.py"),
        "--final-manifest",
        str(final_manifest),
        "--plan",
        str(work_dir / "simulation_plan.json"),
        "--work-dir",
        str(work_dir / "run"),
        "--blender-bin",
        args.blender_bin,
        "--conda-bin",
        args.sim_conda_bin,
        "--conda-env",
        args.sim_conda_env,
        "--reuse-upstream-sapien-export",
        args.reuse_upstream_sapien_export,
        "--convex-decomposition-method",
        args.convex_decomposition_method,
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
        "--render-width",
        str(args.render_width),
        "--render-height",
        str(args.render_height),
        "--render-fps",
        str(args.render_fps),
        "--render-samples",
        str(args.render_samples),
        "--render-max-frames",
        str(args.render_max_frames),
        "--render-compute-backend",
        args.render_compute_backend,
        "--camera-opt-steps",
        str(args.camera_opt_steps),
        "--camera-opt-max-side",
        str(args.camera_opt_max_side),
        "--camera-opt-rot-lr",
        str(args.camera_opt_rot_lr),
        "--camera-opt-trans-lr",
        str(args.camera_opt_trans_lr),
        "--object-pose-workers",
        str(args.object_pose_workers),
        "--object-pose-max-side",
        str(args.object_pose_max_side),
        "--object-position-steps",
        str(args.object_position_steps),
        "--object-yaw-steps",
        str(args.object_yaw_steps),
        "--object-scale-steps",
        str(args.object_scale_steps),
        "--object-translation-mode",
        str(args.object_translation_mode),
        "--object-translation-initial-grid",
        str(args.object_translation_initial_grid),
        "--object-center-weight",
        str(args.object_center_weight),
        "--object-area-weight",
        str(args.object_area_weight),
        "--object-yaw-initial-samples",
        str(args.object_yaw_initial_samples),
        "--object-scale-initial-samples",
        str(args.object_scale_initial_samples),
        "--object-phase-accept-iou-drop",
        str(args.object_phase_accept_iou_drop),
        "--object-position-lr",
        str(args.object_position_lr),
        "--object-yaw-lr",
        str(args.object_yaw_lr),
        "--object-scale-lr",
        str(args.object_scale_lr),
        "--object-max-translation",
        str(args.object_max_translation),
        "--object-max-yaw-deg",
        str(args.object_max_yaw_deg),
        "--object-max-scale-delta",
        str(args.object_max_scale_delta),
    ]
    cmd.append("--skip-original-camera-render" if args.skip_original_camera_render else "--no-skip-original-camera-render")
    cmd.append("--skip-camera-pose-optimization" if args.skip_camera_pose_optimization else "--no-skip-camera-pose-optimization")
    cmd.append("--skip-object-pose-optimization" if args.skip_object_pose_optimization else "--no-skip-object-pose-optimization")
    if args.coacd_real_metric:
        cmd.append("--coacd-real-metric")
    if not args.coacd_merge:
        cmd.append("--no-coacd-merge")
    if args.coacd_decimate:
        cmd.append("--coacd-decimate")
    return cmd


def build_single_prompt(session_dir: Path, goal: str, work_dir: Path, final_manifest: Path, scene_digest: Path, no_run: bool) -> str:
    parent_action = (
        "The parent launcher will stop after these files exist because --no-run was set."
        if no_run
        else "The parent launcher will run the deterministic simulation and original-view video renderer after these files exist."
    )
    return f"""Use the simulator_assistance skill located at:
{SKILL_DIR / 'SKILL.md'}

Task:
Convert this user goal into a rigid-body SAPIEN simulation plan, then save the scene description, plan JSON, and summary. Do not run SAPIEN or Blender from this child Codex process. {parent_action}

User goal:
{goal}

Session:
{session_dir}

Final scene manifest:
{final_manifest}

Scene digest:
{scene_digest}

Work directory:
{work_dir}

Required outputs:
- {work_dir / 'scene_description.json'}
- {work_dir / 'simulation_plan.json'}
- {work_dir / 'codex_summary.md'}
- Do not execute SAPIEN or Blender; stop after writing JSON and summary.

Important constraints:
- Ground plane is fixed at z=0 with normal +Z. Do not infer or request a ground plane from upstream.
- Inspect object crops if needed; preserve mask ids and final_3d_object_name exactly.
- For this pipeline, SAPIEN rendering is disabled. Save pose sequences and Blender animation only.
- The parent launcher provides the SAPIEN and Blender executables through CLI arguments or environment variables.
- Keep assumptions explicit in simulation_plan.json and codex_summary.md.
"""


def build_qwen_single_prompt(session_dir: Path, goal: str, work_dir: Path, final_manifest: Path, scene_digest: Path, digest: dict[str, Any], no_run: bool) -> str:
    return f"""You are simulator_assistance planning with an OpenAI-compatible VLM API.

You cannot read local files or write files directly. The parent process will write your returned JSON to disk.
Return JSON only, no Markdown fences.

User goal:
{goal}

Session:
{session_dir}

Final scene manifest:
{final_manifest}

Work directory:
{work_dir}

Scene digest path:
{scene_digest}

Scene digest content:
{json_for_prompt(digest)}

Required JSON response:
{{
  "scene_description": {{
    "schema": "simulator_scene_description.v1",
    "...": "complete scene_description.json object"
  }},
  "simulation_plan": {{
    "schema": "simulator_simulation_plan.v1",
    "...": "complete simulation_plan.json object"
  }},
  "summary_markdown": "# Simulator Assistance Summary\\n..."
}}

Important constraints:
- Ground plane is fixed at z=0 with normal +Z. Do not infer or request a ground plane from upstream.
- Preserve mask ids and final_3d_object_name exactly.
- For this pipeline, SAPIEN rendering is disabled. Save pose sequences and Blender animation only.
- Use engine.name = "sapien", ground_z = 0.0, gravity_z = -9.81.
- Use exact object names from scene digest in simulation_plan.objects.
- In simulation_plan.objects, use "body_type": "dynamic", "static", or "kinematic". Do not use "type" for body dynamics.
- Put initial pose offsets and initial linear/angular velocities inside the corresponding object entry.
- Do not use top-level "actions" for initial velocities; use per-object initial_linear_velocity, initial_angular_velocity, or object.events.
- Make the plan executable and conservative; do not include comments inside JSON.
- The parent launcher will {'stop after writing these files because --no-run was set' if no_run else 'run deterministic simulation and original-view rendering after these files exist'}.
"""


def run_qwen_prompt(args: argparse.Namespace, prompt: str, images: list[Path], response_path: Path, log_path: Path) -> dict[str, Any]:
    api_key = args.api_key or os.environ.get("SIMULATOR_ASSISTANCE_API_KEY") or ""
    if not api_key:
        raise RuntimeError("OpenAI-compatible backend requires SIMULATOR_ASSISTANCE_API_KEY or --api-key")
    if not args.api_base_url:
        raise RuntimeError("OpenAI-compatible backend requires SIMULATOR_ASSISTANCE_API_BASE_URL or --api-base-url")
    if not args.api_model:
        raise RuntimeError("OpenAI-compatible backend requires SIMULATOR_ASSISTANCE_API_MODEL or --api-model")
    content: list[dict[str, Any]] = []
    for image in images:
        if image.is_file():
            content.append({"type": "image_url", "image_url": {"url": image_data_url(image)}})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    call_started = time.perf_counter()
    response_text, metadata = call_openai_compatible_api(
        api_key=api_key,
        base_url=args.api_base_url,
        model=args.api_model,
        messages=messages,
        timeout=args.llm_timeout,
    )
    metadata["latency_seconds"] = round(time.perf_counter() - call_started, 6)
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(response_text, encoding="utf-8")
    log_path.write_text(
        json.dumps(
            {
                "backend": args.llm_backend,
                "base_url": args.api_base_url,
                "model": args.api_model,
                "num_images": len([image for image in images if image.is_file()]),
                "prompt_chars": len(prompt),
                "response_chars": len(response_text),
                "metadata": metadata,
                "response_path": str(response_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return extract_json(response_text)


def normalize_qwen_simulation_plan(plan: dict[str, Any]) -> dict[str, Any]:
    objects = plan.get("objects")
    if not isinstance(objects, list):
        return plan
    by_name: dict[str, dict[str, Any]] = {}
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        if name:
            by_name[str(name)] = obj
        if obj.get("body_type") is None and str(obj.get("type", "")).lower() in {"dynamic", "static", "kinematic"}:
            obj["body_type"] = str(obj.get("type")).lower()
    for action in plan.get("actions") or []:
        if not isinstance(action, dict):
            continue
        action_type = str(action.get("type", "")).lower()
        if action_type not in {"set_initial_velocity", "set_velocity", "initial_velocity"}:
            continue
        name = action.get("object") or action.get("name") or action.get("target")
        if not name or str(name) not in by_name:
            continue
        obj = by_name[str(name)]
        if action.get("linear_velocity") is not None and obj.get("initial_linear_velocity") is None:
            obj["initial_linear_velocity"] = action["linear_velocity"]
        if action.get("angular_velocity") is not None and obj.get("initial_angular_velocity") is None:
            obj["initial_angular_velocity"] = action["angular_velocity"]
    return plan


def _object_center_xy(digest: dict[str, Any], name: str | None) -> tuple[float, float] | None:
    if not name:
        return None
    for obj in digest.get("objects") or []:
        if str(obj.get("name") or obj.get("final_3d_object_name")) != str(name):
            continue
        center = obj.get("bbox_3d_center") or obj.get("pose_translation")
        if isinstance(center, list) and len(center) >= 2:
            try:
                return float(center[0]), float(center[1])
            except (TypeError, ValueError):
                return None
    return None


def _find_plan_object(plan: dict[str, Any], name: str | None) -> dict[str, Any] | None:
    if not name:
        return None
    for obj in plan.get("objects") or []:
        if isinstance(obj, dict) and str(obj.get("name")) == str(name):
            return obj
    return None


def _first_object_with_role(plan: dict[str, Any], role: str) -> str | None:
    for obj in plan.get("objects") or []:
        if isinstance(obj, dict) and str(obj.get("role", "")).lower() == role:
            name = obj.get("name")
            if name:
                return str(name)
    return None


def apply_direction_guard(scene: dict[str, Any], plan: dict[str, Any], digest: dict[str, Any]) -> dict[str, Any]:
    task_roles = scene.get("task_roles") if isinstance(scene.get("task_roles"), dict) else {}
    active_name = task_roles.get("active_object") or _first_object_with_role(plan, "active")
    target_name = task_roles.get("target_object") or _first_object_with_role(plan, "target")
    active_obj = _find_plan_object(plan, active_name)
    if active_obj is None or not target_name:
        return plan

    active_xy = _object_center_xy(digest, active_name)
    target_xy = _object_center_xy(digest, target_name)
    velocity = active_obj.get("initial_linear_velocity")
    if active_xy is None or target_xy is None or not isinstance(velocity, list) or len(velocity) < 2:
        return plan

    dx = target_xy[0] - active_xy[0]
    dy = target_xy[1] - active_xy[1]
    target_norm = math.hypot(dx, dy)
    try:
        vx = float(velocity[0])
        vy = float(velocity[1])
    except (TypeError, ValueError):
        return plan
    speed_xy = math.hypot(vx, vy)
    if target_norm <= 1e-8 or speed_xy <= 1e-8:
        return plan

    direction = (dx / target_norm, dy / target_norm)
    projection = vx * direction[0] + vy * direction[1]
    if projection >= 0.0:
        return plan

    corrected = [direction[0] * speed_xy, direction[1] * speed_xy, *velocity[2:]]
    active_obj["initial_linear_velocity"] = corrected
    diagnostics = plan.setdefault("diagnostics", {})
    diagnostics["direction_guard"] = {
        "status": "corrected",
        "active_object": active_name,
        "target_object": target_name,
        "target_direction_xy": [direction[0], direction[1]],
        "original_initial_linear_velocity": velocity,
        "corrected_initial_linear_velocity": corrected,
        "original_projection_on_target": projection,
    }
    print(
        "Direction guard corrected initial_linear_velocity for "
        f"{active_name} toward {target_name}: {velocity} -> {corrected}",
        flush=True,
    )
    return plan


def run_qwen_single_mode(args: argparse.Namespace, session_dir: Path, final_manifest: Path, work_dir: Path, digest_path: Path, digest: dict[str, Any]) -> None:
    prompt = build_qwen_single_prompt(session_dir, args.goal, work_dir, final_manifest, digest_path, digest, args.no_run)
    prompt_path = work_dir / "api_prompt.txt"
    response_path = work_dir / "api_response.txt"
    log_path = work_dir / "api_call.json"
    prompt_path.write_text(prompt, encoding="utf-8")
    print(f"Work directory: {work_dir}")
    print(f"Prompt: {prompt_path}")
    raw = run_qwen_prompt(args, prompt, collect_image_inputs(digest, args.max_crop_images), response_path, log_path)
    scene = raw.get("scene_description")
    plan = raw.get("simulation_plan")
    summary = raw.get("summary_markdown") or raw.get("summary") or ""
    if not isinstance(scene, dict) or not isinstance(plan, dict):
        raise RuntimeError(f"OpenAI-compatible single mode must return scene_description and simulation_plan objects: {response_path}")
    plan = normalize_qwen_simulation_plan(plan)
    plan = apply_direction_guard(scene, plan, digest)
    write_json(work_dir / "scene_description.json", scene)
    write_json(work_dir / "simulation_plan.json", plan)
    (work_dir / "codex_summary.md").write_text(str(summary).strip() + "\n", encoding="utf-8")
    load_json(work_dir / "scene_description.json")
    load_json(work_dir / "simulation_plan.json")
    if not args.no_run:
        run_command(simulation_runner_cmd(args, final_manifest, work_dir), cwd=WORKFLOW_ROOT)


def run_single_mode(args: argparse.Namespace, session_dir: Path, final_manifest: Path, work_dir: Path, digest_path: Path, digest: dict[str, Any]) -> None:
    if uses_openai_compatible_backend(args.llm_backend):
        run_qwen_single_mode(args, session_dir, final_manifest, work_dir, digest_path, digest)
        return
    prompt = build_single_prompt(session_dir, args.goal, work_dir, final_manifest, digest_path, args.no_run)
    prompt_path = work_dir / "codex_prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    last_message = args.output_last_message or (work_dir / "codex_last_message.txt")
    cmd = codex_base_cmd(args, last_message, collect_image_inputs(digest, args.max_crop_images), args.planner_effort)
    print(f"Work directory: {work_dir}")
    print(f"Prompt: {prompt_path}")
    run_command(cmd, cwd=WORKFLOW_ROOT, env=codex_env(args), input_text=prompt)
    scene = load_json(work_dir / "scene_description.json")
    plan = load_json(work_dir / "simulation_plan.json")
    guarded_plan = apply_direction_guard(scene, plan, digest)
    if guarded_plan is not plan or guarded_plan.get("diagnostics", {}).get("direction_guard"):
        write_json(work_dir / "simulation_plan.json", guarded_plan)
    if not args.no_run:
        run_command(simulation_runner_cmd(args, final_manifest, work_dir), cwd=WORKFLOW_ROOT)


def object_list_for_prompt(digest: dict[str, Any]) -> str:
    cards = []
    for obj in digest.get("objects") or []:
        cards.append(
            {
                "name": obj.get("name"),
                "mask_id": obj.get("mask_id"),
                "mask_name": obj.get("mask_name"),
                "semantic_label": obj.get("semantic_label"),
                "description": obj.get("description"),
                "vlm_confidence": obj.get("vlm_confidence"),
                "vlm_visibility": obj.get("vlm_visibility"),
                "vlm_support_status": obj.get("vlm_support_status"),
                "crop_rgb": obj.get("crop_rgb"),
                "bbox_2d_xyxy": obj.get("bbox_2d_xyxy"),
                "bbox_3d_center": obj.get("bbox_3d_center"),
                "bbox_3d_extent": obj.get("bbox_3d_extent"),
            }
        )
    return json.dumps(cards, ensure_ascii=False, indent=2)


def planner_prompt(goal: str, digest_path: Path, digest: dict[str, Any], output_path: Path) -> str:
    return f"""You are the task planner for simulator_assistance.

Read the scene digest and attached overlay/crop images. Write exactly one JSON file at:
{output_path}

Do not edit any other file and do not run SAPIEN or Blender.

User goal:
{goal}

Scene digest:
{digest_path}

Scene digest content:
{json_for_prompt(digest)}

Object cards:
{object_list_for_prompt(digest)}

Ground plane is fixed at z=0, Z-up. Do not infer ground.

Output schema:
{{
  "schema": "simulator_task_spec.v1",
  "goal": "...",
  "task_type": "fall_and_hit | push | drop | collide | other",
  "active_object": "exact final_3d_object_name",
  "target_object": "exact final_3d_object_name or null",
  "passive_static_objects": ["exact names"],
  "goal_interpretation": "short concrete description",
  "direction_hint": [x, y, z],
  "success_conditions": ["..."],
  "assumptions": ["..."]
}}
"""


def semantics_prompt(goal: str, digest_path: Path, digest: dict[str, Any], output_path: Path) -> str:
    return f"""You are the object semantics agent for simulator_assistance.

Inspect the attached object crop images and scene overlay. Upstream scene_graph labels in the object cards are useful hints; correct them only when the crop/overlay clearly contradicts them. Write exactly one JSON file at:
{output_path}

Do not edit any other file and do not run SAPIEN or Blender.

User goal:
{goal}

Scene digest:
{digest_path}

Scene digest content:
{json_for_prompt(digest)}

Object cards:
{object_list_for_prompt(digest)}

Output schema:
{{
  "schema": "simulator_object_semantics.v1",
  "objects": [
    {{
      "name": "exact final_3d_object_name",
      "mask_id": 1,
      "semantic_label": "short object name",
      "visual_description": "one sentence",
      "material_hint": "plastic | metal | paper | fabric | wood | leather | mixed | unknown",
      "rigid": true,
      "hollow_or_light": true,
      "mass_hint": "light | medium | heavy | unknown",
      "confidence": 0.0,
      "assumptions": ["..."]
    }}
  ]
}}
"""


def physics_prompt(goal: str, digest_path: Path, task_spec: Path, semantics: Path, output_path: Path) -> str:
    return f"""You are the physics-parameter agent for simulator_assistance.

Use the scene digest, task spec, and object semantics to choose executable rigid-body parameters. Write exactly one JSON file at:
{output_path}

Do not edit any other file and do not run SAPIEN or Blender.

User goal:
{goal}

Scene digest:
{digest_path}
Scene digest content:
{json_file_for_prompt(digest_path)}
Task spec:
{task_spec}
Task spec content:
{json_file_for_prompt(task_spec)}
Object semantics:
{semantics}
Object semantics content:
{json_file_for_prompt(semantics)}

Hard constraints:
- Ground plane is fixed z=0. Use engine.ground_z = 0.0.
- Only use fields supported by simulation_plan.json and the SAPIEN runner.
- Preserve exact object names.
- Treat upstream support relations in scene_digest.json as hints for deciding which objects should remain static/supporting.

Output schema:
{{
  "schema": "simulator_physics_params.v1",
  "engine": {{"steps": 360, "timestep": 0.005, "sample_every": 2, "gravity_z": -9.81, "ground_z": 0.0, "physx_gpu": false}},
  "defaults": {{"unmentioned_body_type": "static", "static_friction": 0.8, "dynamic_friction": 0.6, "restitution": 0.05, "density": 700.0, "linear_damping": 0.02, "angular_damping": 0.02}},
  "objects": [
    {{"name": "exact object name", "body_type": "dynamic | static | kinematic", "density": 700.0, "linear_damping": 0.02, "angular_damping": 0.02}}
  ],
  "assumptions": ["..."]
}}
"""


def initial_conditions_prompt(goal: str, digest_path: Path, task_spec: Path, semantics: Path, output_path: Path) -> str:
    return f"""You are the initial-condition agent for simulator_assistance.

Use the scene digest, task spec, and object semantics to choose initial pose/velocity conditions. Write exactly one JSON file at:
{output_path}

Do not edit any other file and do not run SAPIEN or Blender.

User goal:
{goal}

Scene digest:
{digest_path}
Scene digest content:
{json_file_for_prompt(digest_path)}
Task spec:
{task_spec}
Task spec content:
{json_file_for_prompt(task_spec)}
Object semantics:
{semantics}
Object semantics content:
{json_file_for_prompt(semantics)}

Hard constraints:
- Ground plane is fixed z=0 and the environment is Z-up.
- Use exact object names.
- Use world-space linear/angular velocities.
- `initial_pose_offset.translation` is a world-space offset in meters.
- `initial_pose_offset.rotation_euler_deg` is an actor-frame xyz rotation.
- Use scene_digest.json support/contact hints to avoid pushing active objects through supporting objects.

Output schema:
{{
  "schema": "simulator_initial_conditions.v1",
  "objects": [
    {{
      "name": "exact object name",
      "initial_pose_offset": {{"translation": [0, 0, 0], "rotation_euler_deg": [0, 0, 0], "rotation_order": "xyz"}},
      "initial_linear_velocity": [0, 0, 0],
      "initial_angular_velocity": [0, 0, 0],
      "events": []
    }}
  ],
  "assumptions": ["..."]
}}
"""


def start_codex_job(args: argparse.Namespace, job: CodexJob, job_dir: Path) -> tuple[subprocess.Popen[bytes], Any]:
    job_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = job_dir / f"{job.name}_prompt.txt"
    prompt_path.write_text(job.prompt, encoding="utf-8")
    last_message = job_dir / f"{job.name}_last_message.txt"
    log_path = job_dir / f"{job.name}.log"
    cmd = codex_base_cmd(args, last_message, job.images, job.effort)
    print(f"[{job.name}] + " + " ".join(cmd), flush=True)
    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(WORKFLOW_ROOT),
        env=codex_env(args),
        stdin=subprocess.PIPE,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdin is not None
    proc.stdin.write(job.prompt)
    proc.stdin.close()
    return proc, log_file


def wait_jobs(jobs: list[tuple[CodexJob, subprocess.Popen[bytes], Any]]) -> None:
    failures = []
    for job, proc, log_file in jobs:
        code = proc.wait()
        log_file.close()
        if code != 0:
            failures.append((job.name, code))
        elif not job.output_path.is_file():
            failures.append((job.name, "missing output"))
    if failures:
        details = ", ".join(f"{name}={code}" for name, code in failures)
        raise RuntimeError(f"Codex subtask failed: {details}")


def run_qwen_job(args: argparse.Namespace, job: CodexJob, job_dir: Path) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = job_dir / f"{job.name}_prompt.txt"
    response_path = job_dir / f"{job.name}_response.txt"
    log_path = job_dir / f"{job.name}.log"
    prompt_path.write_text(job.prompt, encoding="utf-8")
    print(f"[{job.name}] api model={args.api_model} images={len(job.images)} output={job.output_path}", flush=True)
    raw = run_qwen_prompt(args, job.prompt, job.images, response_path, log_path)
    write_json(job.output_path, raw)
    load_json(job.output_path)


def run_qwen_jobs(args: argparse.Namespace, jobs: list[CodexJob], agents_dir: Path) -> None:
    failures: list[str] = []
    max_workers = max(1, min(len(jobs), 4))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {
            executor.submit(run_qwen_job, args, job, agents_dir / job.name): job
            for job in jobs
        }
        for future in concurrent.futures.as_completed(future_to_job):
            job = future_to_job[future]
            try:
                future.result()
            except Exception as exc:
                failures.append(f"{job.name}: {exc}")
    missing = [job.name for job in jobs if not job.output_path.is_file()]
    failures.extend(f"{name}: missing output" for name in missing)
    if failures:
        raise RuntimeError("OpenAI-compatible API subtask failed: " + "; ".join(failures))


def run_parallel_mode(args: argparse.Namespace, work_dir: Path, final_manifest: Path, digest_path: Path, digest: dict[str, Any]) -> None:
    agents_dir = work_dir / "agents"
    images = collect_image_inputs(digest, args.max_crop_images)

    task_spec = agents_dir / "planner" / "task_spec.json"
    object_semantics = agents_dir / "semantics" / "object_semantics.json"
    physics_params = agents_dir / "physics" / "physics_params.json"
    initial_conditions = agents_dir / "initial_conditions" / "initial_conditions.json"

    phase1 = [
        CodexJob("planner", planner_prompt(args.goal, digest_path, digest, task_spec), task_spec, images, args.planner_effort),
        CodexJob("semantics", semantics_prompt(args.goal, digest_path, digest, object_semantics), object_semantics, images, args.agent_effort),
    ]
    if uses_openai_compatible_backend(args.llm_backend):
        run_qwen_jobs(args, phase1, agents_dir)
    else:
        running1 = [(job, *start_codex_job(args, job, agents_dir / job.name)) for job in phase1]
        wait_jobs(running1)

    phase2 = [
        CodexJob("physics", physics_prompt(args.goal, digest_path, task_spec, object_semantics, physics_params), physics_params, [], args.agent_effort),
        CodexJob("initial_conditions", initial_conditions_prompt(args.goal, digest_path, task_spec, object_semantics, initial_conditions), initial_conditions, [], args.agent_effort),
    ]
    if uses_openai_compatible_backend(args.llm_backend):
        run_qwen_jobs(args, phase2, agents_dir)
    else:
        running2 = [(job, *start_codex_job(args, job, agents_dir / job.name)) for job in phase2]
        wait_jobs(running2)

    run_command(
        [
            "python3",
            str(SKILL_DIR / "scripts" / "merge_agent_outputs.py"),
            "--goal",
            args.goal,
            "--scene-digest",
            str(digest_path),
            "--task-spec",
            str(task_spec),
            "--object-semantics",
            str(object_semantics),
            "--physics-params",
            str(physics_params),
            "--initial-conditions",
            str(initial_conditions),
            "--scene-description-output",
            str(work_dir / "scene_description.json"),
            "--plan-output",
            str(work_dir / "simulation_plan.json"),
            "--summary-output",
            str(work_dir / "codex_summary.md"),
            *(("--no-run",) if args.no_run else ()),
        ],
        cwd=WORKFLOW_ROOT,
    )
    scene = load_json(work_dir / "scene_description.json")
    plan = load_json(work_dir / "simulation_plan.json")
    guarded_plan = apply_direction_guard(scene, plan, digest)
    if guarded_plan is not plan or guarded_plan.get("diagnostics", {}).get("direction_guard"):
        write_json(work_dir / "simulation_plan.json", guarded_plan)

    if not args.no_run:
        run_command(simulation_runner_cmd(args, final_manifest, work_dir), cwd=WORKFLOW_ROOT)


def effective_mode(args: argparse.Namespace, digest: dict[str, Any]) -> str:
    if args.mode != "auto":
        return args.mode
    return "parallel" if len(digest.get("objects") or []) >= 3 else "single"


def main() -> int:
    args = parse_args()
    session_dir = resolve_session(args.session)
    final_manifest = session_dir / "results" / "final_scene_manifest.json"
    if not final_manifest.is_file():
        raise FileNotFoundError(final_manifest)

    work_dir = resolve_work_dir(args, session_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    digest_path = build_scene_digest(final_manifest, work_dir, reuse=not args.force_scene_digest)
    digest = load_json(digest_path)
    mode = effective_mode(args, digest)

    print(f"Work directory: {work_dir}")
    print(f"Scene digest: {digest_path}")
    print(f"Planning mode: {mode}")
    print(f"LLM backend: {args.llm_backend}")

    if mode == "single":
        run_single_mode(args, session_dir, final_manifest, work_dir, digest_path, digest)
    else:
        run_parallel_mode(args, work_dir, final_manifest, digest_path, digest)

    summary = {
        "schema": "simulator_assistance_invocation.v1",
        "status": "ok",
        "mode": mode,
        "llm_backend": args.llm_backend,
        "llm_model": args.api_model if uses_openai_compatible_backend(args.llm_backend) else args.model,
        "session": str(session_dir),
        "goal": args.goal,
        "work_dir": str(work_dir),
        "scene_digest": str(digest_path),
        "outputs": {
            "scene_description": str(work_dir / "scene_description.json"),
            "simulation_plan": str(work_dir / "simulation_plan.json"),
            "summary": str(work_dir / "codex_summary.md"),
            "run_report": optional_existing(work_dir / "run" / "run_report.json"),
            "animation_blend": optional_existing(work_dir / "run" / "simulation_animation.blend"),
            "camera_pose_optimization": optional_existing(work_dir / "run" / "camera_pose_optimization.json"),
            "camera_pose_optimization_preview": optional_existing(work_dir / "run" / "camera_pose_optimization_preview.png"),
            "camera_pose_optimization_mesh": optional_existing(work_dir / "run" / "camera_pose_optimization_mesh.npz"),
            "object_pose_optimization": optional_existing(work_dir / "run" / "object_pose_optimization.json"),
            "object_pose_optimization_mesh": optional_existing(work_dir / "run" / "object_pose_optimization_mesh.npz"),
            "object_pose_optimization_previews": optional_existing_path(work_dir / "run" / "object_pose_optimization_previews"),
            "object_pose_optimized_blend": optional_existing(work_dir / "run" / "object_pose_optimized.blend"),
            "object_pose_apply_report": optional_existing(work_dir / "run" / "object_pose_optimized.object_pose_apply_report.json"),
            "original_camera_render": optional_existing(work_dir / "run" / "simulation_animation_original_view.mp4"),
            "original_camera_render_report": optional_existing(work_dir / "run" / "simulation_animation_original_view_render.json"),
            "original_camera_blend": optional_existing(work_dir / "run" / "simulation_animation_original_view_camera.blend"),
        },
    }
    write_json(work_dir / "invocation.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
