from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

SAM3D_ROOT = Path(os.environ.get("SAM3D_ROOT", str(Path(__file__).resolve().parents[1] / "third_party" / "SAM3D")))
WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
SAM3D_SCRIPT = WORKFLOW_ROOT / "system_infer" / "sam3d_cached_runner.py"
SAM3D_ENV_DIR = Path(
    os.environ.get("SAM3D_ENV_DIR", os.environ.get("CONDA_PREFIX", sys.prefix))
)
SAM3D_DEFAULT_CONFIG = SAM3D_ROOT / "runtime_configs" / "pipeline_hf_cache_moge_modelpt.yaml"
DEFAULT_TIMEOUT_SECONDS = 14_400


def _parse_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return safe[:120] or "scene"


def _numeric_mask_paths(scene_dir: Path) -> List[Path]:
    paths: List[Path] = []
    idx = 0
    while True:
        path = scene_dir / f"{idx}.png"
        if not path.is_file():
            break
        paths.append(path)
        idx += 1
    return paths


def _resolve_python(env_dir: Path) -> str:
    configured = os.environ.get("SAM3D_PYTHON")
    if configured:
        return configured

    env_python = env_dir / "bin" / "python"
    if env_python.exists():
        return str(env_python)
    return sys.executable


def _last_scene_result(output_dir: Path, scene_name: str, stdout: str) -> Optional[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    log_dir = output_dir / "logs"
    if log_dir.is_dir():
        for report_path in sorted(log_dir.glob("worker_*.jsonl")):
            for raw in report_path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item.get("scene_id") == scene_name:
                    results.append(item)

    for raw in stdout.splitlines():
        if "[RESULT]" not in raw:
            continue
        payload = raw.split("[RESULT]", 1)[1].strip()
        try:
            item = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if item.get("scene_id") == scene_name:
            results.append(item)

    return results[-1] if results else None


def _report_paths(output_dir: Path) -> List[str]:
    log_dir = output_dir / "logs"
    if not log_dir.is_dir():
        return []
    return [str(path) for path in sorted(log_dir.glob("worker_*.jsonl"))]


def _read_json(path: Path) -> Dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _tail(text: str, max_chars: int = 4000) -> str:
    text = text or ""
    return text if len(text) <= max_chars else text[-max_chars:]


class SAM3DInfer:
    """Thin wrapper around the real SAM3D 3D-FUTURE CLI.

    The production input directory is the same structure used by
    SAM3D_ROOT/3d_future_test.py:
      scene_dir/image.png
      scene_dir/0.png, scene_dir/1.png, ...
    """

    def __init__(
        self,
        config: Any = None,
        compile_model: bool = False,
        compile: bool | None = None,
        gpu: Any = None,
        root: Any = None,
        env_dir: Any = None,
        timeout_seconds: Any = None,
        require_pointmap_cache: Any = None,
        **_: Any,
    ) -> None:
        self.root = Path(root) if root is not None else SAM3D_ROOT
        self.script = WORKFLOW_ROOT / "system_infer" / "sam3d_cached_runner.py"
        self.env_dir = Path(env_dir) if env_dir is not None else SAM3D_ENV_DIR
        self.python = _resolve_python(self.env_dir)
        self.config = Path(
            config
            or os.environ.get("SAM3D_CONFIG")
            or self.root / "runtime_configs" / "pipeline_hf_cache_moge_modelpt.yaml"
        )
        self.gpu = None if gpu is None else str(gpu)
        self.compile_model = bool(compile_model if compile is None else compile)
        self.timeout_seconds = int(
            timeout_seconds
            if timeout_seconds is not None
            else os.environ.get("SAM3D_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
        )
        self.require_pointmap_cache = _parse_bool(
            require_pointmap_cache
            if require_pointmap_cache is not None
            else os.environ.get("SAM3D_REQUIRE_POINTMAP_CACHE")
        )

    def _validate(self, scene_dir: Path) -> List[Path]:
        if not self.script.is_file():
            raise FileNotFoundError(f"SAM3D CLI not found: {self.script}")
        if not self.config.is_file():
            raise FileNotFoundError(f"SAM3D config not found: {self.config}")

        image_path = scene_dir / "image.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"SAM3D image not found: {image_path}")

        mask_paths = _numeric_mask_paths(scene_dir)
        if not mask_paths:
            raise FileNotFoundError(
                f"No SAM3D masks found under {scene_dir}; expected 0.png, 1.png, ..."
            )
        return mask_paths

    def _build_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        env.setdefault("CONDA_PREFIX", str(self.env_dir))
        env["PATH"] = f"{self.env_dir / 'bin'}:{env.get('PATH', '')}"
        env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sam3d")
        env.setdefault("LIDRA_SKIP_INIT", "true")
        env["PYTHONUNBUFFERED"] = "1"
        env["SAM3D_RANK"] = "0"
        env["SAM3D_WORLD_SIZE"] = "1"
        if self.gpu is not None:
            env["SAM3D_GPU"] = self.gpu
            env["CUDA_VISIBLE_DEVICES"] = self.gpu
        return env

    def _run_cli(
        self,
        scene_dir: Path,
        run_output_dir: Path,
        seed: int,
        require_pointmap_cache: bool,
    ) -> subprocess.CompletedProcess[str]:
        cmd = [
            self.python,
            str(self.script),
            "--input_root",
            str(scene_dir.parent),
            "--output_dir",
            str(run_output_dir),
            "--config",
            str(self.config),
            "--rank",
            "0",
            "--world_size",
            "1",
            "--seed",
            str(seed),
            "--scene_ids",
            scene_dir.name,
            "--overwrite",
            "--fail_fast",
        ]
        if self.gpu is not None:
            cmd.extend(["--gpu", self.gpu])
        if self.compile_model:
            cmd.append("--compile")

        return subprocess.run(
            cmd,
            cwd=str(WORKFLOW_ROOT),
            env=self._build_env(),
            text=True,
            capture_output=True,
            timeout=self.timeout_seconds,
            check=False,
        )

    def generate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        scene_dir = Path(request["scene_dir"]).resolve()
        output_dir = Path(request["output_dir"]).resolve()
        scene_id = str(request.get("scene_id") or scene_dir.name)
        seed = int(request.get("seed", 42))

        mask_paths = self._validate(scene_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        run_id = f"{_safe_name(scene_dir.name)}_{int(time.time())}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        run_output_dir = output_dir / "_sam3d_cli_runs" / run_id
        run_output_dir.mkdir(parents=True, exist_ok=True)

        pointmap_path = scene_dir / "image.npz"
        requested_pointmap = _parse_bool(request.get("require_pointmap_cache"))
        require_pointmap_cache = (
            requested_pointmap
            if requested_pointmap is not None
            else self.require_pointmap_cache
            if self.require_pointmap_cache is not None
            else pointmap_path.is_file()
        )

        try:
            proc = self._run_cli(
                scene_dir=scene_dir,
                run_output_dir=run_output_dir,
                seed=seed,
                require_pointmap_cache=bool(require_pointmap_cache),
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"SAM3D CLI timed out after {self.timeout_seconds}s for {scene_dir}"
            ) from exc

        log_stem = f"sam3d_{_safe_name(scene_id)}_{run_id}"
        stdout_log = logs_dir / f"{log_stem}.stdout.log"
        stderr_log = logs_dir / f"{log_stem}.stderr.log"
        stdout_log.write_text(proc.stdout or "", encoding="utf-8")
        stderr_log.write_text(proc.stderr or "", encoding="utf-8")

        result = _last_scene_result(run_output_dir, scene_dir.name, proc.stdout or "")
        if proc.returncode != 0:
            detail = result.get("error") if result else (proc.stderr or proc.stdout).strip()
            raise RuntimeError(
                f"SAM3D CLI failed for {scene_dir.name} with exit code {proc.returncode}: {detail}\n"
                f"stdout_log={stdout_log}\nstderr_log={stderr_log}\n"
                f"stdout_tail:\n{_tail(proc.stdout or '')}\n"
                f"stderr_tail:\n{_tail(proc.stderr or '')}"
            )
        if result is None:
            raise RuntimeError(
                f"SAM3D CLI finished without a [RESULT] entry for {scene_dir.name}; "
                f"see {stdout_log}\nstdout_tail:\n{_tail(proc.stdout or '')}\n"
                f"stderr_tail:\n{_tail(proc.stderr or '')}"
            )
        if result.get("status") != "ok":
            raise RuntimeError(
                f"SAM3D scene {scene_dir.name} status={result.get('status')}: {result.get('error')}\n"
                f"stdout_log={stdout_log}\nstderr_log={stderr_log}\n"
                f"stdout_tail:\n{_tail(proc.stdout or '')}\n"
                f"stderr_tail:\n{_tail(proc.stderr or '')}"
            )

        source_glb = Path(result.get("output_path") or run_output_dir / f"{scene_dir.name}.glb")
        if not source_glb.is_absolute():
            source_glb = (self.root / source_glb).resolve()
        if not source_glb.is_file():
            fallback = run_output_dir / f"{scene_dir.name}.glb"
            if fallback.is_file():
                source_glb = fallback
            else:
                candidates = sorted(run_output_dir.glob("*.glb"))
                if len(candidates) == 1:
                    source_glb = candidates[0]
                else:
                    raise FileNotFoundError(
                        f"SAM3D result GLB not found for {scene_dir.name}; "
                        f"expected {fallback}, candidates={candidates}"
                    )

        target_glb = output_dir / f"{scene_id}.glb"
        shutil.copy2(source_glb, target_glb)

        num_meshes = int(result.get("num_meshes") or 0)
        input_manifest_path = scene_dir / "mask_manifest.json"
        input_manifest = _read_json(input_manifest_path)
        metadata_path = output_dir / f"{scene_id}_sam3d_pose_metadata.json"
        metadata = {
            "scene_id": scene_id,
            "sam3d_scene_name": scene_dir.name,
            "input_scene_dir": str(scene_dir),
            "seed": seed,
            "num_masks": int(result.get("num_masks") or len(mask_paths)),
            "num_meshes": num_meshes,
            "l2c_applied_to_mesh": True,
            "postprocess": "none",
            "source_glb": str(source_glb),
            "glb": str(target_glb),
            "moge_cache": result.get("pointmap_cache"),
            "sam3d_root": str(self.root),
            "sam3d_script": str(self.script),
            "config": str(self.config),
            "python": self.python,
            "gpu": self.gpu,
            "compile": self.compile_model,
            "require_pointmap_cache": bool(require_pointmap_cache),
            "pointmap_cache": str(pointmap_path) if pointmap_path.is_file() else None,
            "sam3d_moge_cache": result.get("pointmap_cache"),
            "numeric_masks": [str(path) for path in mask_paths],
            "input_manifest": str(input_manifest_path) if input_manifest_path.is_file() else result.get("manifest_path"),
            "objects": result.get("objects") or [],
            "mask_to_3d_objects": [
                {
                    "mask_id": obj.get("mask_id"),
                    "mask_name": obj.get("mask_name"),
                    "sam3d_input_index": obj.get("sam3d_input_index"),
                    "object_name": obj.get("object_name"),
                    "status": obj.get("status"),
                }
                for obj in (result.get("objects") or [])
            ],
            "input_manifest_data": input_manifest,
            "cli_result": result,
            "logs": {
                "stdout": str(stdout_log),
                "stderr": str(stderr_log),
                "reports": _report_paths(run_output_dir),
                "run_output_dir": str(run_output_dir),
            },
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return {
            "status": "ok",
            "glb": str(target_glb),
            "pose_metadata": str(metadata_path),
            "moge_cache": result.get("pointmap_cache"),
            "num_meshes": num_meshes,
            "objects": result.get("objects") or [],
            "input_manifest": str(input_manifest_path) if input_manifest_path.is_file() else result.get("manifest_path"),
            "l2c_applied_to_mesh": True,
            "postprocess": "none",
        }
