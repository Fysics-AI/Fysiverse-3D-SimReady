#!/usr/bin/env python3
"""Interactive Gradio front end for the open-source scene pipeline."""
from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from functools import wraps
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

WORKFLOW_ROOT = Path(__file__).resolve().parent
if str(WORKFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_ROOT))

from pipeline_utils import (  # noqa: E402
    SESSIONS_ROOT,
    collect_existing,
    ensure_dir,
    find_free_port,
    label_ids_by_area,
    mask_to_label_array,
    now_session_id,
    palette,
    save_color_mask,
    save_label_mask,
    save_rgb_image,
    write_mask_crops_and_manifest,
)
from scripts.load_pipeline_config import load_exports_from_config  # noqa: E402
from system_infer.gsam2_infer import GSAM2Infer  # noqa: E402


DEFAULT_CONFIG = WORKFLOW_ROOT / "config" / "main.yaml"
ANNOTATION_CSS = ""


def gr_component(component_cls: Any, *args: Any, **kwargs: Any) -> Any:
    """Create Gradio components across minor API differences."""
    try:
        signature = inspect.signature(component_cls.__init__)
    except (TypeError, ValueError):
        return component_cls(*args, **kwargs)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return component_cls(*args, **kwargs)
    allowed = set(signature.parameters)
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in allowed}
    return component_cls(*args, **filtered_kwargs)


def console_log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def gradio_logged(fn: Any) -> Any:
    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        console_log(f"[gradio-callback:start] {fn.__qualname__}")
        try:
            result = fn(*args, **kwargs)
            console_log(f"[gradio-callback:ok] {fn.__qualname__}")
            return result
        except BaseException:
            console_log(f"[gradio-callback:error] {fn.__qualname__}")
            traceback.print_exc(file=sys.stderr)
            raise

    return wrapper


def apply_pipeline_config(config_path: Path | None) -> None:
    if config_path is None:
        return
    path = config_path.expanduser()
    if not path.is_absolute():
        path = (WORKFLOW_ROOT / path).resolve()
    if not path.is_file():
        return
    for key, value in load_exports_from_config(path).items():
        if value != "" or key not in os.environ:
            os.environ[key] = value
    os.environ["PIPELINE_CONFIG"] = str(path)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def env_str(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    return default if value in (None, "") else str(value)


def normalize_box(box: list[int] | tuple[int, ...], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return [
        int(np.clip(x1, 0, width - 1)),
        int(np.clip(y1, 0, height - 1)),
        int(np.clip(x2, 0, width - 1)),
        int(np.clip(y2, 0, height - 1)),
    ]


def select_event_xy(evt: gr.SelectData, image_path: str | Path) -> tuple[int, int, int, int, Any]:
    raw = getattr(evt, "index", None)
    if raw is None or len(raw) < 2:
        raise gr.Error(f"Invalid click coordinate: {raw}")
    width, height = Image.open(image_path).size
    x = int(np.clip(round(float(raw[0])), 0, width - 1))
    y = int(np.clip(round(float(raw[1])), 0, height - 1))
    return x, y, width, height, raw


def draw_annotation_overlay(
    image_path: str | Path,
    points: list[list[int]] | None = None,
    boxes: list[list[int]] | None = None,
    pending_box: list[int] | None = None,
) -> np.ndarray:
    base = Image.open(image_path).convert("RGB")
    image = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    width, height = image.size
    radius = max(10, min(width, height) // 120)
    line_width = max(3, radius // 4)
    for idx, box in enumerate(boxes or [], start=1):
        x1, y1, x2, y2 = normalize_box(box, width, height)
        color = (30, 144, 255, 255)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        draw.text((x1 + 3, max(0, y1 - radius)), f"B{idx}", fill=color)
    if pending_box:
        x, y = int(pending_box[0]), int(pending_box[1])
        color = (30, 144, 255, 255)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=line_width)
        draw.line((x - radius, y, x + radius, y), fill=color, width=line_width)
        draw.line((x, y - radius, x, y + radius), fill=color, width=line_width)
        draw.text((x + radius + 2, y - radius), "B start", fill=color)
    for idx, item in enumerate(points or [], start=1):
        x, y, label = int(item[0]), int(item[1]), int(item[2])
        color = (0, 220, 0, 255) if label == 1 else (255, 40, 40, 255)
        if label == 1:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=line_width)
            draw.line((x - radius, y, x + radius, y), fill=color, width=line_width)
            draw.line((x, y - radius, x, y + radius), fill=color, width=line_width)
        else:
            draw.line((x - radius, y - radius, x + radius, y + radius), fill=color, width=line_width)
            draw.line((x - radius, y + radius, x + radius, y - radius), fill=color, width=line_width)
        draw.text((x + radius + 2, y - radius), str(idx), fill=color)
    return np.asarray(image)


def overlay_update(state: dict[str, Any] | None) -> Any:
    if not state or not state.get("image_path"):
        return gr.update(value=None, visible=False)
    return gr.update(
        value=draw_annotation_overlay(
            state["image_path"],
            points=state.get("points") or [],
            boxes=state.get("boxes") or [],
            pending_box=state.get("pending_box"),
        ),
        visible=True,
    )


def draw_ground_prompt_image(
    image_path: str | Path,
    ground_points: list[list[int]] | None = None,
    ground_mask_path: str | Path | None = None,
) -> np.ndarray:
    base = Image.open(image_path).convert("RGB")
    image = base.convert("RGBA")
    width, height = image.size
    if ground_mask_path and Path(str(ground_mask_path)).is_file():
        mask = mask_to_label_array(ground_mask_path) > 0
        if mask.shape != (height, width):
            mask = np.asarray(
                Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize((width, height), Image.Resampling.NEAREST)
            ) > 0
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        color = Image.new("RGBA", image.size, (0, 210, 230, 105))
        alpha = Image.fromarray(mask.astype(np.uint8) * 105, mode="L")
        overlay = Image.composite(color, overlay, alpha)
        image = Image.alpha_composite(image, overlay)
    draw = ImageDraw.Draw(image)
    radius = max(10, min(width, height) // 110)
    line_width = max(3, radius // 4)
    for idx, item in enumerate(ground_points or [], start=1):
        x, y, label = int(item[0]), int(item[1]), int(item[2])
        color = (0, 210, 230, 255) if label == 1 else (255, 70, 180, 255)
        if label == 1:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=line_width)
            draw.line((x - radius, y, x + radius, y), fill=color, width=line_width)
            draw.line((x, y - radius, x, y + radius), fill=color, width=line_width)
        else:
            draw.line((x - radius, y - radius, x + radius, y + radius), fill=color, width=line_width)
            draw.line((x - radius, y + radius, x + radius, y - radius), fill=color, width=line_width)
        draw.text((x + radius + 2, y - radius), f"G{idx}", fill=color)
    return np.asarray(image.convert("RGB"))


def ground_prompt_update(state: dict[str, Any] | None) -> Any:
    if not state or not state.get("image_path"):
        return gr.update(value=None, visible=False)
    return gr.update(
        value=draw_ground_prompt_image(
            state["image_path"],
            ground_points=state.get("ground_points") or [],
            ground_mask_path=state.get("ground_mask_path"),
        ),
        visible=True,
    )


def write_combined_segmentation_preview(
    *,
    image_path: Path,
    object_label_path: Path | None,
    ground_mask_path: Path | None,
    output_path: Path,
) -> Path:
    base = Image.open(image_path).convert("RGB")
    image = base.convert("RGBA")
    width, height = image.size
    if object_label_path is not None and object_label_path.is_file():
        label = mask_to_label_array(object_label_path)
        if label.shape != (height, width):
            label = np.asarray(
                Image.fromarray(label.astype(np.uint16)).resize((width, height), Image.Resampling.NEAREST),
                dtype=np.int32,
            )
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        for idx, label_id in enumerate(label_ids_by_area(label, order="index")):
            mask = label == label_id
            color = palette(idx)
            color_img = Image.new("RGBA", image.size, (*color, 145))
            alpha = Image.fromarray(mask.astype(np.uint8) * 145, mode="L")
            overlay = Image.composite(color_img, overlay, alpha)
        image = Image.alpha_composite(image, overlay)
    if ground_mask_path is not None and ground_mask_path.is_file():
        ground = mask_to_label_array(ground_mask_path) > 0
        if ground.shape != (height, width):
            ground = np.asarray(
                Image.fromarray(ground.astype(np.uint8) * 255, mode="L").resize((width, height), Image.Resampling.NEAREST)
            ) > 0
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        ground_img = Image.new("RGBA", image.size, (0, 220, 235, 135))
        alpha = Image.fromarray(ground.astype(np.uint8) * 135, mode="L")
        overlay = Image.composite(ground_img, overlay, alpha)
        image = Image.alpha_composite(image, overlay)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path)
    return output_path


class InteractivePipeline:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.enable_scene_graph = env_bool("ENABLE_SCENE_GRAPH", True)
        self.scene_graph_threads: dict[str, threading.Thread] = {}
        self.scene_graph_lock = threading.Lock()
        self._gsam2: GSAM2Infer | None = None

    def gsam2(self) -> GSAM2Infer:
        if self._gsam2 is None:
            self._gsam2 = GSAM2Infer(gpu=os.environ.get("GSAM_GPU", "0"))
        return self._gsam2

    @staticmethod
    def read_json_file(path: str | Path | None) -> dict[str, Any] | None:
        if not path:
            return None
        p = Path(path)
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def scene_graph_path_for_state(self, state: dict[str, Any]) -> Path:
        return Path(state["session_dir"]) / "results" / "scene_graph.json"

    def scene_graph_visualization_path_for_state(self, state: dict[str, Any]) -> Path:
        return Path(state["session_dir"]) / "results" / "scene_graph.png"

    def scene_graph_status_text(self, state: dict[str, Any] | None) -> str:
        if not state:
            return "scene graph: no session"
        status = state.get("scene_graph_status") or "not_started"
        backend = env_str("SCENE_GRAPH_BACKEND", "openai_compatible")
        lines = [f"scene graph: {status}", f"backend: {backend}"]
        if status == "building":
            lines.append("building in background; preview will refresh automatically")
        elif status == "ok":
            lines.append("preview ready")
        elif status == "failed":
            lines.append("failed; see scene_graph_error.txt")
        return "\n".join(lines)

    def refresh_scene_graph_state(self, state: dict[str, Any] | None) -> bool:
        if not state:
            return False
        if not self.enable_scene_graph:
            state["scene_graph_status"] = "disabled"
            state["scene_graph_path"] = None
            return False
        path = Path(str(state.get("scene_graph_path") or self.scene_graph_path_for_state(state)))
        vis = Path(str(state.get("scene_graph_visualization") or self.scene_graph_visualization_path_for_state(state)))
        state["scene_graph_path"] = str(path)
        state["scene_graph_visualization"] = str(vis)
        session_id = str(state.get("session_id"))
        with self.scene_graph_lock:
            thread = self.scene_graph_threads.get(session_id)
        if path.is_file():
            graph = self.read_json_file(path)
            state["scene_graph_status"] = (graph or {}).get("status") or "invalid"
            return state["scene_graph_status"] == "ok"
        if isinstance(thread, threading.Thread) and thread.is_alive():
            state["scene_graph_status"] = "building"
            return False
        error_path = path.parent / "scene_graph_error.txt"
        if error_path.is_file():
            state["scene_graph_status"] = "failed"
        else:
            state["scene_graph_status"] = state.get("scene_graph_status") or "not_started"
        return False

    def pipeline_action_button_update(self, state: dict[str, Any] | None) -> Any:
        scene_ready = self.refresh_scene_graph_state(state)
        if scene_ready:
            return gr.update(value="Run 3D Inference", interactive=True)
        status = (state or {}).get("scene_graph_status") or "not_started"
        if status == "disabled":
            return gr.update(value="Run 3D Inference (scene graph disabled)", interactive=False)
        if status == "failed":
            return gr.update(value="Run 3D Inference (scene graph failed)", interactive=False)
        return gr.update(value="Run 3D Inference (waiting scene graph)", interactive=False)

    def background_button_update(self, state: dict[str, Any] | None) -> Any:
        if not state:
            return gr.update(value="Run 3DGS Background (after 3D inference)", interactive=False)
        final_manifest = Path(str(state.get("session_dir") or "")) / "results" / "final_scene_manifest.json"
        if final_manifest.is_file():
            return gr.update(value="Run 3DGS Background", interactive=True)
        return gr.update(value="Run 3DGS Background (after 3D inference)", interactive=False)

    def create_session(self, image: Any) -> tuple[dict[str, Any], str, str, Any]:
        if image is None:
            raise gr.Error("Upload an image first.")
        session_id = now_session_id()
        session_dir = ensure_dir(SESSIONS_ROOT / session_id)
        image_path = save_rgb_image(image, session_dir / "input" / "image.png")
        state = {
            "session_id": session_id,
            "session_dir": str(session_dir),
            "image_path": str(image_path),
            "label_mask_path": None,
            "color_mask_path": None,
            "scene_graph_path": None,
            "scene_graph_visualization": None,
            "scene_graph_status": "not_started",
            "points": [],
            "boxes": [],
            "ground_points": [],
            "pending_box": None,
            "annotation_mode": "box",
            "ground_mask_path": None,
            "combined_segmentation_preview": None,
        }
        return state, f"session={session_id}\nimage={image_path}", str(image_path), overlay_update(state)

    def create_session_from_upload(
        self, image: Any
    ) -> tuple[dict[str, Any], str, str, Any, Any, Any, Any, str, str, Any, Any, Any, Any, Any]:
        state, status, preview, overlay = self.create_session(image)
        return (
            state,
            status,
            preview,
            overlay,
            ground_prompt_update(state),
            gr.update(value=None),
            gr.update(value=None),
            "",
            self.scene_graph_status_text(state),
            gr.update(value=None),
            gr.update(value=None),
            self.pipeline_action_button_update(state),
            gr.update(value=None),
            self.background_button_update(state),
        )

    def set_annotation_mode(self, state: dict[str, Any], annotation_mode: str) -> tuple[dict[str, Any], Any, str]:
        if not state:
            return state, gr.update(value=None, visible=False), "Upload an image first."
        state["annotation_mode"] = annotation_mode
        if annotation_mode == "point":
            state["boxes"] = []
            state["pending_box"] = None
        elif annotation_mode == "box":
            state["points"] = []
        return state, overlay_update(state), f"annotation_mode={annotation_mode}; other annotations cleared"

    def add_annotation(
        self,
        state: dict[str, Any],
        image: Any,
        annotation_mode: str,
        point_type: str,
        evt: gr.SelectData,
    ) -> tuple[dict[str, Any], Any, str]:
        if not state:
            state, _, _, _ = self.create_session(image)
        x, y, width, height, raw_xy = select_event_xy(evt, state["image_path"])
        state["annotation_mode"] = annotation_mode
        coord_log = f"click raw={raw_xy} -> image_xy=({x}, {y}), image_size=({width}, {height})"
        if annotation_mode == "box":
            state["points"] = []
            boxes = list(state.get("boxes") or [])
            pending = state.get("pending_box")
            if pending:
                box = normalize_box([int(pending[0]), int(pending[1]), x, y], width, height)
                if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                    state["pending_box"] = None
                    return state, overlay_update(state), f"{coord_log}\nignored tiny box; boxes={boxes}"
                boxes.append(box)
                state["boxes"] = boxes
                state["pending_box"] = None
                return state, overlay_update(state), f"{coord_log}\nboxes={boxes}"
            state["pending_box"] = [x, y]
            return state, overlay_update(state), f"{coord_log}\nbox start={[x, y]}; click opposite corner"
        state["boxes"] = []
        state["pending_box"] = None
        points = list(state.get("points") or [])
        label = 0 if point_type == "background" else 1
        points.append([x, y, label])
        state["points"] = points
        return state, overlay_update(state), f"{coord_log}\npoints={points}"

    def clear_points(self, state: dict[str, Any]) -> tuple[dict[str, Any], Any, str]:
        if state:
            state["points"] = []
        return state, overlay_update(state), "points cleared"

    def clear_boxes(self, state: dict[str, Any]) -> tuple[dict[str, Any], Any, str]:
        if state:
            state["boxes"] = []
            state["pending_box"] = None
        return state, overlay_update(state), "boxes cleared"

    def add_ground_point(
        self,
        state: dict[str, Any],
        image: Any,
        ground_point_type: str,
        evt: gr.SelectData,
    ) -> tuple[dict[str, Any], Any, str]:
        if not state or not state.get("image_path"):
            raise gr.Error("Upload an image first.")
        x, y, _width, _height, raw = select_event_xy(evt, state["image_path"])
        label = 0 if ground_point_type == "background" else 1
        points = list(state.get("ground_points") or [])
        points.append([x, y, label])
        state["ground_points"] = points
        return state, ground_prompt_update(state), f"ground point raw={raw} mapped={[x, y, label]}\nground_points={points}"

    def clear_ground_points(self, state: dict[str, Any]) -> tuple[dict[str, Any], Any, str]:
        if state:
            state["ground_points"] = []
        return state, ground_prompt_update(state), "ground points cleared"

    def build_scene_graph_background(self, state: dict[str, Any], manifest_path: Path, request_id: str) -> None:
        output_path = self.scene_graph_path_for_state(state)
        visualization_path = self.scene_graph_visualization_path_for_state(state)
        log_path = output_path.parent / "scene_graph_stdout.log"
        cmd = [
            sys.executable,
            str(WORKFLOW_ROOT / "scripts" / "build_scene_graph_codex.py"),
            "--backend",
            env_str("SCENE_GRAPH_BACKEND", "openai_compatible"),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            "--log",
            str(log_path),
            "--visualization",
            str(visualization_path),
            "--timeout",
            str(env_int("SCENE_GRAPH_TIMEOUT", 300)),
            "--max-crop-images",
            str(env_int("SCENE_GRAPH_MAX_CROP_IMAGES", 40)),
            "--api-base-url",
            env_str("SCENE_GRAPH_API_BASE_URL", ""),
            "--api-model",
            env_str("SCENE_GRAPH_API_MODEL", ""),
            "--reasoning-effort",
            env_str("SCENE_GRAPH_REASONING_EFFORT", "inherit"),
        ]
        codex_bin = env_str("CODEX_BIN", "codex")
        if codex_bin:
            cmd.extend(["--codex-bin", codex_bin])
        model = env_str("SCENE_GRAPH_MODEL", "")
        if model:
            cmd.extend(["--model", model])
        if env_bool("SCENE_GRAPH_API_RETRY_WITHOUT_CROPS", True):
            cmd.append("--api-retry-without-crops")
        env = os.environ.copy()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(WORKFLOW_ROOT),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=env_int("SCENE_GRAPH_TIMEOUT", 300) + 30,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stdout[-4000:])
            with self.scene_graph_lock:
                is_latest = state.get("scene_graph_request_id") == request_id
            if is_latest:
                graph = self.read_json_file(output_path) or {}
                state["scene_graph_status"] = graph.get("status") or "ok"
        except Exception as exc:
            error_path = output_path.parent / "scene_graph_error.txt"
            error_path.write_text(str(exc), encoding="utf-8")
            with self.scene_graph_lock:
                is_latest = state.get("scene_graph_request_id") == request_id
            if is_latest:
                state["scene_graph_status"] = "failed"

    def start_scene_graph_async(self, state: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
        if not self.enable_scene_graph:
            state["scene_graph_status"] = "disabled"
            state["scene_graph_path"] = None
            return state
        output_path = self.scene_graph_path_for_state(state)
        visualization_path = self.scene_graph_visualization_path_for_state(state)
        state["scene_graph_path"] = str(output_path)
        state["scene_graph_visualization"] = str(visualization_path)
        state["scene_graph_status"] = "building"
        request_id = str(time.time_ns())
        state["scene_graph_request_id"] = request_id
        for stale_path in (output_path, visualization_path, output_path.parent / "scene_graph_error.txt"):
            try:
                stale_path.unlink(missing_ok=True)
            except Exception:
                pass
        thread = threading.Thread(
            target=self.build_scene_graph_background,
            args=(state, manifest_path, request_id),
            name=f"scene-graph-{state.get('session_id')}",
            daemon=True,
        )
        with self.scene_graph_lock:
            self.scene_graph_threads[str(state.get("session_id"))] = thread
            thread.start()
        return state

    def poll_scene_graph(self, state: dict[str, Any]) -> tuple[dict[str, Any], str, Any, list[str], Any]:
        if not state:
            return state, "scene graph: no session", gr.update(value=None), [], self.pipeline_action_button_update(state)
        self.refresh_scene_graph_state(state)
        path = Path(str(state.get("scene_graph_path") or self.scene_graph_path_for_state(state)))
        vis = Path(str(state.get("scene_graph_visualization") or self.scene_graph_visualization_path_for_state(state)))
        return (
            state,
            self.scene_graph_status_text(state),
            gr.update(value=str(vis) if vis.is_file() else None),
            collect_existing([path, vis]),
            self.pipeline_action_button_update(state),
        )

    def segment_ground_prompt(self, state: dict[str, Any]) -> tuple[Path, list[str], str]:
        ground_points = state.get("ground_points") or []
        if not ground_points:
            raise gr.Error("Click a ground point first.")
        session_dir = Path(state["session_dir"])
        ground_dir = ensure_dir(session_dir / "segmentation" / "ground")
        resp = self.gsam2().segment(
            {
                "image_path": state["image_path"],
                "out_dir": str(ground_dir),
                "points": ground_points,
                "manual_boxes": [],
                "text_prompt": "",
                "use_grounding": False,
                "multi_object": False,
                "box_index": 0,
            }
        )
        if resp.get("status") != "ok" or not resp.get("mask_path"):
            raise gr.Error("Ground segmentation produced no mask: " + json.dumps(resp, ensure_ascii=False))
        ground_label = (mask_to_label_array(resp["mask_path"]) > 0).astype(np.uint8)
        ground_mask_path = save_label_mask(ground_label, session_dir / "input" / "ground_mask.png")
        state["ground_mask_path"] = str(ground_mask_path)
        combined_path = write_combined_segmentation_preview(
            image_path=Path(state["image_path"]),
            object_label_path=Path(state["label_mask_path"]) if state.get("label_mask_path") else None,
            ground_mask_path=ground_mask_path,
            output_path=session_dir / "segmentation" / "combined_overlay.png",
        )
        state["combined_segmentation_preview"] = str(combined_path)
        files = collect_existing([resp.get("mask_path"), resp.get("overlay_path"), ground_mask_path, combined_path])
        return combined_path, files, f"{resp.get('message')}\nground_mask={ground_mask_path}"

    def run_segmentation(
        self, state: dict[str, Any], text_prompt: str
    ) -> tuple[dict[str, Any], str | None, str, list[str], str, Any, list[str], Any]:
        if not state:
            raise gr.Error("Upload an image first.")
        if state.get("pending_box"):
            raise gr.Error("Finish the current box by clicking the opposite corner, or clear boxes.")
        points = state.get("points") or []
        boxes = state.get("boxes") or []
        ground_points = state.get("ground_points") or []
        text_prompt = (text_prompt or "").strip()
        if points and boxes:
            raise gr.Error("Use either point prompts or box prompts, not both.")
        if not points and not boxes and not text_prompt and not ground_points:
            raise gr.Error("Add object prompts, a text prompt, or a ground point.")

        session_dir = Path(state["session_dir"])
        seg_dir = ensure_dir(session_dir / "segmentation")
        files: list[str] = []
        status_parts: list[str] = []
        preview_path: Path | None = None

        if points or boxes or text_prompt:
            use_grounding = bool(text_prompt) and not points and not boxes
            resp = self.gsam2().segment(
                {
                    "image_path": state["image_path"],
                    "out_dir": str(seg_dir / "objects"),
                    "points": points,
                    "manual_boxes": boxes,
                    "text_prompt": text_prompt,
                    "box_threshold": 0.35,
                    "text_threshold": 0.25,
                    "use_grounding": use_grounding,
                    "multi_object": True,
                    "box_index": 0,
                }
            )
            if resp.get("status") != "ok" or not resp.get("mask_path"):
                raise gr.Error("Object segmentation produced no mask: " + json.dumps(resp, ensure_ascii=False))
            label = mask_to_label_array(resp["mask_path"])
            label_path = save_label_mask(label, session_dir / "input" / "mask_label.png")
            color_path = save_color_mask(label, session_dir / "input" / "mask.png")
            state["label_mask_path"] = str(label_path)
            state["color_mask_path"] = str(color_path)
            manifest_path = session_dir / "results" / "final_scene_manifest.json"
            write_mask_crops_and_manifest(
                session_dir=session_dir,
                scene_id=state["session_id"],
                image_path=Path(state["image_path"]),
                label_path=label_path,
                manifest_path=manifest_path,
            )
            self.start_scene_graph_async(state, manifest_path)
            preview_path = Path(resp.get("overlay_path"))
            files.extend(collect_existing([resp.get("mask_path"), label_path, color_path, resp.get("overlay_path"), manifest_path]))
            status_parts.append(f"{resp.get('message')}\nlabel_mask={label_path}\nmanifest={manifest_path}")

        if ground_points:
            ground_preview_path, ground_files, ground_message = self.segment_ground_prompt(state)
            preview_path = ground_preview_path
            files.extend(ground_files)
            status_parts.append(ground_message)

        if state.get("ground_mask_path") and state.get("label_mask_path"):
            combined_path = write_combined_segmentation_preview(
                image_path=Path(state["image_path"]),
                object_label_path=Path(state["label_mask_path"]),
                ground_mask_path=Path(state["ground_mask_path"]),
                output_path=session_dir / "segmentation" / "combined_overlay.png",
            )
            state["combined_segmentation_preview"] = str(combined_path)
            preview_path = combined_path
            files.extend(collect_existing([combined_path]))

        files.extend(collect_existing([state.get("ground_mask_path"), state.get("scene_graph_path"), state.get("scene_graph_visualization")]))
        status = "\n\n".join(status_parts)
        status += f"\n\nscene_graph={state.get('scene_graph_path')} ({state.get('scene_graph_status')})"
        return (
            state,
            str(preview_path) if preview_path else None,
            status,
            sorted(set(files)),
            self.scene_graph_status_text(state),
            gr.update(value=None),
            collect_existing([state.get("scene_graph_path"), state.get("scene_graph_visualization")]),
            self.pipeline_action_button_update(state),
        )

    def use_uploaded_mask(
        self, state: dict[str, Any], mask_file: Any
    ) -> tuple[dict[str, Any], str, list[str], str, Any, list[str], Any]:
        if not state:
            raise gr.Error("Upload an image first.")
        if mask_file is None:
            raise gr.Error("No mask uploaded.")
        session_dir = Path(state["session_dir"])
        mask_path = Path(getattr(mask_file, "name", mask_file))
        label = mask_to_label_array(mask_path)
        image = Image.open(state["image_path"]).convert("RGB")
        if label.shape != (image.height, image.width):
            raise gr.Error(f"mask/image size mismatch: mask={label.shape[::-1]} image={image.size}")
        label_path = save_label_mask(label, session_dir / "input" / "mask_label.png")
        color_path = save_color_mask(label, session_dir / "input" / "mask.png")
        state["label_mask_path"] = str(label_path)
        state["color_mask_path"] = str(color_path)
        manifest_path = session_dir / "results" / "final_scene_manifest.json"
        write_mask_crops_and_manifest(
            session_dir=session_dir,
            scene_id=state["session_id"],
            image_path=Path(state["image_path"]),
            label_path=label_path,
            manifest_path=manifest_path,
        )
        self.start_scene_graph_async(state, manifest_path)
        return (
            state,
            f"mask loaded: {label_path}\nmanifest={manifest_path}\nscene_graph={state.get('scene_graph_path')} ({state.get('scene_graph_status')})",
            collect_existing([label_path, color_path, manifest_path, state.get("scene_graph_path")]),
            self.scene_graph_status_text(state),
            gr.update(value=None),
            collect_existing([state.get("scene_graph_path"), state.get("scene_graph_visualization")]),
            self.pipeline_action_button_update(state),
        )

    def await_scene_graph_if_needed(self, state: dict[str, Any]) -> Path:
        if not self.enable_scene_graph:
            raise gr.Error("Scene graph is disabled. Enable it in config/main.yaml for this Gradio flow.")
        path = Path(str(state.get("scene_graph_path") or self.scene_graph_path_for_state(state)))
        session_id = str(state.get("session_id"))
        with self.scene_graph_lock:
            thread = self.scene_graph_threads.get(session_id)
        wait_seconds = env_int("SCENE_GRAPH_WAIT_BEFORE_3D", 30)
        if isinstance(thread, threading.Thread) and thread.is_alive() and wait_seconds > 0:
            thread.join(timeout=float(wait_seconds))
        if isinstance(thread, threading.Thread) and thread.is_alive():
            state["scene_graph_status"] = "still_building"
            raise gr.Error(f"Scene graph is still building. Try again later: {path}")
        if path.is_file():
            graph = self.read_json_file(path)
            if graph and graph.get("status") == "ok":
                state["scene_graph_status"] = "ok"
                state["scene_graph_path"] = str(path)
                return path
            state["scene_graph_status"] = (graph or {}).get("status") or "invalid"
            raise gr.Error(f"Scene graph is invalid or failed: {path}\nstatus={state['scene_graph_status']}")
        error_path = path.parent / "scene_graph_error.txt"
        detail = error_path.read_text(encoding="utf-8") if error_path.is_file() else "no scene_graph_error.txt"
        state["scene_graph_status"] = "missing"
        raise gr.Error(f"Scene graph was not generated.\nexpected={path}\nerror={detail}")

    def run_models(self, state: dict[str, Any], models: list[str]) -> tuple[dict[str, Any], str, list[str], Any, Any, Any]:
        if not state:
            raise gr.Error("Upload an image first.")
        if not state.get("label_mask_path"):
            raise gr.Error("Run segmentation or upload a mask before 3D inference.")
        scene_graph_path = self.await_scene_graph_if_needed(state)
        session_dir = Path(state["session_dir"])
        cmd = [
            sys.executable,
            str(WORKFLOW_ROOT / "app.py"),
            "--config",
            str(self.config_path),
            "--image",
            str(state["image_path"]),
            "--mask",
            str(state["label_mask_path"]),
            "--scene-graph",
            str(scene_graph_path),
            "--session-id",
            str(state["session_id"]),
            "--session-dir",
            str(session_dir),
            "--skip-background-3dgs",
        ]
        if state.get("ground_mask_path"):
            cmd.extend(["--ground-mask", str(state["ground_mask_path"])])
        started = time.time()
        proc = subprocess.run(
            cmd,
            cwd=str(WORKFLOW_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        elapsed = time.time() - started
        log = f"[3D pipeline]\n\n{proc.stdout or ''}\n[exit={proc.returncode} elapsed={elapsed:.1f}s]"
        result_dir = session_dir / "results"
        files = collect_release_files(result_dir)
        scene_graph_vis = state.get("scene_graph_visualization") or str(self.scene_graph_visualization_path_for_state(state))
        diagnostic_blend = diagnostic_blend_for_result(result_dir)
        if proc.returncode != 0:
            raise gr.Error(log[-6000:])
        return (
            state,
            log,
            files,
            gr.update(value=scene_graph_vis if Path(scene_graph_vis).is_file() else None),
            diagnostic_blend,
            self.background_button_update(state),
        )

    def run_background_3dgs(self, state: dict[str, Any]) -> tuple[dict[str, Any], str, list[str], Any, Any]:
        if not state:
            raise gr.Error("Upload an image first.")
        session_dir = Path(str(state.get("session_dir") or ""))
        result_dir = session_dir / "results"
        final_manifest = result_dir / "final_scene_manifest.json"
        if not final_manifest.is_file():
            raise gr.Error("Run 3D Inference before 3DGS background reconstruction.")

        cmd = ["bash", str(WORKFLOW_ROOT / "background" / "run_session_background_3dgs.sh"), str(session_dir)]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = env_str("BACKGROUND_3DGS_GPU", env.get("CUDA_VISIBLE_DEVICES", "0"))
        env["ITERATIONS"] = env_str("BACKGROUND_3DGS_ITERATIONS", env.get("ITERATIONS", "300"))
        env["KSPLAT_COMPRESSION"] = env_str("BACKGROUND_3DGS_KSPLAT_COMPRESSION", env.get("KSPLAT_COMPRESSION", "2"))
        env["KSPLAT_ALPHA_THRESHOLD"] = env_str("BACKGROUND_3DGS_KSPLAT_ALPHA_THRESHOLD", env.get("KSPLAT_ALPHA_THRESHOLD", "5"))
        env["KSPLAT_SH_DEGREE"] = env_str("BACKGROUND_3DGS_KSPLAT_SH_DEGREE", env.get("KSPLAT_SH_DEGREE", "0"))
        timeout_s = env_int("BACKGROUND_3DGS_TIMEOUT", 7200)

        started = time.time()
        proc = subprocess.run(
            cmd,
            cwd=str(WORKFLOW_ROOT),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=max(1, timeout_s),
            check=False,
        )
        elapsed = time.time() - started
        log = f"[3DGS background]\n\n{proc.stdout or ''}\n[exit={proc.returncode} elapsed={elapsed:.1f}s]"
        if proc.returncode != 0:
            raise gr.Error(log[-6000:])

        ksplat = result_dir / "3dgs_bg" / "background.ksplat"
        scene_json = result_dir / "3dgs_bg" / "scene.json"
        manifest_json = result_dir / "3dgs_bg" / "manifest.json"
        if not ksplat.is_file() or not scene_json.is_file() or not manifest_json.is_file():
            raise gr.Error(f"3DGS finished but required outputs are missing:\n{ksplat}\n{scene_json}\n{manifest_json}")
        files = collect_release_files(result_dir)
        return state, log, files, str(ksplat), self.background_button_update(state)

    def ui_worker_state(self, state: dict[str, Any], selected_models: list[str] | None = None) -> tuple[str, Any, Any]:
        lines = [
            "gsam2: local in-process loader (initialized lazily)",
            "sam3d: 3D pipeline subprocess",
            f"scene graph: {(state or {}).get('scene_graph_status', 'no session')}",
        ]
        return "\n".join(lines), gr.update(interactive=True), self.pipeline_action_button_update(state)


def diagnostic_blend_for_result(result_dir: Path) -> Any:
    candidates = [
        result_dir / "pipeline_stage_diagnostic" / "pipeline_stage_diagnostic.blend",
        result_dir / "release_package" / "blends" / "pipeline_animation.blend",
    ]
    files = collect_existing(candidates)
    return files[0] if files else None


def collect_release_files(result_dir: Path) -> list[str]:
    release_dir = result_dir / "release_package"
    candidates: list[Path] = [
        result_dir / "final_scene_manifest.json",
        result_dir / "scene_graph.json",
        result_dir / "scene_graph.png",
        release_dir / "final_scene_manifest.json",
        release_dir / "blends" / "final_state.blend",
        release_dir / "blends" / "pipeline_animation.blend",
        release_dir / "glb" / "final_state.glb",
        release_dir / "web" / "animation_manifest.json",
        result_dir / "3dgs_bg" / "background.ksplat",
        result_dir / "3dgs_bg" / "scene.json",
        result_dir / "3dgs_bg" / "manifest.json",
        result_dir.parent / "input" / "background.png",
        release_dir / "background_3dgs" / "background.ksplat",
        release_dir / "background_3dgs" / "scene.json",
        release_dir / "background_3dgs" / "manifest.json",
        release_dir / "background_3dgs" / "background.png",
    ]
    for subdir in (release_dir / "glb" / "raw_objects", release_dir / "glb" / "web_objects"):
        if subdir.is_dir():
            candidates.extend(sorted(subdir.glob("*.glb")))
    return collect_existing(candidates)


def build_demo(app: InteractivePipeline) -> gr.Blocks:
    with gr.Blocks(title="Image to 3D Scene Pipeline") as demo:
        state = gr.State(value={})
        worker_timer = gr.Timer(value=5.0, active=False)
        scene_graph_timer = gr.Timer(value=2.0, active=True)
        gr.Markdown("## Image to 3D Scene Pipeline")
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr_component(gr.Image, type="numpy", label="Input image", height=460, show_download_button=False, show_fullscreen_button=False, sources=["upload", "clipboard"])
                annotation_overlay = gr_component(gr.Image, type="numpy", label="Annotation preview", height=260, format="png", image_mode="RGBA", visible=False, show_download_button=False, show_fullscreen_button=False)
                session_status = gr.Textbox(label="Session / Status", lines=8)
                annotation_mode = gr.Radio(["point", "box"], value="box", label="Annotation mode")
                point_type = gr.Radio(["foreground", "background"], value="foreground", label="Point type")
                with gr.Row():
                    clear_points_btn = gr.Button("Clear points")
                    clear_boxes_btn = gr.Button("Clear boxes")
                with gr.Accordion("Ground prompt", open=True):
                    ground_prompt_image = gr_component(gr.Image, type="numpy", label="Ground point prompt", height=260, show_download_button=False, show_fullscreen_button=False)
                    ground_point_type = gr.Radio(["foreground", "background"], value="foreground", label="Ground point type")
                    with gr.Row():
                        clear_ground_points_btn = gr.Button("Clear ground points")
                uploaded_mask = gr.File(label="Optional mask.png / label mask")
                use_mask_btn = gr.Button("Use Uploaded Mask")
            with gr.Column(scale=1):
                seg_preview = gr_component(gr.Image, type="filepath", label="Segmentation preview", height=460)
                seg_files = gr.File(label="Segmentation files", file_count="multiple")
        with gr.Row():
            with gr.Column(scale=1):
                scene_graph_status = gr.Textbox(label="Scene graph status", lines=4)
                scene_graph_files = gr.File(label="Scene graph files", file_count="multiple")
            with gr.Column(scale=2):
                scene_graph_preview = gr_component(gr.Image, type="filepath", label="Scene graph preview", height=360, show_download_button=True, show_fullscreen_button=True)
        with gr.Row():
            with gr.Column(scale=1):
                text_prompt = gr.Textbox(value="", label="Optional text prompt", placeholder="Optional. Used only when no points or boxes are selected.")
                segment_btn = gr.Button("Run Segmentation", variant="primary")
            with gr.Column(scale=1):
                models = gr.CheckboxGroup(["SAM3D"], value=["SAM3D"], label="3D models", visible=False)
                infer_btn = gr.Button("Run 3D Inference (waiting scene graph)", variant="primary", interactive=False)
        with gr.Row():
            output_log = gr.Textbox(label="Run log", lines=12)
            with gr.Column():
                pipeline_diagnostic_blend = gr.File(label="Pipeline diagnostic animation (.blend)", file_count="single")
                background_btn = gr.Button("Run 3DGS Background (after 3D inference)", variant="secondary", interactive=False)
                background_ksplat = gr.File(label="3DGS background output (background.ksplat)", file_count="single")
                output_files = gr.File(label="Output files", file_count="multiple")
        worker_btn = gr.Button("Check Worker Status")
        worker_status = gr.Textbox(label="Worker status", lines=8)

        input_image.upload(
            gradio_logged(app.create_session_from_upload),
            [input_image],
            [state, session_status, seg_preview, annotation_overlay, ground_prompt_image, seg_files, output_files, output_log, scene_graph_status, scene_graph_preview, scene_graph_files, infer_btn, background_ksplat, background_btn],
            queue=False,
        )
        annotation_mode.change(gradio_logged(app.set_annotation_mode), [state, annotation_mode], [state, annotation_overlay, session_status], queue=False)
        input_image.select(gradio_logged(app.add_annotation), [state, input_image, annotation_mode, point_type], [state, annotation_overlay, session_status], queue=False)
        clear_points_btn.click(gradio_logged(app.clear_points), [state], [state, annotation_overlay, session_status], queue=False)
        clear_boxes_btn.click(gradio_logged(app.clear_boxes), [state], [state, annotation_overlay, session_status], queue=False)
        ground_prompt_image.select(gradio_logged(app.add_ground_point), [state, ground_prompt_image, ground_point_type], [state, ground_prompt_image, session_status], queue=False)
        clear_ground_points_btn.click(gradio_logged(app.clear_ground_points), [state], [state, ground_prompt_image, session_status], queue=False)
        use_mask_btn.click(gradio_logged(app.use_uploaded_mask), [state, uploaded_mask], [state, session_status, seg_files, scene_graph_status, scene_graph_preview, scene_graph_files, infer_btn])
        segment_btn.click(gradio_logged(app.run_segmentation), [state, text_prompt], [state, seg_preview, session_status, seg_files, scene_graph_status, scene_graph_preview, scene_graph_files, infer_btn])
        infer_btn.click(gradio_logged(app.run_models), [state, models], [state, output_log, output_files, scene_graph_preview, pipeline_diagnostic_blend, background_btn])
        background_btn.click(gradio_logged(app.run_background_3dgs), [state], [state, output_log, output_files, background_ksplat, background_btn])
        worker_btn.click(gradio_logged(app.ui_worker_state), [state, models], [worker_status, segment_btn, infer_btn], queue=False)
        worker_timer.tick(gradio_logged(app.ui_worker_state), [state, models], [worker_status, segment_btn, infer_btn], queue=False)
        scene_graph_timer.tick(gradio_logged(app.poll_scene_graph), [state], [state, scene_graph_status, scene_graph_preview, scene_graph_files, infer_btn], queue=False)
        models.change(gradio_logged(app.ui_worker_state), [state, models], [worker_status, segment_btn, infer_btn], queue=False)
    return demo


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive Gradio app for Fysiverse scene generation.")
    parser.add_argument("--config", type=Path, default=Path(os.environ.get("PIPELINE_CONFIG", str(DEFAULT_CONFIG))))
    parser.add_argument("--server-name", default=os.environ.get("SERVER_NAME", "0.0.0.0"))
    parser.add_argument("--server-port", type=int, default=int(os.environ.get("SERVER_PORT", "7860")))
    parser.add_argument("--port-search-count", type=int, default=int(os.environ.get("PORT_SEARCH_COUNT", "100")))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.expanduser()
    if not config_path.is_absolute():
        config_path = (WORKFLOW_ROOT / config_path).resolve()
    apply_pipeline_config(config_path)
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    server_port = find_free_port(
        int(args.server_port),
        host="" if args.server_name == "0.0.0.0" else args.server_name,
        count=int(args.port_search_count),
    )
    app = InteractivePipeline(config_path)
    demo = build_demo(app)
    print(f"Local URL: http://127.0.0.1:{server_port}", flush=True)
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.server_name,
        server_port=server_port,
        allowed_paths=[str(WORKFLOW_ROOT / "sessions")],
        quiet=False,
        show_error=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
