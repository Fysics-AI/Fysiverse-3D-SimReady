#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPENAI_COMPATIBLE_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_COMPATIBLE_MODEL = ""
DEFAULT_DASHSCOPE_BASE_URL = ""
DEFAULT_DASHSCOPE_MODEL = ""


def artifact_dir_for_output(output_path: Path) -> Path:
    if output_path.parent.name == "scene_graph_inputs":
        return output_path.parent
    return output_path.parent / "scene_graph_inputs"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def configure_codex_home_for_reasoning(effort: str, output_path: Path) -> Path | None:
    if effort == "inherit":
        return None
    source_home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
    runtime_home = artifact_dir_for_output(output_path) / f"codex_home_{effort}"
    runtime_home.mkdir(parents=True, exist_ok=True)
    for name in ("auth.json", "installation_id"):
        src = source_home / name
        if src.is_file():
            dst = runtime_home / name
            if not dst.exists():
                dst.write_bytes(src.read_bytes())
    config_src = source_home / "config.toml"
    config_text = config_src.read_text(encoding="utf-8") if config_src.is_file() else ""
    if re.search(r"(?m)^model_reasoning_effort\s*=", config_text):
        config_text = re.sub(r'(?m)^model_reasoning_effort\s*=.*$', f'model_reasoning_effort = "{effort}"', config_text)
    else:
        config_text = config_text.rstrip() + f'\nmodel_reasoning_effort = "{effort}"\n'
    (runtime_home / "config.toml").write_text(config_text, encoding="utf-8")
    return runtime_home


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_wrapped_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, *, font: ImageFont.ImageFont, fill: tuple[int, int, int], max_width: int, line_gap: int = 4) -> int:
    words = str(text).split()
    if not words:
        return xy[1]
    x, y = xy
    line = ""
    for word in words:
        test = word if not line else f"{line} {word}"
        bbox = draw.textbbox((x, y), test, font=font)
        if bbox[2] - bbox[0] <= max_width or not line:
            line = test
            continue
        draw.text((x, y), line, font=font, fill=fill)
        y += (bbox[3] - bbox[1]) + line_gap
        line = word
    if line:
        bbox = draw.textbbox((x, y), line, font=font)
        draw.text((x, y), line, font=font, fill=fill)
        y += (bbox[3] - bbox[1]) + line_gap
    return y


def render_scene_graph(graph: dict[str, Any], output_path: Path) -> Path:
    objects = sorted(graph.get("objects") or [], key=lambda item: int(item.get("mask_id", 0)))
    support_relations = list(graph.get("support_relations") or [])
    width = 1600
    margin = 44
    gutter = 18
    columns = 3 if len(objects) > 1 else 1
    card_w = (width - 2 * margin - (columns - 1) * gutter) // columns
    card_h = 126
    object_rows_count = max(1, (len(objects) + columns - 1) // columns)
    objects_h = 48 + object_rows_count * card_h + (object_rows_count - 1) * gutter
    support_card_h = 108
    supports_h = 50 + max(1, len(support_relations)) * support_card_h + max(0, len(support_relations) - 1) * 12
    height = margin + 90 + 28 + objects_h + 30 + supports_h + margin
    image = Image.new("RGB", (width, height), (244, 246, 249))
    draw = ImageDraw.Draw(image)
    title_font = load_font(34)
    section_font = load_font(22)
    label_font = load_font(19)
    body_font = load_font(16)
    small_font = load_font(13)

    palette = [
        (31, 111, 235),
        (34, 139, 85),
        (198, 122, 22),
        (126, 87, 194),
        (0, 137, 150),
        (196, 58, 76),
        (80, 104, 140),
        (178, 91, 28),
    ]
    object_by_id = {int(obj.get("mask_id", 0)): obj for obj in objects}
    color_by_id = {int(obj.get("mask_id", 0)): palette[idx % len(palette)] for idx, obj in enumerate(objects)}

    def obj_label(mask_id: int) -> str:
        obj = object_by_id.get(mask_id) or {}
        label = obj.get("semantic_label") or obj.get("mask_name") or f"mask_{mask_id:03d}"
        return str(label)

    def chip_text(value: Any) -> str:
        text = str(value or "unknown").replace("_", " ")
        return text[:26]

    def crop_thumb(obj: dict[str, Any], size: int = 74) -> Image.Image:
        path = obj.get("crop_rgb")
        thumb = Image.new("RGB", (size, size), (229, 233, 239))
        if isinstance(path, str) and Path(path).is_file():
            try:
                crop = Image.open(path).convert("RGB")
                crop.thumbnail((size, size), Image.Resampling.LANCZOS)
                thumb.paste(crop, ((size - crop.width) // 2, (size - crop.height) // 2))
            except Exception:
                pass
        return thumb

    def draw_chip(x: int, y: int, text: str, *, fill: tuple[int, int, int], fg: tuple[int, int, int] = (255, 255, 255)) -> int:
        text = str(text)
        bbox = draw.textbbox((0, 0), text, font=small_font)
        w = bbox[2] - bbox[0] + 20
        h = 24
        draw.rounded_rectangle((x, y, x + w, y + h), radius=12, fill=fill)
        draw.text((x + 10, y + 4), text, font=small_font, fill=fg)
        return x + w + 8

    def draw_panel(x: int, y: int, w: int, h: int, title: str) -> tuple[int, int]:
        draw.rounded_rectangle((x, y, x + w, y + h), radius=10, fill=(255, 255, 255), outline=(218, 224, 232), width=1)
        draw.text((x + 22, y + 16), title, font=section_font, fill=(26, 32, 44))
        return x + 22, y + 52

    def draw_object_card(obj: dict[str, Any], x: int, y: int, w: int, h: int) -> None:
        mask_id = int(obj.get("mask_id", 0))
        color = color_by_id.get(mask_id, (80, 104, 140))
        draw.rounded_rectangle((x, y, x + w, y + h), radius=8, fill=(250, 252, 255), outline=(226, 231, 238), width=1)
        draw.rectangle((x, y, x + 7, y + h), fill=color)
        thumb_size = 76
        tx, ty = x + 20, y + 20
        draw.rounded_rectangle((tx - 2, ty - 2, tx + thumb_size + 2, ty + thumb_size + 2), radius=8, fill=(232, 236, 242))
        image.paste(crop_thumb(obj, thumb_size), (tx, ty))
        text_x = tx + thumb_size + 18
        id_text = f"mask_{mask_id:03d}"
        draw.text((text_x, y + 18), id_text, font=small_font, fill=color)
        draw_wrapped_text(draw, (text_x, y + 40), obj_label(mask_id), font=label_font, fill=(23, 31, 43), max_width=w - (text_x - x) - 20, line_gap=1)
        chip_y = y + h - 34
        next_x = text_x
        support = chip_text(obj.get("support_status"))
        visibility = chip_text(obj.get("visibility"))
        next_x = draw_chip(next_x, chip_y, support, fill=(233, 242, 255), fg=(31, 90, 170))
        if next_x + 120 < x + w:
            draw_chip(next_x, chip_y, visibility, fill=(239, 246, 241), fg=(44, 112, 78))

    backend = graph.get("vlm_backend") or (graph.get("scene_graph_config") or {}).get("backend") or "vlm"
    model = graph.get("vlm_model") or (graph.get("scene_graph_config") or {}).get("model") or ""
    draw.text((margin, 28), "Scene Graph", font=title_font, fill=(20, 27, 38))
    chip_x = margin
    chip_x = draw_chip(chip_x, 76, f"{backend}", fill=(31, 111, 235))
    if model:
        chip_x = draw_chip(chip_x, 76, str(model), fill=(65, 75, 92))
    chip_x = draw_chip(chip_x, 76, f"{len(objects)} masks", fill=(34, 139, 85))
    draw_chip(chip_x, 76, f"{len(support_relations)} supports", fill=(198, 122, 22))

    y0 = 132
    sx, sy = draw_panel(margin, y0, width - 2 * margin, objects_h, "Objects")
    for idx, obj in enumerate(objects):
        row = idx // columns
        col = idx % columns
        x = margin + col * (card_w + gutter)
        y = sy + row * (card_h + gutter)
        draw_object_card(obj, x, y, card_w, card_h)

    y0 += objects_h + 30
    sx, sy = draw_panel(margin, y0, width - 2 * margin, supports_h, "Support")
    support_w = width - 2 * margin - 44
    y = sy
    if support_relations:
        for idx, rel in enumerate(support_relations):
            upper = int(rel.get("upper_mask_id"))
            lower = int(rel.get("lower_mask_id"))
            draw.rounded_rectangle((sx, y, sx + support_w, y + support_card_h), radius=8, fill=(248, 252, 249), outline=(209, 226, 215), width=1)
            upper_color = color_by_id.get(upper, (31, 111, 235))
            lower_color = color_by_id.get(lower, (34, 139, 85))
            left_x = sx + 22
            right_x = sx + support_w - 520
            image.paste(crop_thumb(object_by_id.get(upper, {}), 56), (left_x, y + 22))
            image.paste(crop_thumb(object_by_id.get(lower, {}), 56), (right_x, y + 22))
            draw.text((left_x + 72, y + 22), f"mask_{upper:03d}", font=small_font, fill=upper_color)
            draw_wrapped_text(draw, (left_x + 72, y + 42), obj_label(upper), font=body_font, fill=(23, 31, 43), max_width=360, line_gap=1)
            draw.text((right_x + 72, y + 22), f"mask_{lower:03d}", font=small_font, fill=lower_color)
            draw_wrapped_text(draw, (right_x + 72, y + 42), obj_label(lower), font=body_font, fill=(23, 31, 43), max_width=360, line_gap=1)
            arrow_y = y + 50
            ax1 = sx + 540
            ax2 = sx + support_w - 560
            draw.line((ax1, arrow_y, ax2, arrow_y), fill=(34, 139, 85), width=4)
            draw.polygon([(ax2, arrow_y), (ax2 - 16, arrow_y - 9), (ax2 - 16, arrow_y + 9)], fill=(34, 139, 85))
            draw.text((ax1 + 28, y + 24), "supported by", font=label_font, fill=(34, 110, 75))
            reason = str(rel.get("reason") or "")
            if reason:
                draw_wrapped_text(draw, (ax1 + 28, y + 58), reason, font=small_font, fill=(76, 88, 104), max_width=max(160, ax2 - ax1 - 56), line_gap=1)
            y += support_card_h + 12
    else:
        draw.rounded_rectangle((sx, y, sx + support_w, y + support_card_h), radius=8, fill=(250, 252, 255), outline=(226, 231, 238), width=1)
        draw.text((sx + 24, y + 38), "No mask-to-mask support relation detected.", font=label_font, fill=(76, 88, 104))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def object_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for obj in manifest.get("objects") or []:
        if obj.get("mask_id") is None:
            continue
        rows.append(
            {
                "mask_id": int(obj["mask_id"]),
                "mask_name": obj.get("mask_name") or f"mask_{int(obj['mask_id']):03d}",
                "bbox_2d_xyxy": obj.get("bbox_2d_xyxy"),
                "bbox_2d_xywh": obj.get("bbox_2d_xywh"),
                "area_pixels": obj.get("area_pixels"),
                "crop_rgb": obj.get("crop_rgb"),
                "sam3d_input_index": obj.get("sam3d_input_index"),
            }
        )
    return sorted(rows, key=lambda item: int(item["mask_id"]))


def create_scope_image(image_path: Path, objects: list[dict[str, Any]], output_path: Path) -> Path:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    line_width = max(3, min(width, height) // 220)
    label_h = max(18, min(width, height) // 35)
    palette = [
        (220, 20, 60),
        (30, 80, 220),
        (0, 150, 80),
        (210, 150, 0),
        (120, 55, 190),
        (0, 135, 145),
        (205, 85, 0),
        (25, 115, 175),
    ]
    for idx, obj in enumerate(objects):
        bbox = obj.get("bbox_2d_xyxy")
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
        x1, x2 = sorted((max(0, min(width - 1, x1)), max(0, min(width - 1, x2))))
        y1, y2 = sorted((max(0, min(height - 1, y1)), max(0, min(height - 1, y2))))
        color = palette[idx % len(palette)]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        text = f"mask_{int(obj['mask_id']):03d}"
        tx1, ty1 = x1, max(0, y1 - label_h)
        tx2 = min(width - 1, x1 + max(label_h * 5, 95))
        ty2 = min(height - 1, ty1 + label_h)
        draw.rectangle((tx1, ty1, tx2, ty2), fill=color)
        draw.text((tx1 + 3, ty1 + 2), text, fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def backend_file_prefix(backend: str) -> str:
    return "codex" if backend == "codex" else backend


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


def compact_prompt(manifest: dict[str, Any], objects: list[dict[str, Any]], crop_attachments: list[tuple[int, Path]]) -> str:
    allowed_ids = [int(obj["mask_id"]) for obj in objects]
    crop_lines = []
    for image_number, crop_path in crop_attachments:
        mask_match = re.search(r"mask_(\d+)", crop_path.name)
        mask_id = int(mask_match.group(1)) if mask_match else None
        crop_lines.append(f"- Image #{image_number}: crop for mask_id={mask_id}, path={crop_path}")
    crop_text = "\n".join(crop_lines) if crop_lines else "- No crop images attached."
    object_json = json.dumps(objects, ensure_ascii=False, indent=2)
    session_id = manifest.get("session_id")
    image_path = (manifest.get("inputs") or {}).get("image")
    return f"""You are acting as a vision-language scene-graph annotator for an image-to-3D pipeline.

Answer quickly while preserving correctness. Do not perform unnecessary long reasoning; inspect the provided images and metadata, then return the JSON directly.

Your scope is STRICTLY limited to the already segmented mask objects listed below.
Allowed mask_id values: {allowed_ids}

Do not invent background objects, floor, walls, hands, shelves, or any object that is not one of these mask_id values.
If an unsegmented object appears to support a segmented object, mention it only in notes; do not create a node or a support relation to it.
Every object and relation in your JSON must refer only to the allowed mask_id values.

Images:
- Image #1: original full input image.
- Image #2: the same image annotated with bbox rectangles and mask_id labels for the allowed objects only.
{crop_text}

Session: {session_id}
Full image path: {image_path}
Allowed segmented objects with 2D bbox/crop metadata:
{object_json}

Return JSON only, with this exact top-level shape:
{{
  "schema": "fysiverse_mask_scene_graph.v1",
  "status": "ok",
  "scope": {{
    "only_mask_objects": true,
    "allowed_mask_ids": {allowed_ids},
    "unsegmented_objects_excluded": true
  }},
  "objects": [
    {{
      "mask_id": 1,
      "semantic_label": "short noun phrase for this segmented object",
      "description": "one concise visual description",
      "confidence": 0.0,
      "visibility": "visible|partially_occluded|mostly_occluded|unknown",
      "support_status": "root_or_floor|supported_by_mask|unknown"
    }}
  ],
  "relations": [
    {{
      "subject_mask_id": 1,
      "object_mask_id": 2,
      "relation": "supported_by|contains|inside|leaning_on|in_front_of|behind|next_to|touching|overlaps_2d|unknown",
      "confidence": 0.0,
      "evidence": "brief visual reason"
    }}
  ],
  "support_relations": [
    {{
      "upper_mask_id": 1,
      "lower_mask_id": 2,
      "confidence": 0.0,
      "reason": "why mask 1 should be vertically supported by mask 2"
    }}
  ],
  "root_mask_ids": [],
  "notes": []
}}

Support relation means: under gravity, the upper object should be kept vertically on top of the lower segmented object during 3D postprocessing. Be conservative: only output support_relations when the visual evidence is reasonably clear.
"""


def extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if fence:
        stripped = fence.group(1)
    else:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            stripped = stripped[start : end + 1]
    return json.loads(stripped)


def relation_mask_id(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def normalize_scene_graph(
    *,
    raw: dict[str, Any],
    manifest: dict[str, Any],
    objects: list[dict[str, Any]],
    source_manifest: Path,
    source_image: Path,
    scope_image: Path,
    backend: str,
    backend_model: str | None,
    backend_stdout: Path | None,
    backend_response: Path | None,
) -> dict[str, Any]:
    allowed = {int(obj["mask_id"]) for obj in objects}
    object_meta = {int(obj["mask_id"]): obj for obj in objects}

    vlm_objects_by_id: dict[int, dict[str, Any]] = {}
    for obj in raw.get("objects") or []:
        mask_id = relation_mask_id(obj.get("mask_id"))
        if mask_id not in allowed:
            continue
        vlm_objects_by_id[mask_id] = dict(obj)

    normalized_objects: list[dict[str, Any]] = []
    for mask_id in sorted(allowed):
        meta = dict(object_meta[mask_id])
        vlm = vlm_objects_by_id.get(mask_id) or {}
        normalized_objects.append(
            {
                **meta,
                "semantic_label": vlm.get("semantic_label"),
                "description": vlm.get("description"),
                "confidence": vlm.get("confidence"),
                "visibility": vlm.get("visibility"),
                "support_status": vlm.get("support_status"),
            }
        )

    relations: list[dict[str, Any]] = []
    for rel in raw.get("relations") or []:
        subj = relation_mask_id(rel.get("subject_mask_id"))
        obj = relation_mask_id(rel.get("object_mask_id"))
        if subj not in allowed or obj not in allowed or subj == obj:
            continue
        relations.append(
            {
                "subject_mask_id": subj,
                "object_mask_id": obj,
                "relation": rel.get("relation") or "unknown",
                "confidence": rel.get("confidence"),
                "evidence": rel.get("evidence") or rel.get("reason"),
            }
        )

    support_relations: list[dict[str, Any]] = []
    relation_source = "codex_vlm" if backend == "codex" else f"{backend}_vlm"
    for rel in raw.get("support_relations") or []:
        upper = relation_mask_id(first_present(rel, ("upper_mask_id", "child_mask_id", "subject_mask_id")))
        lower = relation_mask_id(first_present(rel, ("lower_mask_id", "parent_mask_id", "object_mask_id")))
        if upper not in allowed or lower not in allowed or upper == lower:
            continue
        support_relations.append(
            {
                "upper_mask_id": upper,
                "lower_mask_id": lower,
                "confidence": rel.get("confidence"),
                "reason": rel.get("reason") or rel.get("evidence"),
                "source": relation_source,
            }
        )

    root_mask_ids = []
    for value in raw.get("root_mask_ids") or []:
        mask_id = relation_mask_id(value)
        if mask_id in allowed:
            root_mask_ids.append(mask_id)

    return {
        "schema": "fysiverse_mask_scene_graph.v1",
        "status": "ok",
        "session_id": manifest.get("session_id"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_manifest": str(source_manifest),
        "source_image": str(source_image),
        "scope_image": str(scope_image),
        "vlm_backend": backend,
        "vlm_model": backend_model,
        "vlm_stdout": str(backend_stdout) if backend_stdout else None,
        "vlm_response": str(backend_response) if backend_response else None,
        "codex_stdout": str(backend_stdout) if backend == "codex" and backend_stdout else None,
        "codex_last_message": str(backend_response) if backend == "codex" and backend_response else None,
        "scope": {
            "only_mask_objects": True,
            "allowed_mask_ids": sorted(allowed),
            "unsegmented_objects_excluded": True,
            "note": "All objects and relations are filtered to the segmented mask_id whitelist.",
        },
        "objects": normalized_objects,
        "relations": relations,
        "support_relations": support_relations,
        "root_mask_ids": sorted(set(root_mask_ids)),
        "notes": [str(v) for v in (raw.get("notes") or [])],
    }


def failure_graph(manifest: dict[str, Any], objects: list[dict[str, Any]], source_manifest: Path, source_image: Path, error: str, backend: str = "codex") -> dict[str, Any]:
    return {
        "schema": "fysiverse_mask_scene_graph.v1",
        "status": "failed",
        "session_id": manifest.get("session_id"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_manifest": str(source_manifest),
        "source_image": str(source_image),
        "vlm_backend": backend,
        "scope": {
            "only_mask_objects": True,
            "allowed_mask_ids": [int(obj["mask_id"]) for obj in objects],
            "unsegmented_objects_excluded": True,
        },
        "objects": objects,
        "relations": [],
        "support_relations": [],
        "root_mask_ids": [],
        "notes": [f"scene graph construction failed: {error}"],
    }


def run_codex_backend(args: argparse.Namespace, prompt: str, attachments: list[Path], stdout_path: Path) -> tuple[dict[str, Any], Path]:
    image_args: list[str] = []
    for image in attachments:
        image_args.extend(["--image", str(image)])

    with tempfile.TemporaryDirectory(prefix="codex_scene_graph_") as tmp_dir:
        last_message = Path(tmp_dir) / "last_message.json"
        cmd = [
            args.codex_bin,
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "-C",
            str(WORKFLOW_ROOT),
            "--output-last-message",
            str(last_message),
        ]
        if args.model:
            cmd.extend(["--model", args.model])
        cmd.extend(image_args)
        cmd.append("-")
        env = os.environ.copy()
        env.setdefault("NO_COLOR", "1")
        codex_home = configure_codex_home_for_reasoning(args.reasoning_effort, args.output)
        if codex_home is not None:
            env["CODEX_HOME"] = str(codex_home)
        proc = subprocess.run(cmd, cwd=str(WORKFLOW_ROOT), env=env, input=prompt, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=args.timeout)
        stdout_path.write_text(proc.stdout or "", encoding="utf-8")
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout)
        if not last_message.is_file():
            raise RuntimeError("codex did not write an output-last-message file")
        last_text = last_message.read_text(encoding="utf-8")
        raw = extract_json(last_text)
        stable_last = artifact_dir_for_output(args.output) / "codex_scene_graph_last_message.txt"
        stable_last.write_text(last_text, encoding="utf-8")
    return raw, stable_last


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
        metadata = {
            "client": "openai",
            "id": getattr(completion, "id", None),
            "model": getattr(completion, "model", model),
        }
        return text, metadata

    payload = {"model": model, "messages": messages}
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        chat_completions_url(base_url),
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
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
    metadata = {
        "client": "urllib",
        "id": response_json.get("id"),
        "model": response_json.get("model") or model,
    }
    return text, metadata


def openai_compatible_defaults(backend: str) -> tuple[str, str]:
    if backend == "dashscope_qwen":
        return DEFAULT_DASHSCOPE_BASE_URL, DEFAULT_DASHSCOPE_MODEL
    return DEFAULT_OPENAI_COMPATIBLE_BASE_URL, DEFAULT_OPENAI_COMPATIBLE_MODEL


def run_openai_compatible_backend(args: argparse.Namespace, prompt: str, attachments: list[Path], stdout_path: Path, *, attempt_label: str = "primary") -> tuple[dict[str, Any], Path]:
    api_key = args.api_key or os.environ.get("SCENE_GRAPH_API_KEY") or ""
    if not api_key:
        raise RuntimeError(f"{args.backend} backend requires SCENE_GRAPH_API_KEY or --api-key")

    default_base_url, default_model = openai_compatible_defaults(args.backend)
    base_url = args.api_base_url or default_base_url
    model = args.api_model or default_model
    if not model:
        raise RuntimeError(f"{args.backend} backend requires SCENE_GRAPH_API_MODEL or --api-model")
    content: list[dict[str, Any]] = []
    for image in attachments:
        content.append({"type": "image_url", "image_url": {"url": image_data_url(image)}})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    request_timeout = float(getattr(args, "effective_request_timeout", args.timeout))
    response_text, metadata = call_openai_compatible_api(
        api_key=api_key,
        base_url=base_url,
        model=model,
        messages=messages,
        timeout=request_timeout,
    )
    response_suffix = "" if attempt_label == "primary" else f"_{attempt_label}"
    prefix = backend_file_prefix(args.backend)
    response_path = artifact_dir_for_output(args.output) / f"{prefix}_scene_graph_response{response_suffix}.txt"
    response_path.write_text(response_text, encoding="utf-8")
    stdout_path.write_text(
        json.dumps(
            {
                "backend": args.backend,
                "attempt": attempt_label,
                "base_url": base_url,
                "model": model,
                "num_images": len(attachments),
                "attachments": [str(path) for path in attachments],
                "prompt_chars": len(prompt),
                "request_timeout": request_timeout,
                "response_chars": len(response_text),
                "metadata": metadata,
                "response_path": str(response_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    raw = extract_json(response_text)
    return raw, response_path


def run_openai_compatible_with_optional_crop_fallback(
    args: argparse.Namespace,
    *,
    manifest: dict[str, Any],
    objects: list[dict[str, Any]],
    full_attachments: list[Path],
    full_crop_attachments: list[tuple[int, Path]],
    stdout_path: Path,
) -> tuple[dict[str, Any], Path, int, str | None]:
    prompt = compact_prompt(manifest, objects, full_crop_attachments)
    original_timeout = float(args.timeout)
    primary_timeout = original_timeout
    if args.api_retry_without_crops and full_crop_attachments:
        primary_timeout = min(original_timeout, 120.0)
    try:
        args.effective_request_timeout = primary_timeout
        raw, response_path = run_openai_compatible_backend(args, prompt, full_attachments, stdout_path)
        return raw, response_path, len(full_crop_attachments), None
    except Exception as primary_exc:
        if not args.api_retry_without_crops or not full_crop_attachments:
            raise
        fallback_prompt = compact_prompt(manifest, objects, [])
        artifact_dir = artifact_dir_for_output(args.output)
        prefix = backend_file_prefix(args.backend)
        fallback_prompt_path = artifact_dir / f"{prefix}_scene_graph_prompt_no_crops.txt"
        fallback_prompt_path.write_text(fallback_prompt, encoding="utf-8")
        fallback_attachments = full_attachments[:2]
        fallback_stdout = stdout_path.with_name(stdout_path.stem + "_retry_no_crops" + stdout_path.suffix)
        stdout_path.write_text(
            json.dumps(
                {
                    "backend": args.backend,
                    "attempt": "primary_failed_retry_no_crops_started",
                    "failed_num_crop_images": len(full_crop_attachments),
                    "primary_request_timeout": primary_timeout,
                    "primary_failure": str(primary_exc),
                    "retry_stdout": str(fallback_stdout),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        try:
            args.effective_request_timeout = min(original_timeout, 180.0)
            raw, response_path = run_openai_compatible_backend(args, fallback_prompt, fallback_attachments, fallback_stdout, attempt_label="retry_no_crops")
        except Exception as retry_exc:
            raise RuntimeError(
                f"{args.backend} scene graph failed with crops and retry_without_crops also failed. "
                f"primary={primary_exc}; retry={retry_exc}"
            ) from retry_exc
        retry_note = f"primary {args.backend} request with {len(full_crop_attachments)} crop images failed; retried with original+scope images only: {primary_exc}"
        stdout_path.write_text(
            json.dumps(
                {
                    "backend": args.backend,
                    "attempt": "primary_failed_retry_no_crops_succeeded",
                    "failed_num_crop_images": len(full_crop_attachments),
                    "primary_request_timeout": primary_timeout,
                    "failure": str(primary_exc),
                    "retry_stdout": str(fallback_stdout),
                    "retry_response": str(response_path),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return raw, response_path, 0, retry_note
    finally:
        if hasattr(args, "effective_request_timeout"):
            delattr(args, "effective_request_timeout")


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_json(args.manifest)
    objects = object_rows(manifest)
    if not objects:
        raise RuntimeError(f"manifest has no segmented mask objects: {args.manifest}")
    image_path = Path((manifest.get("inputs") or {}).get("image") or args.image or "")
    if not image_path.is_file():
        raise RuntimeError(f"source image not found: {image_path}")

    artifact_dir = artifact_dir_for_output(args.output)
    scope_image = args.scope_image or (artifact_dir / "mask_scope_annotated.png")
    create_scope_image(image_path, objects, scope_image)

    attachments: list[Path] = [image_path, scope_image]
    crop_attachments: list[tuple[int, Path]] = []
    next_image_number = 3
    for obj in objects[: max(0, int(args.max_crop_images))]:
        crop = Path(str(obj.get("crop_rgb") or ""))
        if crop.is_file():
            attachments.append(crop)
            crop_attachments.append((next_image_number, crop))
            next_image_number += 1

    prompt = compact_prompt(manifest, objects, crop_attachments)
    prefix = backend_file_prefix(args.backend)
    prompt_path = artifact_dir / f"{prefix}_scene_graph_prompt.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt, encoding="utf-8")

    stdout_path = args.log or (artifact_dir / f"{prefix}_scene_graph_stdout.log")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    effective_crop_images = len(crop_attachments)
    retry_note = None
    if args.backend == "codex":
        raw, response_path = run_codex_backend(args, prompt, attachments, stdout_path)
        backend_model = args.model or None
    elif args.backend in {"openai_compatible", "dashscope_qwen"}:
        raw, response_path, effective_crop_images, retry_note = run_openai_compatible_with_optional_crop_fallback(
            args,
            manifest=manifest,
            objects=objects,
            full_attachments=attachments,
            full_crop_attachments=crop_attachments,
            stdout_path=stdout_path,
        )
        _, default_model = openai_compatible_defaults(args.backend)
        backend_model = args.api_model or default_model
    else:
        raise RuntimeError(f"unsupported scene graph backend: {args.backend}")

    graph = normalize_scene_graph(
        raw=raw,
        manifest=manifest,
        objects=objects,
        source_manifest=args.manifest,
        source_image=image_path,
        scope_image=scope_image,
        backend=args.backend,
        backend_model=backend_model,
        backend_stdout=stdout_path,
        backend_response=response_path,
    )
    graph["visualization"] = str(args.visualization) if args.visualization else str(args.output.with_suffix(".png"))
    graph["scene_graph_config"] = {
        "backend": args.backend,
        "model": backend_model,
        "max_crop_images": int(args.max_crop_images),
        "effective_crop_images": effective_crop_images,
    }
    if retry_note:
        graph["notes"].append(retry_note)
    if args.backend == "codex":
        graph["codex_config"] = {
            "reasoning_effort": args.reasoning_effort,
            "model": args.model or None,
            "codex_home": str(artifact_dir / f"codex_home_{args.reasoning_effort}") if args.reasoning_effort != "inherit" else None,
        }
    elif args.backend in {"openai_compatible", "dashscope_qwen"}:
        default_base_url, default_model = openai_compatible_defaults(args.backend)
        graph["openai_compatible_config"] = {
            "backend_alias": args.backend,
            "base_url": args.api_base_url or default_base_url,
            "model": args.api_model or default_model,
            "api_key": "set" if (args.api_key or os.environ.get("SCENE_GRAPH_API_KEY")) else "unset",
            "retry_without_crops": bool(args.api_retry_without_crops),
        }
    render_scene_graph(graph, Path(graph["visualization"]))
    write_json(args.output, graph)
    return graph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a mask-scoped scene graph with a selectable VLM backend.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--scope-image", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--visualization", type=Path, default=None)
    parser.add_argument("--backend", default=os.environ.get("SCENE_GRAPH_BACKEND", "openai_compatible"), choices=["codex", "openai_compatible", "dashscope_qwen"])
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--model", default="")
    parser.add_argument("--reasoning-effort", default="inherit", choices=["inherit", "low", "medium", "high", "xhigh"], help="Codex reasoning effort for this simple VLM scene graph task. 'inherit' uses the current Codex default configuration.")
    parser.add_argument("--api-key", default="", help="External API key. Prefer SCENE_GRAPH_API_KEY so the key is not visible in process args.")
    parser.add_argument("--api-base-url", default=os.environ.get("SCENE_GRAPH_API_BASE_URL", ""))
    parser.add_argument("--api-model", default=os.environ.get("SCENE_GRAPH_API_MODEL", ""))
    parser.add_argument("--api-retry-without-crops", action="store_true", help="For external APIs, retry with only original+scope images if the crop-image request fails.")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-crop-images", type=int, default=20)
    parser.add_argument("--allow-failed-placeholder", action="store_true", help="Write status=failed JSON and exit 0 instead of failing. Not used by the main Gradio pipeline.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        graph = run(args)
        print(f"Wrote mask-scoped scene graph with backend={args.backend}: {args.output} ({len(graph.get('objects') or [])} objects, {len(graph.get('support_relations') or [])} support relations)")
        return 0
    except Exception as exc:
        if not args.allow_failed_placeholder:
            print(f"scene graph construction failed: {exc}", file=sys.stderr)
            return 1
        try:
            manifest = load_json(args.manifest)
            objects = object_rows(manifest)
            image_path = Path((manifest.get("inputs") or {}).get("image") or args.image or "")
            write_json(args.output, failure_graph(manifest, objects, args.manifest, image_path, str(exc), backend=args.backend))
            print(f"Wrote failed scene graph placeholder: {args.output}: {exc}")
            return 0
        except Exception:
            print(f"scene graph construction failed: {exc}", file=sys.stderr)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
