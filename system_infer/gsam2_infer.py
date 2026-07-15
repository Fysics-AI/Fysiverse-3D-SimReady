from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(os.environ.get("GSAM2_ROOT", str(Path(__file__).resolve().parents[1] / "third_party" / "Grounded-SAM-2")))
SAM2_CHECKPOINT = ROOT / "checkpoints" / "sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = ROOT / "grounding_dino" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = ROOT / "gdino_checkpoints" / "groundingdino_swint_ogc.pth"

MASK_COLORS = [
    (220, 20, 60),
    (30, 80, 220),
    (0, 150, 80),
    (210, 150, 0),
    (120, 55, 190),
    (0, 135, 145),
    (205, 85, 0),
    (25, 115, 175),
]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _loads_maybe(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _to_uint8_rgb(image: Any) -> np.ndarray:
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"))
    else:
        arr = np.asarray(Image.open(image).convert("RGB"))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        if arr.max(initial=0) <= 1.0:
            arr = arr * 255.0
        return np.clip(arr, 0, 255).astype(np.uint8)
    info = np.iinfo(arr.dtype)
    if info.max > 255:
        arr = arr.astype(np.float32) * (255.0 / float(info.max))
    return np.clip(arr, 0, 255).astype(np.uint8)


def _normalize_points(points_value: Any, width: int, height: int) -> list[tuple[tuple[int, int], int]]:
    points_value = _loads_maybe(points_value)
    if points_value is None:
        return []
    if isinstance(points_value, dict):
        points_value = points_value.get("points") or points_value.get("point")
    if not isinstance(points_value, (list, tuple)):
        return []
    if points_value and len(points_value) >= 2 and all(isinstance(v, (int, float)) for v in points_value[:2]):
        points_iter = [points_value]
    else:
        points_iter = list(points_value)

    points: list[tuple[tuple[int, int], int]] = []
    for point in points_iter:
        if isinstance(point, dict):
            point = [point.get("x"), point.get("y"), point.get("label", point.get("point_label", 1))]
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        x, y = float(point[0]), float(point[1])
        if max(abs(x), abs(y)) <= 1.0:
            x, y = x * width, y * height
        label = int(point[2]) if len(point) >= 3 and point[2] is not None else 1
        label = 0 if label == 0 or str(label).lower() in {"background", "bg"} else 1
        points.append(
            (
                (
                    max(0, min(width - 1, int(round(x)))),
                    max(0, min(height - 1, int(round(y)))),
                ),
                label,
            )
        )
    return points


def _normalize_boxes(
    boxes_value: Any,
    width: int,
    height: int,
    box_index: Optional[int] = None,
) -> np.ndarray:
    boxes_value = _loads_maybe(boxes_value)
    if boxes_value is None:
        return np.empty((0, 4), dtype=np.float32)
    if isinstance(boxes_value, dict):
        boxes_value = boxes_value.get("boxes") or boxes_value.get("manual_boxes") or boxes_value.get("bbox")
    if not isinstance(boxes_value, (list, tuple)):
        return np.empty((0, 4), dtype=np.float32)
    if boxes_value and len(boxes_value) == 4 and all(isinstance(v, (int, float)) for v in boxes_value):
        boxes_iter = [boxes_value]
    else:
        boxes_iter = list(boxes_value)
    if box_index is not None and not isinstance(box_index, str):
        idx = int(box_index)
        if 0 <= idx < len(boxes_iter):
            boxes_iter = [boxes_iter[idx]]

    boxes: list[list[float]] = []
    for box in boxes_iter:
        if isinstance(box, dict):
            box = box.get("box") or box.get("bbox") or [
                box.get("x1"),
                box.get("y1"),
                box.get("x2"),
                box.get("y2"),
            ]
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        left = max(0.0, min(float(width - 1), left))
        right = max(0.0, min(float(width - 1), right))
        top = max(0.0, min(float(height - 1), top))
        bottom = max(0.0, min(float(height - 1), bottom))
        if right - left >= 2 and bottom - top >= 2:
            boxes.append([left, top, right, bottom])
    if not boxes:
        return np.empty((0, 4), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32)


def _choose_best_masks(masks: Any, scores: Any) -> np.ndarray:
    masks = np.asarray(masks)
    scores = np.asarray(scores)
    if masks.ndim == 4:
        if scores.ndim == 2:
            best = np.argmax(scores, axis=1)
        else:
            best = np.zeros((masks.shape[0],), dtype=np.int64)
        masks = masks[np.arange(masks.shape[0]), best]
    elif masks.ndim == 3 and scores.ndim == 1 and masks.shape[0] > 1:
        masks = masks[np.argmax(scores)][None, ...]
    elif masks.ndim == 2:
        masks = masks[None, ...]
    return masks.astype(bool)


def _build_mask_with_id(masks: list[np.ndarray]) -> np.ndarray | None:
    if not masks:
        return None
    dtype = np.uint16 if len(masks) > 255 else np.uint8
    out = np.zeros(np.asarray(masks[0]).shape, dtype=dtype)
    for idx, mask in enumerate(masks, start=1):
        out[np.asarray(mask).astype(bool)] = idx
    return out


def _save_label_mask(mask: np.ndarray, path: Path) -> None:
    if mask.dtype == np.uint16:
        Image.fromarray(mask, mode="I;16").save(path)
    else:
        Image.fromarray(mask.astype(np.uint8), mode="L").save(path)


def _save_binary_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(np.asarray(mask).astype(np.uint8) * 255, mode="L").save(path)


def _overlay_masks(
    image_rgb: np.ndarray,
    masks: list[np.ndarray],
    points: list[tuple[tuple[int, int], int]] | None = None,
    boxes: np.ndarray | None = None,
    labels: list[str] | None = None,
) -> Image.Image:
    base = Image.fromarray(image_rgb, mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    mask_alpha = 190
    for idx, mask in enumerate(masks):
        color = MASK_COLORS[idx % len(MASK_COLORS)]
        color_img = Image.new("RGBA", base.size, (*color, mask_alpha))
        alpha = Image.fromarray(np.asarray(mask).astype(np.uint8) * mask_alpha, mode="L")
        overlay = Image.composite(color_img, overlay, alpha)
    image = Image.alpha_composite(base, overlay)
    draw = ImageDraw.Draw(image)
    width, height = base.size
    radius = max(6, min(width, height) // 140)
    line_width = max(2, radius // 3)
    if boxes is not None:
        for idx, box in enumerate(np.asarray(boxes, dtype=np.int32)):
            x1, y1, x2, y2 = box.tolist()
            color = (30, 144, 255, 255)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
            label = f"box {idx}"
            if labels and idx < len(labels):
                label = labels[idx]
            draw.text((x1 + 3, max(0, y1 - radius * 2)), label, fill=color)
    for idx, (point, label) in enumerate(points or [], start=1):
        x, y = point
        color = (0, 220, 0, 255) if int(label) == 1 else (255, 40, 40, 255)
        if int(label) == 1:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=line_width)
            draw.line((x - radius, y, x + radius, y), fill=color, width=line_width)
            draw.line((x, y - radius, x, y + radius), fill=color, width=line_width)
        else:
            draw.line((x - radius, y - radius, x + radius, y + radius), fill=color, width=line_width)
            draw.line((x - radius, y + radius, x + radius, y - radius), fill=color, width=line_width)
        draw.text((x + radius + 2, y - radius), str(idx), fill=color)
    return image.convert("RGB")


class GSAM2Infer:
    def __init__(self, gpu: Optional[Any] = None, **_: Any):
        if gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        import torch
        import grounding_dino.groundingdino.datasets.transforms as grounding_transforms
        from grounding_dino.groundingdino.util.inference import load_model, predict
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from torchvision.ops import box_convert

        self.torch = torch
        self.grounding_transforms = grounding_transforms
        self.grounding_predict = predict
        self.box_convert = box_convert
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.sam2_predictor = SAM2ImagePredictor(
            build_sam2(SAM2_MODEL_CONFIG, str(SAM2_CHECKPOINT), device=self.device)
        )
        self.grounding_model = load_model(
            model_config_path=str(GROUNDING_DINO_CONFIG),
            model_checkpoint_path=str(GROUNDING_DINO_CHECKPOINT),
            device=self.device,
        )

    def _grounding_transform(self, image_rgb: np.ndarray) -> Any:
        transform = self.grounding_transforms.Compose(
            [
                self.grounding_transforms.RandomResize([800], max_size=1333),
                self.grounding_transforms.ToTensor(),
                self.grounding_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_transformed, _ = transform(Image.fromarray(image_rgb).convert("RGB"), None)
        return image_transformed

    def _set_sam_image(self, image_rgb: np.ndarray) -> None:
        if self.device == "cuda":
            with self.torch.autocast(device_type="cuda", dtype=self.torch.bfloat16):
                self.sam2_predictor.set_image(image_rgb)
        else:
            self.sam2_predictor.set_image(image_rgb)

    def _detect_boxes(
        self,
        image_rgb: np.ndarray,
        text_prompt: str,
        box_threshold: float,
        text_threshold: float,
    ) -> tuple[np.ndarray, list[float], list[str]]:
        image_tensor = self._grounding_transform(image_rgb)
        boxes, confidences, labels = self.grounding_predict(
            model=self.grounding_model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )
        if boxes.shape[0] == 0:
            return np.empty((0, 4), dtype=np.float32), [], []
        h, w = image_rgb.shape[:2]
        boxes = boxes * self.torch.Tensor([w, h, w, h])
        boxes_xyxy = self.box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, w - 1)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, h - 1)
        return boxes_xyxy.astype(np.float32), confidences.numpy().tolist(), list(labels)

    def _predict_sam(
        self,
        point_coords: np.ndarray | None = None,
        point_labels: np.ndarray | None = None,
        boxes: np.ndarray | None = None,
        multimask_output: bool = True,
    ) -> np.ndarray:
        if self.device == "cuda":
            with self.torch.autocast(device_type="cuda", dtype=self.torch.bfloat16):
                masks, scores, _ = self.sam2_predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=boxes,
                    multimask_output=multimask_output,
                )
        else:
            masks, scores, _ = self.sam2_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=boxes,
                multimask_output=multimask_output,
            )
        return _choose_best_masks(masks, scores)

    def _segment_from_points(
        self,
        selected_points: list[tuple[tuple[int, int], int]],
        multi_object: bool,
    ) -> list[np.ndarray]:
        if not selected_points:
            return []
        points = np.array([point for point, _ in selected_points], dtype=np.float32)
        labels = np.array([label for _, label in selected_points], dtype=np.int32)
        if not multi_object:
            return list(self._predict_sam(points, labels, boxes=None, multimask_output=True))

        masks: list[np.ndarray] = []
        bg_points = [(point, label) for point, label in selected_points if int(label) == 0]
        for fg_point, fg_label in selected_points:
            if int(fg_label) != 1:
                continue
            object_points = [fg_point] + [point for point, _ in bg_points]
            object_labels = [1] + [0 for _ in bg_points]
            object_masks = self._predict_sam(
                np.array(object_points, dtype=np.float32),
                np.array(object_labels, dtype=np.int32),
                boxes=None,
                multimask_output=True,
            )
            masks.extend(list(object_masks[:1]))
        return masks

    def _segment_from_boxes(
        self,
        boxes: np.ndarray,
        selected_points: list[tuple[tuple[int, int], int]],
        multi_object: bool,
        box_index: int,
    ) -> list[np.ndarray]:
        if boxes.shape[0] == 0:
            return []
        if not multi_object:
            box_index = int(np.clip(int(box_index), 0, boxes.shape[0] - 1))
            boxes = boxes[box_index : box_index + 1]
            if selected_points:
                points = np.array([point for point, _ in selected_points], dtype=np.float32)
                labels = np.array([label for _, label in selected_points], dtype=np.int32)
                return list(self._predict_sam(points, labels, boxes=boxes, multimask_output=False))
            return list(self._predict_sam(None, None, boxes=boxes, multimask_output=False))

        masks: list[np.ndarray] = []
        for box in boxes:
            x1, y1, x2, y2 = box
            box_points = [
                (point, label)
                for point, label in selected_points
                if x1 <= point[0] <= x2 and y1 <= point[1] <= y2
            ]
            if box_points:
                points = np.array([point for point, _ in box_points], dtype=np.float32)
                labels = np.array([label for _, label in box_points], dtype=np.int32)
                object_masks = self._predict_sam(points, labels, boxes=box[None, :], multimask_output=False)
            else:
                object_masks = self._predict_sam(None, None, boxes=box[None, :], multimask_output=False)
            masks.extend(list(object_masks[:1]))
        return masks

    def _save_outputs(
        self,
        image_rgb: np.ndarray,
        out_dir: Path,
        masks: list[np.ndarray],
        selected_points: list[tuple[tuple[int, int], int]],
        boxes: np.ndarray,
        labels: list[str],
    ) -> dict[str, Any]:
        out_dir.mkdir(parents=True, exist_ok=True)
        files: list[str] = []
        mask_paths: list[str] = []
        for idx, mask in enumerate(masks):
            path = out_dir / f"mask_{idx:03d}.png"
            _save_binary_mask(mask, path)
            files.append(str(path))
            mask_paths.append(str(path))

        label_mask = _build_mask_with_id(masks)
        mask_path = None
        if label_mask is not None:
            mask_path = out_dir / "mask.png"
            _save_label_mask(label_mask, mask_path)
            files.append(str(mask_path))

        overlay_path = out_dir / "overlay.png"
        _overlay_masks(image_rgb, masks, selected_points, boxes=boxes, labels=labels).save(overlay_path)
        files.append(str(overlay_path))

        return {
            "mask_path": str(mask_path) if mask_path else None,
            "mask_paths": mask_paths,
            "overlay_path": str(overlay_path),
            "files": files,
        }

    def segment(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if "jobs" in payload:
            results = [self.segment(job) for job in payload["jobs"]]
            return {"status": "ok", "results": results, "num_jobs": len(results)}

        image_path = Path(payload["image_path"])
        out_dir = Path(payload.get("out_dir") or payload.get("output_dir") or image_path.parent)
        image_rgb = _to_uint8_rgb(image_path)
        height, width = image_rgb.shape[:2]
        selected_points = _normalize_points(payload.get("points"), width, height)
        box_index = int(payload.get("box_index", 0) or 0)
        manual_boxes = _normalize_boxes(
            payload.get("manual_boxes") or payload.get("boxes"),
            width,
            height,
            None,
        )
        text_prompt = str(payload.get("text_prompt") or "").strip()
        use_grounding = bool(payload.get("use_grounding"))
        multi_object = bool(payload.get("multi_object", True))
        box_threshold = float(payload.get("box_threshold", 0.35))
        text_threshold = float(payload.get("text_threshold", 0.25))

        self._set_sam_image(image_rgb)

        boxes = np.empty((0, 4), dtype=np.float32)
        scores: list[float] = []
        det_labels: list[str] = []
        if manual_boxes.shape[0] > 0:
            boxes = manual_boxes
            masks = self._segment_from_boxes(boxes, selected_points, multi_object, box_index)
            message_prefix = f"Generated {len(masks)} mask(s) from {boxes.shape[0]} manual box(es)."
        elif use_grounding:
            if not text_prompt:
                raise ValueError("use_grounding=True requires text_prompt")
            boxes, scores, det_labels = self._detect_boxes(image_rgb, text_prompt, box_threshold, text_threshold)
            if boxes.shape[0] == 0:
                overlay_path = out_dir / "overlay.png"
                out_dir.mkdir(parents=True, exist_ok=True)
                _overlay_masks(image_rgb, [], selected_points, boxes=boxes, labels=det_labels).save(overlay_path)
                return {
                    "status": "ok",
                    "message": f"No boxes detected for prompt: {text_prompt}",
                    "mask_path": None,
                    "overlay_path": str(overlay_path),
                    "files": [str(overlay_path)],
                    "num_masks": 0,
                    "boxes": [],
                    "labels": [],
                    "scores": [],
                }
            masks = self._segment_from_boxes(boxes, selected_points, multi_object, box_index)
            message_prefix = f"Generated {len(masks)} mask(s). Detected {boxes.shape[0]} box(es)."
        else:
            masks = self._segment_from_points(selected_points, multi_object)
            message_prefix = f"Generated {len(masks)} mask(s) from {len(selected_points)} point(s)."

        if not masks:
            overlay_path = out_dir / "overlay.png"
            out_dir.mkdir(parents=True, exist_ok=True)
            labels = [f"{label} {score:.2f}" for label, score in zip(det_labels, scores)]
            _overlay_masks(image_rgb, [], selected_points, boxes=boxes, labels=labels).save(overlay_path)
            return {
                "status": "ok",
                "message": "No masks generated.",
                "mask_path": None,
                "overlay_path": str(overlay_path),
                "files": [str(overlay_path)],
                "num_masks": 0,
                "boxes": boxes.tolist(),
                "labels": det_labels,
                "scores": scores,
            }

        labels = [f"{label} {score:.2f}" for label, score in zip(det_labels, scores)]
        shown_boxes = boxes
        if use_grounding and not multi_object and boxes.shape[0] > 0:
            idx = int(np.clip(box_index, 0, boxes.shape[0] - 1))
            shown_boxes = boxes[idx : idx + 1]
            labels = labels[idx : idx + 1]

        saved = self._save_outputs(image_rgb, out_dir, masks, selected_points, shown_boxes, labels)
        return {
            "status": "ok",
            "message": message_prefix,
            "num_masks": len(masks),
            "boxes": shown_boxes.tolist(),
            "labels": det_labels,
            "scores": scores,
            **saved,
        }
