#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import requests


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKEND = "qwen"
DEFAULT_QWEN_MODEL = "qwen-image-2.0-pro"
DEFAULT_SEEDDREAM_MODEL = "doubao-seedream-5-0-pro-260628"
DEFAULT_SEEDDREAM_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_PROMPT = "删除前景物体，保留其他背景。"
SESSION_PROMPT_TEMPLATE = "删除前景物体，包括{{{objects}}}，保留其他背景。"
DEFAULT_FALLBACK_SIZE = "512*512"


class QwenImageEditError(RuntimeError):
    pass


def image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise QwenImageEditError(f"Pillow is required to inspect image size: {exc}") from exc
    with Image.open(path) as image:
        return image.size


def image_to_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"

def image_to_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def background_image_config() -> dict[str, Any]:
    data = load_yaml(WORKFLOW_ROOT / "config" / "background_3dgs.yaml")
    config = data.get("background_image") if isinstance(data.get("background_image"), dict) else {}
    return config


def scene_graph_api_key() -> str | None:
    config_path = WORKFLOW_ROOT / "config" / "scene_graph.yaml"
    if not config_path.is_file():
        return None
    data = load_yaml(config_path)
    api = data.get("api") if isinstance(data, dict) else None
    key = api.get("key") if isinstance(api, dict) else None
    return str(key).strip() if key else None


def provider_config(backend: str) -> dict[str, Any]:
    config = background_image_config()
    provider = config.get(backend) if isinstance(config.get(backend), dict) else {}
    return provider


def validate_external_api_key(api_key: str, *, env_name: str) -> str:
    key = str(api_key or "").strip()
    if not key:
        raise QwenImageEditError(f"{env_name} is empty")
    lower = key.lower()
    looks_placeholder = (
        "<" in key
        or ">" in key
        or "your" in lower
        or "api key" in lower
        or "你的" in key
        or "占位" in key
    )
    if looks_placeholder or not key.isascii():
        raise QwenImageEditError(
            f"{env_name} looks like a placeholder or contains non-ASCII characters. "
            "Set it to the real provider key before running background image editing."
        )
    return key


def resolve_api_key(backend: str, explicit_key: str | None = None) -> str:
    provider = provider_config(backend)
    env_name = "QWEN_IMAGE_API_KEY"
    if backend == "seeddream":
        env_name = "ARK_API_KEY"
        key = (
            explicit_key
            or os.environ.get("ARK_API_KEY")
            or os.environ.get("BACKGROUND_IMAGE_SEEDDREAM_API_KEY")
            or provider.get("api_key")
        )
        message = (
            "Seedream API key is not set; use ARK_API_KEY, "
            "BACKGROUND_IMAGE_SEEDDREAM_API_KEY, or config/background_3dgs.yaml."
        )
    else:
        key = (
            explicit_key
            or os.environ.get("QWEN_IMAGE_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY")
            or os.environ.get("SCENE_GRAPH_API_KEY")
            or provider.get("api_key")
            or scene_graph_api_key()
        )
        message = (
            "DashScope API key is not set; use QWEN_IMAGE_API_KEY, DASHSCOPE_API_KEY, "
            "SCENE_GRAPH_API_KEY, config/background_3dgs.yaml, or config/scene_graph.yaml."
        )
    if not key:
        raise QwenImageEditError(message)
    return validate_external_api_key(str(key), env_name=env_name)


def resolve_paths(
    *,
    session: Path | None,
    input_path: Path | None,
    output_path: Path | None,
    report_path: Path | None,
) -> tuple[Path, Path, Path | None]:
    if session is not None:
        session = session.expanduser().resolve()
        if input_path is None:
            input_path = session / "input" / "image.png"
        if output_path is None:
            output_path = session / "input" / "background.png"
        if report_path is None:
            report_path = session / "input" / "background_edit_report.json"

    if input_path is None:
        raise QwenImageEditError("missing --input or --session")

    input_path = input_path.expanduser().resolve()
    if output_path is None:
        output_path = input_path.parent / "background.png"
    else:
        output_path = output_path.expanduser().resolve()
    if report_path is not None:
        report_path = report_path.expanduser().resolve()
    return input_path, output_path, report_path


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def compact_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return " ".join(text.split())


def object_prompt_label(item: dict[str, Any]) -> str | None:
    for key in ("semantic_label", "description", "mask_name", "final_3d_object_name"):
        label = compact_label(item.get(key))
        if label:
            return label
    mask_id = item.get("mask_id")
    if mask_id is not None:
        return f"mask {mask_id}"
    return None


def foreground_objects_from_session(session: Path | None) -> dict[str, Any]:
    if session is None:
        return {"source": None, "objects": []}

    session = session.expanduser().resolve()
    candidates = [
        session / "results" / "scene_graph.json",
        session / "results" / "final_scene_manifest.json",
    ]
    output_by_key: dict[str, dict[str, Any]] = {}
    sources: list[str] = []
    for path in candidates:
        data = read_json(path)
        objects = data.get("objects") if isinstance(data, dict) else None
        if not isinstance(objects, list):
            continue
        sources.append(str(path))

        for index, raw in enumerate(objects):
            if not isinstance(raw, dict):
                continue
            label = object_prompt_label(raw)
            if not label:
                continue
            mask_id = raw.get("mask_id")
            key = f"mask:{mask_id}" if mask_id is not None else f"{path}:{index}:{label.casefold()}"
            if key in output_by_key:
                continue
            output_by_key[key] = {
                "mask_id": mask_id,
                "semantic_label": raw.get("semantic_label"),
                "description": raw.get("description"),
                "prompt_label": label,
            }

    return {"source": sources[0] if sources else None, "sources": sources, "objects": list(output_by_key.values())}


def build_prompt_from_objects(objects: list[dict[str, Any]]) -> str:
    label_counts: dict[str, int] = {}
    for item in objects:
        label = compact_label(item.get("prompt_label"))
        if label:
            key = label.casefold()
            label_counts[key] = label_counts.get(key, 0) + 1

    labels: list[str] = []
    for item in objects:
        label = compact_label(item.get("prompt_label"))
        if not label:
            continue
        if label_counts.get(label.casefold(), 0) > 1:
            label = compact_label(item.get("description")) or label
        labels.append(label)
    if not labels:
        return DEFAULT_PROMPT
    return SESSION_PROMPT_TEMPLATE.format(objects="，".join(labels))


def resolve_prompt(explicit_prompt: str | None, session: Path | None) -> tuple[str, dict[str, Any]]:
    if explicit_prompt:
        return explicit_prompt, {"source": "explicit", "objects": []}
    foreground = foreground_objects_from_session(session)
    prompt = build_prompt_from_objects(foreground["objects"])
    source = foreground["source"] or "default"
    return prompt, {"source": source, "objects": foreground["objects"]}


def response_image_urls(response: Any) -> list[str]:
    choices = getattr(getattr(response, "output", None), "choices", None)
    if not choices:
        return []
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return []
    urls: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("image"):
            urls.append(str(item["image"]))
    return urls


def normalize_size(size: str, input_path: Path) -> str:
    if size.lower() != "auto":
        return size
    width, height = image_size(input_path)
    return f"{width}*{height}"


def download_image(url: str, output_path: Path, timeout: float) -> dict[str, Any]:
    response = requests.get(url, timeout=timeout)
    if response.status_code != 200:
        raise QwenImageEditError(f"image download failed with HTTP {response.status_code}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix or ".png"
    with tempfile.NamedTemporaryFile(
        prefix=f"{output_path.stem}.",
        suffix=suffix,
        dir=str(output_path.parent),
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(response.content)
    tmp_path.replace(output_path)
    return {
        "download_url": url,
        "content_type": response.headers.get("Content-Type"),
        "bytes": int(output_path.stat().st_size),
    }


def preserve_image_size(output_path: Path, target_size: tuple[int, int]) -> bool:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise QwenImageEditError(f"Pillow is required to resize edited image: {exc}") from exc
    with Image.open(output_path) as image:
        if image.size == target_size:
            return False
        resized = image.convert("RGB").resize(target_size, Image.Resampling.LANCZOS)
        suffix = output_path.suffix or ".png"
        with tempfile.NamedTemporaryFile(
            prefix=f"{output_path.stem}.resize.",
            suffix=suffix,
            dir=str(output_path.parent),
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
        resized.save(tmp_path)
        tmp_path.replace(output_path)
        return True


def call_qwen_image_edit(
    *,
    input_path: Path,
    prompt: str,
    model: str,
    size: str,
    api_key: str,
    n: int,
) -> Any:
    try:
        import dashscope
        from dashscope import MultiModalConversation
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise QwenImageEditError(f"dashscope SDK is required: {exc}") from exc

    dashscope.api_key = api_key
    messages = [
        {
            "role": "user",
            "content": [
                {"image": image_to_data_url(input_path)},
                {"text": prompt},
            ],
        }
    ]
    return MultiModalConversation.call(
        model=model,
        messages=messages,
        n=n,
        size=size,
        stream=False,
    )


def call_seedream_image_edit(
    *,
    input_path: Path,
    prompt: str,
    model: str,
    size: str,
    api_key: str,
    base_url: str,
    response_format: str,
    watermark: bool,
    timeout: float,
) -> Any:
    endpoint = f"{base_url.rstrip('/')}/images/generations"
    payload = {
        "model": model,
        "prompt": prompt,
        "response_format": response_format,
        "size": size,
        "stream": False,
        "watermark": bool(watermark),
        "image": image_to_data_url(input_path),
    }
    response = requests.post(
        endpoint,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        json=payload,
        timeout=timeout,
    )
    try:
        data = response.json()
    except Exception:
        data = {"raw_text": response.text}
    if response.status_code >= 400:
        raise QwenImageEditError(f"Seedream HTTP {response.status_code}: {data}")
    return data


def seedream_image_urls(response: Any) -> list[str]:
    urls: list[str] = []
    if isinstance(response, dict):
        for item in response.get("data") or []:
            if isinstance(item, dict):
                url = item.get("url")
                if url:
                    urls.append(str(url))
        return urls
    for item in getattr(response, "data", None) or []:
        url = getattr(item, "url", None)
        if url:
            urls.append(str(url))
    return urls


def edit_background(
    *,
    input_path: Path,
    output_path: Path,
    prompt: str = DEFAULT_PROMPT,
    backend: str = DEFAULT_BACKEND,
    model: str | None = None,
    size: str = "auto",
    fallback_size: str = DEFAULT_FALLBACK_SIZE,
    retry_fallback_size: bool = True,
    preserve_size: bool = True,
    timeout: float = 300.0,
    n: int = 1,
    api_key: str | None = None,
    base_url: str = DEFAULT_SEEDDREAM_BASE_URL,
    response_format: str = "url",
    watermark: bool = True,
) -> dict[str, Any]:
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    target_size = image_size(input_path)
    backend = backend.strip().lower()
    if backend not in {"qwen", "seeddream"}:
        raise QwenImageEditError(f"unsupported background image backend: {backend}")
    if model is None:
        model = DEFAULT_SEEDDREAM_MODEL if backend == "seeddream" else DEFAULT_QWEN_MODEL
    resolved_api_key = resolve_api_key(backend, api_key)
    requested_sizes = [normalize_size(size, input_path) if backend == "qwen" else size]
    if backend == "qwen" and retry_fallback_size and fallback_size and fallback_size not in requested_sizes:
        requested_sizes.append(fallback_size)

    last_error: str | None = None
    response = None
    used_size = requested_sizes[0]
    for candidate_size in requested_sizes:
        used_size = candidate_size
        if backend == "seeddream":
            response = call_seedream_image_edit(
                input_path=input_path,
                prompt=prompt,
                model=model,
                size=candidate_size,
                api_key=resolved_api_key,
                base_url=base_url,
                response_format=response_format,
                watermark=watermark,
                timeout=timeout,
            )
            break
        response = call_qwen_image_edit(
            input_path=input_path,
            prompt=prompt,
            model=model,
            size=candidate_size,
            api_key=resolved_api_key,
            n=n,
        )
        if getattr(response, "status_code", None) == 200:
            break
        last_error = f"{getattr(response, 'code', None)} - {getattr(response, 'message', None)}"
        response = None

    if response is None:
        raise QwenImageEditError(f"{backend} image edit failed: {last_error}")

    image_urls = seedream_image_urls(response) if backend == "seeddream" else response_image_urls(response)
    if not image_urls:
        raise QwenImageEditError(f"{backend} response did not contain an edited image URL")

    download_info = download_image(image_urls[0], output_path, timeout=timeout)
    resized = preserve_image_size(output_path, target_size) if preserve_size else False
    output_size = image_size(output_path)

    return {
        "status": "ok",
        "backend": backend,
        "input": str(input_path),
        "output": str(output_path),
        "model": model,
        "prompt": prompt,
        "requested_size": used_size,
        "base_url": base_url if backend == "seeddream" else None,
        "response_format": response_format if backend == "seeddream" else None,
        "watermark": bool(watermark) if backend == "seeddream" else None,
        "input_size": [int(target_size[0]), int(target_size[1])],
        "output_size": [int(output_size[0]), int(output_size[1])],
        "preserve_size": bool(preserve_size),
        "resized_to_input_size": bool(resized),
        "download": download_info,
    }


def write_report(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    config = background_image_config()
    default_backend = str(os.environ.get("BACKGROUND_IMAGE_BACKEND") or config.get("backend") or DEFAULT_BACKEND)
    parser = argparse.ArgumentParser(description="Remove foreground objects from a session image with Qwen or Seedream image edit.")
    parser.add_argument("--session", type=Path, default=None, help="Session directory. Uses input/image.png -> input/background.png.")
    parser.add_argument("--input", type=Path, default=None, help="Input image. Overrides --session default.")
    parser.add_argument("--output", type=Path, default=None, help="Edited output image. Defaults to input/background.png.")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON report path.")
    parser.add_argument("--prompt", default=None, help="Override the auto session prompt.")
    parser.add_argument("--backend", choices=["qwen", "seeddream"], default=default_backend)
    parser.add_argument("--model", default=None)
    parser.add_argument("--size", default=None, help='Qwen supports "auto"; Seedream commonly uses "2K".')
    parser.add_argument("--fallback-size", default=None)
    parser.add_argument("--no-retry-fallback-size", action="store_true")
    parser.add_argument("--no-preserve-size", action="store_true")
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--response-format", default=None)
    parser.add_argument("--watermark", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--api-key", default=None, help="Avoid using this in logs; environment variables are preferred.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = background_image_config()
        provider = config.get(args.backend) if isinstance(config.get(args.backend), dict) else {}
        model = args.model or os.environ.get("BACKGROUND_IMAGE_MODEL") or provider.get("model")
        size = args.size or os.environ.get("BACKGROUND_IMAGE_SIZE") or provider.get("size") or ("2K" if args.backend == "seeddream" else "auto")
        fallback_size = args.fallback_size or provider.get("fallback_size") or DEFAULT_FALLBACK_SIZE
        timeout = float(args.timeout if args.timeout is not None else config.get("timeout", 300.0))
        base_url = args.base_url or os.environ.get("BACKGROUND_IMAGE_SEEDDREAM_BASE_URL") or provider.get("base_url") or DEFAULT_SEEDDREAM_BASE_URL
        response_format = args.response_format or provider.get("response_format") or "url"
        watermark = args.watermark if args.watermark is not None else bool(provider.get("watermark", True))
        input_path, output_path, report_path = resolve_paths(
            session=args.session,
            input_path=args.input,
            output_path=args.output,
            report_path=args.report,
        )
        prompt, prompt_info = resolve_prompt(args.prompt, args.session)
        payload = edit_background(
            input_path=input_path,
            output_path=output_path,
            prompt=prompt,
            backend=args.backend,
            model=str(model) if model else None,
            size=str(size),
            fallback_size=str(fallback_size),
            retry_fallback_size=not args.no_retry_fallback_size,
            preserve_size=not args.no_preserve_size,
            timeout=timeout,
            n=int(args.n),
            api_key=args.api_key,
            base_url=str(base_url),
            response_format=str(response_format),
            watermark=watermark,
        )
        payload["prompt_source"] = prompt_info["source"]
        payload["foreground_objects"] = prompt_info["objects"]
        write_report(report_path, payload)
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    except Exception as exc:
        payload = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        try:
            _, _, report_path = resolve_paths(
                session=args.session,
                input_path=args.input,
                output_path=args.output,
                report_path=args.report,
            )
            write_report(report_path, payload)
        except Exception:
            pass
        print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)
        return 2 if isinstance(exc, (FileNotFoundError, QwenImageEditError)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
