from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

import cv2
import numpy as np
import torch
from manga_ocr import MangaOcr
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from app.core.custom_conf import custom_conf
from app.core.logger import logger
from app.core.paths import MODELS_DIR

TEXT_BUBBLE_MODEL_PATH = MODELS_DIR / "comic-text-and-bubble-detector"
TEXT_BUBBLE_LABEL = "text_bubble"
TEXT_BUBBLE_CONFIDENCE = 0.8
MOCR_MODEL_PATH = MODELS_DIR / "manga-ocr-base"

_MODEL_LOCK = Lock()
_DET_MODEL: TextBubbleDetector | None = None
_MOCR: MangaOcr | None = None
_MOCR_DEVICE: str | None = None
_GPU_FALLBACK_REASON = ""


@dataclass(frozen=True)
class TextBubbleDetector:
    processor: Any
    model: torch.nn.Module
    device: torch.device
    label_id: int


def _is_cuda_related_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "cuda",
            "cudnn",
            "no kernel image",
            "driver",
            "device-side assert",
        )
    )


def _is_cuda_runtime_usable() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() = False"
    try:
        a = torch.tensor([1, 2, 3], device="cuda")
        b = torch.tensor([2], device="cuda")
        _ = torch.isin(a, b)
        torch.cuda.synchronize()
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _resolve_device() -> tuple[torch.device, bool]:
    global _GPU_FALLBACK_REASON

    gpu_enabled = custom_conf.use_gpu
    cuda_usable, cuda_fail_reason = _is_cuda_runtime_usable() if gpu_enabled else (False, "")
    use_cuda = gpu_enabled and cuda_usable
    _GPU_FALLBACK_REASON = cuda_fail_reason if gpu_enabled and not use_cuda else ""

    if gpu_enabled and not use_cuda:
        logger.warning(f"检测到 GPU 已启用但 CUDA 不可用，自动回退 CPU。原因：{cuda_fail_reason}")

    return torch.device("cuda:0") if use_cuda else torch.device("cpu"), use_cuda


def _resolve_label_id(model: torch.nn.Module) -> int:
    label2id = getattr(model.config, "label2id", {}) or {}
    if TEXT_BUBBLE_LABEL in label2id:
        return int(label2id[TEXT_BUBBLE_LABEL])

    id2label = getattr(model.config, "id2label", {}) or {}
    for raw_id, label in id2label.items():
        if label == TEXT_BUBBLE_LABEL:
            return int(raw_id)

    raise RuntimeError(f"Detector label not found: {TEXT_BUBBLE_LABEL}")


def warmup_models() -> tuple[TextBubbleDetector, MangaOcr]:
    global _DET_MODEL, _GPU_FALLBACK_REASON, _MOCR, _MOCR_DEVICE

    with _MODEL_LOCK:
        if _DET_MODEL is not None and _MOCR is not None:
            return _DET_MODEL, _MOCR

        device, use_cuda = _resolve_device()

        if _DET_MODEL is None:
            processor = AutoImageProcessor.from_pretrained(
                str(TEXT_BUBBLE_MODEL_PATH),
                local_files_only=True,
                use_fast=False,
            )
            model = AutoModelForObjectDetection.from_pretrained(
                str(TEXT_BUBBLE_MODEL_PATH),
                local_files_only=True,
            )
            try:
                model = model.to(device)
            except Exception as exc:
                if not use_cuda or not _is_cuda_related_error(exc):
                    raise
                logger.warning(f"文字气泡检测模型 CUDA 初始化失败，自动回退 CPU。原因：{exc}")
                _GPU_FALLBACK_REASON = str(exc)
                device = torch.device("cpu")
                use_cuda = False
                model = model.to(device)
            model.eval()
            _DET_MODEL = TextBubbleDetector(
                processor=processor,
                model=model,
                device=device,
                label_id=_resolve_label_id(model),
            )
            logger.info(f"Text bubble detector loaded on {_DET_MODEL.device}")

        if _MOCR is None:
            if use_cuda:
                try:
                    _MOCR = MangaOcr(pretrained_model_name_or_path=str(MOCR_MODEL_PATH), force_cpu=False)
                    _MOCR_DEVICE = "cuda"
                    logger.info("MangaOCR 加载成功，使用：cuda")
                except Exception as exc:
                    if not _is_cuda_related_error(exc):
                        raise
                    logger.warning(f"MangaOCR CUDA 初始化失败，自动回退 CPU。原因：{exc}")
                    _GPU_FALLBACK_REASON = str(exc)
                    _MOCR = MangaOcr(pretrained_model_name_or_path=str(MOCR_MODEL_PATH), force_cpu=True)
                    _MOCR_DEVICE = "cpu"
                    logger.info("MangaOCR 加载成功，使用：cpu")
            else:
                _MOCR = MangaOcr(pretrained_model_name_or_path=str(MOCR_MODEL_PATH), force_cpu=True)
                _MOCR_DEVICE = "cpu"
                logger.info("MangaOCR 加载成功，使用：cpu")

        return _DET_MODEL, _MOCR


def reset_models() -> None:
    """释放模型单例，使下一次 OCR 请求按最新设备配置重新加载。"""
    global _DET_MODEL, _GPU_FALLBACK_REASON, _MOCR, _MOCR_DEVICE

    with _MODEL_LOCK:
        _DET_MODEL = None
        _MOCR = None
        _MOCR_DEVICE = None
        _GPU_FALLBACK_REASON = ""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("OCR 模型设备配置已更新，将在下一次请求时重新加载")


def get_gpu_status() -> dict[str, Any]:
    requested = bool(custom_conf.use_gpu)
    with _MODEL_LOCK:
        detector_device = str(_DET_MODEL.device) if _DET_MODEL is not None else None
        mocr_device = _MOCR_DEVICE
        loaded = _DET_MODEL is not None and _MOCR is not None
        fallback_reason = _GPU_FALLBACK_REASON

    if loaded:
        detector_uses_gpu = bool(detector_device and detector_device.startswith("cuda"))
        mocr_uses_gpu = mocr_device == "cuda"
        uses_gpu = detector_uses_gpu or mocr_uses_gpu
        fully_on_gpu = detector_uses_gpu and mocr_uses_gpu
        available = uses_gpu if requested else None
        if requested and not fully_on_gpu:
            message = "GPU 未能用于全部 OCR 模型，已自动回退到 CPU"
        else:
            message = ""
    elif requested:
        available, fallback_reason = _is_cuda_runtime_usable()
        uses_gpu = available
        message = "" if available else f"GPU 不可用，将自动使用 CPU：{fallback_reason}"
    else:
        available = None
        uses_gpu = False
        message = ""

    return {
        "requested": requested,
        "available": available,
        "device": "gpu" if uses_gpu else "cpu",
        "models_loaded": loaded,
        "detector_device": detector_device,
        "mocr_device": mocr_device,
        "fallback_reason": fallback_reason if requested and not uses_gpu else "",
        "message": message,
    }


def get_det_model() -> TextBubbleDetector:
    det_model, _ = warmup_models()
    return det_model


def detect_text_bubbles(image_cv: np.ndarray) -> np.ndarray:
    detector = get_det_model()
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_rgb)
    inputs = detector.processor(images=image, return_tensors="pt")
    inputs = {
        key: value.to(detector.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = detector.model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=detector.device)
    result = detector.processor.post_process_object_detection(
        outputs,
        threshold=TEXT_BUBBLE_CONFIDENCE,
        target_sizes=target_sizes,
    )[0]

    labels = result["labels"].detach().cpu().numpy()
    boxes = result["boxes"].detach().cpu().numpy()
    text_bubble_boxes = boxes[labels == detector.label_id]
    return text_bubble_boxes.astype(np.float32, copy=False)


def get_mocr() -> MangaOcr:
    _, mocr = warmup_models()
    return mocr
