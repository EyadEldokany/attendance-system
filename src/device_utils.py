"""
Device auto-detection utility.

Resolves the compute device ("cuda" or "cpu") based on what's actually
available on the machine, so the app prefers GPU when present and silently
falls back to CPU when it isn't - instead of crashing inside ultralytics /
onnxruntime with "Invalid CUDA 'device=0' requested".

Resolution rules for a requested device string:
  - "auto"  : use CUDA if torch reports a usable GPU, else CPU.
  - "cuda"  : use CUDA if available, else fall back to CPU (with a warning).
  - "cpu"   : always CPU.
  - other   : treated as "auto".

The chosen device is cached after the first call so the log message only
appears once and the (relatively expensive) availability checks run once.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("attendance.device")

_cached_device: str | None = None


def _cuda_available() -> bool:
    """True if torch can actually use a CUDA device."""
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception as e:  # pragma: no cover - torch should be installed
        logger.warning(f"torch not available for CUDA check: {e}")
        return False


def onnxruntime_gpu_available() -> bool:
    """True if onnxruntime exposes a CUDA execution provider."""
    try:
        import onnxruntime as ort  # type: ignore
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception as e:  # pragma: no cover
        logger.warning(f"onnxruntime not available for GPU check: {e}")
        return False


def resolve_device(requested: str = "auto") -> str:
    """
    Resolve the final device string ("cuda" or "cpu") to hand to
    DetectorTracker / FaceRecognizer.

    Call this once at startup; subsequent calls return the cached result
    (ignoring `requested`) so every component agrees on the same device.
    """
    global _cached_device
    if _cached_device is not None:
        return _cached_device

    requested = (requested or "auto").strip().lower()
    cuda_ok = _cuda_available()
    onnx_gpu_ok = onnxruntime_gpu_available()

    if requested == "cpu":
        chosen = "cpu"
    elif requested == "cuda":
        if cuda_ok:
            chosen = "cuda"
        else:
            logger.warning(
                "GPU requested but not available (torch.cuda.is_available()=False). "
                "Falling back to CPU."
            )
            chosen = "cpu"
    else:  # "auto" or anything else
        chosen = "cuda" if cuda_ok else "cpu"

    # InsightFace uses onnxruntime; warn if we picked CUDA but onnxruntime
    # can't actually use it (e.g. only onnxruntime-cpu installed). We still
    # keep "cuda" for YOLO and let InsightFace fall back to CPU internally.
    if chosen == "cuda" and not onnx_gpu_ok:
        logger.warning(
            "CUDA selected for YOLO, but onnxruntime has no CUDAExecutionProvider "
            "(is onnxruntime-gpu installed?). InsightFace will run on CPU."
        )

    _cached_device = chosen
    logger.info(f"Compute device resolved: '{chosen}' (requested='{requested}')")
    return chosen


def reset_device_cache() -> None:
    """Clear the cached device choice (mainly for tests)."""
    global _cached_device
    _cached_device = None
