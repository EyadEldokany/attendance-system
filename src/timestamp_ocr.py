"""
Reads the burned-in date/time overlay that CCTV footage stamps onto every
frame, so archived video (which isn't named with its recording date) can be
converted into real-world timestamps.

IMPORTANT - read this before trusting this module's output blindly:
Testing against the actual sample footage from this project showed that
this camera's on-screen timestamp uses a digital/seven-segment-style font,
rendered through a semi-transparent overlay box whose contrast varies a lot
depending on what's behind it (a dark door vs. a bright marble floor, for
example). Standard OCR engines (including Tesseract) are trained on normal
fonts, not seven-segment displays, and their read quality on this specific
overlay is inconsistent - sometimes a full clean read, sometimes partial or
wrong digits.

Because of that, this module is deliberately NOT used as a silent, fully-
automatic source of truth. It produces a *best-effort guess* with an honest
confidence flag, meant to pre-fill a human confirmation step (see
archive_pipeline.py): a person glances at the cropped timestamp image next
to the OCR guess and confirms or corrects it ONCE per video file (not once
per event), then the system derives every subsequent event's exact
timestamp from that single confirmed start time plus the video's frame
rate - which is far more reliable than trusting repeated OCR reads.
"""
import re
import logging
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytesseract

try:
    import easyocr
except ImportError:  # pragma: no cover - dependency is installed from requirements
    easyocr = None

logger = logging.getLogger("attendance.timestamp_ocr")

# Matches things like "08-06-2026 09:30:08", "08/06/2026 09:30:08", "2026-08-06 09:30:08", etc.
_FULL_PATTERN = re.compile(
    r"(\d{2,4})[-/](\d{2})[-/](\d{2,4})\D{0,3}(\d{2}):(\d{2}):(\d{2})"
)
_DIGIT_WHITELIST = "0123456789:-/ "
_EASYOCR_READER = None


def configure_tesseract_path(tesseract_cmd: str = ""):
    """
    Call this once at startup if pytesseract can't auto-detect the Tesseract
    engine (common on Windows when it's not added to PATH during install).
    """
    candidates = [tesseract_cmd] if tesseract_cmd else [
        shutil.which("tesseract"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            pytesseract.pytesseract.tesseract_cmd = candidate
            logger.info(f"Using Tesseract path: {candidate}")
            return candidate
    logger.warning(
        "Tesseract OCR engine was not found. Install Tesseract-OCR or set "
        "archive.tesseract_cmd in config.yaml."
    )
    return None


def _candidate_thresholds(gray: np.ndarray):
    """
    Produces several binarized versions of the cropped timestamp region using
    different strategies, because no single threshold works well across every
    lighting condition this overlay can appear against. Returns a list of
    (label, image) pairs to try OCR on.
    """
    candidates = []

    # Seven-segment digits have crisp bright edges; mild sharpening helps
    # separate them from the semi-transparent background.
    sharpened = cv2.GaussianBlur(gray, (0, 0), 3)
    sharpened = cv2.addWeighted(gray, 1.8, sharpened, -0.8, 0)
    candidates.append(("sharpened", sharpened))

    # Plain fixed thresholds - works well when the overlay sits on a dark background.
    for tv in (150, 180, 200):
        _, t = cv2.threshold(gray, tv, 255, cv2.THRESH_BINARY)
        candidates.append((f"fixed_{tv}", t))

    # Illumination-normalized version - divides out the local background
    # brightness first, which helps a lot when the overlay sits on a bright
    # background (e.g. marble floor) where plain thresholding loses the text.
    gray_f = gray.astype(np.float32)
    bg = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=25)
    norm = cv2.divide(gray_f, bg, scale=255)
    norm = np.clip(norm, 0, 255).astype(np.uint8)
    for tv in (150, 160, 170, 180):
        _, t = cv2.threshold(norm, tv, 255, cv2.THRESH_BINARY)
        candidates.append((f"norm_{tv}", t))

    # Otsu automatic threshold on the normalized image as a final fallback.
    _, otsu = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates.append(("norm_otsu", otsu))

    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 41, 7,
    )
    candidates.append(("adaptive", adaptive))

    inverted = cv2.bitwise_not(sharpened)
    candidates.append(("inverted", inverted))
    _, inverted_binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates.append(("inverted_otsu", inverted_binary))

    return candidates


def _ocr_candidates(image, tesseract_lang):
    """Return OCR strings from several layouts and preprocessing variants."""
    texts = []
    for label, candidate in _candidate_thresholds(image):
        for psm in (6, 7, 8, 13):
            try:
                text = pytesseract.image_to_string(
                    candidate,
                    lang=tesseract_lang,
                    config=f"--psm {psm} -c tessedit_char_whitelist={_DIGIT_WHITELIST}",
                ).strip()
            except Exception as e:
                logger.debug(f"OCR attempt '{label}' psm={psm} failed: {e}")
                continue
            if text:
                texts.append(text)
    return texts


def _easyocr_candidates(image):
    """Read the complete timestamp crop with a digit-only OCR model."""
    global _EASYOCR_READER
    if easyocr is None:
        return []
    if _EASYOCR_READER is None:
        _EASYOCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)

    results = _EASYOCR_READER.readtext(
        image,
        allowlist="0123456789-/:",
        detail=1,
        paragraph=False,
        text_threshold=0.35,
        low_text=0.25,
        link_threshold=0.25,
    )
    results.sort(key=lambda item: min(point[0] for point in item[0]))
    text = " ".join(item[1] for item in results).strip()
    confidence = min((float(item[2]) for item in results), default=0.0)
    return [(text, confidence)] if text else []


def _preprocess_region(frame, region_norm):
    """
    region_norm: (x1, y1, x2, y2) as ratios (0-1) of frame width/height,
    matching the convention used for the door line in config.yaml.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = region_norm
    crop = frame[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]
    if crop.size == 0:
        return None, None
    # Upscaling substantially improves OCR accuracy on small burned-in text.
    big = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    return crop, gray


def read_timestamp(frame, region_norm, tesseract_lang: str = "eng"):
    """
    Attempts to read the burned-in timestamp inside region_norm.

    Returns a dict:
        {
            "datetime": datetime or None,   # parsed result, if a full match was found
            "raw_text": str,                # the raw OCR text that produced the best match
            "confidence": float,            # 0.0-1.0, heuristic based on match completeness
            "crop_bgr": np.ndarray or None, # the cropped region image, for showing to a human
        }

    A confidence of 1.0 means a full, well-formed date+time pattern was
    matched. Lower values (or a None datetime) mean the caller should not
    trust this automatically - route it to a human confirmation step instead.
    """
    crop, gray = _preprocess_region(frame, region_norm)
    if gray is None:
        return {"datetime": None, "raw_text": "", "confidence": 0.0, "crop_bgr": None}

    best = {"datetime": None, "raw_text": "", "confidence": 0.0, "crop_bgr": crop}

    easy_results = _easyocr_candidates(cv2.resize(
        crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC
    ))
    for text, ocr_confidence in easy_results:
        parsed, parse_confidence = _parse_candidate(text)
        if parsed is not None:
            # A complete valid parse from EasyOCR is stronger than a partial
            # Tesseract string, but retain the OCR confidence for the UI.
            best = {
                "datetime": parsed, "raw_text": text,
                "confidence": max(0.8, min(1.0, ocr_confidence)),
                "crop_bgr": crop,
            }
            return best

    candidates = _ocr_candidates(gray, tesseract_lang)
    # The date and time are visually separated. OCRing each half independently
    # prevents a missing separator from merging the date and time into garbage.
    midpoint = int(gray.shape[1] * 0.50)
    date_region = gray[:, :midpoint]
    time_region = gray[:, midpoint:]
    date_candidates = _ocr_candidates(date_region, tesseract_lang)
    time_candidates = _ocr_candidates(time_region, tesseract_lang)

    for date_text in date_candidates:
        for time_text in time_candidates:
            combined = f"{date_text} {time_text}"
            parsed, confidence = _parse_candidate(combined)
            if confidence > best["confidence"]:
                best = {
                    "datetime": parsed, "raw_text": combined,
                    "confidence": confidence, "crop_bgr": crop,
                }

    for text in candidates:
        parsed, confidence = _parse_candidate(text)
        if confidence > best["confidence"]:
            best = {
                "datetime": parsed, "raw_text": text,
                "confidence": confidence, "crop_bgr": crop,
            }
        if confidence >= 1.0:
            break

    return best


def _parse_candidate(text: str):
    """
    Tries to extract a full datetime from a raw OCR string. Returns
    (datetime_or_None, confidence). A partial/garbled string that still
    contains recognizable digit groups gets a low but non-zero confidence,
    which is enough to flag "OCR found something, but needs a human check"
    rather than "nothing readable at all".
    """
    match = _FULL_PATTERN.search(text)
    if match:
        a, b, c, hh, mm, ss = match.groups()
        # Figure out which group is the 4-digit year.
        parts = [a, b, c]
        year_idx = next((i for i, p in enumerate(parts) if len(p) == 4), None)
        try:
            if year_idx == 2:  # DD-MM-YYYY or MM-DD-YYYY
                first, second, year = int(a), int(b), int(c)
                # This camera uses MM-DD-YYYY. If the first field is above
                # 12, it is unambiguously DD-MM-YYYY instead.
                if first > 12:
                    month, day = second, first
                else:
                    month, day = first, second
                dt = datetime(year, month, day, int(hh), int(mm), int(ss))
            elif year_idx == 0:  # YYYY-MM-DD
                dt = datetime(int(a), int(b), int(c), int(hh), int(mm), int(ss))
            else:
                return None, 0.2
            return dt, 1.0
        except ValueError:
            return None, 0.2

    # Tesseract often drops all separators on this display, producing a
    # compact 14-digit value such as 08062026093004.
    digits = re.sub(r"\D", "", text)
    for candidate in (digits, digits.replace("0", "0")):
        if len(candidate) < 14:
            continue
        for start in range(len(candidate) - 13):
            chunk = candidate[start:start + 14]
            try:
                first, second, year = int(chunk[0:2]), int(chunk[2:4]), int(chunk[4:8])
                month, day = (second, first) if first > 12 else (first, second)
                dt = datetime(
                    year, month, day, int(chunk[8:10]),
                    int(chunk[10:12]), int(chunk[12:14]),
                )
                return dt, 1.0
            except ValueError:
                continue

    # No full pattern - still give partial credit if it at least looks like
    # digit groups resembling a date/time, so the human reviewer sees "close".
    digit_groups = re.findall(r"\d+", text)
    if len(digit_groups) >= 3:
        return None, 0.4
    if digit_groups:
        return None, 0.15
    return None, 0.0
