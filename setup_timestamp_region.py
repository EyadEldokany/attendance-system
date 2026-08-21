"""
Interactive setup tool: pick a sample video, draw a box around the burned-in
date/time overlay, and immediately see what the OCR module reads from it -
so you can adjust the box until the read looks reasonable before running the
tool against the full archive. Saves the confirmed region into config.yaml.

Usage:
    python setup_timestamp_region.py "D:/path/to/sample_video.mp4"

- Click and drag to draw a box around the timestamp text.
- Press 't' to test-read the OCR result for the current box (shown in the console).
- Press 's' to save the box to config.yaml once you're happy with the OCR read.
- Press 'r' to redraw the box, 'q' to quit without saving.
"""
import sys
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config
from src.timestamp_ocr import read_timestamp

_drawing = False
_box = None  # (x1, y1, x2, y2) in pixel coords


def _on_mouse(event, x, y, flags, param):
    global _drawing, _box
    if event == cv2.EVENT_LBUTTONDOWN:
        _drawing = True
        _box = [x, y, x, y]
    elif event == cv2.EVENT_MOUSEMOVE and _drawing:
        _box[2], _box[3] = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        _drawing = False
        _box[2], _box[3] = x, y


def main():
    if len(sys.argv) < 2:
        print("Usage: python setup_timestamp_region.py <path_to_sample_video>")
        sys.exit(1)

    video_path = sys.argv[1]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        sys.exit(1)

    # Grab a frame a couple seconds in (avoids black/transition first frames).
    cap.set(cv2.CAP_PROP_POS_MSEC, 2000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("Could not read a frame from the video.")
        sys.exit(1)

    h, w = frame.shape[:2]
    config = load_config("config.yaml")

    from src.timestamp_ocr import configure_tesseract_path
    configure_tesseract_path(config.get("archive.tesseract_cmd", ""))

    window_name = "Draw a box around the timestamp - t=test OCR / s=save / r=redo / q=quit"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, _on_mouse)

    print("Click and drag a box around the burned-in date/time text.")
    print("Press 't' to test the OCR read, 's' to save once it looks right.")

    global _box
    while True:
        display = frame.copy()
        if _box:
            x1, y1, x2, y2 = _box
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("r"):
            _box = None
        elif key == ord("q"):
            break
        elif key == ord("t"):
            if not _box:
                print("Draw a box first.")
                continue
            region_norm = _box_to_norm(_box, w, h)
            result = read_timestamp(frame, region_norm)
            print(f"\nOCR raw text: {result['raw_text']!r}")
            print(f"Parsed datetime: {result['datetime']}")
            print(f"Confidence: {result['confidence']:.2f}")
            if result["confidence"] < 1.0:
                print(
                    "Note: this is a best-effort guess, not a guarantee - low/partial "
                    "confidence just means a human should confirm it during archive "
                    "processing, which is expected and fine.\n"
                )
        elif key == ord("s"):
            if not _box:
                print("Draw a box first.")
                continue
            region_norm = _box_to_norm(_box, w, h)
            config.set("archive.timestamp_region", list(region_norm), persist=True)
            print(f"\nSaved timestamp region to config.yaml: {region_norm}")
            break

    cv2.destroyAllWindows()


def _box_to_norm(box, w, h):
    x1, y1, x2, y2 = box
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return (x1 / w, y1 / h, x2 / w, y2 / h)


if __name__ == "__main__":
    main()
