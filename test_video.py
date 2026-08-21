#!/usr/bin/env python3
"""
Video debug test script — replays a local video through the attendance pipeline
with live cv2 overlays. Uses a temp SQLite DB; never touches data/attendance.db.

Usage:
    python test_video.py <path/to/video.mp4>
"""
import sys
import os
import tempfile
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import Config, load_config
from src.detector_tracker import DetectorTracker
from src.door_crossing import DoorCrossingTracker
from src.face_recognizer import FaceRecognizer
from src.database import Database

FLASH_DURATION_FRAMES = 30
BOX_COLOR      = (0, 255, 0)
LINE_COLOR     = (255, 255, 255)
IN_FLASH_COLOR = (255, 100, 0)
OUT_FLASH_COLOR= (0, 50, 255)
PANEL_ALPHA    = 0.5
FONT           = cv2.FONT_HERSHEY_SIMPLEX


def draw_bounding_boxes(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        tid = det["track_id"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        label = f"Track #{tid}"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), BOX_COLOR, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), FONT, 0.55, (0, 0, 0), 1)


def draw_door_line(frame, p1_norm, p2_norm, frame_h, frame_w):
    p1 = (int(p1_norm[0] * frame_w), int(p1_norm[1] * frame_h))
    p2 = (int(p2_norm[0] * frame_w), int(p2_norm[1] * frame_h))
    cv2.line(frame, p1, p2, LINE_COLOR, 2)
    cv2.putText(frame, "door line", (p1[0] + 6, p1[1] + 16), FONT, 0.45, LINE_COLOR, 1)


def draw_flash(frame, flash_events, frame_h, frame_w):
    for key in list(flash_events.keys()):
        direction, name, counter = flash_events[key]
        if counter <= 0:
            del flash_events[key]
            continue
        color = IN_FLASH_COLOR if direction == "in" else OUT_FLASH_COLOR
        label = f"{'ENTRY' if direction == 'in' else 'EXIT'}  {name}"
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, frame_h // 2 - 40), (frame_w, frame_h // 2 + 40), color, -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        (tw, th), _ = cv2.getTextSize(label, FONT, 1.2, 2)
        tx = (frame_w - tw) // 2
        cv2.putText(frame, label, (tx, frame_h // 2 + th // 2), FONT, 1.2, (255, 255, 255), 2)
        flash_events[key] = (direction, name, counter - 1)


def draw_event_log(frame, event_log, frame_h, frame_w):
    if not event_log:
        return
    recent = event_log[-5:]
    panel_h = len(recent) * 22 + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, frame_h - panel_h), (frame_w, frame_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, PANEL_ALPHA, frame, 1 - PANEL_ALPHA, 0, frame)
    for i, ev in enumerate(recent):
        y = frame_h - panel_h + 18 + i * 22
        dir_label = "IN " if ev["event_type"] == "in" else "OUT"
        name = ev.get("name", "Unknown")
        conf_str = f"{ev['confidence']:.2f}" if ev.get("confidence") is not None else "--"
        text = (f"{ev['timestamp'][11:19]}  {dir_label}  {name:<20}"
                f"  conf:{conf_str}  [{ev['status']}]")
        cv2.putText(frame, text, (8, y), FONT, 0.42, (200, 255, 200), 1)


def draw_frame_info(frame, frame_num, fps):
    text = f"Frame: {frame_num}  |  FPS: {fps:.1f}"
    cv2.putText(frame, text, (8, 22), FONT, 0.55, (220, 220, 220), 1)


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_video.py <path/to/video.mp4>")
        sys.exit(1)

    video_path = sys.argv[1]
    if not Path(video_path).exists():
        print(f"Error: video file not found: {video_path}")
        sys.exit(1)

    # Reset Config singleton so we always load fresh from project config.yaml
    Config._instance = None
    config_path = str(Path(__file__).resolve().parent / "config.yaml")
    cfg = load_config(config_path)

    # Temp database — deleted in the finally block
    tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp_db_path = tmp_db.name
    tmp_db.close()

    cap = None
    db = None
    frame_num = 0
    crossings = []

    try:
        db = Database(tmp_db_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: cannot open video: {video_path}")
            sys.exit(1)

        video_fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        wait_ms    = max(1, int(1000 / video_fps))

        detector = DetectorTracker(
            model_path=str(
                Path(__file__).resolve().parent /
                cfg.get("detection.model_path", "models/yolov8n.pt")
            ),
            confidence_threshold=cfg.get("detection.confidence_threshold", 0.5),
            device="cpu",
            tracker_type=cfg.get("tracking.tracker_type", "bytetrack"),
        )

        crossing_tracker = DoorCrossingTracker(
            point1=cfg.get("door_line.point1", [0.5, 0.0]),
            point2=cfg.get("door_line.point2", [0.5, 1.0]),
            entry_direction=cfg.get("door_line.entry_direction", "positive_to_negative"),
            min_crossing_confidence_frames=cfg.get("door_line.min_crossing_confidence_frames", 3),
            frame_width=frame_w,
            frame_height=frame_h,
        )

        recognizer = FaceRecognizer(
            model_name=cfg.get("recognition.model_name", "buffalo_l"),
            detector=cfg.get("recognition.detector", "scrfd"),
            embedding_dim=cfg.get("recognition.embedding_dim", 512),
            match_threshold=cfg.get("recognition.match_threshold", 0.62),
            review_threshold=cfg.get("recognition.review_threshold", 0.45),
            min_face_size_px=cfg.get("recognition.min_face_size_px", 40),
            device="cpu",
            model_root=cfg.get("recognition.model_root", "~/.insightface"),
        )
        recognizer.rebuild_index(db.get_all_embeddings())

        print(f"Components loaded. Video: {frame_w}x{frame_h} @ {video_fps:.1f} fps")
        print("Controls: SPACE = pause/resume | Q = quit")

        p1_norm         = cfg.get("door_line.point1", [0.5, 0.0])
        p2_norm         = cfg.get("door_line.point2", [0.5, 1.0])
        flash_events    = {}
        event_log       = []
        paused          = False
        frame           = None
        last_detections = []

        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_num += 1
                last_detections = detector.track(frame)

                for det in last_detections:
                    direction = crossing_tracker.update(det["track_id"], det["bbox"])
                    if direction:
                        ts = datetime.now().isoformat(timespec="seconds")
                        face_result = recognizer.extract_best_face(frame, det["bbox"])
                        if face_result:
                            emp_id, conf, status = recognizer.match(face_result["embedding"])
                            emp  = db.get_employee(emp_id) if emp_id else None
                            name = emp["full_name"] if emp else "Unknown"
                        else:
                            emp_id, conf, status, name = None, None, "needs_review", "Unknown"

                        db.log_event(direction, ts, emp_id, det["track_id"], conf, status)
                        ev = {
                            "event_type": direction,
                            "timestamp":  ts,
                            "name":       name,
                            "confidence": conf,
                            "status":     status,
                        }
                        event_log.append(ev)
                        crossings.append(ev)
                        flash_events[det["track_id"]] = (direction, name, FLASH_DURATION_FRAMES)
                        print(f"  Crossing: {'IN ' if direction == 'in' else 'OUT'}  {name}  [{status}]")

            if frame is None:
                key = cv2.waitKey(wait_ms) & 0xFF
                if key == ord("q") or key == 27:
                    break
                continue

            display = frame.copy()
            draw_bounding_boxes(display, last_detections)
            draw_door_line(display, p1_norm, p2_norm, frame_h, frame_w)
            draw_flash(display, flash_events, frame_h, frame_w)
            draw_event_log(display, event_log, frame_h, frame_w)
            draw_frame_info(display, frame_num, video_fps)

            cv2.imshow("Attendance System — Debug", display)
            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord(" "):
                paused = not paused

    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

        unresolved = db.get_pending_labeling(min_captures=0) if db is not None else []

        print("\n=== Test Run Summary ===")
        print(f"Video            : {video_path}")
        print(f"Frames played    : {frame_num}")
        print(f"Crossings        : {len(crossings)}")
        for ev in crossings:
            dir_label = "IN " if ev["event_type"] == "in" else "OUT"
            conf_str  = f"{ev['confidence']:.2f}" if ev.get("confidence") is not None else " -- "
            print(f"  {ev['timestamp']}  {dir_label}  {ev['name']:<20}  conf:{conf_str}  [{ev['status']}]")
        print(f"Events in DB     : {len(crossings)}")
        print(f"Unresolved tracks: {len(unresolved)}")

        try:
            os.unlink(tmp_db_path)
            print("Temp DB          : deleted")
        except OSError:
            print(f"Temp DB          : could not delete {tmp_db_path}")


if __name__ == "__main__":
    main()
