"""
End-to-end test harness: runs the real pipeline (detection + tracking + door
crossing + face recognition + logging) on a video file instead of a live
camera stream, with a live visual overlay (bounding boxes, door line, entry/
exit events) so you can visually confirm every component is working before
handing the system off to the client.

Writes to a completely separate test database and Excel file (test_data/ and
test_exports/) so the test run never touches real production data.

Usage:
    python test_with_video.py "D:/videos/sample.mp4"

Shortcuts while the video window is open:
    q       -> stop the test and show the final report
    space   -> pause / resume
    s       -> save a screenshot of the current frame to test_data/screenshots/
"""
import sys
import cv2
import time
from pathlib import Path
from datetime import datetime
from collections import Counter

from src.config import load_config
from src.database import Database
from src.excel_sync import ExcelSync
from src.detector_tracker import DetectorTracker
from src.door_crossing import DoorCrossingTracker
from src.face_recognizer import FaceRecognizer
from src.enrollment_manager import EnrollmentManager
from src.logging_setup import setup_logging

PROJECT_ROOT = Path(__file__).resolve().parent


def draw_overlay(frame, detections, door, direction_events, w, h):
    """Draws person bounding boxes, the door line, and recent crossing events on the frame for live viewing."""
    p1 = (int(door.p1[0]), int(door.p1[1]))
    p2 = (int(door.p2[0]), int(door.p2[1]))
    cv2.line(frame, p1, p2, (0, 255, 255), 3)
    cv2.putText(frame, "Door line", (p1[0] + 5, p1[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(frame, f"ID:{det['track_id']}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

    y_offset = 30
    for text in direction_events[-5:]:
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)
        y_offset += 30
    return frame


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_with_video.py <path_to_video_file>")
        sys.exit(1)

    video_path = sys.argv[1]
    if not Path(video_path).exists():
        print(f"File not found: {video_path}")
        sys.exit(1)

    setup_logging(log_dir="test_data/logs", level="INFO")
    config = load_config(str(PROJECT_ROOT / "config.yaml"))

    # completely separate test database and Excel file - the test run never
    # touches real production data
    test_db_path = PROJECT_ROOT / "test_data" / "test_attendance.db"
    test_excel_path = PROJECT_ROOT / "test_exports" / "test_attendance_log.xlsx"
    test_db_path.parent.mkdir(parents=True, exist_ok=True)
    screenshots_dir = PROJECT_ROOT / "test_data" / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    db = Database(str(test_db_path))
    excel_sync = ExcelSync(db=db, excel_path=str(test_excel_path))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open the video file - check the path and video format.")
        sys.exit(1)

    frame_w = config.get("camera.frame_width", 1280)
    frame_h = config.get("camera.frame_height", 720)

    print("Loading models (YOLOv8n + face recognition)... this may take a few seconds the first time.")
    detector = DetectorTracker(
        model_path=str(PROJECT_ROOT / config.get("detection.model_path")),
        confidence_threshold=config.get("detection.confidence_threshold", 0.5),
        device=config.get("detection.device", "auto"),
        tracker_type=config.get("tracking.tracker_type", "bytetrack"),
    )
    door = DoorCrossingTracker(
        point1=config.get("door_line.point1"),
        point2=config.get("door_line.point2"),
        entry_direction=config.get("door_line.entry_direction"),
        min_crossing_confidence_frames=config.get("door_line.min_crossing_confidence_frames", 3),
        frame_width=frame_w, frame_height=frame_h,
    )
    recognizer = FaceRecognizer(
        model_pack=config.get("recognition.model_pack", "buffalo_l"),
        embedding_dim=config.get("recognition.embedding_dim", 512),
        match_threshold=config.get("recognition.match_threshold", 0.62),
        review_threshold=config.get("recognition.review_threshold", 0.45),
        min_face_size_px=config.get("recognition.min_face_size_px", 40),
        device=config.get("detection.device", "auto"),
    )
    # Use the real production database's embeddings if available (so if some
    # employees are already enrolled, we can confirm recognition works for
    # them too). Otherwise it stays empty, which is expected on a fresh setup.
    prod_db_path = PROJECT_ROOT / config.get("database.sqlite_path")
    if prod_db_path.exists():
        prod_db = Database(str(prod_db_path))
        recognizer.rebuild_index(prod_db.get_all_embeddings())
    else:
        recognizer.rebuild_index([])

    enrollment = EnrollmentManager(
        db=db, recognizer=recognizer, faces_dir=str(PROJECT_ROOT / "test_data" / "captured_faces"),
        min_captures_before_prompt=config.get("enrollment.min_captures_before_prompt", 3),
    )

    print(f"Starting test run on: {video_path}")
    print("Press 'q' to stop and show the report, 'space' to pause, 's' to save a screenshot.\n")

    direction_log = []       # text lines drawn on screen
    event_counter = Counter()
    frame_idx = 0
    process_every_n = config.get("camera.process_every_n_frames", 3)
    paused = False
    start_time = time.time()
    display = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\nEnd of video reached.")
                break
            frame = cv2.resize(frame, (frame_w, frame_h))
            frame_idx += 1

            if frame_idx % process_every_n == 0:
                detections = detector.track(frame)

                for det in detections:
                    direction = door.update(det["track_id"], det["bbox"])
                    if direction is None:
                        continue

                    face_result = recognizer.extract_best_face(frame, det["bbox"])
                    timestamp = datetime.now().isoformat()

                    if face_result is None:
                        db.log_event(event_type=direction, timestamp=timestamp,
                                     track_id=det["track_id"], status="needs_review")
                        msg = f"[{direction.upper()}] ID:{det['track_id']} - no clear face"
                    else:
                        employee_id, confidence, status = recognizer.match(face_result["embedding"])
                        db.log_event(event_type=direction, timestamp=timestamp,
                                     employee_id=employee_id, track_id=det["track_id"],
                                     confidence=confidence, status=status)
                        if status != "confirmed":
                            enrollment.capture_unknown_face(
                                det["track_id"], frame, face_result["bbox"],
                                embedding=face_result["embedding"],
                            )
                        name = f"Employee #{employee_id}" if employee_id else "Unknown"
                        msg = (f"[{direction.upper()}] ID:{det['track_id']} - {name} "
                               f"(confidence: {confidence:.2f}, {status})")

                    print(msg)
                    direction_log.append(msg)
                    event_counter[direction] += 1

                display = draw_overlay(frame.copy(), detections, door, direction_log, frame_w, frame_h)
            else:
                display = frame.copy()

        if display is not None:
            cv2.imshow("System test on video - press q to stop", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("s") and display is not None:
            shot_path = screenshots_dir / f"shot_{int(time.time())}.jpg"
            cv2.imwrite(str(shot_path), display)
            print(f"Screenshot saved: {shot_path}")

    cap.release()
    cv2.destroyAllWindows()

    excel_sync.sync_once()
    elapsed = time.time() - start_time

    # ---------------- Final report ----------------
    print("\n" + "=" * 50)
    print("Test Results Report")
    print("=" * 50)
    print(f"Run duration: {elapsed:.1f}s | Frames processed: {frame_idx // process_every_n}")
    print(f"Entry events detected: {event_counter.get('in', 0)}")
    print(f"Exit events detected: {event_counter.get('out', 0)}")

    review_count = len(db.get_events_needing_review())
    print(f"Events needing manual review: {review_count}")
    pending_labeling = enrollment.get_pending_for_labeling()
    print(f"New people needing labels (seen 3+ times): {len(pending_labeling)}")

    print(f"\nTest database: {test_db_path}")
    print(f"Test Excel file: {test_excel_path}")
    print(f"Captured face crops: {PROJECT_ROOT / 'test_data' / 'captured_faces'}")
    print("\nNote: this is separate test data from the production database ")
    print("(data/attendance.db) and will not affect any real data in actual use.")


if __name__ == "__main__":
    main()
