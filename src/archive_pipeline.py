"""
Batch pipeline for analyzing a folder of pre-recorded CCTV footage (e.g. a
year of archived video) instead of a live camera stream.

Design decision - why "confirm once per video" instead of "OCR every event":
The burned-in timestamp on this footage is not reliably machine-readable on
every single frame (see timestamp_ocr.py for why). But within one video
file, the camera's internal clock runs continuously and its frame rate is
stable, so the timestamp of ANY frame in that file can be computed precisely
from a single known reference point: (video_start_time + frame_index / fps).
So instead of re-running fragile OCR for every detected crossing event, the
system only needs ONE trustworthy timestamp per video file - and a human
glancing at one cropped image per file to confirm/correct the OCR's best
guess is both faster and far more reliable than hoping OCR gets every event
right on its own.

Flow:
  1. scan_folder()      - registers every video file found, attempts an OCR
                           guess of its start time from the first few seconds.
  2. (human step)        - confirm_video_start() is called once per video,
                           either accepting the OCR guess or typing the
                           correct start date/time after looking at the crop.
  3. process_confirmed_videos() - runs detection + tracking + door-crossing +
                           face recognition across every confirmed video,
                           computing each event's real timestamp from the
                           confirmed start time + frame offset / fps.
"""
import logging
from pathlib import Path
from datetime import datetime, timedelta

import cv2
import numpy as np

from src.path_utils import resolve_path
from src.detector_tracker import DetectorTracker
from src.door_crossing import DoorCrossingTracker
from src.device_utils import resolve_device
from src.face_recognizer import FaceRecognizer
from src.enrollment_manager import EnrollmentManager
from src.timestamp_ocr import read_timestamp

logger = logging.getLogger("attendance.archive")

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".ts"}

# How many seconds into the video to sample while looking for the clearest
# possible frame to OCR - the very first frame is sometimes a black/blank
# transition frame, so trying a few spread across the first 10s is more robust.
_OCR_SAMPLE_SECONDS = [0.5, 2, 5, 10]


class ArchiveProcessor:
    def __init__(self, config, db):
        self.config = config
        self.db = db
        device = resolve_device(config.get("detection.device", "auto"))

        self.timestamp_region = config.get(
            "archive.timestamp_region", [0.02, 0.955, 0.30, 0.995]
        )
        self.thumbnails_dir = resolve_path(config.get("archive.thumbnails_dir", "archive_thumbnails"))
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)

        self.frame_width = config.get("camera.frame_width", 1280)
        self.frame_height = config.get("camera.frame_height", 720)
        self.process_every_n_frames = config.get("camera.process_every_n_frames", 3)

        self.detector = DetectorTracker(
            model_path=str(resolve_path(config.get("detection.model_path", "models/yolov8n.pt"))),
            confidence_threshold=config.get("detection.confidence_threshold", 0.5),
            device=device,
            tracker_type=config.get("tracking.tracker_type", "bytetrack"),
        )
        self.recognizer = FaceRecognizer(
            model_name=config.get("recognition.model_name", "buffalo_l"),
            model_root=config.get("recognition.model_root", "~/.insightface"),
            embedding_dim=config.get("recognition.embedding_dim", 512),
            match_threshold=config.get("recognition.match_threshold", 0.50),
            review_threshold=config.get("recognition.review_threshold", 0.35),
            min_face_size_px=config.get("recognition.min_face_size_px", 40),
            device=device,
            face_crop_padding=config.get("recognition.face_crop_padding", 0.20),
            face_crop_scale=config.get("recognition.face_crop_scale", 2.0),
        )
        embeddings = self.db.get_all_embeddings()
        self.recognizer.rebuild_index(embeddings)
        logger.info(
            "Recognition index ready with %d saved embedding(s) for %d employee(s).",
            len(embeddings), len({employee_id for employee_id, _ in embeddings}),
        )

        self.enrollment = EnrollmentManager(
            db=self.db,
            recognizer=self.recognizer,
            faces_dir=resolve_path(config.get("enrollment.faces_dir", "enrollment_faces")),
            min_captures_before_prompt=config.get("enrollment.min_captures_before_prompt", 3),
            max_embeddings_per_person=config.get("recognition.max_embeddings_per_person", 8),
        )

    # ---------------- Step 1: scan the folder ----------------

    def scan_folder(self, folder_path: str, progress_callback=None, stop_callback=None):
        """
        Walks the folder (recursively) for video files, registers each one in
        the video_files table, and attempts an OCR guess for its start time.
        Safe to re-run on the same folder - already-registered files are
        skipped (see Database.add_or_get_video).
        """
        folder = Path(folder_path)
        video_paths = sorted(
            p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS
        )
        logger.info(f"Found {len(video_paths)} video file(s) under {folder_path}")

        for idx, video_path in enumerate(video_paths):
            if stop_callback and stop_callback():
                logger.info("Archive scan stopped by user.")
                break
            if progress_callback:
                progress_callback(idx + 1, len(video_paths), str(video_path))
            self._register_video(video_path)

        return len(video_paths)

    def _register_video(self, video_path: Path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            self.db.add_or_get_video(str(video_path), fps=0, total_frames=0)
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        best_result = {"datetime": None, "raw_text": "", "confidence": 0.0, "crop_bgr": None}
        valid_results = []
        for sec in _OCR_SAMPLE_SECONDS:
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            result = read_timestamp(frame, self.timestamp_region)
            if result["datetime"] is not None:
                valid_results.append((sec, result))
            if result["confidence"] > best_result["confidence"]:
                best_result = result
        cap.release()

        # A single seven-segment OCR read can confidently confuse one digit.
        # Require two samples with the same calendar date before auto-accepting.
        if valid_results:
            date_counts = {}
            for _, result in valid_results:
                date_key = result["datetime"].date()
                date_counts[date_key] = date_counts.get(date_key, 0) + 1
            agreed_date, agreement_count = max(date_counts.items(), key=lambda item: item[1])
            agreed = [item for item in valid_results if item[1]["datetime"].date() == agreed_date]
            if agreement_count >= 2:
                # The earliest sample is closest to the video's start.
                _, best_result = min(agreed, key=lambda item: item[0])
                best_result = dict(best_result)
                best_result["confidence"] = 1.0
            else:
                best_result = dict(best_result)
                best_result["confidence"] = min(best_result["confidence"], 0.8)

        thumb_path = None
        if best_result["crop_bgr"] is not None:
            thumb_path = str(self.thumbnails_dir / f"{video_path.stem}_ts.jpg")
            cv2.imwrite(thumb_path, best_result["crop_bgr"])

        guess_text = (
            best_result["datetime"].isoformat()
            if best_result["datetime"] else best_result["raw_text"]
        )

        record = self.db.add_or_get_video(
            file_path=str(video_path), fps=fps, total_frames=total_frames,
            ocr_guess_text=guess_text, ocr_confidence=best_result["confidence"],
            thumbnail_path=thumb_path,
        )

        # High-confidence full matches can be auto-confirmed; anything else
        # is left as 'pending_confirmation' for a human to check - see the
        # module docstring for why this matters more than it might seem.
        if best_result["confidence"] >= 1.0 and best_result["datetime"] and record["status"] == "pending_confirmation":
            self.db.confirm_video_start(record["video_id"], best_result["datetime"].isoformat())
            logger.info(f"Auto-confirmed start time for {video_path.name}: {best_result['datetime']}")
        elif record["status"] == "pending_confirmation":
            logger.info(
                f"{video_path.name}: OCR guess='{guess_text}' confidence={best_result['confidence']:.2f} "
                f"- needs human confirmation."
            )
        else:
            logger.info(
                f"{video_path.name}: already has status '{record['status']}', "
                "leaving its existing archive state unchanged."
            )

    # ---------------- Step 2: human confirmation (called from the GUI) ----------------

    def get_pending_confirmations(self):
        return self.db.get_pending_confirmation_videos()

    def confirm_video_start(self, video_id: int, start_datetime: datetime):
        self.db.confirm_video_start(video_id, start_datetime.isoformat())

    def skip_video(self, video_id: int):
        self.db.skip_video(video_id)

    # ---------------- Step 3: process confirmed videos ----------------

    def process_confirmed_videos(self, progress_callback=None, stop_event=None, frame_callback=None):
        """
        Runs the full detection/tracking/recognition pipeline over every
        confirmed-but-unprocessed video. progress_callback(video_index,
        video_total, video_name, frame_index, frame_total) is called
        periodically so a GUI can show progress. frame_callback(frame_bgr,
        video_name), if given, is called with an annotated preview frame
        (bounding boxes + door line drawn on it) so a GUI can display the
        video being analyzed live while it processes.
        """
        self.recognizer.rebuild_index(self.db.get_all_embeddings())
        self._reconcile_review_events()
        videos = self.db.get_confirmed_unprocessed_videos()
        logger.info(f"Processing {len(videos)} confirmed video(s)...")

        for v_idx, video in enumerate(videos):
            if stop_event is not None and stop_event.is_set():
                logger.info("Archive processing stopped by user.")
                return
            try:
                # Rebuild index before each video so newly labeled employees
                # (from the Review tab) are picked up for subsequent videos.
                self.recognizer.rebuild_index(self.db.get_all_embeddings())
                self._process_one_video(video, v_idx, len(videos), progress_callback, stop_event, frame_callback)
                self.db.mark_video_processed(video["video_id"])
            except Exception as e:
                logger.exception(f"Error processing {video['file_path']}")
                self.db.mark_video_error(video["video_id"], str(e))

    def _process_one_video(self, video, v_idx, v_total, progress_callback, stop_event, frame_callback=None):
        path = video["file_path"]
        fps = video["fps"] or 25.0
        start_dt = datetime.fromisoformat(video["start_timestamp"])
        total_frames = video["total_frames"] or 0

        logger.info(f"Processing {path} (start={start_dt}, fps={fps:.1f})")

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not reopen video file: {path}")

        door = DoorCrossingTracker(
            point1=self.config.get("door_line.point1"),
            point2=self.config.get("door_line.point2"),
            entry_direction=self.config.get("door_line.entry_direction"),
            min_crossing_confidence_frames=self.config.get("door_line.min_crossing_confidence_frames", 3),
            frame_width=self.frame_width, frame_height=self.frame_height,
        )
        door_p1 = (int(door.p1[0]), int(door.p1[1]))
        door_p2 = (int(door.p2[0]), int(door.p2[1]))
        recent_events = []  # short rolling log shown on the preview overlay
        best_faces = {}
        face_sample_interval = max(
            1, int(self.config.get("recognition.face_sample_interval", 5))
        )

        frame_idx = 0
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            if progress_callback and frame_idx % 50 == 0:
                progress_callback(v_idx + 1, v_total, Path(path).name, frame_idx, total_frames)

            if frame_idx % self.process_every_n_frames != 0:
                continue

            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            detections = self.detector.track(frame)
            event_time = start_dt + timedelta(seconds=frame_idx / fps)
            room_faces = (
                self.recognizer.extract_faces(frame)
                if frame_idx % face_sample_interval == 0 else []
            )

            for det in detections:
                track_id = det["track_id"]
                face_result = self._face_for_detection(room_faces, det["bbox"])
                if face_result is not None:
                    face_area = max(0, face_result["bbox"][2] - face_result["bbox"][0]) * max(
                        0, face_result["bbox"][3] - face_result["bbox"][1]
                    )
                    previous = best_faces.get(track_id)
                    previous_area = previous["area"] if previous else 0
                    if face_area > previous_area:
                        best_faces[track_id] = {
                            "result": face_result, "area": face_area, "frame": frame.copy()
                        }

                direction = door.update(det["track_id"], det["bbox"])
                if direction is None:
                    continue
                cached = best_faces.pop(track_id, None)
                employee_label = self._handle_crossing(
                    det, direction, frame, event_time,
                    cached["result"] if cached else None,
                    cached["frame"] if cached else None,
                )
                recent_events.append(f"[{direction.upper()}] {event_time.strftime('%H:%M:%S')} {employee_label}")
                recent_events = recent_events[-5:]

            if frame_callback is not None:
                annotated = self._draw_overlay(frame, detections, door_p1, door_p2, recent_events)
                frame_callback(annotated, Path(path).name)

        cap.release()

    @staticmethod
    def _face_for_detection(faces, person_bbox):
        """Choose the largest room-detected face whose center belongs to a person box."""
        x1, y1, x2, y2 = person_bbox
        matching = []
        for face in faces:
            fx1, fy1, fx2, fy2 = face["bbox"]
            center_x = (fx1 + fx2) / 2
            center_y = (fy1 + fy2) / 2
            if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                area = max(0, fx2 - fx1) * max(0, fy2 - fy1)
                matching.append((area, face))
        return max(matching, key=lambda item: item[0])[1] if matching else None

    @staticmethod
    def _draw_overlay(frame, detections, door_p1, door_p2, recent_events):
        """Draws person bounding boxes, the door line, and recent crossing events for a live preview."""
        annotated = frame.copy()
        cv2.line(annotated, door_p1, door_p2, (0, 255, 255), 3)
        cv2.putText(annotated, "Door line", (door_p1[0] + 5, door_p1[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(annotated, f"ID:{det['track_id']}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        y_offset = 30
        for text in recent_events[-5:]:
            cv2.putText(annotated, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)
            y_offset += 26
        return annotated

    def _handle_crossing(self, det, direction, frame, event_time, face_result=None,
                         face_frame=None):
        source_frame = face_frame if face_frame is not None else frame
        if face_result is None:
            face_result = self.recognizer.extract_best_face(frame, det["bbox"])
        timestamp_str = event_time.isoformat()

        if face_result is None:
            logger.warning(f"Track {det['track_id']} crossed ({direction}) at {event_time} with no clear face.")
            self.db.log_event(
                event_type=direction, timestamp=timestamp_str, employee_id=None,
                track_id=det["track_id"], confidence=None, status="needs_review",
            )
            return "no clear face"

        employee_id, confidence, status = self.recognizer.match(face_result["embedding"])
        face_crop_path = None
        if status != "confirmed":
            face_crop_path = self.enrollment.capture_unknown_face(
                det["track_id"], source_frame, face_result["bbox"],
                embedding=face_result["embedding"], replace_best=True,
            )

        self.db.log_event(
            event_type=direction, timestamp=timestamp_str, employee_id=employee_id,
            track_id=det["track_id"], confidence=confidence, status=status,
            face_crop_path=face_crop_path,
        )
        logger.info(
            f"[{direction.upper()}] {event_time} | track {det['track_id']} | "
            f"{'employee #' + str(employee_id) if employee_id else 'unknown'} | "
            f"confidence={confidence:.2f} | status={status}"
        )
        return f"employee #{employee_id}" if employee_id else "unknown"

    def _reconcile_review_events(self):
        """Match stored face vectors from earlier runs after employees are labeled."""
        repaired = 0
        for event in self.db.get_events_needing_review():
            track_id = event.get("track_id")
            blobs = self.db.get_pending_track_embeddings(track_id) if track_id is not None else []
            face_result = None
            if not blobs:
                face_path = event.get("face_crop_path") or event.get("resolved_face_path")
                if face_path:
                    face_path = Path(face_path)
                    if not face_path.is_absolute():
                        face_path = Path(__file__).resolve().parent.parent / face_path
                    image = cv2.imread(str(face_path))
                    if image is not None:
                        face_result = self.recognizer.extract_best_face(image)
                        if face_result is not None:
                            blobs = [face_result["embedding"].tobytes()]

            for blob in blobs:
                embedding = (
                    face_result["embedding"] if face_result is not None
                    else np.frombuffer(blob, dtype=np.float32)
                )
                employee_id, confidence, status = self.recognizer.match(embedding)
                if status == "confirmed" and employee_id is not None:
                    self.db.resolve_review(
                        event["event_id"], employee_id,
                        "automatic_reconciliation", confidence,
                    )
                    repaired += 1
                    break
        if repaired:
            logger.info("Automatically reconciled %d earlier attendance event(s).", repaired)
