import logging
import time
from datetime import datetime

from src.path_utils import resolve_path
from src.camera_stream import CameraStream
from src.detector_tracker import DetectorTracker
from src.door_crossing import DoorCrossingTracker
from src.device_utils import resolve_device
from src.face_recognizer import FaceRecognizer
from src.enrollment_manager import EnrollmentManager

logger = logging.getLogger("attendance.pipeline")


class AttendancePipeline:
    def __init__(self, config, db):
        self.config = config
        self.db = db
        device = resolve_device(config.get("detection.device", "auto"))

        self.camera = CameraStream(
            source=config.get("camera.source"),
            reconnect_attempts=config.get("camera.reconnect_attempts", 10),
            reconnect_delay_seconds=config.get("camera.reconnect_delay_seconds", 5),
            frame_width=config.get("camera.frame_width", 1280),
            frame_height=config.get("camera.frame_height", 720),
        )
        self.detector = DetectorTracker(
            model_path=str(resolve_path(config.get("detection.model_path", "models/yolov8n.pt"))),
            confidence_threshold=config.get("detection.confidence_threshold", 0.5),
            device=device,
            tracker_type=config.get("tracking.tracker_type", "bytetrack"),
        )
        self.door = DoorCrossingTracker(
            point1=config.get("door_line.point1"),
            point2=config.get("door_line.point2"),
            entry_direction=config.get("door_line.entry_direction"),
            min_crossing_confidence_frames=config.get("door_line.min_crossing_confidence_frames", 3),
            frame_width=config.get("camera.frame_width", 1280),
            frame_height=config.get("camera.frame_height", 720),
        )
        self.recognizer = FaceRecognizer(
            model_name=config.get("recognition.model_name", "buffalo_l"),
            detector=config.get("recognition.detector", "scrfd"),
            embedding_dim=config.get("recognition.embedding_dim", 512),
            match_threshold=config.get("recognition.match_threshold", 0.50),
            review_threshold=config.get("recognition.review_threshold", 0.35),
            min_face_size_px=config.get("recognition.min_face_size_px", 40),
            device=device,
            model_root=config.get("recognition.model_root", "~/.insightface"),
            face_crop_padding=config.get("recognition.face_crop_padding", 0.20),
            face_crop_scale=config.get("recognition.face_crop_scale", 2.0),
        )
        self.recognizer.rebuild_index(self.db.get_all_embeddings())

        self.enrollment = EnrollmentManager(
            db=self.db,
            recognizer=self.recognizer,
            faces_dir=resolve_path(config.get("enrollment.faces_dir", "enrollment_faces")),
            min_captures_before_prompt=config.get("enrollment.min_captures_before_prompt", 3),
            max_embeddings_per_person=config.get("recognition.max_embeddings_per_person", 8),
        )

        self.frame_callback = None   # Callable[[frame, detections], None] | None
        self.event_callback = None   # Callable[[dict], None] | None

        self.process_every_n_frames = config.get("camera.process_every_n_frames", 3)
        self._frame_counter = 0
        self._track_last_seen = {}
        self._best_faces = {}
        self._face_sample_interval = max(
            1, int(config.get("recognition.face_sample_interval", 5))
        )
        self._stale_track_timeout_seconds = 300

    def run_forever(self, stop_event=None):
        logger.info("Starting the real-time attendance and departure system...")
        while stop_event is None or not stop_event.is_set():
            try:
                self._process_one_frame()
            except Exception:
                logger.exception("An error occurred while processing a frame - will continue on the next frame.")
                time.sleep(0.5)

    def _process_one_frame(self):
        ok, frame = self.camera.read()
        if not ok:
            time.sleep(1)
            return

        self._frame_counter += 1
        if self._frame_counter % self.process_every_n_frames != 0:
            return  # ignore this frame to save performance (one sample every N frames is enough)

        detections = self.detector.track(frame)
        room_faces = (
            self.recognizer.extract_faces(frame)
            if self._frame_counter % self._face_sample_interval == 0 else []
        )
        if self.frame_callback:
            self.frame_callback(frame, list(detections))
        now = time.time()

        for det in detections:
            track_id = det["track_id"]
            self._track_last_seen[track_id] = now

            face_result = self._face_for_detection(room_faces, det["bbox"])
            if face_result is not None:
                face_area = max(0, face_result["bbox"][2] - face_result["bbox"][0]) * max(
                    0, face_result["bbox"][3] - face_result["bbox"][1]
                )
                previous = self._best_faces.get(track_id)
                if previous is None or face_area > previous["area"]:
                    self._best_faces[track_id] = {
                        "result": face_result, "area": face_area, "frame": frame.copy()
                    }

            direction = self.door.update(track_id, det["bbox"])
            if direction is None:
                continue  # Person has not crossed the line yet

            cached = self._best_faces.pop(track_id, None)
            self._handle_crossing(
                track_id, direction, frame, det["bbox"],
                cached["result"] if cached else None,
                cached["frame"] if cached else None,
            )

        self._cleanup_stale_tracks(now)

    @staticmethod
    def _face_for_detection(faces, person_bbox):
        x1, y1, x2, y2 = person_bbox
        matching = []
        for face in faces:
            fx1, fy1, fx2, fy2 = face["bbox"]
            center_x = (fx1 + fx2) / 2
            center_y = (fy1 + fy2) / 2
            if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                matching.append((max(0, fx2 - fx1) * max(0, fy2 - fy1), face))
        return max(matching, key=lambda item: item[0])[1] if matching else None

    def _handle_crossing(self, track_id, direction, frame, bbox,
                         face_result=None, face_frame=None):
        timestamp = datetime.now().isoformat()
        source_frame = face_frame if face_frame is not None else frame
        if face_result is None:
            face_result = self.recognizer.extract_best_face(frame, bbox)

        if face_result is None:
            logger.warning(f"Track {track_id} crossed the door ({direction}) without a clear face - needs manual review.")
            self.db.log_event(
                event_type=direction, timestamp=timestamp, employee_id=None,
                track_id=track_id, confidence=None, status="needs_review",
            )
            return

        employee_id, confidence, status = self.recognizer.match(face_result["embedding"])

        face_crop_path = None
        if status != "confirmed":
            face_crop_path = self.enrollment.capture_unknown_face(
                track_id, source_frame, face_result["bbox"],
                embedding=face_result["embedding"], replace_best=True,
            )

        self.db.log_event(
            event_type=direction, timestamp=timestamp, employee_id=employee_id,
            track_id=track_id, confidence=confidence, status=status,
            face_crop_path=face_crop_path,
        )

        if self.event_callback:
            self.event_callback({
                "event_type": direction, "timestamp": timestamp,
                "employee_id": employee_id, "confidence": confidence,
                "status": status, "track_id": track_id,
            })

        logger.info(
            f"[{direction.upper()}] Track {track_id} | "
            f"{'Employee #' + str(employee_id) if employee_id else 'Unknown'} | "
            f"Confidence={confidence:.2f} | Status={status}"
        )

    def _cleanup_stale_tracks(self, now):
        stale = [tid for tid, last_seen in self._track_last_seen.items()
                 if now - last_seen > self._stale_track_timeout_seconds]
        for tid in stale:
            self.door.forget_track(tid)
            self._track_last_seen.pop(tid, None)

    def reload_recognition_index(self):
        self.recognizer.rebuild_index(self.db.get_all_embeddings())
        logger.info("Face recognition index updated after adding new data.")
