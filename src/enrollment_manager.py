import logging
import cv2
from pathlib import Path
from datetime import datetime

from src.path_utils import get_app_dir, resolve_path

logger = logging.getLogger("attendance.enrollment")


class EnrollmentManager:
    def __init__(self, db, recognizer, faces_dir="enrollment_faces",
                 min_captures_before_prompt=3, max_embeddings_per_person=8):
        self.db = db
        self.recognizer = recognizer
        self.faces_dir = resolve_path(faces_dir)
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        self.min_captures_before_prompt = min_captures_before_prompt
        self.max_embeddings_per_person = max_embeddings_per_person

    def capture_unknown_face(self, track_id: int, frame, face_bbox, embedding=None,
                             replace_best: bool = False):
        """
        Saves the unknown face image and logs its appearance in unresolved_tracks.
        Called by the main pipeline whenever an "uncertain" person passes through the door.
        """
        timestamp = datetime.now()
        filename = f"track_{track_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = self.faces_dir / filename

        x1, y1, x2, y2 = face_bbox
        face_crop = frame[max(0, y1):y2, max(0, x1):x2]
        if face_crop.size > 0:
            cv2.imwrite(str(filepath), face_crop)

        if face_crop.size > 0 and filepath.exists():
            try:
                saved_path = str(filepath.relative_to(get_app_dir())).replace("\\", "/")
            except ValueError:
                saved_path = str(filepath)
        else:
            saved_path = None

        self.db.upsert_unresolved_track(
            track_id, saved_path, timestamp.isoformat(), replace_best=replace_best
        )

        if embedding is not None:
            from src.face_recognizer import embedding_to_bytes
            self.db.add_pending_track_embedding(
                track_id, embedding_to_bytes(embedding)
            )

        return saved_path

    def get_pending_for_labeling(self):
        return self.db.get_pending_labeling(self.min_captures_before_prompt)

    def label_track_as_new_employee(self, track_id: int, full_name: str, reviewer: str = "admin"):
        employee_id = self.db.add_employee(full_name)
        self._commit_embeddings(track_id, employee_id)
        logger.info(f"New employee recorded: {full_name} (ID={employee_id}) from track {track_id}")
        return employee_id

    def label_track_as_existing_employee(self, track_id: int, employee_id: int):
        self._commit_embeddings(track_id, employee_id)
        logger.info(f"Added additional reference images for employee ID={employee_id} from track {track_id}")

    def _commit_embeddings(self, track_id, employee_id):
        from src.face_recognizer import bytes_to_embedding, embedding_to_bytes
        pending = [
            bytes_to_embedding(blob)
            for blob in self.db.get_pending_track_embeddings(track_id)
        ]
        existing_count = self.db.count_embeddings_for(employee_id)
        for emb in pending:
            if existing_count >= self.max_embeddings_per_person:
                break
            self.db.add_embedding(employee_id, embedding_to_bytes(emb))
            existing_count += 1
        self.db.mark_track_labeled(track_id)
        self.db.clear_pending_track_embeddings(track_id)

    def ignore_track(self, track_id: int):
        self.db.mark_track_labeled(track_id)
        self.db.clear_pending_track_embeddings(track_id)
