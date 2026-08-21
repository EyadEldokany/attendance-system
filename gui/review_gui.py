import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QPushButton, QListWidget, QListWidgetItem, QLineEdit, QComboBox,
    QMessageBox, QGroupBox,
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt

import cv2
from src.path_utils import get_app_dir, resolve_path
from src.config import load_config
from src.database import Database
from src.face_recognizer import FaceRecognizer, embedding_to_bytes, bytes_to_embedding
from src.device_utils import resolve_device


PROJECT_ROOT = get_app_dir()
RELOAD_SIGNAL_FILE = PROJECT_ROOT / "data" / ".reload_index"


def _media_path(value):
    if not value:
        return None
    p = Path(value)
    if p.is_absolute() and p.exists():
        return p
    resolved = resolve_path(value)
    if resolved.exists():
        return resolved
    face_fallback = resolve_path(Path("enrollment_faces") / p.name)
    if face_fallback.exists():
        return face_fallback
    return resolved


class ReviewWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Review Attendance System")
        self.resize(900, 600)
        self.setLayoutDirection(Qt.LayoutDirection.RightToLeft)

        self.config = load_config("config.yaml")
        self.db = Database(str(resolve_path(self.config.get("database.sqlite_path", "data/attendance.db"))))

        self.recognizer = FaceRecognizer(
            model_name=self.config.get("recognition.model_name", "adaface_ir101"),
            detector=self.config.get("recognition.detector", "scrfd"),
            embedding_dim=self.config.get("recognition.embedding_dim", 512),
            match_threshold=self.config.get("recognition.match_threshold", 0.50),
            review_threshold=self.config.get("recognition.review_threshold", 0.35),
            min_face_size_px=self.config.get("recognition.min_face_size_px", 40),
            device=resolve_device(self.config.get("detection.device", "auto")),
        )
        self.enrollment = EnrollmentManagerLite(self.db, self.recognizer)

        tabs = QTabWidget()
        tabs.addTab(self._build_new_faces_tab(), "New Faces")
        tabs.addTab(self._build_review_tab(), "Review Attendance")
        self.setCentralWidget(tabs)


    def _build_new_faces_tab(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)

        self.pending_list = QListWidget()
        self._refresh_pending_list()
        layout.addWidget(self.pending_list, 1)

        panel = QGroupBox("Label the selected person")
        panel_layout = QVBoxLayout(panel)

        self.face_preview = QLabel("Choose a person from the list")
        self.face_preview.setFixedSize(220, 220)
        self.face_preview.setAlignment(Qt.AlignmentType.AlignCenter if hasattr(Qt, "AlignmentType") else Qt.AlignCenter)
        panel_layout.addWidget(self.face_preview)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("New employee name")
        panel_layout.addWidget(QLabel("New employee name:"))
        panel_layout.addWidget(self.name_input)

        self.existing_employee_combo = QComboBox()
        self._refresh_employee_combo()
        panel_layout.addWidget(QLabel("Or select an existing employee:"))
        panel_layout.addWidget(self.existing_employee_combo)

        btn_row = QHBoxLayout()
        save_new_btn = QPushButton("Save as new employee")
        save_new_btn.clicked.connect(self._save_as_new_employee)
        save_existing_btn = QPushButton("Save as existing employee")
        save_existing_btn.clicked.connect(self._save_as_existing_employee)
        ignore_btn = QPushButton("Ignore (not an employee)")
        ignore_btn.clicked.connect(self._ignore_selected)
        btn_row.addWidget(save_new_btn)
        btn_row.addWidget(save_existing_btn)
        btn_row.addWidget(ignore_btn)
        panel_layout.addLayout(btn_row)

        panel_layout.addStretch()
        layout.addWidget(panel, 1)

        self.pending_list.currentItemChanged.connect(self._on_pending_selected)
        return widget

    def _refresh_pending_list(self):
        self.pending_list.clear()
        for track in self.db.get_pending_labeling(
            self.config.get("enrollment.min_captures_before_prompt", 3)
        ):
            item = QListWidgetItem(
                f"Track #{track['track_id']} — {track['capture_count']} times "
                f"— First seen: {track['first_seen'][:19]}"
            )
            item.setData(Qt.ItemDataRole.UserRole if hasattr(Qt, "ItemDataRole") else Qt.UserRole, track)
            self.pending_list.addItem(item)

    def _refresh_employee_combo(self):
        self.existing_employee_combo.clear()
        self.existing_employee_combo.addItem("-- Select employee --", None)
        for emp in self.db.list_employees():
            self.existing_employee_combo.addItem(emp["full_name"], emp["employee_id"])

    def _on_pending_selected(self, current, previous):
        if current is None:
            return
        role = Qt.ItemDataRole.UserRole if hasattr(Qt, "ItemDataRole") else Qt.UserRole
        track = current.data(role)
        face_path = _media_path(track.get("best_face_path"))
        if face_path and face_path.exists():
            pixmap = QPixmap(str(face_path)).scaled(
                220, 220, Qt.AspectRatioMode.KeepAspectRatio if hasattr(Qt, "AspectRatioMode") else Qt.KeepAspectRatio
            )
            self.face_preview.setPixmap(pixmap)
        else:
            self.face_preview.setText("No image found")

    def _current_track_id(self):
        item = self.pending_list.currentItem()
        if item is None:
            return None
        role = Qt.ItemDataRole.UserRole if hasattr(Qt, "ItemDataRole") else Qt.UserRole
        return item.data(role)["track_id"]

    def _save_as_new_employee(self):
        track_id = self._current_track_id()
        name = self.name_input.text().strip()
        if track_id is None:
            QMessageBox.warning(self, "Warning", "Select a person from the list first.")
            return
        if not name:
            QMessageBox.warning(self, "Warning", "Enter the employee name first.")
            return

        employee_id = self.enrollment.label_as_new(track_id, name)
        confirmed = self.db.confirm_events_for_track(track_id, employee_id)
        self._signal_reload()
        self.name_input.clear()
        self._refresh_pending_list()
        self._refresh_employee_combo()
        QMessageBox.information(
            self, "Done", f"Employee '{name}' added. {confirmed} attendance event(s) confirmed."
        )

    def _save_as_existing_employee(self):
        track_id = self._current_track_id()
        employee_id = self.existing_employee_combo.currentData()
        if track_id is None or employee_id is None:
            QMessageBox.warning(self, "Warning", "Select a track and existing employee.")
            return
        self.enrollment.label_as_existing(track_id, employee_id)
        confirmed = self.db.confirm_events_for_track(track_id, employee_id)
        self._signal_reload()
        self._refresh_pending_list()
        QMessageBox.information(
            self, "Done", f"Face added. {confirmed} attendance event(s) confirmed."
        )

    def _ignore_selected(self):
        track_id = self._current_track_id()
        if track_id is None:
            return
        self.db.mark_track_labeled(track_id)
        self._refresh_pending_list()

    def _signal_reload(self):
      
        RELOAD_SIGNAL_FILE.parent.mkdir(parents=True, exist_ok=True)
        RELOAD_SIGNAL_FILE.touch()

    def _build_review_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.review_list = QListWidget()
        self._refresh_review_list()
        layout.addWidget(self.review_list)

        refresh_btn = QPushButton("Refresh the list")
        refresh_btn.clicked.connect(self._refresh_review_list)
        layout.addWidget(refresh_btn)

        return widget

    def _refresh_review_list(self):
        self.review_list.clear()
        for ev in self.db.get_events_needing_review():
            event_ar = "in" if ev["event_type"] == "in" else "out"
            text = (f"{ev['event_timestamp'][:19]} — {event_ar} — "
                    f"confidence: {ev['confidence']:.2f}" if ev["confidence"] else
                    f"{ev['event_timestamp'][:19]} — {event_ar} — no clear face")
            self.review_list.addItem(text)


class EnrollmentManagerLite:
  
    def __init__(self, db, recognizer: FaceRecognizer):
        self.db = db
        self.recognizer = recognizer

    def _extract_embedding_from_file(self, face_path):
        face_path = _media_path(face_path)
        if not face_path or not face_path.exists():
            return None
        image = cv2.imread(str(face_path))
        if image is None:
            return None
        result = self.recognizer.extract_best_face(image)
        return result["embedding"] if result else None

    def _get_embeddings(self, track_id, face_path):
        stored = self.db.get_pending_track_embeddings(track_id)
        if stored:
            return [bytes_to_embedding(blob) for blob in stored]
        fallback = self._extract_embedding_from_file(face_path)
        return [fallback] if fallback is not None else []

    def _get_track_face_path(self, track_id):
        pending = self.db.get_pending_labeling(min_captures=0)
        for t in pending:
            if t["track_id"] == track_id:
                return t["best_face_path"]
        return None

    def label_as_new(self, track_id, full_name):
        employee_id = self.db.add_employee(full_name)
        face_path = self._get_track_face_path(track_id)
        for embedding in self._get_embeddings(track_id, face_path):
            self.db.add_embedding(employee_id, embedding_to_bytes(embedding), source_image=face_path)
        # if no clear face in the saved image, the employee is registered without an initial embedding
        # and will be automatically captured again the next time they pass through the door
        self.db.mark_track_labeled(track_id)
        return employee_id

    def label_as_existing(self, track_id, employee_id):
        face_path = self._get_track_face_path(track_id)
        max_embeddings = 8
        for embedding in self._get_embeddings(track_id, face_path):
            if self.db.count_embeddings_for(employee_id) >= max_embeddings:
                break
            self.db.add_embedding(employee_id, embedding_to_bytes(embedding), source_image=face_path)
        self.db.mark_track_labeled(track_id)


def main():
    app = QApplication(sys.argv)
    window = ReviewWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
