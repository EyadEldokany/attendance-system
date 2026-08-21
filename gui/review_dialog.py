import logging
from pathlib import Path

import cv2
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QLabel, QPushButton, QListWidget, QListWidgetItem, QLineEdit, QComboBox,
    QMessageBox, QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView,
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt

from src.face_recognizer import FaceRecognizer, embedding_to_bytes, bytes_to_embedding
from src.device_utils import resolve_device

from src.path_utils import get_app_dir, resolve_path

logger = logging.getLogger("attendance.gui.review")

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


class ReviewWidget(QWidget):
    """
    The actual reviewing UI (New Faces + Review Attendance tabs), factored
    out as a plain QWidget so it can be embedded either inside its own
    standalone dialog (ReviewDialog, below) or directly as a tab inside
    another window (e.g. the Archive processing dialog).
    """
    def __init__(self, config, db, parent=None):
        super().__init__(parent)
        self._config = config
        self._db = db
        self._recognizer = FaceRecognizer(
            model_name=config.get("recognition.model_name", "buffalo_l"),
            detector=config.get("recognition.detector", "scrfd"),
            embedding_dim=config.get("recognition.embedding_dim", 512),
            match_threshold=config.get("recognition.match_threshold", 0.50),
            review_threshold=config.get("recognition.review_threshold", 0.35),
            min_face_size_px=config.get("recognition.min_face_size_px", 40),
            device=resolve_device(config.get("detection.device", "auto")),
            model_root=config.get("recognition.model_root", "~/.insightface"),
        )
        self._enrollment = _EnrollmentHelper(self._db, self._recognizer)

        tabs = QTabWidget()
        tabs.addTab(self._build_new_faces_tab(), "New Faces")
        tabs.addTab(self._build_review_tab(), "Review Attendance")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(tabs)

    def refresh_all(self):
        """Called externally (e.g. by the Archive dialog) to pick up newly captured faces/events."""
        self._refresh_pending_list()
        self._refresh_employee_combo()
        self._refresh_review_table()

    # ------------------------------------------------------------------ Tab 1

    def _build_new_faces_tab(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)

        self._pending_list = QListWidget()
        self._refresh_pending_list()
        self._pending_status = QLabel()
        self._refresh_pending_list()
        refresh_faces = QPushButton("Refresh faces")
        refresh_faces.clicked.connect(self._refresh_pending_list)

        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Unlabeled face captures:"))
        left_layout.addWidget(self._pending_status)
        left_layout.addWidget(self._pending_list)
        left_layout.addWidget(refresh_faces)
        layout.addLayout(left_layout, 1)

        panel = QGroupBox("Label the selected person")
        panel_layout = QVBoxLayout(panel)

        self._face_preview = QLabel("Select a person from the list")
        self._face_preview.setFixedSize(220, 220)
        self._face_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        panel_layout.addWidget(self._face_preview)

        self._name_input = QLineEdit()
        self._name_input.setPlaceholderText("New employee name…")
        panel_layout.addWidget(QLabel("New employee name:"))
        panel_layout.addWidget(self._name_input)

        self._emp_combo = QComboBox()
        self._refresh_employee_combo()
        panel_layout.addWidget(QLabel("Or select an existing employee:"))
        panel_layout.addWidget(self._emp_combo)

        btn_row = QHBoxLayout()
        btn_new = QPushButton("Save as new employee")
        btn_new.clicked.connect(self._save_as_new)
        btn_existing = QPushButton("Add to existing employee")
        btn_existing.clicked.connect(self._save_as_existing)
        btn_ignore = QPushButton("Ignore")
        btn_ignore.clicked.connect(self._ignore_selected)
        btn_row.addWidget(btn_new)
        btn_row.addWidget(btn_existing)
        btn_row.addWidget(btn_ignore)
        panel_layout.addLayout(btn_row)
        panel_layout.addStretch()
        layout.addWidget(panel, 1)

        self._pending_list.currentItemChanged.connect(self._on_pending_selected)
        return widget

    def _refresh_pending_list(self):
        self._pending_list.clear()
        tracks = self._db.get_pending_labeling(
            self._config.get("enrollment.min_captures_before_prompt", 3)
        )
        for track in tracks:
            item = QListWidgetItem(
                f"Track #{track['track_id']} — seen {track['capture_count']}× "
                f"— first: {track['first_seen'][:19]}"
            )
            item.setData(Qt.ItemDataRole.UserRole, track)
            self._pending_list.addItem(item)
        if hasattr(self, "_pending_status"):
            self._pending_status.setText(f"{len(tracks)} face capture(s) waiting for labeling")

    def _refresh_employee_combo(self):
        self._emp_combo.clear()
        self._emp_combo.addItem("-- Select employee --", None)
        for emp in self._db.list_employees():
            self._emp_combo.addItem(emp["full_name"], emp["employee_id"])

    def _on_pending_selected(self, current, _prev):
        if current is None:
            return
        track = current.data(Qt.ItemDataRole.UserRole)
        face_path = _media_path(track.get("best_face_path"))
        if face_path and face_path.exists():
            pix = QPixmap(str(face_path)).scaled(
                220, 220, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._face_preview.setPixmap(pix)
        else:
            self._face_preview.setText("No image available")

    def _current_track_id(self):
        item = self._pending_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole)["track_id"] if item else None

    def _save_as_new(self):
        track_id = self._current_track_id()
        name = self._name_input.text().strip()
        if track_id is None:
            QMessageBox.warning(self, "Warning", "Select a person first.")
            return
        if not name:
            QMessageBox.warning(self, "Warning", "Enter a name.")
            return
        employee_id = self._enrollment.label_as_new(track_id, name)
        confirmed = self._db.confirm_events_for_track(track_id, employee_id)
        self._signal_reload()
        self._name_input.clear()
        self._refresh_pending_list()
        self._refresh_employee_combo()
        QMessageBox.information(
            self, "Done",
            f"Employee '{name}' added. {confirmed} attendance event(s) confirmed.",
        )

    def _save_as_existing(self):
        track_id = self._current_track_id()
        emp_id = self._emp_combo.currentData()
        if track_id is None or emp_id is None:
            QMessageBox.warning(self, "Warning", "Select a track and an employee.")
            return
        self._enrollment.label_as_existing(track_id, emp_id)
        confirmed = self._db.confirm_events_for_track(track_id, emp_id)
        self._signal_reload()
        self._refresh_pending_list()
        self._refresh_review_table()
        QMessageBox.information(
            self, "Done",
            f"Face added to employee profile. {confirmed} attendance event(s) confirmed.",
        )

    def _ignore_selected(self):
        track_id = self._current_track_id()
        if track_id is None:
            return
        self._db.mark_track_labeled(track_id)
        self._refresh_pending_list()

    def _signal_reload(self):
        try:
            self._recognizer.rebuild_index(self._db.get_all_embeddings())
        except Exception:
            pass
        RELOAD_SIGNAL_FILE.parent.mkdir(parents=True, exist_ok=True)
        RELOAD_SIGNAL_FILE.touch()

    # ------------------------------------------------------------------ Tab 2

    def _build_review_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self._review_table = QTableWidget(0, 5)
        self._review_table.setHorizontalHeaderLabels(
            ["Time", "Direction", "Confidence", "Face", "Actions"]
        )
        self._review_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._review_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._review_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self._review_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._review_table.verticalHeader().setVisible(False)
        layout.addWidget(self._review_table)

        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self._refresh_review_table)
        layout.addWidget(btn_refresh)

        self._refresh_review_table()
        return widget

    def _refresh_review_table(self):
        self._review_table.setRowCount(0)
        for ev in self._db.get_events_needing_review():
            row = self._review_table.rowCount()
            self._review_table.insertRow(row)

            ts  = ev.get("event_timestamp", "")[:19]
            dir_label = "IN" if ev.get("event_type") == "in" else "OUT"
            conf = ev.get("confidence")
            conf_str = f"{conf:.2f}" if conf is not None else "—"

            self._review_table.setItem(row, 0, QTableWidgetItem(ts))
            self._review_table.setItem(row, 1, QTableWidgetItem(dir_label))
            self._review_table.setItem(row, 2, QTableWidgetItem(conf_str))

            face_path = _media_path(ev.get("face_crop_path") or ev.get("resolved_face_path"))
            if face_path and face_path.exists():
                thumb = QLabel()
                pix = QPixmap(str(face_path)).scaled(
                    60, 60, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                thumb.setPixmap(pix)
                self._review_table.setCellWidget(row, 3, thumb)
                self._review_table.setRowHeight(row, 64)
            else:
                self._review_table.setItem(row, 3, QTableWidgetItem("—"))

            actions = QWidget()
            action_layout = QHBoxLayout(actions)
            action_layout.setContentsMargins(2, 2, 2, 2)

            btn_assign = QPushButton("Assign")
            btn_reject = QPushButton("Reject")
            event_id = ev["event_id"]
            btn_assign.clicked.connect(lambda _, eid=event_id: self._assign_event(eid))
            btn_reject.clicked.connect(lambda _, eid=event_id: self._reject_event(eid))
            action_layout.addWidget(btn_assign)
            action_layout.addWidget(btn_reject)
            self._review_table.setCellWidget(row, 4, actions)

    def _assign_event(self, event_id):
        employees = self._db.list_employees()
        if not employees:
            QMessageBox.warning(self, "No employees", "Add employees first.")
            return
        from PyQt6.QtWidgets import QInputDialog
        names = [e["full_name"] for e in employees]
        name, ok = QInputDialog.getItem(self, "Assign to employee", "Select employee:", names, 0, False)
        if not ok:
            return
        emp = next(e for e in employees if e["full_name"] == name)
        event = next(
            (row for row in self._db.get_events_needing_review()
             if row["event_id"] == event_id),
            None,
        )
        if event is not None:
            self._save_event_embeddings(event, emp["employee_id"])
        self._db.resolve_review(event_id, emp["employee_id"], "manual")
        self._refresh_review_table()

    def _save_event_embeddings(self, event, employee_id):
        """Persist the track vectors when an attendance event is manually assigned."""
        blobs = self._db.get_pending_track_embeddings(event.get("track_id"))
        max_embeddings = self._config.get("recognition.max_embeddings_per_person", 8)
        saved = 0
        for blob in blobs:
            if self._db.count_embeddings_for(employee_id) >= max_embeddings:
                break
            self._db.add_embedding(
                employee_id, blob,
                source_image=event.get("face_crop_path") or event.get("resolved_face_path"),
            )
            saved += 1
        if saved:
            self._signal_reload()
            logger.info(
                "Saved %s face embedding(s) for employee #%s from event #%s",
                saved, employee_id, event["event_id"],
            )

    def _reject_event(self, event_id):
        self._db.reject_review(event_id, "manual")
        self._refresh_review_table()


class ReviewDialog(QDialog):
    """Standalone dialog wrapper around ReviewWidget (used from the main menu)."""
    def __init__(self, config, db, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Review")
        self.resize(960, 620)
        layout = QVBoxLayout(self)
        self.widget = ReviewWidget(config, db, parent=self)
        layout.addWidget(self.widget)


class _EnrollmentHelper:
    def __init__(self, db, recognizer):
        self.db = db
        self.recognizer = recognizer

    def _get_face_path(self, track_id):
        for t in self.db.get_pending_labeling(min_captures=0):
            if t["track_id"] == track_id:
                path = _media_path(t.get("best_face_path"))
                return str(path) if path else None
        return None

    def _extract_embedding(self, face_path):
        if not face_path or not face_path.exists():
            return None
        img = cv2.imread(str(face_path))
        if img is None:
            return None
        result = self.recognizer.extract_best_face(img)
        return result["embedding"] if result else None

    def _get_embeddings(self, track_id, face_path):
        stored = self.db.get_pending_track_embeddings(track_id)
        if stored:
            return [bytes_to_embedding(blob) for blob in stored]
        fallback = self._extract_embedding(face_path)
        return [fallback] if fallback is not None else []

    def label_as_new(self, track_id, full_name):
        emp_id = self.db.add_employee(full_name)
        face_path = self._get_face_path(track_id)
        for embedding in self._get_embeddings(track_id, face_path):
            self.db.add_embedding(emp_id, embedding_to_bytes(embedding), source_image=face_path)
        self.db.mark_track_labeled(track_id)
        return emp_id

    def label_as_existing(self, track_id, employee_id):
        face_path = self._get_face_path(track_id)
        max_emb = 8
        for embedding in self._get_embeddings(track_id, face_path):
            if self.db.count_embeddings_for(employee_id) >= max_emb:
                break
            self.db.add_embedding(employee_id, embedding_to_bytes(embedding), source_image=face_path)
        self.db.mark_track_labeled(track_id)
