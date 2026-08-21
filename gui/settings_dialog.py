from PyQt6.QtWidgets import (
    QDialog, QFormLayout, QVBoxLayout, QHBoxLayout,
    QLineEdit, QDoubleSpinBox, QSpinBox, QPushButton,
    QDialogButtonBox, QGroupBox, QLabel,
)
from PyQt6.QtCore import Qt


class SettingsDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(500, 440)
        self._config = config
        self._build_ui()
        self._load_values()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Camera group
        cam_group = QGroupBox("Camera")
        cam_form = QFormLayout(cam_group)
        self._rtsp_input = QLineEdit()
        self._rtsp_input.setMinimumWidth(300)
        self._n_frames_spin = QSpinBox()
        self._n_frames_spin.setRange(1, 30)
        cam_form.addRow("RTSP URL:", self._rtsp_input)
        cam_form.addRow("Process every N frames:", self._n_frames_spin)
        layout.addWidget(cam_group)

        # Detection group
        det_group = QGroupBox("Detection")
        det_form = QFormLayout(det_group)
        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.1, 1.0)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setDecimals(2)
        det_form.addRow("Confidence threshold:", self._conf_spin)
        layout.addWidget(det_group)

        # Recognition group
        rec_group = QGroupBox("Face Recognition")
        rec_form = QFormLayout(rec_group)
        self._match_spin = QDoubleSpinBox()
        self._match_spin.setRange(0.1, 1.0)
        self._match_spin.setSingleStep(0.01)
        self._match_spin.setDecimals(2)
        self._review_spin = QDoubleSpinBox()
        self._review_spin.setRange(0.1, 1.0)
        self._review_spin.setSingleStep(0.01)
        self._review_spin.setDecimals(2)
        self._min_face_spin = QSpinBox()
        self._min_face_spin.setRange(10, 200)
        rec_form.addRow("Match threshold (confirmed):", self._match_spin)
        rec_form.addRow("Review threshold (needs review):", self._review_spin)
        rec_form.addRow("Min face size (px):", self._min_face_spin)
        layout.addWidget(rec_group)

        # Excel group
        excel_group = QGroupBox("Excel Export")
        excel_form = QFormLayout(excel_group)
        self._excel_path_input = QLineEdit()
        excel_form.addRow("Export path:", self._excel_path_input)
        layout.addWidget(excel_group)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _load_values(self):
        c = self._config
        self._rtsp_input.setText(c.get("camera.source", ""))
        self._n_frames_spin.setValue(c.get("camera.process_every_n_frames", 3))
        self._conf_spin.setValue(c.get("detection.confidence_threshold", 0.5))
        self._match_spin.setValue(c.get("recognition.match_threshold", 0.50))
        self._review_spin.setValue(c.get("recognition.review_threshold", 0.35))
        self._min_face_spin.setValue(c.get("recognition.min_face_size_px", 40))
        self._excel_path_input.setText(c.get("excel_sync.excel_path", "exports/attendance_log.xlsx"))

    def _save(self):
        c = self._config
        c.set("camera.source",                   self._rtsp_input.text().strip(), persist=True)
        c.set("camera.process_every_n_frames",   self._n_frames_spin.value(),     persist=True)
        c.set("detection.confidence_threshold",  self._conf_spin.value(),          persist=True)
        c.set("recognition.match_threshold",     self._match_spin.value(),         persist=True)
        c.set("recognition.review_threshold",    self._review_spin.value(),        persist=True)
        c.set("recognition.min_face_size_px",    self._min_face_spin.value(),      persist=True)
        c.set("excel_sync.excel_path",           self._excel_path_input.text().strip(), persist=True)
        self.accept()
