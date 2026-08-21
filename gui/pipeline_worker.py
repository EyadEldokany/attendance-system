import threading
import logging
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

logger = logging.getLogger("attendance.gui.worker")

BOX_COLOR  = (0, 255, 0)
LINE_COLOR = (255, 255, 255)
FONT       = cv2.FONT_HERSHEY_SIMPLEX


def _draw_frame(frame, detections, p1_norm, p2_norm):
    h, w = frame.shape[:2]
    display = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(display, (x1, y1), (x2, y2), BOX_COLOR, 2)
        label = f"#{det['track_id']}"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
        cv2.rectangle(display, (x1, y1 - th - 4), (x1 + tw + 4, y1), BOX_COLOR, -1)
        cv2.putText(display, label, (x1 + 2, y1 - 3), FONT, 0.5, (0, 0, 0), 1)
    p1 = (int(p1_norm[0] * w), int(p1_norm[1] * h))
    p2 = (int(p2_norm[0] * w), int(p2_norm[1] * h))
    cv2.line(display, p1, p2, LINE_COLOR, 2)
    cv2.putText(display, "door line", (p1[0] + 6, p1[1] + 16), FONT, 0.45, LINE_COLOR, 1)
    return display


def _to_qimage(bgr_frame) -> QImage:
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()


class PipelineWorker(QThread):
    frame_ready            = pyqtSignal(QImage)
    event_logged           = pyqtSignal(dict)
    camera_status_changed  = pyqtSignal(bool)

    def __init__(self, config, db, parent=None):
        super().__init__(parent)
        self._config = config
        self._db = db
        self._stop_event = threading.Event()
        self._camera_connected = None

    def run(self):
        from src.pipeline import AttendancePipeline

        p1 = self._config.get("door_line.point1", [0.5, 0.0])
        p2 = self._config.get("door_line.point2", [0.5, 1.0])

        try:
            pipeline = AttendancePipeline(self._config, self._db)
            pipeline.frame_callback = lambda frame, dets: self._on_frame(frame, dets, p1, p2)
            pipeline.event_callback = self._on_event
            self._emit_camera_state(True)
            pipeline.run_forever(self._stop_event)
        except Exception:
            logger.exception("Pipeline worker crashed")
            self._emit_camera_state(False)

    def _on_frame(self, frame, detections, p1, p2):
        try:
            display = _draw_frame(frame, detections, p1, p2)
            self.frame_ready.emit(_to_qimage(display))
        except Exception:
            logger.debug("Frame conversion error", exc_info=True)

    def _on_event(self, event_dict):
        self.event_logged.emit(event_dict)

    def _emit_camera_state(self, connected: bool):
        if connected != self._camera_connected:
            self._camera_connected = connected
            self.camera_status_changed.emit(connected)

    def stop(self):
        self._stop_event.set()
        self.wait(5000)
