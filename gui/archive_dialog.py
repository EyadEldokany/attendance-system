import logging
import numpy as np
from pathlib import Path
from datetime import datetime

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QLabel, QPushButton, QListWidget, QListWidgetItem, QLineEdit,
    QMessageBox, QGroupBox, QFileDialog, QProgressBar, QDateTimeEdit,
    QTextEdit,
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QDateTime

from gui.review_dialog import ReviewWidget

logger = logging.getLogger("attendance.gui.archive")


def _bgr_to_pixmap(frame_bgr: np.ndarray, max_height: int = 360) -> QPixmap:
    """Converts an OpenCV BGR frame to a QPixmap for display in a QLabel."""
    rgb = frame_bgr[:, :, ::-1].copy()  # BGR -> RGB
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    pix = QPixmap.fromImage(qimg)
    if pix.height() > max_height:
        pix = pix.scaledToHeight(max_height, Qt.TransformationMode.SmoothTransformation)
    return pix


class _ScanWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished_scan = pyqtSignal(int)
    failed = pyqtSignal(str)

    def __init__(self, processor, folder_path, parent=None):
        super().__init__(parent)
        self._processor = processor
        self._folder_path = folder_path

    def run(self):
        try:
            count = self._processor.scan_folder(
                self._folder_path,
                progress_callback=lambda i, t, name: self.progress.emit(i, t, name),
                stop_callback=self.isInterruptionRequested,
            )
            self.finished_scan.emit(count)
        except Exception as e:
            logger.exception("Archive scan failed")
            self.failed.emit(str(e))


class _ProcessWorker(QThread):
    progress = pyqtSignal(int, int, str, int, int)
    frame_ready = pyqtSignal(object, str)
    finished_processing = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, processor, stop_event, parent=None):
        super().__init__(parent)
        self._processor = processor
        self._stop_event = stop_event

    def run(self):
        try:
            self._processor.process_confirmed_videos(
                progress_callback=lambda vi, vt, name, fi, ft: self.progress.emit(vi, vt, name, fi, ft),
                stop_event=self._stop_event,
                frame_callback=lambda frame, name: self.frame_ready.emit(frame, name),
            )
            self.finished_processing.emit()
        except Exception as e:
            logger.exception("Archive processing failed")
            self.failed.emit(str(e))


class ArchiveDialog(QDialog):
    """
    Two-step workflow for analyzing a folder of archived footage:
      1. "Scan Folder" tab - pick the folder, the tool finds every video and
         attempts to read its burned-in start timestamp automatically.
      2. "Confirm Videos" tab - for each video the tool wasn't fully sure
         about, glance at the cropped timestamp image, accept or correct the
         guessed date/time once, then start batch processing.
    """

    def __init__(self, config, db, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Process Archived Footage")
        self.setMinimumSize(900, 600)
        screen = self.screen().availableGeometry() if self.screen() else None
        if screen:
            self.resize(
                min(1150, max(900, screen.width() - 40)),
                min(760, max(600, screen.height() - 80)),
            )
        else:
            self.resize(1150, 760)

        self._config = config
        self._db = db
        self._processor = None
        self._scan_worker = None
        self._process_worker = None
        self._stop_event = None
        self._closing = False

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_scan_tab(), "1. Scan Folder")
        self._tabs.addTab(self._build_confirm_tab(), "2. Confirm & Process")
        self._review_widget = ReviewWidget(config, db, parent=self)
        self._tabs.addTab(self._review_widget, "3. Review New Faces")
        layout = QVBoxLayout(self)
        layout.addWidget(self._tabs)

    # ---------------- Tab 1: scan ----------------

    def _build_scan_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        info = QLabel(
            "Select the folder containing last year's recordings. The tool will scan every "
            "video file and try to read its burned-in date/time overlay automatically.\n"
            "This can take a while for a large archive - it only needs to run once per folder."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        row = QHBoxLayout()
        self._folder_input = QLineEdit()
        self._folder_input.setPlaceholderText("Select a folder…")
        self._folder_input.setReadOnly(True)
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse_folder)
        row.addWidget(self._folder_input)
        row.addWidget(btn_browse)
        layout.addLayout(row)

        self._scan_progress = QProgressBar()
        layout.addWidget(self._scan_progress)

        self._scan_log = QTextEdit()
        self._scan_log.setReadOnly(True)
        layout.addWidget(self._scan_log)

        self._btn_scan = QPushButton("Start Scan")
        self._btn_scan.clicked.connect(self._start_scan)
        layout.addWidget(self._btn_scan)

        return widget

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select archive folder")
        if folder:
            self._folder_input.setText(folder)

    def _ensure_processor(self):
        if self._processor is None:
            from src.archive_pipeline import ArchiveProcessor
            self._scan_log.append("Loading detection and recognition models (first run only takes longer)…")
            self._processor = ArchiveProcessor(self._config, self._db)
        return self._processor

    def _start_scan(self):
        folder = self._folder_input.text().strip()
        if not folder:
            QMessageBox.warning(self, "Warning", "Select a folder first.")
            return
        self._btn_scan.setEnabled(False)
        processor = self._ensure_processor()
        self._scan_worker = _ScanWorker(processor, folder, self)
        self._scan_worker.progress.connect(self._on_scan_progress)
        self._scan_worker.finished_scan.connect(self._on_scan_finished)
        self._scan_worker.failed.connect(self._on_worker_failed)
        self._scan_worker.start()

    def _on_scan_progress(self, i, total, name):
        self._scan_progress.setMaximum(max(total, 1))
        self._scan_progress.setValue(i)
        self._scan_log.append(f"[{i}/{total}] {name}")

    def _on_scan_finished(self, count):
        self._btn_scan.setEnabled(True)
        self._scan_log.append(f"\nScan complete - {count} video file(s) found.")
        QMessageBox.information(
            self, "Scan complete",
            f"Found {count} video file(s).\nGo to the 'Confirm & Process' tab to review "
            f"start times and begin analysis."
        )
        self._refresh_pending_list()
        self._tabs.setCurrentIndex(1)

    def _on_worker_failed(self, message):
        self._btn_scan.setEnabled(True)
        if hasattr(self, "_btn_stop"):
            self._btn_stop.setEnabled(False)
        self._update_processing_button()
        QMessageBox.critical(self, "Error", message)

    # ---------------- Tab 2: confirm + process ----------------

    def _build_confirm_tab(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)

        left = QVBoxLayout()
        left.addWidget(QLabel("Videos needing a start-time confirmation:"))
        self._pending_list = QListWidget()
        self._pending_list.currentItemChanged.connect(self._on_pending_selected)
        left.addWidget(self._pending_list)
        btn_refresh = QPushButton("Refresh list")
        btn_refresh.clicked.connect(self._refresh_pending_list)
        left.addWidget(btn_refresh)
        layout.addLayout(left, 1)

        right = QVBoxLayout()
        panel = QGroupBox("Confirm start date/time")
        panel_layout = QVBoxLayout(panel)

        self._thumb_label = QLabel("Select a video from the list")
        self._thumb_label.setFixedHeight(80)
        self._thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        panel_layout.addWidget(self._thumb_label)

        self._ocr_guess_label = QLabel("")
        self._ocr_guess_label.setWordWrap(True)
        panel_layout.addWidget(self._ocr_guess_label)

        panel_layout.addWidget(QLabel("Start date/time (edit if the guess above is wrong):"))
        self._datetime_edit = QDateTimeEdit()
        self._datetime_edit.setCalendarPopup(True)
        self._datetime_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        panel_layout.addWidget(self._datetime_edit)

        btn_row = QHBoxLayout()
        btn_confirm = QPushButton("Confirm this video")
        btn_confirm.clicked.connect(self._confirm_selected)
        btn_skip = QPushButton("Skip this video")
        btn_skip.clicked.connect(self._skip_selected)
        btn_row.addWidget(btn_confirm)
        btn_row.addWidget(btn_skip)
        panel_layout.addLayout(btn_row)

        right.addWidget(panel)

        process_group = QGroupBox("Batch processing")
        process_layout = QVBoxLayout(process_group)

        self._video_preview = QLabel("Video preview appears here while processing runs")
        self._video_preview.setFixedHeight(280)
        self._video_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_preview.setStyleSheet("background-color: #111; color: #888; border: 1px solid #333;")
        process_layout.addWidget(self._video_preview)

        self._process_progress = QProgressBar()
        process_layout.addWidget(self._process_progress)
        self._process_log = QTextEdit()
        self._process_log.setReadOnly(True)
        process_layout.addWidget(self._process_log)
        btn_row2 = QHBoxLayout()
        self._btn_start = QPushButton("Start Processing Confirmed Videos")
        self._btn_start.clicked.connect(self._start_processing)
        btn_stop = QPushButton("Stop")
        btn_stop.clicked.connect(self._stop_processing)
        self._btn_stop = btn_stop
        self._btn_start.setEnabled(False)
        btn_row2.addWidget(self._btn_start)
        btn_row2.addWidget(btn_stop)
        process_layout.addLayout(btn_row2)
        right.addWidget(process_group)

        layout.addLayout(right, 1)
        return widget

    def _refresh_pending_list(self):
        self._pending_list.clear()
        processor = self._ensure_processor()
        for video in processor.get_pending_confirmations():
            item = QListWidgetItem(f"{Path(video['file_path']).name}")
            item.setData(Qt.ItemDataRole.UserRole, video)
            self._pending_list.addItem(item)
        self._update_processing_button()

    def _update_processing_button(self):
        if not hasattr(self, "_btn_start"):
            return
        running = self._process_worker is not None and self._process_worker.isRunning()
        has_confirmed = bool(self._db.get_confirmed_unprocessed_videos())
        self._btn_start.setEnabled(has_confirmed and not running)

    def _on_pending_selected(self, current, _prev):
        if current is None:
            return
        video = current.data(Qt.ItemDataRole.UserRole)
        thumb_path = video.get("thumbnail_path")
        if thumb_path and Path(thumb_path).exists():
            pix = QPixmap(thumb_path).scaledToHeight(
                76, Qt.TransformationMode.SmoothTransformation
            )
            self._thumb_label.setPixmap(pix)
        else:
            self._thumb_label.setText("No preview available")

        guess = video.get("ocr_guess_text") or ""
        confidence = video.get("ocr_confidence") or 0.0
        self._ocr_guess_label.setText(
            f"Automatic read: {guess or '(nothing readable)'}\n"
            f"Confidence: {confidence:.0%} — please verify against the image above."
        )

        parsed = None
        if guess:
            try:
                parsed = datetime.fromisoformat(guess)
            except ValueError:
                parsed = None
        qdt = QDateTime(parsed) if parsed else QDateTime.currentDateTime()
        self._datetime_edit.setDateTime(qdt)

    def _current_video(self):
        item = self._pending_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _confirm_selected(self):
        video = self._current_video()
        if video is None:
            QMessageBox.warning(self, "Warning", "Select a video first.")
            return
        qdt = self._datetime_edit.dateTime()
        dt = qdt.toPyDateTime()
        processor = self._ensure_processor()
        processor.confirm_video_start(video["video_id"], dt)
        self._refresh_pending_list()

    def _skip_selected(self):
        video = self._current_video()
        if video is None:
            return
        processor = self._ensure_processor()
        processor.skip_video(video["video_id"])
        self._refresh_pending_list()

    def _start_processing(self):
        if self._process_worker is not None and self._process_worker.isRunning():
            return
        processor = self._ensure_processor()
        import threading
        self._stop_event = threading.Event()
        self._process_worker = _ProcessWorker(processor, self._stop_event, self)
        self._process_worker.progress.connect(self._on_process_progress)
        self._process_worker.frame_ready.connect(self._on_frame_ready)
        self._process_worker.finished_processing.connect(self._on_process_finished)
        self._process_worker.failed.connect(self._on_worker_failed)
        self._process_worker.start()
        self._update_processing_button()
        self._btn_stop.setEnabled(True)

        # Refresh the "Review New Faces" tab periodically while processing
        # runs, so newly captured unresolved faces show up without the user
        # having to close and reopen the dialog.
        from PyQt6.QtCore import QTimer
        self._review_refresh_timer = QTimer(self)
        self._review_refresh_timer.timeout.connect(self._review_widget.refresh_all)
        self._review_refresh_timer.start(10_000)

    def _on_process_progress(self, v_idx, v_total, name, frame_idx, frame_total):
        self._process_progress.setMaximum(max(frame_total, 1))
        self._process_progress.setValue(min(frame_idx, frame_total) if frame_total else 0)
        self._process_log.append(f"[Video {v_idx}/{v_total}] {name} — frame {frame_idx}/{frame_total or '?'}")

    def _on_frame_ready(self, frame_bgr, video_name):
        self._video_preview.setPixmap(_bgr_to_pixmap(frame_bgr))

    def _on_process_finished(self):
        self._btn_stop.setEnabled(False)
        self._update_processing_button()
        self._process_log.append("\nBatch processing complete.")
        if hasattr(self, "_review_refresh_timer"):
            self._review_refresh_timer.stop()
        self._review_widget.refresh_all()
        self._video_preview.setText("Processing complete")
        QMessageBox.information(self, "Done", "Archive processing complete.")

    def _stop_processing(self):
        if self._stop_event is not None:
            self._stop_event.set()
            self._btn_stop.setEnabled(False)
            self._process_log.append("Stopping after the current frame…")
            if hasattr(self, "_review_refresh_timer"):
                self._review_refresh_timer.stop()

    def closeEvent(self, event):
        workers = [worker for worker in (self._scan_worker, self._process_worker)
                   if worker is not None and worker.isRunning()]
        if not workers:
            event.accept()
            return

        reply = QMessageBox.question(
            self,
            "Operation still running",
            "An archive operation is still running. Stop it and close this window?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            event.ignore()
            return

        self._closing = True
        self._stop_processing()
        if self._scan_worker is not None and self._scan_worker.isRunning():
            self._scan_worker.requestInterruption()
        for worker in workers:
            if not worker.wait(15_000):
                QMessageBox.warning(
                    self, "Still processing",
                    "The operation has not stopped yet. Keep this window open until it finishes.",
                )
                event.ignore()
                return
        event.accept()
