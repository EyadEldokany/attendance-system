import logging
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QTableWidget, QTableWidgetItem, QPushButton,
    QStatusBar, QMenuBar, QMessageBox, QHeaderView, QSizePolicy,
)
from PyQt6.QtGui import QPixmap, QAction, QFont
from PyQt6.QtCore import Qt, QTimer

logger = logging.getLogger("attendance.gui.main")

STATUS_COLORS = {
    "confirmed":    "#2ecc71",
    "needs_review": "#f39c12",
    "rejected":     "#e74c3c",
}


class MainWindow(QMainWindow):
    def __init__(self, config, db, worker=None):
        super().__init__()
        self._config = config
        self._db = db
        self._worker = worker

        self.setWindowTitle("Attendance System")
        self.resize(1280, 720)

        self._build_menu()
        self._build_central()
        self._build_status_bar()

        if worker is not None:
            # Live mode: connect to the real-time camera pipeline.
            worker.frame_ready.connect(self._on_frame)
            worker.event_logged.connect(self._on_event)
            worker.camera_status_changed.connect(self._on_camera_status)
            worker.start()
        else:
            # Archive mode: no live camera - show a clear call to action
            # instead of an empty/confusing feed panel.
            self._feed_label.setText(
                "Archive mode\n\nNo live camera connected.\n\n"
                "Click \"Process Archive Footage\" to analyze a folder of\n"
                "recorded videos instead."
            )
            self._lbl_camera.setText("Mode: Archive (no live camera)")
            self._lbl_camera.setStyleSheet("color: #f39c12;")

        # Periodic log refresh
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._refresh_log)
        self._refresh_timer.start(15_000)

        self._refresh_log()

    # ------------------------------------------------------------------ layout

    def _build_menu(self):
        menu = self.menuBar()
        file_menu = menu.addMenu("File")

        act_settings = QAction("Settings", self)
        act_settings.triggered.connect(self._open_settings)
        file_menu.addAction(act_settings)

        act_export = QAction("Export Excel now", self)
        act_export.triggered.connect(self._export_excel)
        file_menu.addAction(act_export)

        archive_menu = menu.addMenu("Archive")
        act_archive = QAction("Process Archive Footage…", self)
        act_archive.triggered.connect(self._open_archive)
        archive_menu.addAction(act_archive)

        help_menu = menu.addMenu("Help")
        act_about = QAction("About", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    def _build_central(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

        # Left — live feed
        self._feed_label = QLabel("Connecting to camera…")
        self._feed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._feed_label.setStyleSheet("background:#111; color:#888;")
        self._feed_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        root.addWidget(self._feed_label, 65)

        # Right — log + buttons
        right = QVBoxLayout()
        right.setSpacing(4)

        title = QLabel("Recent Attendance Events")
        title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        right.addWidget(title)

        self._log_table = QTableWidget(0, 5)
        self._log_table.setHorizontalHeaderLabels(["Date", "Time", "Name", "Dir", "Status"])
        self._log_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._log_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._log_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._log_table.verticalHeader().setVisible(False)
        self._log_table.setAlternatingRowColors(True)
        right.addWidget(self._log_table)

        btn_review = QPushButton("Review Faces")
        btn_review.clicked.connect(self._open_review)
        btn_manage = QPushButton("Manage Employees")
        btn_manage.clicked.connect(self._open_employees)
        btn_archive = QPushButton("Process Archive Footage…")
        btn_archive.setStyleSheet("font-weight: bold;")
        btn_archive.clicked.connect(self._open_archive)
        right.addWidget(btn_archive)
        right.addWidget(btn_review)
        right.addWidget(btn_manage)

        right_widget = QWidget()
        right_widget.setLayout(right)
        root.addWidget(right_widget, 35)

    def _build_status_bar(self):
        sb = self.statusBar()
        self._lbl_camera  = QLabel("Camera: connecting…")
        self._lbl_empcount = QLabel()
        self._lbl_last    = QLabel()
        sb.addWidget(self._lbl_camera)
        sb.addPermanentWidget(self._lbl_empcount)
        sb.addPermanentWidget(self._lbl_last)
        self._refresh_status_bar()

    # ------------------------------------------------------------------ slots

    def _on_frame(self, qimage):
        size = self._feed_label.size()
        pix = QPixmap.fromImage(qimage).scaled(
            size, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._feed_label.setPixmap(pix)

    def _on_event(self, ev):
        self._add_log_row(ev)
        emp_count = len(self._db.list_employees())
        self._lbl_empcount.setText(f"Employees: {emp_count}")
        ts = ev.get("timestamp", "")
        self._lbl_last.setText(f"Last event: {ts[11:19]}" if ts else "")

    def _on_camera_status(self, connected: bool):
        if connected:
            self._lbl_camera.setText("Camera: connected")
            self._lbl_camera.setStyleSheet("color: #2ecc71;")
        else:
            self._lbl_camera.setText("Camera: disconnected")
            self._lbl_camera.setStyleSheet("color: #e74c3c;")
            self._feed_label.setText("Camera disconnected — reconnecting…")

    # ------------------------------------------------------------------ helpers

    def _refresh_log(self):
        from src.database import Database
        with self._db._lock, self._db._connect() as conn:
            rows = conn.execute(
                """SELECT e.event_timestamp, e.event_type, e.confidence, e.status,
                          emp.full_name
                   FROM attendance_events e
                   LEFT JOIN employees emp ON emp.employee_id = e.employee_id
                   ORDER BY e.event_timestamp DESC
                   LIMIT 200"""
            ).fetchall()

        self._log_table.setRowCount(0)
        for row in rows:
            self._add_log_row({
                "timestamp":  row["event_timestamp"],
                "event_type": row["event_type"],
                "confidence": row["confidence"],
                "status":     row["status"],
                "name":       row["full_name"] or "Unknown",
            })

    def _add_log_row(self, ev):
        row = self._log_table.rowCount()
        self._log_table.insertRow(0)           # newest on top
        ts   = ev.get("timestamp", "")
        name = ev.get("name") or (
            self._db.get_employee(ev.get("employee_id")) or {}
        ).get("full_name") or "Unknown"
        direction = "→ IN" if ev.get("event_type") == "in" else "← OUT"
        status    = ev.get("status", "")
        conf      = ev.get("confidence")
        status_display = status if conf is None else f"{status} ({conf:.2f})"

        self._log_table.setItem(0, 0, QTableWidgetItem(ts[:10] if ts else ""))
        self._log_table.setItem(0, 1, QTableWidgetItem(ts[11:19] if ts else ""))
        self._log_table.setItem(0, 2, QTableWidgetItem(name))
        self._log_table.setItem(0, 3, QTableWidgetItem(direction))
        status_item = QTableWidgetItem(status_display)
        color = STATUS_COLORS.get(status, "#aaa")
        status_item.setForeground(
            __import__("PyQt6.QtGui", fromlist=["QColor"]).QColor(color)
        )
        self._log_table.setItem(0, 4, status_item)

    def _refresh_status_bar(self):
        emp_count = len(self._db.list_employees())
        self._lbl_empcount.setText(f"Employees: {emp_count}")

    # ------------------------------------------------------------------ dialogs

    def _open_archive(self):
        try:
            from gui.archive_dialog import ArchiveDialog
            dlg = ArchiveDialog(self._config, self._db, parent=self)
            dlg.exec()
            self._refresh_log()
            self._refresh_status_bar()
        except Exception as exc:
            logger.exception("Could not open archive dialog")
            QMessageBox.critical(
                self,
                "Archive unavailable",
                "The archive window could not be opened.\n\n"
                f"{exc}\n\nSee logs/attendance_system.log for details.",
            )

    def _open_review(self):
        from gui.review_dialog import ReviewDialog
        dlg = ReviewDialog(self._config, self._db, parent=self)
        dlg.exec()
        self._refresh_log()

    def _open_employees(self):
        from gui.employees_dialog import EmployeesDialog
        dlg = EmployeesDialog(self._db, parent=self)
        dlg.exec()
        self._refresh_status_bar()

    def _open_settings(self):
        from gui.settings_dialog import SettingsDialog
        dlg = SettingsDialog(self._config, parent=self)
        if dlg.exec():
            QMessageBox.information(
                self, "Restart required",
                "Settings saved. Restart the application for camera changes to take effect.",
            )

    def _export_excel(self):
        try:
            from src.excel_sync import ExcelSync
            from src.path_utils import resolve_path
            excel_path = resolve_path(self._config.get("excel_sync.excel_path", "exports/attendance_log.xlsx"))
            sync = ExcelSync(self._db, excel_path=excel_path)
            count = sync.sync_now(force_all=True)
            QMessageBox.information(
                self, "Export Successful",
                f"Successfully exported {count} record(s) to:\n{excel_path}"
            )
        except Exception as exc:
            logger.exception("Excel export failed")
            QMessageBox.warning(self, "Export failed", str(exc))

    def _show_about(self):
        QMessageBox.about(
            self, "About",
            "Attendance System\n\nFace-recognition based door attendance tracking.\n"
            "Uses YOLOv8 + ByteTrack + InsightFace."
        )

    # ------------------------------------------------------------------ close

    def closeEvent(self, event):
        if self._worker is not None:
            self._worker.stop()
        event.accept()
