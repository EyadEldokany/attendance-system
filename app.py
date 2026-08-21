#!/usr/bin/env python3
"""
Main entry point for the Attendance System GUI application.
Run this file directly:  python app.py
"""
import logging
import os
import sys
import traceback
from pathlib import Path

import torch
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

from src.path_utils import get_app_dir, get_bundle_dir, resolve_path

# 1. ضبط مسار المشروع الأساسي والتنفيذي
BASE_DIR = get_app_dir()
BUNDLE_DIR = get_bundle_dir()

# 2. توجيه InsightFace لقراءة النماذج المحزومة داخل الـ EXE
if getattr(sys, "frozen", False):
    os.environ["INSIGHTFACE_HOME"] = str(BUNDLE_DIR / ".insightface")

# Ensure src/ and gui/ are importable when launched as a script or packaged .exe
sys.path.insert(0, str(BUNDLE_DIR))
sys.path.insert(0, str(BASE_DIR))

from gui.main_window import MainWindow
from src.config import Config, load_config
from src.database import Database
from src.logging_setup import setup_logging


def main():
    Config._instance = None
    # قراءة ملف الإعدادات من المسار المحزوم أو المحلي
    cfg = load_config("config.yaml")

    setup_logging(
        log_dir=cfg.get("logging.log_dir", "logs"),
        level=cfg.get("logging.level", "INFO"),
        max_log_size_mb=cfg.get("logging.max_log_size_mb", 20),
        backup_count=cfg.get("logging.backup_count", 10),
    )

    def report_unhandled_exception(exc_type, exc_value, exc_traceback):
        details = "".join(
            traceback.format_exception(exc_type, exc_value, exc_traceback)
        )
        logging.getLogger("attendance.app").critical(
            "Unhandled application exception:\n%s", details
        )
        from PyQt6.QtWidgets import QMessageBox

        QMessageBox.critical(
            None,
            "Attendance System Error",
            f"The application encountered an error:\n\n{exc_value}\n\n"
            "See logs/attendance_system.log for details.",
        )

    sys.excepthook = report_unhandled_exception

    # ضبط مسار قاعدة البيانات SQLite
    db_rel_path = cfg.get("database.sqlite_path", "data/attendance.db")
    db_path = str(resolve_path(db_rel_path))

    # التأكد من وجود مجلد قاعدة البيانات في حال لم يكن موجوداً
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    db = Database(db_path)

    from src.timestamp_ocr import configure_tesseract_path

    configure_tesseract_path(cfg.get("archive.tesseract_cmd", ""))

    app = QApplication(sys.argv)
    app.setApplicationName("Attendance System")
    app.setOrganizationName("AttendanceCo")

    icon_path = resolve_path("assets/icon.ico")
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    mode = cfg.get("app.mode", "archive")
    worker = None
    if mode == "live":
        from gui.pipeline_worker import PipelineWorker

        worker = PipelineWorker(cfg, db)

    window = MainWindow(cfg, db, worker)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()