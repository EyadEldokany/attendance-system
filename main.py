"""
نقطة التشغيل الرئيسية لنظام الحضور والانصراف.

بيشغل في الخلفية:
1. الـ pipeline الأساسي (كاميرا -> كشف -> تتبع -> تعرف -> تسجيل)
2. thread مزامنة Excel المستمرة
3. مراقبة ملف إشارة إعادة تحميل فهرس التعرف (لما حد يسمي وجه جديد من الواجهة)

الاستخدام:
    python main.py

للمراجعة والتسمية اليدوية شغّل في نافذة/جهاز تاني (أو في أي وقت):
    python gui/review_gui.py
"""
import sys
import time
import threading
import logging
from pathlib import Path

from src.path_utils import get_app_dir, resolve_path
from src.config import load_config
from src.database import Database
from src.excel_sync import ExcelSync
from src.pipeline import AttendancePipeline
from src.logging_setup import setup_logging

PROJECT_ROOT = get_app_dir()
RELOAD_SIGNAL_FILE = PROJECT_ROOT / "data" / ".reload_index"


def watch_reload_signal(pipeline: AttendancePipeline, stop_event: threading.Event):
    """يراقب ملف الإشارة اللي بتكتبه واجهة المراجعة بعد أي تسمية جديدة، ويعيد بناء فهرس التعرف فورًا."""
    logger = logging.getLogger("attendance.reload_watcher")
    while not stop_event.is_set():
        if RELOAD_SIGNAL_FILE.exists():
            try:
                pipeline.reload_recognition_index()
                RELOAD_SIGNAL_FILE.unlink()
            except Exception:
                logger.exception("فشل تحديث فهرس التعرف بعد إشارة إعادة التحميل.")
        stop_event.wait(5)


def run_once(config_path=None):
    config_file = resolve_path(config_path) if config_path else "config.yaml"
    config = load_config(config_file)
    setup_logging(
        log_dir=resolve_path(config.get("logging.log_dir", "logs")),
        level=config.get("logging.level", "INFO"),
        max_log_size_mb=config.get("logging.max_log_size_mb", 20),
        backup_count=config.get("logging.backup_count", 10),
    )
    logger = logging.getLogger("attendance.main")
    logger.info("=" * 60)
    logger.info("بدء تشغيل نظام الحضور والانصراف")
    logger.info("=" * 60)

    db_path = resolve_path(config.get("database.sqlite_path", "data/attendance.db"))
    db = Database(str(db_path))
    pipeline = AttendancePipeline(config, db)

    excel_sync = None
    if config.get("excel_sync.enabled", True):
        excel_sync = ExcelSync(
            db=db,
            excel_path=resolve_path(config.get("excel_sync.excel_path", "exports/attendance_log.xlsx")),
            retry_max_attempts=config.get("excel_sync.retry_max_attempts", 30),
            retry_delay_seconds=config.get("excel_sync.retry_delay_seconds", 10),
        )

    stop_event = threading.Event()
    threads = []

    if excel_sync:
        excel_thread = threading.Thread(
            target=excel_sync.run_forever, args=(5, stop_event), daemon=True,
            name="excel-sync",
        )
        excel_thread.start()
        threads.append(excel_thread)

    reload_thread = threading.Thread(
        target=watch_reload_signal, args=(pipeline, stop_event), daemon=True,
        name="reload-watcher",
    )
    reload_thread.start()
    threads.append(reload_thread)

    try:
        pipeline.run_forever(stop_event)
    except KeyboardInterrupt:
        logger.info("تم إيقاف النظام يدويًا (Ctrl+C).")
    finally:
        stop_event.set()
        pipeline.camera.release()
        for t in threads:
            t.join(timeout=10)


def main():
    """
    حلقة خارجية بتعيد تشغيل النظام تلقائيًا لو حصل عطل غير متوقع (crash) بالكامل،
    عشان النظام يفضل شغال 24/7 من غير تدخل يدوي.
    """
    config = load_config("config.yaml")
    auto_restart = config.get("app.auto_restart_on_crash", True)

    while True:
        try:
            run_once()
            break  # خرج بشكل طبيعي (Ctrl+C) - مش عطل
        except Exception:
            logging.getLogger("attendance.main").exception(
                "توقف النظام بشكل غير متوقع!"
            )
            if not auto_restart:
                raise
            logging.getLogger("attendance.main").info(
                "إعادة تشغيل النظام تلقائيًا خلال 15 ثانية..."
            )
            time.sleep(15)


if __name__ == "__main__":
    main()
