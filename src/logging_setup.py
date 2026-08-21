import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from src.path_utils import resolve_path


def setup_logging(log_dir="logs", level="INFO", max_log_size_mb=20, backup_count=10):
    log_dir_path = resolve_path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_file = log_dir_path / "attendance_system.log"

    root_logger = logging.getLogger("attendance")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        log_file, maxBytes=max_log_size_mb * 1024 * 1024, backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    return root_logger
