"""
Sync attendance records from SQLite (primary source) to an Excel file.

SQLite writes each event immediately and securely. This thread runs in the background,
adding any new events (that haven't been synced yet) to the Excel file. If the file is
open (e.g., HR is reviewing it), it retries automatically instead of failing or losing data.
"""
import time
import logging
import threading
from pathlib import Path
from datetime import datetime

from openpyxl import Workbook, load_workbook

from src.path_utils import resolve_path

logger = logging.getLogger("attendance.excel_sync")

HEADERS = ["Employee Name", "Date", "Time", "Event Type", "Confidence", "Status"]
EVENT_TYPE_AR = {"in": "In", "out": "Out"}
STATUS_AR = {
    "confirmed": "Confirmed",
    "needs_review": "Needs Review",
    "rejected": "Rejected",
}


class ExcelSync:
    def __init__(self, db, excel_path="exports/attendance_log.xlsx",
                 retry_max_attempts=30, retry_delay_seconds=10):
        self.db = db
        self.excel_path = resolve_path(excel_path)
        self.excel_path.parent.mkdir(parents=True, exist_ok=True)
        self.retry_max_attempts = retry_max_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self._ensure_workbook_exists()

    def _ensure_workbook_exists(self):
        if not self.excel_path.exists():
            wb = Workbook()
            ws = wb.active
            ws.title = "Attendance Log"
            ws.sheet_view.rightToLeft = False
            ws.append(HEADERS)
            for col_idx, _ in enumerate(HEADERS, start=1):
                ws.column_dimensions[chr(64 + col_idx)].width = 20
            wb.save(self.excel_path)
            logger.info(f"New Excel file created: {self.excel_path}")

    def _append_rows(self, events: list):
        """
        Tries to open the file and append new rows. If the file is closed (opened in Excel
        by the user), it retries every retry_delay_seconds up to retry_max_attempts.
        """
        attempts = 0
        while attempts < self.retry_max_attempts:
            try:
                wb = load_workbook(self.excel_path)
                ws = wb.active
                for ev in events:
                    dt = datetime.fromisoformat(ev["event_timestamp"])
                    ws.append([
                        ev.get("full_name") or "Unknown - Needs Review",
                        dt.strftime("%Y-%m-%d"),
                        dt.strftime("%H:%M:%S"),
                        EVENT_TYPE_AR.get(ev["event_type"], ev["event_type"]),
                        round(ev["confidence"], 3) if ev["confidence"] is not None else "",
                        STATUS_AR.get(ev["status"], ev["status"]),
                    ])
                wb.save(self.excel_path)
                return True
            except PermissionError:
                attempts += 1
                logger.warning(
                    f"Excel file is currently open, retrying "
                    f"({attempts}/{self.retry_max_attempts}) after {self.retry_delay_seconds} seconds..."
                )
                time.sleep(self.retry_delay_seconds)
            except Exception as e:
                logger.error(f"Unexpected error while writing to Excel: {e}")
                return False
        logger.error(
            "All attempts to write to Excel failed because the file is closed - "
            "events are safely stored in the database and will be synced later."
        )
        return False

    def sync_once(self):
        """Syncs all events that are pending (not yet written to Excel)."""
        pending = self.db.get_unsynced_events()
        if not pending:
            return 0
        success = self._append_rows(pending)
        if success:
            self.db.mark_synced([ev["event_id"] for ev in pending])
            logger.info(f"Synced {len(pending)} new event(s) to Excel.")
            return len(pending)
        return 0

    def export_all(self):
        """Rebuilds and exports all attendance records from SQLite database to the Excel sheet."""
        events = self.db.get_all_events() if hasattr(self.db, "get_all_events") else self.db.get_unsynced_events()
        if not events:
            self._ensure_workbook_exists()
            return 0

        attempts = 0
        while attempts < self.retry_max_attempts:
            try:
                wb = Workbook()
                ws = wb.active
                ws.title = "Attendance Log"
                ws.sheet_view.rightToLeft = False
                ws.append(HEADERS)
                for col_idx, _ in enumerate(HEADERS, start=1):
                    ws.column_dimensions[chr(64 + col_idx)].width = 20

                for ev in events:
                    dt = datetime.fromisoformat(ev["event_timestamp"])
                    ws.append([
                        ev.get("full_name") or "Unknown - Needs Review",
                        dt.strftime("%Y-%m-%d"),
                        dt.strftime("%H:%M:%S"),
                        EVENT_TYPE_AR.get(ev["event_type"], ev["event_type"]),
                        round(ev["confidence"], 3) if ev["confidence"] is not None else "",
                        STATUS_AR.get(ev["status"], ev["status"]),
                    ])
                wb.save(self.excel_path)
                self.db.mark_synced([ev["event_id"] for ev in events])
                logger.info(f"Exported {len(events)} attendance event(s) to Excel: {self.excel_path}")
                return len(events)
            except PermissionError:
                attempts += 1
                logger.warning(
                    f"Excel file is open, retrying export ({attempts}/{self.retry_max_attempts}) after {self.retry_delay_seconds}s..."
                )
                time.sleep(self.retry_delay_seconds)
            except Exception as e:
                logger.error(f"Error exporting Excel: {e}")
                return 0
        return 0

    def sync_now(self, force_all: bool = True):
        """
        Public method to trigger sync or export immediately.
        When force_all is True (default for manual export button), exports all events from database.
        """
        if force_all:
            return self.export_all()
        return self.sync_once()

    def run_forever(self, interval_seconds=5, stop_event: threading.Event = None):
        """
        Runs a continuous sync loop in the background. The default value (every 5 seconds)
        is very close to "instant" without burdening the system by writing too frequently for each event.
        """
        stop_event = stop_event or threading.Event()
        while not stop_event.is_set():
            try:
                self.sync_once()
            except Exception as e:
                logger.exception(f"Error in Excel sync loop: {e}")
            stop_event.wait(interval_seconds)
