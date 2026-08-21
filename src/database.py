"""
Database Layer - SQLite as the Source of Truth.
All real-time write operations happen here to ensure speed and data integrity,
even during event congestion or sudden shutdowns. The Excel file is synchronized from here
(see excel_sync.py), not the other way around - any manual edits in Excel by the client
do not count as truth.
"""
import sqlite3
import threading
import numpy as np
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS employees (
    employee_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name       TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    is_active       INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    embedding_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id     INTEGER NOT NULL REFERENCES employees(employee_id),
    embedding_blob  BLOB NOT NULL,
    source_image    TEXT,
    quality_score   REAL,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS attendance_events (
    event_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id     INTEGER REFERENCES employees(employee_id),  -- NULL 
    track_id        INTEGER,
    event_type      TEXT NOT NULL,        
    event_timestamp TEXT NOT NULL,
    confidence      REAL,
    status          TEXT NOT NULL DEFAULT 'confirmed',  -- 'confirmed' / 'needs_review' / 'rejected'
    face_crop_path  TEXT,
    reviewed_by     TEXT,
    reviewed_at     TEXT,
    synced_to_excel INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS unresolved_tracks (
    track_id        INTEGER PRIMARY KEY,
    first_seen      TEXT NOT NULL,
    last_seen       TEXT NOT NULL,
    capture_count   INTEGER NOT NULL DEFAULT 1,
    best_face_path  TEXT,
    status          TEXT NOT NULL DEFAULT 'pending_label'  -- 'pending_label' / 'labeled' / 'ignored'
);

CREATE TABLE IF NOT EXISTS pending_track_embeddings (
    pending_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id        INTEGER NOT NULL REFERENCES unresolved_tracks(track_id),
    embedding_blob  BLOB NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS video_files (
    video_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path            TEXT NOT NULL UNIQUE,
    fps                  REAL,
    total_frames         INTEGER,
    ocr_guess_text       TEXT,             -- best-effort OCR read of the burned-in timestamp
    ocr_confidence       REAL,
    start_timestamp       TEXT,             -- confirmed/corrected real-world start time of the recording
    status                TEXT NOT NULL DEFAULT 'pending_confirmation',
        -- 'pending_confirmation' / 'confirmed' / 'processed' / 'skipped' / 'error'
    thumbnail_path        TEXT,             -- crop of the burned-in timestamp region, shown for confirmation
    error_message          TEXT,
    processed_at           TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_timestamp ON attendance_events(event_timestamp);
CREATE INDEX IF NOT EXISTS idx_events_status ON attendance_events(status);
CREATE INDEX IF NOT EXISTS idx_events_synced ON attendance_events(synced_to_excel);
CREATE INDEX IF NOT EXISTS idx_video_status ON video_files(status);
"""


from src.path_utils import resolve_path


class Database:
    """
    SQLite wrapper handling all read/write operations for the system.
    Thread-safe using a simple lock because the system runs with multiple threads
    (real-time processing thread + Excel sync thread + GUI).
    """

    def __init__(self, db_path="data/attendance.db"):
        self.db_path = str(resolve_path(db_path))
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")  
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self):
        with self._lock, self._connect() as conn:
            conn.executescript(SCHEMA)

    # ---------------- Employees ----------------

    def add_employee(self, full_name: str) -> int:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO employees (full_name, created_at) VALUES (?, ?)",
                (full_name, datetime.now().isoformat()),
            )
            return cur.lastrowid

    def get_employee(self, employee_id: int):
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM employees WHERE employee_id = ?", (employee_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_employees(self, active_only=True):
        query = "SELECT * FROM employees"
        if active_only:
            query += " WHERE is_active = 1"
        with self._lock, self._connect() as conn:
            return [dict(r) for r in conn.execute(query)]

    def deactivate_employee(self, employee_id: int):
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE employees SET is_active = 0 WHERE employee_id = ?",
                (employee_id,),
            )

    # ---------------- Embeddings ----------------

    def add_embedding(self, employee_id: int, embedding_bytes: bytes,
                       source_image: str = None, quality_score: float = None):
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO face_embeddings
                   (employee_id, embedding_blob, source_image, quality_score, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (employee_id, embedding_bytes, source_image, quality_score,
                 datetime.now().isoformat()),
            )

    def get_all_embeddings(self):
        """بيرجع كل الـ embeddings مع employee_id المرتبط بيها - يستخدم لبناء فهرس FAISS عند بدء التشغيل."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT employee_id, embedding_blob FROM face_embeddings"
            ).fetchall()
            return [(r["employee_id"], r["embedding_blob"]) for r in rows]

    def count_embeddings_for(self, employee_id: int) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) c FROM face_embeddings WHERE employee_id = ?",
                (employee_id,),
            ).fetchone()
            return row["c"]

    # ---------------- Attendance Events ----------------

    def log_event(self, event_type: str, timestamp: str, employee_id: int = None,
                   track_id: int = None, confidence: float = None,
                   status: str = "confirmed", face_crop_path: str = None) -> int:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO attendance_events
                   (employee_id, track_id, event_type, event_timestamp, confidence,
                    status, face_crop_path, synced_to_excel)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 0)""",
                (employee_id, track_id, event_type, timestamp, confidence,
                 status, face_crop_path),
            )
            return cur.lastrowid

    def get_unsynced_events(self):
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """SELECT e.*, emp.full_name FROM attendance_events e
                   LEFT JOIN employees emp ON emp.employee_id = e.employee_id
                   WHERE e.synced_to_excel = 0
                   ORDER BY e.event_timestamp ASC"""
            ).fetchall()
            return [dict(r) for r in rows]

    def get_all_events(self):
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """SELECT e.*, emp.full_name FROM attendance_events e
                   LEFT JOIN employees emp ON emp.employee_id = e.employee_id
                   ORDER BY e.event_timestamp ASC"""
            ).fetchall()
            return [dict(r) for r in rows]

    def mark_synced(self, event_ids: list):
        if not event_ids:
            return
        with self._lock, self._connect() as conn:
            conn.executemany(
                "UPDATE attendance_events SET synced_to_excel = 1 WHERE event_id = ?",
                [(eid,) for eid in event_ids],
            )

    def get_events_needing_review(self):
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """SELECT e.*, COALESCE(e.face_crop_path, u.best_face_path) AS resolved_face_path
                   FROM attendance_events e
                   LEFT JOIN unresolved_tracks u ON u.track_id = e.track_id
                   WHERE e.status = 'needs_review' """
                "ORDER BY event_timestamp DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def resolve_review(self, event_id: int, employee_id: int, reviewer: str,
                       confidence: float = None):
        with self._lock, self._connect() as conn:
            conn.execute(
                """UPDATE attendance_events
                   SET employee_id = ?, status = 'confirmed',
                       confidence = COALESCE(?, confidence),
                       reviewed_by = ?, reviewed_at = ?, synced_to_excel = 0
                   WHERE event_id = ?""",
                (employee_id, confidence, reviewer, datetime.now().isoformat(), event_id),
            )

    def confirm_events_for_track(self, track_id: int, employee_id: int,
                                 reviewer: str = "face_labeling"):
        """Confirm review events after their track is assigned to an employee."""
        with self._lock, self._connect() as conn:
            pending_rows = conn.execute(
                "SELECT embedding_blob FROM pending_track_embeddings WHERE track_id = ?",
                (track_id,),
            ).fetchall()
            employee_rows = conn.execute(
                "SELECT embedding_blob FROM face_embeddings WHERE employee_id = ?",
                (employee_id,),
            ).fetchall()
            confidence = None
            if pending_rows and employee_rows:
                reference = np.vstack([
                    np.frombuffer(row["embedding_blob"], dtype=np.float32)
                    for row in employee_rows
                ])
                queries = np.vstack([
                    np.frombuffer(row["embedding_blob"], dtype=np.float32)
                    for row in pending_rows
                ])
                reference /= np.linalg.norm(reference, axis=1, keepdims=True).clip(min=1e-12)
                queries /= np.linalg.norm(queries, axis=1, keepdims=True).clip(min=1e-12)
                confidence = float(np.max(queries @ reference.T))

            cur = conn.execute(
                """UPDATE attendance_events
                   SET employee_id = ?, status = 'confirmed',
                       confidence = COALESCE(?, confidence),
                       reviewed_by = ?, reviewed_at = ?, synced_to_excel = 0
                   WHERE track_id = ? AND status = 'needs_review'""",
                (employee_id, confidence, reviewer, datetime.now().isoformat(), track_id),
            )
            return cur.rowcount

    def reject_review(self, event_id: int, reviewer: str):
        with self._lock, self._connect() as conn:
            conn.execute(
                """UPDATE attendance_events
                   SET status = 'rejected', reviewed_by = ?, reviewed_at = ?
                   WHERE event_id = ?""",
                (reviewer, datetime.now().isoformat(), event_id),
            )

    # ---------------- Unresolved tracks for enrollment ----------------

    def upsert_unresolved_track(self, track_id: int, face_path: str, timestamp: str,
                                replace_best: bool = False):
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM unresolved_tracks WHERE track_id = ?", (track_id,)
            ).fetchone()
            if existing:
                if replace_best and face_path:
                    conn.execute(
                        """UPDATE unresolved_tracks
                           SET last_seen = ?, capture_count = capture_count + 1,
                               best_face_path = ?
                           WHERE track_id = ?""",
                        (timestamp, face_path, track_id),
                    )
                else:
                    conn.execute(
                        """UPDATE unresolved_tracks
                           SET last_seen = ?, capture_count = capture_count + 1,
                               best_face_path = COALESCE(best_face_path, ?)
                           WHERE track_id = ?""",
                        (timestamp, face_path, track_id),
                    )
            else:
                conn.execute(
                    """INSERT INTO unresolved_tracks
                       (track_id, first_seen, last_seen, capture_count, best_face_path, status)
                       VALUES (?, ?, ?, 1, ?, 'pending_label')""",
                    (track_id, timestamp, timestamp, face_path),
                )

    def get_pending_labeling(self, min_captures: int = 3):
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM unresolved_tracks
                   WHERE status = 'pending_label' AND capture_count >= ?
                   ORDER BY first_seen ASC""",
                (min_captures,),
            ).fetchall()
            return [dict(r) for r in rows]

    def mark_track_labeled(self, track_id: int):
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE unresolved_tracks SET status = 'labeled' WHERE track_id = ?",
                (track_id,),
            )

    def add_pending_track_embedding(self, track_id: int, embedding_bytes: bytes):
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO pending_track_embeddings
                   (track_id, embedding_blob, created_at) VALUES (?, ?, ?)""",
                (track_id, embedding_bytes, datetime.now().isoformat()),
            )

    def get_pending_track_embeddings(self, track_id: int):
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """SELECT embedding_blob FROM pending_track_embeddings
                   WHERE track_id = ? ORDER BY pending_id ASC""",
                (track_id,),
            ).fetchall()
            return [r["embedding_blob"] for r in rows]

    def clear_pending_track_embeddings(self, track_id: int):
        with self._lock, self._connect() as conn:
            conn.execute(
                "DELETE FROM pending_track_embeddings WHERE track_id = ?",
                (track_id,),
            )

    # ---------------- Archive video files (batch/folder mode) ----------------

    def add_or_get_video(self, file_path: str, fps: float, total_frames: int,
                          ocr_guess_text: str = None, ocr_confidence: float = None,
                          thumbnail_path: str = None):
        """
        Registers a video file the first time it's scanned. If it was scanned
        before (same file_path), returns the existing row instead of
        duplicating it - this makes re-scanning the same archive folder safe.
        """
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM video_files WHERE file_path = ?", (file_path,)
            ).fetchone()
            if existing:
                # A failed processing attempt must be recoverable by scanning
                # the folder again; put it back in the human-confirmation queue.
                if existing["status"] in ("error", "pending_confirmation"):
                    conn.execute(
                        """UPDATE video_files
                           SET fps = ?, total_frames = ?, ocr_guess_text = ?,
                               ocr_confidence = ?, thumbnail_path = ?,
                               error_message = NULL,
                               status = 'pending_confirmation'
                           WHERE video_id = ?""",
                        (fps, total_frames, ocr_guess_text, ocr_confidence,
                         thumbnail_path, existing["video_id"]),
                    )
                    refreshed = conn.execute(
                        "SELECT * FROM video_files WHERE video_id = ?",
                        (existing["video_id"],),
                    ).fetchone()
                    return dict(refreshed)
                return dict(existing)
            cur = conn.execute(
                """INSERT INTO video_files
                   (file_path, fps, total_frames, ocr_guess_text, ocr_confidence,
                    thumbnail_path, status)
                   VALUES (?, ?, ?, ?, ?, ?, 'pending_confirmation')""",
                (file_path, fps, total_frames, ocr_guess_text, ocr_confidence, thumbnail_path),
            )
            return {
                "video_id": cur.lastrowid, "file_path": file_path, "fps": fps,
                "total_frames": total_frames, "ocr_guess_text": ocr_guess_text,
                "ocr_confidence": ocr_confidence, "start_timestamp": None,
                "status": "pending_confirmation", "thumbnail_path": thumbnail_path,
            }

    def get_pending_confirmation_videos(self):
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM video_files WHERE status = 'pending_confirmation' "
                "ORDER BY file_path ASC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_confirmed_unprocessed_videos(self):
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM video_files WHERE status = 'confirmed' "
                "ORDER BY file_path ASC"
            ).fetchall()
            return [dict(r) for r in rows]

    def confirm_video_start(self, video_id: int, start_timestamp: str):
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE video_files SET start_timestamp = ?, status = 'confirmed' "
                "WHERE video_id = ?",
                (start_timestamp, video_id),
            )

    def mark_video_processed(self, video_id: int):
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE video_files SET status = 'processed', processed_at = ? "
                "WHERE video_id = ?",
                (datetime.now().isoformat(), video_id),
            )

    def mark_video_error(self, video_id: int, error_message: str):
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE video_files SET status = 'error', error_message = ? "
                "WHERE video_id = ?",
                (error_message, video_id),
            )

    def skip_video(self, video_id: int):
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE video_files SET status = 'skipped' WHERE video_id = ?",
                (video_id,),
            )

    def get_video_stats(self):
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) c FROM video_files GROUP BY status"
            ).fetchall()
            return {r["status"]: r["c"] for r in rows}
