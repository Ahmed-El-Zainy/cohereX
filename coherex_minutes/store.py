"""SQLite job store with atomic FIFO claiming and crash recovery."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass
class Job:
    meeting_id: str
    video_url: str
    language: str
    status: str
    stage: str | None
    progress: int
    retry_count: int
    download_status: str
    error_code: str | None
    error_message: str | None
    result: dict[str, Any] | None
    created_at: str
    updated_at: str
    generated_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JobStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    meeting_id TEXT PRIMARY KEY,
                    video_url TEXT NOT NULL,
                    language TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stage TEXT,
                    progress INTEGER NOT NULL DEFAULT 0
                        CHECK (progress BETWEEN 0 AND 100),
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    download_status TEXT NOT NULL DEFAULT 'PENDING',
                    error_code TEXT,
                    error_message TEXT,
                    result_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    generated_at TEXT
                )
                """
            )
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(jobs)").fetchall()
            }
            if "retry_count" not in columns:
                connection.execute(
                    "ALTER TABLE jobs ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0"
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(status, download_status, created_at)"
            )

    @staticmethod
    def _row_to_job(row: sqlite3.Row | None) -> Job | None:
        if row is None:
            return None
        values = dict(row)
        raw_result = values.pop("result_json")
        values["result"] = json.loads(raw_result) if raw_result else None
        return Job(**values)

    def get(self, meeting_id: str) -> Job | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE meeting_id = ?", (meeting_id,)
            ).fetchone()
        return self._row_to_job(row)

    def create(self, meeting_id: str, video_url: str, language: str = "ar") -> tuple[Job, bool]:
        now = utc_now()
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO jobs (
                    meeting_id, video_url, language, status, stage, progress,
                    download_status, created_at, updated_at
                ) VALUES (?, ?, ?, 'QUEUED', 'QUEUED', 0, 'PENDING', ?, ?)
                """,
                (meeting_id, video_url, language, now, now),
            )
            row = connection.execute(
                "SELECT * FROM jobs WHERE meeting_id = ?", (meeting_id,)
            ).fetchone()
        job = self._row_to_job(row)
        assert job is not None
        return job, cursor.rowcount == 1

    def pending_downloads(self) -> list[Job]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM jobs
                WHERE status = 'QUEUED' AND download_status IN ('PENDING', 'DOWNLOADING')
                ORDER BY created_at, rowid
                """
            ).fetchall()
        return [job for row in rows if (job := self._row_to_job(row)) is not None]

    def terminal_meeting_ids(self) -> list[str]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT meeting_id FROM jobs WHERE status IN ('COMPLETED', 'FAILED')"
            ).fetchall()
        return [str(row["meeting_id"]) for row in rows]

    def set_download_status(self, meeting_id: str, status: str) -> None:
        self._update(meeting_id, download_status=status)

    def claim_next(self) -> Job | None:
        """Resume an interrupted PROCESSING job, otherwise claim FIFO READY job."""
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM jobs
                WHERE status = 'PROCESSING'
                ORDER BY created_at, rowid
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                # Strict FIFO: a later ready download may not overtake the oldest
                # accepted job while its signed URL is still downloading.
                row = connection.execute(
                    """
                    SELECT * FROM jobs
                    WHERE status = 'QUEUED'
                    ORDER BY created_at, rowid
                    LIMIT 1
                    """
                ).fetchone()
            if row is None:
                return None
            if row["status"] == "QUEUED" and row["download_status"] != "READY":
                return None
            if row["status"] == "QUEUED":
                connection.execute(
                    """
                    UPDATE jobs
                    SET status = 'PROCESSING', stage = 'TRANSCRIBING', updated_at = ?
                    WHERE meeting_id = ?
                    """,
                    (now, row["meeting_id"]),
                )
                row = connection.execute(
                    "SELECT * FROM jobs WHERE meeting_id = ?", (row["meeting_id"],)
                ).fetchone()
        return self._row_to_job(row)

    def update_progress(self, meeting_id: str, stage: str, progress: int) -> None:
        """Record progress, resetting the retry counter only on real advancement.

        ``retry_count`` counts *consecutive* failures, so forward progress is
        allowed to forgive earlier ones. A resumed job, however, replays the
        progress it already reached — ``MeetingProcessor`` re-emits stage and
        percent from durable checkpoints on every attempt — and treating a
        replay as advancement would clear the counter each time, so a job that
        fails deterministically after its first progress write would retry for
        ever instead of failing at ``max_processing_retries``.

        ``progress`` is therefore a high-water mark (also what the platform
        contract wants: it must never appear to go backwards), and only a value
        strictly above that mark counts as advancement. Both expressions below
        read the pre-update row, so they compare against the stored value.
        """
        value = max(0, min(99, int(progress)))
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET status = 'PROCESSING',
                    stage = ?,
                    progress = MAX(progress, ?),
                    retry_count = CASE WHEN ? > progress THEN 0 ELSE retry_count END,
                    updated_at = ?
                WHERE meeting_id = ?
                """,
                (stage, value, value, utc_now(), meeting_id),
            )

    def increment_retry(self, meeting_id: str) -> int:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET retry_count = retry_count + 1, updated_at = ?
                WHERE meeting_id = ?
                """,
                (utc_now(), meeting_id),
            )
            row = connection.execute(
                "SELECT retry_count FROM jobs WHERE meeting_id = ?", (meeting_id,)
            ).fetchone()
        return int(row["retry_count"]) if row else 0

    def complete(self, meeting_id: str, result: dict[str, Any]) -> None:
        now = utc_now()
        self._update(
            meeting_id,
            status="COMPLETED",
            stage="COMPLETED",
            progress=100,
            result_json=json.dumps(result, ensure_ascii=False),
            generated_at=now,
            error_code=None,
            error_message=None,
        )

    def fail(
        self,
        meeting_id: str,
        code: str,
        message: str,
        *,
        stage: str | None = None,
    ) -> None:
        values: dict[str, Any] = {
            "status": "FAILED",
            "error_code": code,
            "error_message": message,
        }
        if stage:
            values["stage"] = stage
        self._update(meeting_id, **values)

    def _update(self, meeting_id: str, **values: Any) -> None:
        if not values:
            return
        values["updated_at"] = utc_now()
        assignments = ", ".join(f"{key} = ?" for key in values)
        params = [*values.values(), meeting_id]
        with self.connect() as connection:
            connection.execute(
                f"UPDATE jobs SET {assignments} WHERE meeting_id = ?",  # noqa: S608
                params,
            )
