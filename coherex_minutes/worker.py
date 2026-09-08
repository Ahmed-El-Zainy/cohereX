"""Single durable FIFO worker for meeting-minutes generation."""

from __future__ import annotations

import logging
import shutil
import signal
import time
from contextlib import contextmanager
from typing import Iterator, TextIO

from .config import Settings
from .processor import FatalJobError, MeetingProcessor
from .store import JobStore

logger = logging.getLogger(__name__)


@contextmanager
def single_worker_lock(settings: Settings) -> Iterator[TextIO]:
    """Prevent two worker processes from resuming the same PROCESSING job."""
    settings.prepare()
    handle = (settings.data_dir / "worker.lock").open("a+", encoding="ascii")
    try:
        try:
            import fcntl
        except ImportError:
            # Production is Linux. Tests and local Windows development do not
            # need an inter-process lock.
            pass
        else:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another coherex-minutes worker is already running") from exc
        yield handle
    finally:
        handle.close()


class Worker:
    def __init__(
        self,
        settings: Settings,
        store: JobStore | None = None,
        processor: MeetingProcessor | None = None,
    ):
        settings.prepare()
        self.settings = settings
        self.store = store or JobStore(settings.database_path)
        self.processor = processor or MeetingProcessor(settings, self.store)
        self.running = True
        for meeting_id in self.store.terminal_meeting_ids():
            self._delete_video(meeting_id)

    def stop(self, *_: object) -> None:
        self.running = False

    def run_once(self) -> bool:
        job = self.store.claim_next()
        if job is None:
            return False
        logger.info("Processing meeting %s at stage %s", job.meeting_id, job.stage)
        try:
            result = self.processor.process(job)
            self.store.complete(job.meeting_id, result)
            self._delete_video(job.meeting_id)
            self._restore_asr()
            logger.info("Completed meeting %s", job.meeting_id)
        except FatalJobError as exc:
            current = self.store.get(job.meeting_id)
            self.store.fail(
                job.meeting_id,
                exc.code,
                str(exc),
                stage=current.stage if current else job.stage,
            )
            self._delete_video(job.meeting_id)
            self._restore_asr()
            logger.exception("Meeting %s failed permanently: %s", job.meeting_id, exc)
        except Exception:
            attempt = self.store.increment_retry(job.meeting_id)
            if attempt >= self.settings.max_processing_retries:
                current = self.store.get(job.meeting_id)
                stage = current.stage if current else job.stage
                code = (
                    "MINUTES_GENERATION_FAILED"
                    if stage == "GENERATING_MINUTES"
                    else "TRANSCRIPTION_FAILED"
                )
                self.store.fail(
                    job.meeting_id,
                    code,
                    f"Processing failed after {attempt} consecutive retries.",
                    stage=stage,
                )
                self._delete_video(job.meeting_id)
                self._restore_asr()
                logger.exception(
                    "Meeting %s exhausted %s retries", job.meeting_id, attempt
                )
            else:
                # Leave PROCESSING untouched. The next iteration resumes from
                # durable ASR/LLM checkpoints.
                delay = min(
                    300,
                    self.settings.worker_poll_seconds * (2 ** max(0, attempt - 1)),
                )
                logger.exception(
                    "Transient failure for meeting %s; retry %s/%s in %.0fs",
                    job.meeting_id,
                    attempt,
                    self.settings.max_processing_retries,
                    delay,
                )
                time.sleep(delay)
        return True

    def run_forever(self) -> None:
        self._restore_asr()
        while self.running:
            worked = self.run_once()
            if not worked:
                time.sleep(self.settings.worker_poll_seconds)

    def _delete_video(self, meeting_id: str) -> None:
        job_dir = self.settings.jobs_dir / meeting_id
        for name in ("source.video", "source.video.part"):
            (job_dir / name).unlink(missing_ok=True)
        shutil.rmtree(job_dir / "chunks", ignore_errors=True)

    def _restore_asr(self) -> None:
        services = getattr(self.processor, "services", None)
        if services is None:
            return
        try:
            services.switch_to("asr")
        except Exception:
            # The job result is already durable. Keep it COMPLETED/FAILED and
            # let systemd/ops restore the preferred idle backend.
            logger.exception("Could not restore ASR as the idle service")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = Settings.from_env()
    with single_worker_lock(settings):
        worker = Worker(settings)
        signal.signal(signal.SIGTERM, worker.stop)
        signal.signal(signal.SIGINT, worker.stop)
        worker.run_forever()


if __name__ == "__main__":
    main()
