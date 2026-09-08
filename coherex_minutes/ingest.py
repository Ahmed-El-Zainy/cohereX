"""Background video ingestion started immediately after POST acceptance."""

from __future__ import annotations

import ipaddress
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from urllib.parse import urljoin, urlparse

import httpx

from .config import Settings
from .store import Job, JobStore


class IngestError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class IngestManager:
    """Download accepted URLs without blocking API requests or the ASR worker."""

    def __init__(self, settings: Settings, store: JobStore, max_workers: int = 2):
        self.settings = settings
        self.store = store
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="video-ingest")
        self._scheduled: set[str] = set()
        self._lock = Lock()

    def recover(self) -> None:
        for job in self.store.pending_downloads():
            self.schedule(job)

    def schedule(self, job: Job) -> None:
        with self._lock:
            if job.meeting_id in self._scheduled:
                return
            self._scheduled.add(job.meeting_id)
        self.executor.submit(self._run, job)

    def _run(self, job: Job) -> None:
        try:
            self.store.set_download_status(job.meeting_id, "DOWNLOADING")
            for attempt in range(3):
                try:
                    self.download(job)
                    break
                except IngestError as exc:
                    if not exc.retryable or attempt == 2:
                        raise
                    time.sleep(2**attempt)
            self.store.set_download_status(job.meeting_id, "READY")
        except IngestError as exc:
            self.store.fail(job.meeting_id, exc.code, str(exc), stage="TRANSCRIBING")
            self._delete_video(job.meeting_id)
        except Exception as exc:
            self.store.fail(
                job.meeting_id,
                "INVALID_VIDEO_URL",
                f"Video download failed: {exc}",
                stage="TRANSCRIBING",
            )
            self._delete_video(job.meeting_id)
        finally:
            with self._lock:
                self._scheduled.discard(job.meeting_id)

    def download(self, job: Job) -> Path:
        job_dir = self.settings.jobs_dir / job.meeting_id
        job_dir.mkdir(parents=True, exist_ok=True)
        destination = job_dir / "source.video"
        temporary = job_dir / "source.video.part"
        temporary.unlink(missing_ok=True)

        headers = {"Accept-Encoding": "identity"}
        try:
            timeout = httpx.Timeout(30.0, read=300.0)
            url = job.video_url
            with httpx.Client(timeout=timeout, follow_redirects=False) as client:
                for redirect_count in range(6):
                    self._validate_public_url(url)
                    with client.stream("GET", url, headers=headers) as response:
                        if response.is_redirect:
                            location = response.headers.get("location")
                            if not location or redirect_count == 5:
                                raise IngestError(
                                    "INVALID_VIDEO_URL",
                                    "The video URL has an invalid redirect chain.",
                                )
                            url = urljoin(url, location)
                            continue
                        response.raise_for_status()
                        declared = response.headers.get("content-length")
                        if declared and int(declared) > self.settings.max_video_bytes:
                            raise IngestError(
                                "VIDEO_TOO_LARGE",
                                f"Video exceeds the {self.settings.max_video_bytes} byte limit.",
                            )
                        total = 0
                        with temporary.open("wb") as output:
                            for block in response.iter_bytes(chunk_size=1024 * 1024):
                                total += len(block)
                                if total > self.settings.max_video_bytes:
                                    raise IngestError(
                                        "VIDEO_TOO_LARGE",
                                        f"Video exceeds the {self.settings.max_video_bytes} byte limit.",
                                    )
                                output.write(block)
                            output.flush()
                            os.fsync(output.fileno())
                        break
        except httpx.HTTPError as exc:
            raise IngestError(
                "INVALID_VIDEO_URL",
                "The video URL is invalid or cannot be accessed.",
                retryable=not isinstance(exc, httpx.HTTPStatusError)
                or exc.response.status_code >= 500,
            ) from exc

        temporary.replace(destination)
        return destination

    def _validate_public_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise IngestError("INVALID_VIDEO_URL", "The video URL must be an http(s) URL.")
        hostname = parsed.hostname.rstrip(".").lower()
        if not self.settings.video_allowed_hosts:
            raise IngestError(
                "VIDEO_HOST_NOT_ALLOWED",
                "No video download hosts are configured.",
            )
        if hostname not in self.settings.video_allowed_hosts:
            raise IngestError(
                "VIDEO_HOST_NOT_ALLOWED",
                "The video URL hostname is not in the configured allowlist.",
            )
        try:
            addresses = {
                ipaddress.ip_address(item[4][0])
                for item in socket.getaddrinfo(hostname, parsed.port or 443)
            }
        except (OSError, ValueError) as exc:
            raise IngestError(
                "INVALID_VIDEO_URL",
                "The video URL hostname could not be resolved.",
                retryable=True,
            ) from exc
        if any(
            not address.is_global
            or (
                (mapped := getattr(address, "ipv4_mapped", None)) is not None
                and not mapped.is_global
            )
            for address in addresses
        ):
            raise IngestError(
                "INVALID_VIDEO_URL",
                "Private, loopback, link-local, and reserved video URLs are not allowed.",
            )

    def _delete_video(self, meeting_id: str) -> None:
        job_dir = self.settings.jobs_dir / meeting_id
        for name in ("source.video", "source.video.part"):
            (job_dir / name).unlink(missing_ok=True)

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=False)
