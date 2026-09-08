"""Environment-backed configuration shared by the API and worker."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value else default


# Transcript characters per LLM slice, and the ceiling used when compacting
# notes back down for the final sections pass.
#
# Qwen2.5 tokenizes this project's Arabic transcripts at ~2.7 chars/token
# (measured against out-ar/saudi_business_03min.txt), so 10,000 chars is
# ~3,700 prompt tokens. The decisions and sections prompts also reserve
# COHEREX_MINUTES_LLM_MAX_TOKENS (1800) for the response, which leaves real
# headroom inside the llama.cpp server's `-c 8192` context window (see
# docs/LLM_DEPLOYMENT.md). Raising this without also raising `-c` overflows
# that window on every attempt, and the failure is deterministic, so the job
# never succeeds.
DEFAULT_SLICE_CHARS = 10_000


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    api_key: str
    vllm_url: str
    vllm_api_key: str
    vllm_model: str
    llm_url: str
    llm_api_key: str
    max_video_bytes: int
    min_free_bytes: int
    ingest_sweep_seconds: float
    max_duration_seconds: float
    chunk_seconds: int
    asr_timeout_seconds: float
    service_timeout_seconds: float
    worker_poll_seconds: float
    max_processing_retries: int
    transcript_slice_chars: int
    llm_max_tokens: int
    llm_timeout_seconds: float
    manage_services: bool
    video_allowed_hosts: tuple[str, ...]
    asr_service: str
    llm_service: str

    @property
    def database_path(self) -> Path:
        return self.data_dir / "jobs.sqlite3"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @classmethod
    def from_env(cls) -> "Settings":
        default_data_dir = (
            Path("/var/lib/coherex-minutes")
            if os.name != "nt"
            else Path.home() / ".coherex-minutes"
        )
        return cls(
            data_dir=Path(os.environ.get("COHEREX_MINUTES_DATA_DIR", default_data_dir)),
            api_key=os.environ.get("AI_SERVICE_API_KEY", ""),
            vllm_url=os.environ.get("COHEREX_VLLM_URL", "http://127.0.0.1:8000"),
            vllm_api_key=os.environ.get("COHEREX_VLLM_API_KEY", ""),
            vllm_model=os.environ.get(
                "COHEREX_VLLM_MODEL",
                "CohereLabs/cohere-transcribe-arabic-07-2026",
            ),
            llm_url=os.environ.get("COHEREX_LLM_URL", "http://127.0.0.1:8001"),
            llm_api_key=os.environ.get("COHEREX_LLM_API_KEY", ""),
            max_video_bytes=_int_env("COHEREX_MINUTES_MAX_VIDEO_BYTES", 2_000_000_000),
            # Disk the ingester will not consume. Downloads begin at POST but
            # the worker is serial, so queued videos accumulate; without a
            # floor a deep queue fills the disk, and once SQLite cannot write
            # the API stops accepting requests at all. Below this floor a
            # download is deferred, never failed -- see IngestManager.
            min_free_bytes=_int_env("COHEREX_MINUTES_MIN_FREE_BYTES", 5_000_000_000),
            # How often deferred or interrupted downloads are retried.
            ingest_sweep_seconds=_float_env("COHEREX_MINUTES_INGEST_SWEEP_SECONDS", 60),
            max_duration_seconds=_float_env("COHEREX_MINUTES_MAX_DURATION_SECONDS", 10_800),
            chunk_seconds=_int_env("COHEREX_MINUTES_CHUNK_SECONDS", 30),
            asr_timeout_seconds=_float_env("COHEREX_MINUTES_ASR_TIMEOUT_SECONDS", 900),
            service_timeout_seconds=_float_env("COHEREX_MINUTES_SERVICE_TIMEOUT_SECONDS", 600),
            worker_poll_seconds=_float_env("COHEREX_MINUTES_WORKER_POLL_SECONDS", 2),
            max_processing_retries=_int_env("COHEREX_MINUTES_MAX_RETRIES", 10),
            transcript_slice_chars=_int_env(
                "COHEREX_MINUTES_SLICE_CHARS", DEFAULT_SLICE_CHARS
            ),
            llm_max_tokens=_int_env("COHEREX_MINUTES_LLM_MAX_TOKENS", 1800),
            llm_timeout_seconds=_float_env("COHEREX_MINUTES_LLM_TIMEOUT_SECONDS", 900),
            manage_services=os.environ.get("COHEREX_MINUTES_MANAGE_SERVICES", "true").lower()
            in {"1", "true", "yes"},
            video_allowed_hosts=tuple(
                host.strip().lower()
                for host in os.environ.get("COHEREX_VIDEO_ALLOWED_HOSTS", "").split(",")
                if host.strip()
            ),
            asr_service=os.environ.get("COHEREX_ASR_SERVICE", "coherex-vllm"),
            llm_service=os.environ.get("COHEREX_LLM_SERVICE", "coherex-llm"),
        )

    def prepare(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
