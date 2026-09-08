from __future__ import annotations

from dataclasses import replace

import pytest

from coherex_minutes.config import Settings
from coherex_minutes.ingest import IngestError, IngestManager
from coherex_minutes.processor import MeetingProcessor
from coherex_minutes.store import JobStore
from coherex_minutes.worker import Worker


def make_settings(tmp_path):
    return replace(
        Settings.from_env(),
        data_dir=tmp_path,
        api_key="secret",
        manage_services=False,
        worker_poll_seconds=0,
    )


def test_store_claims_fifo_and_resumes_processing_first(tmp_path):
    settings = make_settings(tmp_path)
    store = JobStore(settings.database_path)
    store.create("first", "https://example.com/1.mp4")
    store.create("second", "https://example.com/2.mp4")
    store.set_download_status("first", "READY")
    store.set_download_status("second", "READY")

    first = store.claim_next()
    resumed = store.claim_next()

    assert first and first.meeting_id == "first"
    assert resumed and resumed.meeting_id == "first"


def test_video_download_requires_explicit_hostname_allowlist(tmp_path):
    settings = make_settings(tmp_path)
    store = JobStore(settings.database_path)
    ingest = IngestManager(settings, store)
    with pytest.raises(IngestError) as caught:
        ingest._validate_public_url("https://storage.example.com/meeting.mp4")
    ingest.close()
    assert caught.value.code == "VIDEO_HOST_NOT_ALLOWED"


def test_ready_job_cannot_overtake_older_downloading_job(tmp_path):
    settings = make_settings(tmp_path)
    store = JobStore(settings.database_path)
    store.create("downloading", "https://example.com/1.mp4")
    store.create("ready", "https://example.com/2.mp4")
    store.set_download_status("downloading", "DOWNLOADING")
    store.set_download_status("ready", "READY")

    assert store.claim_next() is None
    store.set_download_status("downloading", "READY")
    assert store.claim_next().meeting_id == "downloading"


def test_failed_job_is_sticky_and_never_claimed_again(tmp_path):
    settings = make_settings(tmp_path)
    store = JobStore(settings.database_path)
    original, _ = store.create("failed", "https://example.com/a.mp4")
    store.fail("failed", "INVALID_VIDEO_URL", "bad URL", stage="TRANSCRIBING")

    same, created = store.create("failed", "https://example.com/new.mp4")

    assert not created
    assert same.created_at == original.created_at
    assert same.status == "FAILED"
    assert same.video_url == original.video_url
    assert store.claim_next() is None


def test_asr_checkpoint_ignores_partial_last_line(tmp_path):
    checkpoint = tmp_path / "asr-checkpoint.jsonl"
    checkpoint.write_text(
        '{"index": 0, "text": "الأول"}\n{"index": 1, "text": "الثاني"}\n{"index":',
        encoding="utf-8",
    )
    assert MeetingProcessor._read_asr_checkpoint(checkpoint) == {
        0: "الأول",
        1: "الثاني",
    }


def test_regenerating_chunks_discards_downstream_checkpoints(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    store = JobStore(settings.database_path)
    job, _ = store.create("regenerate", "https://example.com/a.mp4")
    job_dir = settings.jobs_dir / job.meeting_id
    job_dir.mkdir(parents=True)
    source = job_dir / "source.video"
    source.write_bytes(b"video")
    (job_dir / "asr-checkpoint.jsonl").write_text(
        '{"index": 0, "text": "stale"}\n', encoding="utf-8"
    )
    (job_dir / "transcript.txt").write_text("stale", encoding="utf-8")
    (job_dir / "sections.json").write_text("[]", encoding="utf-8")
    (job_dir / "llm-slices").mkdir()
    (job_dir / "llm-slices" / "0000-notes.txt").write_text("stale", encoding="utf-8")

    monkeypatch.setattr(MeetingProcessor, "_probe_duration", lambda *_: 60.0)

    def fake_ffmpeg(command, check):
        assert check
        output_pattern = command[-1]
        chunk = output_pattern.replace("%06d", "000000")
        with open(chunk, "wb") as stream:
            stream.write(b"wav")

    monkeypatch.setattr("coherex_minutes.processor.subprocess.run", fake_ffmpeg)
    processor = MeetingProcessor(settings, store)
    chunks = processor._prepare_chunks(job, source, job_dir)

    assert len(chunks) == 1
    assert not (job_dir / "asr-checkpoint.jsonl").exists()
    assert not (job_dir / "transcript.txt").exists()
    assert not (job_dir / "sections.json").exists()
    assert not (job_dir / "llm-slices").exists()


def test_sections_are_ordered_and_missing_sections_are_empty():
    sections = MeetingProcessor._validate_sections(
        [
            {"key": "main_items", "title": "المناقشات", "content": "نص"},
            {"key": "meeting_info", "title": "البيانات", "content": "معلومات"},
            {"key": "illegal", "content": "drop"},
        ]
    )
    assert [section["key"] for section in sections] == [
        "meeting_info",
        "attendance",
        "introduction",
        "agenda",
        "main_items",
    ]
    assert sections[1]["content"] == ""


def test_decision_validation_drops_invalid_fields_and_deduplicates():
    decisions = MeetingProcessor._validate_decisions(
        [
            {
                "title": "اعتماد الخطة",
                "kind": "RESOLUTION",
                "type": "WRONG",
                "completionDuration": 7,
                "completionDurationUnit": "DAYS",
            },
            {
                "title": "إعداد التقرير",
                "kind": "ASSIGNMENT",
                "responsiblePersonName": "",
                "completionDuration": 14,
                "completionDurationUnit": "DAYS",
            },
            {"title": "نقاش فقط", "kind": "DISCUSSION"},
        ]
    )
    assert decisions[0] == {
        "title": "اعتماد الخطة",
        "kind": "RESOLUTION",
        "type": "FOR_EXECUTION",
        "responsiblePersonName": None,
    }
    assert decisions[1]["completionDuration"] == 14
    assert decisions[1]["responsiblePersonName"] is None
    assert MeetingProcessor._deduplicate_decisions(decisions + [decisions[0]]) == decisions


class FakeProcessor:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    def process(self, _job):
        if self.error:
            raise self.error
        return self.result


def test_worker_completes_and_persists_result(tmp_path):
    settings = make_settings(tmp_path)
    store = JobStore(settings.database_path)
    store.create("work", "https://example.com/a.mp4")
    store.set_download_status("work", "READY")
    result = {
        "meetingId": "work",
        "language": "ar",
        "content": {"sections": []},
        "decisions": [],
        "generatedAt": "2026-09-07T10:00:00Z",
    }
    worker = Worker(settings, store, FakeProcessor(result=result))

    assert worker.run_once()
    completed = store.get("work")
    assert completed and completed.status == "COMPLETED"
    assert completed.result == result


def test_transient_worker_error_leaves_job_processing_for_resume(tmp_path):
    settings = make_settings(tmp_path)
    store = JobStore(settings.database_path)
    store.create("resume", "https://example.com/a.mp4")
    store.set_download_status("resume", "READY")
    worker = Worker(settings, store, FakeProcessor(error=RuntimeError("temporary")))

    assert worker.run_once()
    job = store.get("resume")
    assert job and job.status == "PROCESSING"
    assert store.claim_next().meeting_id == "resume"


def test_worker_fails_job_after_bounded_consecutive_retries(tmp_path):
    settings = replace(make_settings(tmp_path), max_processing_retries=1)
    store = JobStore(settings.database_path)
    store.create("poisoned", "https://example.com/a.mp4")
    store.set_download_status("poisoned", "READY")
    worker = Worker(settings, store, FakeProcessor(error=RuntimeError("permanent")))

    assert worker.run_once()
    job = store.get("poisoned")
    assert job and job.status == "FAILED"
    assert job.error_code == "TRANSCRIPTION_FAILED"
    assert store.claim_next() is None
