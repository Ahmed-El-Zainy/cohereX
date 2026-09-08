"""In-process functional path: POST → ingest → worker → GET minutes."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from coherex_minutes.api import create_app
from coherex_minutes.config import Settings
from coherex_minutes.ingest import IngestManager
from coherex_minutes.processor import MeetingProcessor, SECTION_KEYS
from coherex_minutes.store import JobStore
from coherex_minutes.worker import Worker

AUTH = {"Authorization": "Bearer minutes-secret"}
VIDEO_URL = "https://storage.example.com/meetings/board.mp4"


def _settings(tmp_path: Path) -> Settings:
    return replace(
        Settings.from_env(),
        data_dir=tmp_path,
        api_key="minutes-secret",
        manage_services=False,
        worker_poll_seconds=0,
        vllm_url="http://127.0.0.1:8000",
        vllm_api_key="asr-key",
        llm_url="http://127.0.0.1:8001",
        llm_api_key="llm-key",
        video_allowed_hosts=("storage.example.com",),
        max_video_bytes=1024 * 1024,
    )


def _llm_content(prompt: str) -> str:
    if "meeting_info" in prompt:
        sections = [
            {
                "key": key,
                "title": key.replace("_", " "),
                "content": "محتوى مدعوم من النص" if key == "main_items" else "",
            }
            for key in SECTION_KEYS
        ]
        return json.dumps(sections, ensure_ascii=False)
    if "RESOLUTION" in prompt or "ASSIGNMENT" in prompt:
        return json.dumps(
            [
                {
                    "title": "اعتماد الميزانية",
                    "kind": "RESOLUTION",
                    "type": "FOR_EXECUTION",
                    "agendaItemOrder": 1,
                }
            ],
            ensure_ascii=False,
        )
    return "حضور المجلس ونوقشت الميزانية مع ذكر BIC كما في النص."


def _handler(request: httpx.Request) -> httpx.Response:
    if request.method == "GET":
        return httpx.Response(200, content=b"fake-board-video")
    path = request.url.path
    if path.endswith("/v1/audio/transcriptions"):
        return httpx.Response(200, json={"text": "اعتمد المجلس الميزانية BIC"})
    if path.endswith("/v1/chat/completions"):
        prompt = json.loads(request.content)["messages"][0]["content"]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": _llm_content(prompt)}}]},
        )
    return httpx.Response(404, json={"detail": "unexpected request"})


def _patch_runtime(monkeypatch) -> None:
    transport = httpx.MockTransport(_handler)

    class FakeClient(httpx.Client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            kwargs.pop("timeout", None)
            super().__init__(*args, timeout=None, **kwargs)

    def fake_post(*args, **kwargs):
        with FakeClient() as client:
            return client.post(*args, **kwargs)

    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [(0, 0, 0, 0, ("8.8.8.8", port or 443))]

    def fake_run(command, check=True, capture_output=False, text=False, **kwargs):
        if command[0] == "ffprobe":
            return subprocess.CompletedProcess(command, 0, stdout="60.0\n", stderr="")
        if command[0] == "ffmpeg":
            pattern = Path(command[-1])
            pattern.parent.mkdir(parents=True, exist_ok=True)
            (pattern.parent / "000000.wav").write_bytes(b"RIFF")
            (pattern.parent / "000001.wav").write_bytes(b"RIFF")
            return subprocess.CompletedProcess(command, 0)
        raise AssertionError(command)

    monkeypatch.setattr("coherex_minutes.ingest.httpx.Client", FakeClient)
    monkeypatch.setattr("coherex_minutes.processor.httpx.Client", FakeClient)
    monkeypatch.setattr("coherex_minutes.processor.httpx.post", fake_post)
    monkeypatch.setattr("coherex_minutes.ingest.socket.getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr("coherex_minutes.processor.subprocess.run", fake_run)


def _wait_ready(store: JobStore, meeting_id: str, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = store.get(meeting_id)
        if job and job.download_status == "READY":
            return job
        if job and job.status == "FAILED":
            raise AssertionError(f"ingest failed: {job.error_code} {job.error_message}")
        time.sleep(0.05)
    raise AssertionError("timed out waiting for video ingest")


def test_post_ingest_worker_get_minutes_end_to_end(tmp_path, monkeypatch):
    _patch_runtime(monkeypatch)
    settings = _settings(tmp_path)
    store = JobStore(settings.database_path)
    ingest = IngestManager(settings, store, max_workers=1)
    client = TestClient(create_app(settings, store, ingest))

    try:
        not_ready = client.get("/v1/meeting-minutes/func-001", headers=AUTH)
        assert not_ready.status_code == 404

        submitted = client.post(
            "/v1/meeting-minutes",
            headers=AUTH,
            json={"meetingId": "func-001", "videoUrl": VIDEO_URL, "language": "ar"},
        )
        assert submitted.status_code == 202
        assert submitted.json()["data"]["status"] == "QUEUED"

        hidden = client.get("/v1/meeting-minutes/func-001", headers=AUTH)
        assert hidden.json()["error"]["code"] == "MINUTES_NOT_READY"

        _wait_ready(store, "func-001")
        queued = client.get("/v1/meeting-minutes/func-001/status", headers=AUTH)
        assert queued.json()["data"]["status"] == "QUEUED"

        worker = Worker(settings, store, MeetingProcessor(settings, store))
        assert worker.run_once() is True
    finally:
        ingest.close()

    completed = store.get("func-001")
    assert completed and completed.status == "COMPLETED"
    assert completed.progress == 100
    assert completed.stage == "COMPLETED"

    status = client.get("/v1/meeting-minutes/func-001/status", headers=AUTH)
    assert status.json()["data"]["status"] == "COMPLETED"
    assert status.json()["data"]["progress"] == 100
    assert status.json()["data"]["stage"] == "COMPLETED"

    minutes = client.get("/v1/meeting-minutes/func-001", headers=AUTH)
    body = minutes.json()
    assert minutes.status_code == 200
    assert body["success"] is True
    data = body["data"]
    assert data["meetingId"] == "func-001"
    assert data["language"] == "ar"
    assert [section["key"] for section in data["content"]["sections"]] == SECTION_KEYS
    assert data["decisions"][0]["kind"] == "RESOLUTION"
    assert data["decisions"][0]["type"] == "FOR_EXECUTION"

    replay = client.post(
        "/v1/meeting-minutes",
        headers=AUTH,
        json={"meetingId": "func-001", "videoUrl": VIDEO_URL, "language": "ar"},
    )
    assert replay.status_code == 202
    assert replay.json()["data"]["status"] == "COMPLETED"
    assert client.get("/v1/meeting-minutes/func-001", headers=AUTH).json() == body

    job_dir = settings.jobs_dir / "func-001"
    assert not (job_dir / "source.video").exists()
    assert not (job_dir / "chunks").exists()
    assert (job_dir / "transcript.txt").is_file()


def test_disallowed_host_fails_job_without_processing(tmp_path, monkeypatch):
    _patch_runtime(monkeypatch)
    settings = replace(_settings(tmp_path), video_allowed_hosts=("cdn.allowed.test",))
    store = JobStore(settings.database_path)
    ingest = IngestManager(settings, store, max_workers=1)
    client = TestClient(create_app(settings, store, ingest))

    try:
        response = client.post(
            "/v1/meeting-minutes",
            headers=AUTH,
            json={"meetingId": "blocked-001", "videoUrl": VIDEO_URL, "language": "ar"},
        )
        assert response.status_code == 202
        deadline = time.monotonic() + 5
        job = None
        while time.monotonic() < deadline:
            job = store.get("blocked-001")
            if job and job.status == "FAILED":
                break
            time.sleep(0.05)
    finally:
        ingest.close()

    assert job and job.status == "FAILED"
    assert job.error_code == "VIDEO_HOST_NOT_ALLOWED"
    status = client.get("/v1/meeting-minutes/blocked-001/status", headers=AUTH)
    assert status.status_code == 200
    assert status.json()["data"]["status"] == "FAILED"
    minutes = client.get("/v1/meeting-minutes/blocked-001", headers=AUTH)
    assert minutes.json()["error"]["code"] == "GENERATION_FAILED"
    retry = client.post(
        "/v1/meeting-minutes",
        headers=AUTH,
        json={"meetingId": "blocked-001", "videoUrl": VIDEO_URL, "language": "ar"},
    )
    assert retry.json()["data"]["status"] == "FAILED"
