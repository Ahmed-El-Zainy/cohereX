from __future__ import annotations

from dataclasses import replace

from fastapi.testclient import TestClient

from coherex_minutes.api import create_app
from coherex_minutes.config import Settings
from coherex_minutes.store import Job, JobStore


class FakeIngest:
    def __init__(self) -> None:
        self.scheduled: list[str] = []

    def recover(self) -> None:
        pass

    def schedule(self, job: Job) -> None:
        self.scheduled.append(job.meeting_id)

    def close(self) -> None:
        pass


def make_settings(tmp_path):
    return replace(
        Settings.from_env(),
        data_dir=tmp_path,
        api_key="minutes-secret",
        manage_services=False,
    )


def make_client(tmp_path):
    settings = make_settings(tmp_path)
    store = JobStore(settings.database_path)
    ingest = FakeIngest()
    client = TestClient(create_app(settings, store, ingest))
    return client, store, ingest


def auth():
    return {"Authorization": "Bearer minutes-secret"}


def test_requires_bearer_authentication(tmp_path):
    client, _, _ = make_client(tmp_path)
    response = client.get("/v1/meeting-minutes/unknown/status")
    wrong_length = client.get(
        "/v1/meeting-minutes/unknown/status",
        headers={"Authorization": "Bearer x"},
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHORIZED"
    assert wrong_length.status_code == 401
    assert wrong_length.json()["error"]["code"] == "UNAUTHORIZED"


def test_submit_is_idempotent_and_schedules_only_once(tmp_path):
    client, _, ingest = make_client(tmp_path)
    payload = {
        "meetingId": "meeting-123",
        "videoUrl": "https://storage.example.com/meeting.mp4?signature=secret",
        "language": "ar",
    }

    first = client.post("/v1/meeting-minutes", json=payload, headers=auth())
    second = client.post("/v1/meeting-minutes", json=payload, headers=auth())

    assert first.status_code == 202
    assert first.json()["data"]["status"] == "QUEUED"
    assert second.status_code == 202
    assert second.json()["data"]["createdAt"] == first.json()["data"]["createdAt"]
    assert ingest.scheduled == ["meeting-123"]


def test_submit_rejects_path_traversal_and_unsupported_language(tmp_path):
    client, _, _ = make_client(tmp_path)
    invalid_id = client.post(
        "/v1/meeting-minutes",
        json={"meetingId": "../escape", "videoUrl": "https://example.com/a.mp4"},
        headers=auth(),
    )
    unsupported = client.post(
        "/v1/meeting-minutes",
        json={
            "meetingId": "english",
            "videoUrl": "https://example.com/a.mp4",
            "language": "auto",
        },
        headers=auth(),
    )
    invalid_url = client.post(
        "/v1/meeting-minutes",
        json={"meetingId": "invalid-url", "videoUrl": "not-a-url"},
        headers=auth(),
    )

    assert invalid_id.status_code == 422
    assert invalid_id.json()["error"]["code"] == "INVALID_REQUEST"
    assert invalid_url.status_code == 400
    assert invalid_url.json()["error"]["code"] == "INVALID_VIDEO_URL"
    assert unsupported.status_code == 400
    assert unsupported.json()["error"]["code"] == "UNSUPPORTED_LANGUAGE"


def test_status_matches_processing_and_failed_contract(tmp_path):
    client, store, _ = make_client(tmp_path)
    store.create("meeting-status", "https://example.com/a.mp4")
    store.set_download_status("meeting-status", "READY")
    store.claim_next()
    store.update_progress("meeting-status", "TRANSCRIBING", 37)

    processing = client.get(
        "/v1/meeting-minutes/meeting-status/status", headers=auth()
    )
    data = processing.json()["data"]
    assert data["meetingId"] == "meeting-status"
    assert data["status"] == "PROCESSING"
    assert data["progress"] == 37
    assert data["stage"] == "TRANSCRIBING"

    store.fail(
        "meeting-status",
        "TRANSCRIPTION_FAILED",
        "ASR unavailable",
        stage="TRANSCRIBING",
    )
    failed = client.get(
        "/v1/meeting-minutes/meeting-status/status", headers=auth()
    )
    assert failed.status_code == 200
    assert failed.json()["data"]["status"] == "FAILED"
    assert failed.json()["data"]["error"]["code"] == "TRANSCRIPTION_FAILED"


def test_minutes_are_hidden_until_completed_then_repeatable(tmp_path):
    client, store, _ = make_client(tmp_path)
    store.create("meeting-result", "https://example.com/a.mp4")

    not_ready = client.get("/v1/meeting-minutes/meeting-result", headers=auth())
    assert not_ready.status_code == 200
    assert not_ready.json()["error"]["code"] == "MINUTES_NOT_READY"

    result = {
        "meetingId": "meeting-result",
        "language": "ar",
        "content": {"sections": []},
        "decisions": [],
        "generatedAt": "2026-09-07T10:36:10Z",
    }
    store.complete("meeting-result", result)
    first = client.get("/v1/meeting-minutes/meeting-result", headers=auth())
    second = client.get("/v1/meeting-minutes/meeting-result", headers=auth())

    assert first.status_code == 200
    assert first.json() == second.json() == {"success": True, "data": result}


def test_unknown_meeting_returns_not_found(tmp_path):
    client, _, _ = make_client(tmp_path)
    response = client.get("/v1/meeting-minutes/missing", headers=auth())
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"
