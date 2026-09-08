"""FastAPI implementation of required_intergration.md."""

from __future__ import annotations

import hashlib
import hmac
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import Settings
from .ingest import IngestManager
from .store import Job, JobStore


class SubmitMeeting(BaseModel):
    meetingId: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    videoUrl: HttpUrl
    language: str = "ar"


def success(data: dict[str, Any]) -> dict[str, Any]:
    return {"success": True, "data": data}


def error(code: str, message: str) -> dict[str, Any]:
    return {"success": False, "error": {"code": code, "message": message}}


def _submitted(job: Job) -> dict[str, Any]:
    return success(
        {
            "meetingId": job.meeting_id,
            "status": job.status,
            "createdAt": job.created_at,
        }
    )


def _store(request: Request) -> JobStore:
    """The store exists only after startup; see ``lifespan`` below."""
    store = request.app.state.store
    if store is None:
        raise HTTPException(status_code=503, detail="Service is still starting")
    return store


def create_app(
    settings: Settings | None = None,
    store: JobStore | None = None,
    ingest: IngestManager | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Creating the data directory and the SQLite file is *startup* work,
        # never *import* work. `uvicorn coherex_minutes.api:app` runs this;
        # a bare `import coherex_minutes.api` must not touch the filesystem,
        # otherwise the module cannot be imported by anyone without write
        # access to COHEREX_MINUTES_DATA_DIR (CI, tests, local development).
        settings.prepare()
        if app.state.store is None:
            app.state.store = JobStore(settings.database_path)
        if app.state.ingest is None:
            app.state.ingest = IngestManager(settings, app.state.store)
        app.state.ingest.recover()
        yield
        app.state.ingest.close()

    app = FastAPI(title="CohereX Meeting Minutes API", version="1.0.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.store = store
    app.state.ingest = ingest

    def authenticate(
        authorization: Annotated[str | None, Header()] = None,
    ) -> None:
        expected = settings.api_key
        if not expected:
            raise HTTPException(status_code=503, detail="AI_SERVICE_API_KEY is not configured")
        scheme, _, token = (authorization or "").partition(" ")
        token_digest = hashlib.sha256(token.encode("utf-8")).digest()
        expected_digest = hashlib.sha256(expected.encode("utf-8")).digest()
        if scheme.lower() != "bearer" or not hmac.compare_digest(token_digest, expected_digest):
            raise HTTPException(status_code=401, detail="Unauthorized")

    # Registered against Starlette's class, not FastAPI's subclass: the router
    # raises the parent for an unrouted path (404) or a bad method (405), and a
    # handler bound to the subclass would let those through as Starlette's bare
    # {"detail": ...}. The contract promises one error envelope everywhere, so
    # a client may parse error.code on any failure, including a mistyped URL.
    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        if exc.status_code == 401:
            return JSONResponse(error("UNAUTHORIZED", "Missing or invalid Bearer token."), 401)
        if exc.status_code == 503:
            return JSONResponse(error("SERVICE_UNAVAILABLE", str(exc.detail)), 503)
        return JSONResponse(error("REQUEST_FAILED", str(exc.detail)), exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
        message = first.get("msg", "Invalid request.")
        detail = f"{location}: {message}" if location else str(message)
        if location == "videoUrl":
            return JSONResponse(error("INVALID_VIDEO_URL", detail), 400)
        return JSONResponse(error("INVALID_REQUEST", detail), 422)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return success({"status": "ok"})

    @app.post(
        "/v1/meeting-minutes",
        status_code=202,
        dependencies=[Depends(authenticate)],
    )
    def submit(request: Request, payload: SubmitMeeting) -> JSONResponse:
        if payload.language != "ar":
            return JSONResponse(
                error(
                    "UNSUPPORTED_LANGUAGE",
                    "Version 1 accepts Arabic-primary mixed Arabic/English meetings with language 'ar'.",
                ),
                status_code=400,
            )
        job, created = _store(request).create(payload.meetingId, str(payload.videoUrl), "ar")
        if created:
            request.app.state.ingest.schedule(job)
        return JSONResponse(_submitted(job), status_code=202)

    @app.get(
        "/v1/meeting-minutes/{meeting_id}/status",
        dependencies=[Depends(authenticate)],
    )
    def status(request: Request, meeting_id: str) -> JSONResponse:
        job = _store(request).get(meeting_id)
        if job is None:
            return JSONResponse(error("NOT_FOUND", "Meeting job was not found."), 404)
        data: dict[str, Any] = {
            "meetingId": job.meeting_id,
            "status": job.status,
            "progress": job.progress,
            "stage": job.stage,
            "updatedAt": job.updated_at,
        }
        if job.status == "FAILED":
            data["error"] = {
                "code": job.error_code or "GENERATION_FAILED",
                "message": job.error_message or "Meeting minutes generation failed.",
            }
        return JSONResponse(success(data))

    @app.get(
        "/v1/meeting-minutes/{meeting_id}",
        dependencies=[Depends(authenticate)],
    )
    def minutes(request: Request, meeting_id: str) -> JSONResponse:
        job = _store(request).get(meeting_id)
        if job is None:
            return JSONResponse(error("NOT_FOUND", "Meeting job was not found."), 404)
        if job.status in {"QUEUED", "PROCESSING"}:
            return JSONResponse(
                error("MINUTES_NOT_READY", "Meeting minutes are not ready yet.")
            )
        if job.status == "FAILED":
            return JSONResponse(
                error("GENERATION_FAILED", "Meeting minutes generation failed.")
            )
        if job.result is None:
            return JSONResponse(
                error("GENERATION_FAILED", "Completed job has no stored result."),
                status_code=500,
            )
        return JSONResponse(success(job.result))

    return app


app = create_app()
