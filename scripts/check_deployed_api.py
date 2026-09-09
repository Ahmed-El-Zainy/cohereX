#!/usr/bin/env python3
"""
Fast conformance check of the three deployed endpoints.

Answers "is the live service healthy and still behaving to contract?" in
seconds, where scripts/smoke_test_api.py answers "does a real meeting come out
right?" in ~16 minutes. Run this after every deploy, restart, or config change.

    scripts/check_deployed_api.py

With no arguments it reads COHEREX_SERVER_SSH from .env, opens an SSH tunnel to
the API on the box (it listens on localhost only), reads the bearer key from
/etc/coherex-minutes.env, runs the checks, and closes the tunnel.

Already on the box, or have a public URL:

    scripts/check_deployed_api.py --base-url http://127.0.0.1:8080 --api-key ...

Almost every check is side-effect free. Rejections (401/422/400) never create a
job, and the success paths are exercised against a meeting that already
completed -- re-POSTing an existing meetingId returns the existing job without
creating anything, which is exactly the idempotency rule worth testing. Only
`--submit` creates a new job.

Needs httpx and nothing else. Exit status is 0 only if every check passed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
SECTION_KEYS = ["meeting_info", "attendance", "introduction", "agenda", "main_items"]
STATUSES = {"QUEUED", "PROCESSING", "COMPLETED", "FAILED"}
STAGES = {"QUEUED", "TRANSCRIBING", "GENERATING_MINUTES", "COMPLETED"}
DURATION_UNITS = {"DAYS", "WEEKS", "MONTHS"}
ISO_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")
# Qwen has leaked Mandarin into Arabic before; the grammar should stop it now.
CJK = re.compile(r"[　-〿぀-ヿ㐀-䶿一-鿿가-힯＀-￯]")
FORBIDDEN_DECISION_KEYS = {
    "id", "_id", "dbId", "databaseId", "userId", "user_id", "responsiblePersonId",
    "status", "votes", "votingResult", "votingResults", "vote",
}
REMOTE_ENV = "/etc/coherex-minutes.env"
REMOTE_DB = "/var/lib/coherex-minutes/jobs.sqlite3"


def load_dotenv(path: Path = REPO_ROOT / ".env") -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.passed = 0
        self.tty = sys.stdout.isatty()

    def group(self, title: str) -> None:
        print(f"\n{title}")

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        mark = ("\033[32mPASS\033[0m" if ok else "\033[31mFAIL\033[0m") if self.tty \
            else ("PASS" if ok else "FAIL")
        if ok:
            self.passed += 1
        else:
            self.failures.append(label)
        print(f"  {mark} {label}" + (f" -- {detail}" if detail and not ok else ""))
        return ok

    def skip(self, label: str, why: str) -> None:
        mark = "\033[33mSKIP\033[0m" if self.tty else "SKIP"
        print(f"  {mark} {label} -- {why}")


def ssh(target: str, command: str, stdin: str | None = None) -> str:
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", target, command],
        capture_output=True, text=True, input=stdin,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ssh failed: {result.stderr.strip()[:200]}")
    return result.stdout.strip()


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def open_tunnel(target: str, remote_port: int) -> tuple[str, subprocess.Popen]:
    """The API binds to localhost on the box, so reaching it needs a tunnel."""
    local = free_port()
    proc = subprocess.Popen(
        ["ssh", "-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
         "-L", f"{local}:127.0.0.1:{remote_port}", target],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    base = f"http://127.0.0.1:{local}"
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"tunnel died: {proc.stderr.read().decode()[:200]}")
        try:
            httpx.get(f"{base}/health", timeout=2)
            return base, proc
        except httpx.HTTPError:
            time.sleep(0.4)
    proc.terminate()
    raise RuntimeError(f"tunnel to {target}:{remote_port} never became usable")


def newest_completed(target: str) -> str | None:
    """Reuse a finished meeting so the success paths cost nothing."""
    # Fed over stdin rather than `python -c "..."`: the program contains both
    # quote styles, and nesting them inside an ssh command line silently mangles
    # the SQL.
    program = (
        "import sqlite3\n"
        f"c = sqlite3.connect('file:{REMOTE_DB}?mode=ro', uri=True)\n"
        "r = c.execute(\"SELECT meeting_id FROM jobs WHERE status = 'COMPLETED' \"\n"
        "              \"ORDER BY created_at DESC LIMIT 1\").fetchone()\n"
        "print(r[0] if r else '')\n"
    )
    try:
        return ssh(target, "/opt/coherex-venv/bin/python -", stdin=program) or None
    except RuntimeError:
        return None


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ssh", default=os.environ.get("COHEREX_SERVER_SSH"),
                        help="ssh target hosting the API (default: COHEREX_SERVER_SSH)")
    parser.add_argument("--remote-port", type=int, default=8080)
    parser.add_argument("--base-url", help="skip the tunnel and use this URL directly")
    # Deliberately NOT defaulted from the environment: when we tunnel to a
    # server, that server's own key is authoritative. A stale AI_SERVICE_API_KEY
    # left over from a local dev stack would otherwise shadow it and every
    # authenticated check would fail with 401 against a perfectly healthy box.
    parser.add_argument("--api-key", help="override; by default the key is read "
                                          "from the server being tested")
    parser.add_argument("--meeting-id", help="completed meeting to validate (default: newest)")
    parser.add_argument("--submit", action="store_true",
                        help="also POST a real job. Creates a row that will FAIL on "
                             "download; only useful when no completed meeting exists yet")
    args = parser.parse_args()

    report = Report()
    tunnel = None
    target = args.ssh

    try:
        # ---------- connect ----------
        if args.base_url:
            base = args.base_url.rstrip("/")
            key = args.api_key or os.environ.get("AI_SERVICE_API_KEY")
            if not key:
                print("--api-key is required with --base-url", file=sys.stderr)
                return 2
        else:
            if not target:
                print("Set COHEREX_SERVER_SSH in .env, or pass --base-url and --api-key.",
                      file=sys.stderr)
                return 2
            if shutil.which("ssh") is None:
                print("ssh not found", file=sys.stderr)
                return 2
            print(f"Tunnelling to {target.split('@')[-1]}:{args.remote_port} ...")
            base, tunnel = open_tunnel(target, args.remote_port)
            key = args.api_key or ssh(
                target, f"grep ^AI_SERVICE_API_KEY= {REMOTE_ENV} | cut -d= -f2")
            print(f"Using the key from {REMOTE_ENV} on the server "
                  f"({len(key)} chars).")
            if not key:
                print(f"could not read AI_SERVICE_API_KEY from {REMOTE_ENV}", file=sys.stderr)
                return 2

        auth = {"Authorization": f"Bearer {key}"}
        client = httpx.Client(base_url=base, timeout=30.0)

        def get(path, headers=auth):
            return client.get(path, headers=headers)

        def post(payload, headers=auth):
            return client.post("/v1/meeting-minutes", headers=headers, json=payload)

        # ---------- service ----------
        report.group("Service")
        try:
            health = get("/health", headers={})
            report.check(health.status_code == 200, "GET /health is 200",
                         str(health.status_code))
        except httpx.HTTPError as exc:
            report.check(False, "GET /health is 200", str(exc))
            return 1
        if target:
            try:
                units = ssh(target, "systemctl is-active coherex-minutes-api "
                                    "coherex-minutes-worker | paste -sd' ' -")
                report.check(units.split() == ["active", "active"],
                             "API and worker units are both active", units)
            except RuntimeError as exc:
                report.skip("systemd unit states", str(exc)[:60])

        # ---------- 1. POST ----------
        report.group("1. POST /v1/meeting-minutes")
        unknown = f"nonexistent-{uuid.uuid4().hex[:8]}"
        good_url = "https://storage.example.com/meetings/board.mp4"

        anon = post({"meetingId": unknown, "videoUrl": good_url}, headers={})
        report.check(anon.status_code == 401, "no bearer is rejected 401", str(anon.status_code))
        report.check(anon.json().get("error", {}).get("code") == "UNAUTHORIZED",
                     "401 body uses the error envelope")

        wrong = post({"meetingId": unknown, "videoUrl": good_url},
                     headers={"Authorization": "Bearer not-the-key"})
        report.check(wrong.status_code == 401, "wrong bearer is rejected 401",
                     str(wrong.status_code))

        bad_id = post({"meetingId": "has spaces!", "videoUrl": good_url})
        report.check(bad_id.status_code == 422
                     and bad_id.json().get("error", {}).get("code") == "INVALID_REQUEST",
                     "illegal meetingId is 422 INVALID_REQUEST", str(bad_id.status_code))

        bad_url = post({"meetingId": unknown, "videoUrl": "not-a-url"})
        report.check(bad_url.status_code == 400
                     and bad_url.json().get("error", {}).get("code") == "INVALID_VIDEO_URL",
                     "malformed videoUrl is 400 INVALID_VIDEO_URL", str(bad_url.status_code))

        english = post({"meetingId": unknown, "videoUrl": good_url, "language": "en"})
        report.check(english.status_code == 400
                     and english.json().get("error", {}).get("code") == "UNSUPPORTED_LANGUAGE",
                     "language 'en' is refused 400", str(english.status_code))

        # ---------- pick a finished meeting ----------
        meeting = args.meeting_id or (newest_completed(target) if target else None)
        if not meeting and args.submit:
            meeting = f"check-{time.strftime('%H%M%S')}-{uuid.uuid4().hex[:4]}"
            created = post({"meetingId": meeting, "videoUrl": good_url, "language": "ar"})
            report.check(created.status_code == 202, "POST creates a job (202)",
                         str(created.status_code))
            print(f"       created {meeting} -- it will FAIL on download; that is expected")

        if meeting:
            # Re-POSTing an existing id must return that job, not make another --
            # so these exercise the success path without creating anything.
            replay = post({"meetingId": meeting, "videoUrl": good_url, "language": "ar"})
            ok = report.check(replay.status_code == 202, "re-POST of a known id is 202",
                              str(replay.status_code))
            if ok:
                data = replay.json()["data"]
                report.check(data.get("meetingId") == meeting, "echoes the meetingId back")
                report.check(data.get("status") in STATUSES, "status is a valid enum value",
                             str(data.get("status")))
                report.check(bool(ISO_UTC.match(str(data.get("createdAt")))),
                             "createdAt is ISO 8601 UTC", str(data.get("createdAt")))
                first_created = data.get("createdAt")

                auto = post({"meetingId": meeting, "videoUrl": good_url, "language": "auto"})
                report.check(auto.status_code == 202,
                             "language 'auto' is accepted (the spec's default)",
                             str(auto.status_code))
                omitted = post({"meetingId": meeting, "videoUrl": good_url})
                report.check(omitted.status_code == 202, "language may be omitted",
                             str(omitted.status_code))
                report.check(omitted.json()["data"].get("createdAt") == first_created,
                             "re-POST never starts a second job (createdAt unchanged)")
        else:
            report.skip("POST success paths", "no completed meeting found; pass --submit")

        # ---------- 2. GET status ----------
        report.group("2. GET /v1/meeting-minutes/{id}/status")
        missing = get(f"/v1/meeting-minutes/{unknown}/status")
        report.check(missing.status_code == 404
                     and missing.json().get("error", {}).get("code") == "NOT_FOUND",
                     "unknown meetingId is 404 NOT_FOUND", str(missing.status_code))

        if meeting:
            resp = get(f"/v1/meeting-minutes/{meeting}/status")
            if report.check(resp.status_code == 200, "known meetingId is 200",
                            str(resp.status_code)):
                data = resp.json()["data"]
                report.check(data.get("status") in STATUSES, "status is one of the four values",
                             str(data.get("status")))
                report.check(data.get("stage") in STAGES, "stage is a closed enum",
                             str(data.get("stage")))
                progress = data.get("progress")
                report.check(isinstance(progress, int) and not isinstance(progress, bool)
                             and 0 <= progress <= 100,
                             "progress is an integer 0-100", repr(progress))
                report.check(bool(ISO_UTC.match(str(data.get("updatedAt")))),
                             "updatedAt is ISO 8601 UTC", str(data.get("updatedAt")))
                if data.get("status") == "FAILED":
                    report.check(isinstance(data.get("error"), dict)
                                 and "code" in data["error"],
                                 "FAILED carries an error object")
        else:
            report.skip("status of a real job", "no meeting available")

        # ---------- 3. GET minutes ----------
        report.group("3. GET /v1/meeting-minutes/{id}")
        gone = get(f"/v1/meeting-minutes/{unknown}")
        report.check(gone.status_code == 404
                     and gone.json().get("error", {}).get("code") == "NOT_FOUND",
                     "unknown meetingId is 404 NOT_FOUND", str(gone.status_code))

        if not meeting:
            report.skip("minutes payload", "no completed meeting available")
        else:
            resp = get(f"/v1/meeting-minutes/{meeting}")
            body = resp.json()
            state = get(f"/v1/meeting-minutes/{meeting}/status").json()["data"]["status"]
            if state != "COMPLETED":
                report.check(body.get("success") is False,
                             f"{state} job returns success:false, not a payload")
                report.check(body.get("error", {}).get("code")
                             in {"MINUTES_NOT_READY", "GENERATION_FAILED"},
                             "not-ready/failed code is correct",
                             str(body.get("error", {}).get("code")))
                report.skip("payload conformance", f"newest meeting is {state}, not COMPLETED")
            elif report.check(resp.status_code == 200 and body.get("success") is True,
                              "COMPLETED meeting returns 200 success:true",
                              str(resp.status_code)):
                data = body["data"]
                report.check(data.get("meetingId") == meeting, "meetingId matches")
                report.check(data.get("language") == "ar", "language is ar",
                             str(data.get("language")))
                report.check(bool(ISO_UTC.match(str(data.get("generatedAt")))),
                             "generatedAt is ISO 8601 UTC", str(data.get("generatedAt")))
                report.check(isinstance(data.get("content"), dict),
                             "content is a JSON object, not a string",
                             type(data.get("content")).__name__)

                sections = data.get("content", {}).get("sections") or []
                report.check([s.get("key") for s in sections] == SECTION_KEYS,
                             "all five section keys, in contract order",
                             str([s.get("key") for s in sections]))
                extra = sorted({k for s in sections for k in s} - {"key", "title", "content"})
                report.check(not extra, "sections carry only key/title/content", str(extra))
                report.check(not any("decisions" in s for s in sections),
                             "decisions are not embedded in sections")

                decisions = data.get("decisions")
                report.check(isinstance(decisions, list), "decisions is an array")
                bad, leaked = [], set()
                for d in decisions or []:
                    if d.get("kind") not in {"RESOLUTION", "ASSIGNMENT"}:
                        bad.append(f"kind={d.get('kind')!r}")
                    if d.get("type") != "FOR_EXECUTION":
                        bad.append(f"type={d.get('type')!r}")
                    if not str(d.get("title") or "").strip():
                        bad.append("empty title")
                    if d.get("kind") == "RESOLUTION" and "completionDuration" in d:
                        bad.append("RESOLUTION has completionDuration")
                    order = d.get("agendaItemOrder")
                    if order is not None and not (isinstance(order, int)
                                                  and not isinstance(order, bool) and order >= 1):
                        bad.append(f"agendaItemOrder={order!r}")
                    unit = d.get("completionDurationUnit")
                    if unit is not None and unit not in DURATION_UNITS:
                        bad.append(f"unit={unit!r}")
                    leaked |= FORBIDDEN_DECISION_KEYS & set(d)
                report.check(not bad, "every decision matches the documented shape",
                             "; ".join(bad[:3]))
                report.check(not leaked, "no voting results, status or internal IDs",
                             str(sorted(leaked)))

                # The Qwen CJK leak, checked against live output rather than assumed fixed.
                text = json.dumps(data, ensure_ascii=False)
                found = sorted(set(CJK.findall(text)))
                report.check(not found,
                             "generated text is free of CJK characters", "".join(found))

                again = get(f"/v1/meeting-minutes/{meeting}")
                report.check(again.json() == body, "repeat read returns an identical payload")

                filled = sum(1 for s in sections if (s.get("content") or "").strip())
                print(f"\n  Validated against : {meeting}")
                print(f"  Sections filled   : {filled}/{len(sections)}")
                print(f"  Decisions         : {len(decisions or [])}")

    except (RuntimeError, httpx.HTTPError) as exc:
        print(f"\nAborted: {exc}", file=sys.stderr)
        return 1
    finally:
        if tunnel is not None:
            tunnel.terminate()

    print("\n" + "=" * 58)
    if report.failures:
        print(f"FAILED -- {report.passed} passed, {len(report.failures)} failed:")
        for failure in report.failures:
            print(f"  - {failure}")
        return 1
    print(f"OK -- all {report.passed} checks passed against the deployed service")
    return 0


if __name__ == "__main__":
    sys.exit(main())
