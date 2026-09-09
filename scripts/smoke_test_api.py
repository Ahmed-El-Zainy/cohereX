#!/usr/bin/env python3
"""
End-to-end smoke test for a deployed meeting-minutes API.

Drives the three endpoints from required_intergration.md against a real
server with real audio, then checks the response actually matches the
contract in docs/PLATFORM_TEAM_GUIDE.md -- section keys and order, decision
key presence rules, idempotent replay, and stable repeat reads.

    scripts/smoke_test_api.py --video-url https://storage.example.com/board.mp4

`--serve` publishes a local file over HTTP for the service to download, for
when you are testing on the box before the platform's signed URLs exist:

    scripts/smoke_test_api.py --serve samples/saudi_business_03min.mp3

That requires BOTH of these on the server, and both must be reverted after:

    COHEREX_VIDEO_ALLOWED_HOSTS=127.0.0.1
    COHEREX_MINUTES_ALLOW_PRIVATE_VIDEO_HOSTS=true

`.mp3` is fine -- the worker runs ffmpeg with `-vn`, so an audio-only file
needs no conversion.

Base URL and key come from .env (AI_SERVICE_BASE_URL / AI_SERVICE_API_KEY,
the same names the platform uses) or --base-url / --api-key.

Exit status is 0 only if every check passed.
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import re
import socketserver
import sys
import threading
import time
import uuid
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
SECTION_KEYS = ["meeting_info", "attendance", "introduction", "agenda", "main_items"]
TERMINAL = {"COMPLETED", "FAILED"}
STATUSES = {"QUEUED", "PROCESSING", "COMPLETED", "FAILED"}
STAGES = {"QUEUED", "TRANSCRIBING", "GENERATING_MINUTES", "COMPLETED"}
DURATION_UNITS = {"DAYS", "WEEKS", "MONTHS"}
# "All timestamps use ISO 8601 UTC, for example 2026-09-07T10:30:00Z"
ISO_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")
# "The AI must not generate voting results, status, internal user IDs, or
# database IDs." Anything resembling these leaking into a decision is a defect.
FORBIDDEN_DECISION_KEYS = {
    "id", "_id", "dbId", "databaseId", "userId", "user_id", "responsiblePersonId",
    "status", "votes", "votingResult", "votingResults", "vote",
}


def load_dotenv(path: Path = REPO_ROOT / ".env") -> None:
    """Same minimal loader main.py and llm_client.py use; env always wins."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class Checks:
    """Records pass/fail so one bad assertion does not hide the rest."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.passed = 0

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        if ok:
            self.passed += 1
            print(f"  \033[32mPASS\033[0m {label}")
        else:
            self.failures.append(label)
            print(f"  \033[31mFAIL\033[0m {label}" + (f" -- {detail}" if detail else ""))
        return ok


def serve_file(path: Path, host: str, port: int) -> tuple[str, socketserver.TCPServer]:
    """Serve exactly one file, so a stray path cannot expose the repo."""
    payload = path.read_bytes()
    name = path.name

    class OneFile(http.server.BaseHTTPRequestHandler):
        def do_HEAD(self):  # noqa: N802
            self._respond(head_only=True)

        def do_GET(self):  # noqa: N802
            self._respond(head_only=False)

        def _respond(self, head_only: bool):
            if self.path.lstrip("/") != name:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if not head_only:
                self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = socketserver.TCPServer((host, port), OneFile)
    server.allow_reuse_address = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    actual = server.server_address[1]
    return f"http://{host}:{actual}/{name}", server


def render_markdown(data: dict) -> str:
    lines = [f"# {data['meetingId']}", "", f"_generated {data.get('generatedAt')}_", ""]
    for section in data.get("content", {}).get("sections", []):
        lines += [f"## {section.get('title') or section['key']}  `{section['key']}`", "",
                  section.get("content") or "_(empty)_", ""]
    lines += ["## decisions", ""]
    for decision in data.get("decisions", []) or ["_(none)_"]:
        if isinstance(decision, str):
            lines.append(decision)
        else:
            lines.append(f"- **{decision['title']}** ({decision['kind']})")
            for field in ("responsiblePersonName", "completionDuration",
                          "completionDurationUnit", "agendaItemOrder"):
                if field in decision and decision[field] is not None:
                    lines.append(f"  - {field}: {decision[field]}")
    return "\n".join(lines) + "\n"


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-url", default=os.environ.get("AI_SERVICE_BASE_URL"),
                        help="e.g. https://minutes.example.com (or AI_SERVICE_BASE_URL / .env)")
    parser.add_argument("--api-key", default=os.environ.get("AI_SERVICE_API_KEY"),
                        help="bearer token (or AI_SERVICE_API_KEY / .env)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video-url", help="URL the AI service downloads from")
    source.add_argument("--serve", type=Path,
                        help="publish this local file over HTTP and submit that URL")
    parser.add_argument("--serve-host", default="127.0.0.1",
                        help="interface for --serve (default: 127.0.0.1)")
    parser.add_argument("--serve-port", type=int, default=0, help="0 picks a free port")
    parser.add_argument("--meeting-id", help="default: smoke-<timestamp>-<random>")
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=6 * 3600,
                        help="give up after this many seconds (default: 6h)")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "smoke-out")
    parser.add_argument("--full", action="store_true",
                        help="print whole sections instead of the first 600 chars")
    parser.add_argument("--local", action="store_true",
                        help="start a throwaway local stack (stubbed models), test it, "
                             "then shut it down -- no server, no second terminal")
    args = parser.parse_args()

    stack = None
    if args.local:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        try:
            from dev_stack import free_port, start_stack
        except ImportError as exc:
            print(f"--local needs scripts/dev_stack.py: {exc}", file=sys.stderr)
            return 2
        try:
            stack = start_stack(port=free_port())
        except ImportError as exc:
            print(f'--local needs the API extra: pip install -e ".[minutes-api]"\n  ({exc})',
                  file=sys.stderr)
            return 2
        args.base_url, args.api_key = stack.base_url, stack.api_key
        print(f"Local stack up at {stack.base_url} (models STUBBED -- contract only).")

    if not args.base_url or not args.api_key:
        missing = [n for n, v in (("AI_SERVICE_BASE_URL", args.base_url),
                                  ("AI_SERVICE_API_KEY", args.api_key)) if not v]
        print(f"Not configured: {', '.join(missing)}\n", file=sys.stderr)
        print(
            "There is no third party to request a token from -- this service issues\n"
            "its own. Generate one, put it on the server, and share it with the\n"
            "platform team:\n"
            "\n"
            "  openssl rand -hex 32                 # the value of AI_SERVICE_API_KEY\n"
            "\n"
            "Server (/etc/coherex-minutes.env, mode 0600) and this repo's .env must\n"
            "carry the SAME value. AI_SERVICE_BASE_URL is your own deployment's DNS\n"
            "name, e.g. https://minutes.example.com.\n"
            "\n"
            "  AI_SERVICE_BASE_URL=https://minutes.example.com\n"
            "  AI_SERVICE_API_KEY=<the value you just generated>\n"
            "\n"
            "Not deployed yet? See deploy/README.md. To exercise this script before\n"
            "then, run the local stack -- it prints the two lines to paste:\n"
            "\n"
            "  scripts/dev_stack.py\n",
            file=sys.stderr,
        )
        return 2
    if args.serve and not args.serve.is_file():
        parser.error(f"file not found: {args.serve}")

    base = args.base_url.rstrip("/")
    auth = {"Authorization": f"Bearer {args.api_key}"}
    meeting_id = args.meeting_id or f"smoke-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    checks = Checks()
    server = None

    try:
        if args.serve:
            video_url, server = serve_file(args.serve, args.serve_host, args.serve_port)
            size_mb = args.serve.stat().st_size / 1_000_000
            print(f"Serving {args.serve.name} ({size_mb:.1f} MB) at {video_url}")
        else:
            video_url = args.video_url

        print(f"\nServer   : {base}")
        print(f"meetingId: {meeting_id}")
        print(f"videoUrl : {video_url}\n")

        with httpx.Client(timeout=60.0) as client:
            # --- 1. reachability and auth -------------------------------
            print("1. Preflight")
            try:
                health = client.get(f"{base}/health")
                checks.check(health.status_code == 200, "GET /health is 200",
                             f"got {health.status_code}")
            except httpx.HTTPError as exc:
                checks.check(False, "GET /health is 200", str(exc))
                raise SystemExit(1)

            unauth = client.get(f"{base}/v1/meeting-minutes/{meeting_id}/status")
            checks.check(unauth.status_code == 401, "no bearer is rejected 401",
                         f"got {unauth.status_code}")
            checks.check(unauth.json().get("error", {}).get("code") == "UNAUTHORIZED",
                         "401 uses the error envelope")

            unknown = client.get(f"{base}/v1/meeting-minutes/{meeting_id}/status", headers=auth)
            checks.check(unknown.status_code == 404, "unknown meetingId is 404",
                         f"got {unknown.status_code}")

            # --- 2. submit ----------------------------------------------
            print("\n2. Submit")
            started = time.monotonic()
            body = {"meetingId": meeting_id, "videoUrl": video_url, "language": "ar"}
            submit = client.post(f"{base}/v1/meeting-minutes", headers=auth, json=body)
            ok = checks.check(submit.status_code == 202, "POST returns 202",
                              f"got {submit.status_code}: {submit.text[:300]}")
            if not ok:
                raise SystemExit(1)
            accepted = submit.json()["data"]
            checks.check(accepted["meetingId"] == meeting_id, "echoes our meetingId")
            checks.check(accepted["status"] == "QUEUED", "initial status is QUEUED",
                         str(accepted.get("status")))

            replay = client.post(f"{base}/v1/meeting-minutes", headers=auth, json=body)
            checks.check(replay.json()["data"]["createdAt"] == accepted["createdAt"],
                         "re-POST reuses the job (same createdAt)")

            early = client.get(f"{base}/v1/meeting-minutes/{meeting_id}", headers=auth)
            checks.check(early.json().get("error", {}).get("code") == "MINUTES_NOT_READY",
                         "GET before COMPLETED is MINUTES_NOT_READY")

            # --- 3. poll -------------------------------------------------
            print("\n3. Processing (Ctrl-C to stop polling; the job keeps running)")
            deadline = time.monotonic() + args.timeout
            interactive = sys.stdout.isatty()
            stage_started = {"QUEUED": started}
            last = None
            shown = None
            status_data = {}
            while time.monotonic() < deadline:
                response = client.get(f"{base}/v1/meeting-minutes/{meeting_id}/status",
                                      headers=auth)
                if response.status_code != 200:
                    checks.check(False, "status poll stayed 200", str(response.status_code))
                    break
                status_data = response.json()["data"]
                state = (status_data.get("status"), status_data.get("stage"))
                if state != last:
                    stage_started.setdefault(status_data.get("stage") or "", time.monotonic())
                    last = state
                elapsed = time.monotonic() - started
                line = (f"  [{elapsed / 60:6.1f} min] {status_data.get('status'):<10} "
                        f"{status_data.get('stage') or '-':<20} "
                        f"{status_data.get('progress', 0):3d}%")
                if interactive:
                    print(f"\r{line}", end="", flush=True)
                elif (status_data.get("status"), status_data.get("stage"),
                      status_data.get("progress")) != shown:
                    # Piped to a log or CI: one line per real change, not per poll.
                    shown = (status_data.get("status"), status_data.get("stage"),
                             status_data.get("progress"))
                    print(line, flush=True)
                if status_data.get("status") in TERMINAL:
                    break
                time.sleep(args.poll_interval)
            if interactive:
                print()
            total = time.monotonic() - started

            if status_data.get("status") == "FAILED":
                err = status_data.get("error", {})
                print(f"\n  Job FAILED at stage {status_data.get('stage')}: "
                      f"{err.get('code')} -- {err.get('message')}")
                checks.check(False, "job reached COMPLETED", err.get("code", "FAILED"))
                raise SystemExit(1)
            if not checks.check(status_data.get("status") == "COMPLETED",
                                "job reached COMPLETED",
                                f"still {status_data.get('status')} after {total / 60:.1f} min"):
                raise SystemExit(1)
            checks.check(status_data.get("progress") == 100, "COMPLETED reports progress 100")
            checks.check(status_data.get("stage") == "COMPLETED", "COMPLETED reports stage COMPLETED")
            checks.check(status_data.get("status") in STATUSES,
                         "status is one of the four allowed values",
                         str(status_data.get("status")))
            checks.check(status_data.get("stage") in STAGES,
                         "stage is a closed enum a client can switch on",
                         str(status_data.get("stage")))
            progress = status_data.get("progress")
            checks.check(isinstance(progress, int) and not isinstance(progress, bool)
                         and 0 <= progress <= 100,
                         "progress is an integer 0-100", repr(progress))
            checks.check(bool(ISO_UTC.match(str(status_data.get("updatedAt")))),
                         "updatedAt is ISO 8601 UTC", str(status_data.get("updatedAt")))
            checks.check(bool(ISO_UTC.match(str(accepted.get("createdAt")))),
                         "createdAt is ISO 8601 UTC", str(accepted.get("createdAt")))

            # --- 4. minutes ---------------------------------------------
            print("\n4. Minutes")
            got = client.get(f"{base}/v1/meeting-minutes/{meeting_id}", headers=auth)
            if not checks.check(got.status_code == 200, "GET minutes is 200",
                                f"got {got.status_code}: {got.text[:300]}"):
                raise SystemExit(1)
            payload = got.json()
            checks.check(payload.get("success") is True, "success is true")
            data = payload["data"]
            checks.check(data.get("meetingId") == meeting_id, "meetingId matches")
            checks.check(data.get("language") == "ar", "language is ar")
            checks.check(bool(ISO_UTC.match(str(data.get("generatedAt")))),
                         "generatedAt is ISO 8601 UTC", str(data.get("generatedAt")))
            checks.check(isinstance(data.get("content"), dict),
                         "content is a JSON object, not a JSON-encoded string",
                         type(data.get("content")).__name__)

            sections = data.get("content", {}).get("sections")
            checks.check(isinstance(sections, list), "content.sections is an array")
            checks.check([s.get("key") for s in sections or []] == SECTION_KEYS,
                         "all five section keys, in contract order",
                         str([s.get("key") for s in sections or []]))
            checks.check(any((s.get("content") or "").strip() for s in sections or []),
                         "at least one section has content")
            stray = [s.get("key") for s in sections or []
                     if "decisions" in s or "decision" in (s.get("key") or "")]
            checks.check(not stray, "decisions are not embedded in content.sections", str(stray))
            extra_keys = sorted({k for s in sections or [] for k in s} - {"key", "title", "content"})
            checks.check(not extra_keys,
                         "sections carry only key/title/content", str(extra_keys))

            decisions = data.get("decisions")
            checks.check(isinstance(decisions, list), "decisions is an array")
            shape_ok, detail = True, ""
            order_ok, order_detail = True, ""
            unit_ok, unit_detail = True, ""
            leaked = set()
            for decision in decisions or []:
                if decision.get("kind") not in {"RESOLUTION", "ASSIGNMENT"}:
                    shape_ok, detail = False, f"bad kind {decision.get('kind')!r}"
                if decision.get("type") != "FOR_EXECUTION":
                    shape_ok, detail = False, f"bad type {decision.get('type')!r}"
                if not str(decision.get("title") or "").strip():
                    shape_ok, detail = False, "empty title"
                if decision.get("kind") == "RESOLUTION" and "completionDuration" in decision:
                    shape_ok, detail = False, "RESOLUTION must omit completionDuration"
                order = decision.get("agendaItemOrder")
                if order is not None and not (
                    isinstance(order, int) and not isinstance(order, bool) and order >= 1
                ):
                    order_ok, order_detail = False, f"agendaItemOrder {order!r} is not one-based"
                unit = decision.get("completionDurationUnit")
                if unit is not None and unit not in DURATION_UNITS:
                    unit_ok, unit_detail = False, f"bad unit {unit!r}"
                leaked |= FORBIDDEN_DECISION_KEYS & set(decision)
            checks.check(shape_ok, "every decision matches the documented shape", detail)
            checks.check(order_ok, "agendaItemOrder is a one-based integer", order_detail)
            checks.check(unit_ok, "completionDurationUnit is DAYS/WEEKS/MONTHS", unit_detail)
            checks.check(not leaked,
                         "no voting results, status, or internal IDs in decisions",
                         str(sorted(leaked)))

            again = client.get(f"{base}/v1/meeting-minutes/{meeting_id}", headers=auth)
            checks.check(again.json() == payload, "repeat read returns an identical payload")

            # --- 5. save --------------------------------------------------
            args.output_dir.mkdir(parents=True, exist_ok=True)
            json_path = args.output_dir / f"{meeting_id}.json"
            md_path = args.output_dir / f"{meeting_id}.md"
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            md_path.write_text(render_markdown(data), encoding="utf-8")

            # --- 6. show what the models actually produced ----------------
            # A smoke test that only prints PASS lines cannot tell "the
            # pipeline ran" from "the pipeline ran and produced nothing
            # usable". Both are green on shape alone, so print the content.
            filled = sum(1 for s in sections if (s.get("content") or "").strip())
            print("\n5. Generated content")
            print(f"  {'section':<14} {'title':<24} chars")
            for section in sections:
                body = section.get("content") or ""
                marker = " " if body.strip() else "!"
                print(f" {marker}{section['key']:<14} {(section.get('title') or '-')[:24]:<24} "
                      f"{len(body):>5}")
            if filled < len(sections):
                print(f"\n  NOTE: {len(sections) - filled} section(s) marked ! are empty. "
                      "That is legitimate\n  when the recording never states them, but all-empty "
                      "output usually means the\n  transcript was silent or the LLM returned "
                      "nothing usable -- check the worker\n  log and the job's transcript.txt "
                      "before accepting this run.")

            body_limit = None if args.full else 600
            for section in sections:
                body = (section.get("content") or "").strip()
                if not body:
                    continue
                shown = body if body_limit is None else body[:body_limit]
                print(f"\n  --- {section['key']} ---")
                for line in shown.splitlines():
                    print(f"  {line}")
                if body_limit is not None and len(body) > body_limit:
                    print(f"  ... (+{len(body) - body_limit} chars; --full to see all)")

            print(f"\n  --- decisions ({len(decisions or [])}) ---")
            for index, decision in enumerate(decisions or [], 1):
                print(f"  {index}. [{decision['kind']}] {decision['title']}")
                for field in ("description", "responsiblePersonName", "agendaItemOrder",
                              "completionDuration", "completionDurationUnit"):
                    if decision.get(field) is not None:
                        print(f"       {field}: {decision[field]}")
            if not decisions:
                print("  (none extracted -- legitimate if the meeting agreed nothing)")

            print(f"\n  Wall clock : {total / 60:.1f} min")
            print(f"  Sections   : {filled}/{len(sections)} non-empty")
            print(f"  Decisions  : {len(decisions or [])}")
            print(f"  Saved      : {json_path}")
            print(f"               {md_path}")

    except KeyboardInterrupt:
        print("\nInterrupted. The job keeps running on the server; re-poll with:")
        print(f"  curl -H 'Authorization: Bearer $AI_SERVICE_API_KEY' "
              f"{base}/v1/meeting-minutes/{meeting_id}/status")
        return 130
    except SystemExit:
        pass
    finally:
        if server is not None:
            server.shutdown()
        if stack is not None:
            stack.stop()
            print("Local stack stopped.")

    print(f"\n{'=' * 60}")
    if checks.failures:
        print(f"FAILED -- {checks.passed} passed, {len(checks.failures)} failed:")
        for failure in checks.failures:
            print(f"  - {failure}")
        return 1
    print(f"OK -- all {checks.passed} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
