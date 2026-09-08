#!/usr/bin/env python3
"""
Run the whole meeting-minutes service locally, with the two model servers stubbed.

The real API, the real worker, the real SQLite store, the real video download
and the real ffmpeg chunking all run. Only the ASR and LLM HTTP calls are
answered by a stub, because those need the 8GB box and hours of CPU.

Use it to exercise scripts/smoke_test_api.py, or to give the platform team
something to integrate against, before the server is deployed.

    scripts/dev_stack.py

It prints an AI_SERVICE_BASE_URL / AI_SERVICE_API_KEY pair to paste into .env,
then serves until Ctrl-C. In another shell:

    scripts/smoke_test_api.py --serve samples/saudi_business_03min.mp3

This is a development tool. It stubs the models, so it proves the API contract
and the job lifecycle -- never transcription or minutes quality.

Needs the API extra:  pip install -e ".[minutes-api]"
"""
from __future__ import annotations

import json
import os
import secrets
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SECTION_KEYS = ["meeting_info", "attendance", "introduction", "agenda", "main_items"]
TITLES = {
    "meeting_info": "بيانات الاجتماع",
    "attendance": "الحضور",
    "introduction": "المقدمة",
    "agenda": "جدول الأعمال",
    "main_items": "البنود الرئيسية",
}


class StubModels(BaseHTTPRequestHandler):
    """Answers the two endpoints the worker calls: vLLM ASR and llama.cpp chat."""

    def do_GET(self):  # noqa: N802
        self._json({"status": "ok"}) if self.path == "/health" else self.send_error(404)

    def do_POST(self):  # noqa: N802
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.path.endswith("/v1/audio/transcriptions"):
            self._json({"text": "اعتمد المجلس الميزانية السنوية وكلّف أحمد علي بإعداد خطة التنفيذ."})
        elif self.path.endswith("/v1/chat/completions"):
            self._json({"choices": [{"message": {"content": self._reply(body)}}]})
        else:
            self.send_error(404)

    @staticmethod
    def _reply(body: bytes) -> str:
        prompt = json.loads(body)["messages"][0]["content"]
        if "meeting_info" in prompt:  # sections pass
            return json.dumps(
                [
                    {
                        "key": key,
                        "title": TITLES[key],
                        "content": "استعرض المجلس الميزانية السنوية المقترحة."
                        if key == "main_items"
                        else "",
                    }
                    for key in SECTION_KEYS
                ],
                ensure_ascii=False,
            )
        if "RESOLUTION" in prompt:  # decisions pass
            return json.dumps(
                [
                    {
                        "title": "اعتماد الميزانية السنوية",
                        "kind": "RESOLUTION",
                        "type": "FOR_EXECUTION",
                        "agendaItemOrder": 1,
                    },
                    {
                        "title": "إعداد خطة التنفيذ النهائية",
                        "kind": "ASSIGNMENT",
                        "type": "FOR_EXECUTION",
                        "responsiblePersonName": "أحمد علي",
                        "completionDuration": 14,
                        "completionDurationUnit": "DAYS",
                    },
                ],
                ensure_ascii=False,
            )
        return "ملاحظات: نوقشت الميزانية السنوية وخطة التنفيذ."

    def _json(self, payload):
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *_args):
        pass


def main() -> int:
    api_key = secrets.token_hex(32)
    data_dir = Path(tempfile.mkdtemp(prefix="coherex-dev-"))

    models = ThreadingHTTPServer(("127.0.0.1", 0), StubModels)
    threading.Thread(target=models.serve_forever, daemon=True).start()
    models_url = f"http://127.0.0.1:{models.server_address[1]}"

    # Set before importing the app: Settings.from_env() reads these at startup.
    os.environ.update(
        COHEREX_MINUTES_DATA_DIR=str(data_dir),
        AI_SERVICE_API_KEY=api_key,
        COHEREX_VLLM_URL=models_url,
        COHEREX_LLM_URL=models_url,
        COHEREX_VLLM_API_KEY="",
        COHEREX_LLM_API_KEY="",
        # No systemd here, so the worker must not try to toggle services.
        COHEREX_MINUTES_MANAGE_SERVICES="false",
        # --serve publishes the file on loopback; both SSRF gates must open.
        COHEREX_VIDEO_ALLOWED_HOSTS="127.0.0.1,localhost",
        COHEREX_MINUTES_ALLOW_PRIVATE_VIDEO_HOSTS="true",
        COHEREX_MINUTES_MIN_FREE_BYTES="0",
        COHEREX_MINUTES_WORKER_POLL_SECONDS="1",
    )

    try:
        import uvicorn
        from coherex_minutes.api import create_app
        from coherex_minutes.config import Settings
        from coherex_minutes.worker import Worker
    except ImportError as exc:
        print(f"Missing dependency: {exc}\nInstall with: pip install -e \".[minutes-api]\"",
              file=sys.stderr)
        return 2

    settings = Settings.from_env()
    settings.prepare()
    worker = Worker(settings)
    threading.Thread(target=worker.run_forever, daemon=True).start()

    config = uvicorn.Config(create_app(), host="127.0.0.1", port=8080, log_level="warning")
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None  # not the main thread's job here
    threading.Thread(target=server.run, daemon=True).start()

    base_url = "http://127.0.0.1:8080"
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    else:
        print("API did not start", file=sys.stderr)
        return 1

    print(f"""
Local stack is up (models are STUBBED -- contract only, not real transcription).

  API        {base_url}
  models     {models_url}
  data       {data_dir}

Paste into .env, or export:

  AI_SERVICE_BASE_URL={base_url}
  AI_SERVICE_API_KEY={api_key}

Then, in another shell:

  scripts/smoke_test_api.py --serve samples/saudi_business_03min.mp3 \\
    --base-url {base_url} --api-key {api_key}

Ctrl-C to stop.""", flush=True)   # flush: the key must appear even when redirected

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        worker.stop()
        models.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
