# Deploying the meeting-minutes API

This directory contains deployable templates for the design in
[`docs/MEETING_MINUTES_API.md`](../docs/MEETING_MINUTES_API.md).

## 1. Install code and API dependencies

On the existing Ubuntu box:

```bash
cd /opt/coherex
git pull
source /opt/coherex-venv/bin/activate
pip install -e ".[minutes-api]" --no-deps
pip install "fastapi>=0.115" "httpx>=0.27" "uvicorn[standard]>=0.30"
mkdir -p /var/lib/coherex-minutes/jobs
chmod 700 /var/lib/coherex-minutes
```

The split install deliberately avoids letting pip replace the locally built
CPU vLLM package (see `docs/DEPLOYMENT.md`).

## 2. Make model servers private

Change both existing systemd units from `--host 0.0.0.0` to:

```text
--host 127.0.0.1
```

The units are `/etc/systemd/system/coherex-vllm.service` and
`/etc/systemd/system/coherex-llm.service`. Also change llama.cpp context from
`-c 4096` to `-c 8192`.

`-c` and `COHEREX_MINUTES_SLICE_CHARS` must be changed together: a slice is sent
as one prompt and the response budget (`COHEREX_MINUTES_LLM_MAX_TOKENS`) is
reserved on top of it. At ~2.7 Arabic chars per token, the shipped 10000/1800
pair needs ~5500 of the 8192 tokens. Overflowing the window fails every attempt
identically, so the job never completes.

Add `OOMScoreAdjust=500` to the `[Service]` section of **`coherex-vllm.service`**.
The box runs at ~7GB/7.8GB with swap active, so an OOM event is a matter of
when. This makes vLLM the kernel's preferred victim: the worker restarts it on
the next toggle, whereas a killed API silently drops every submission. The
minutes API and worker units carry the matching negative adjustments.

Reload, but do **not** enable `coherex-llm`; the worker keeps only one model
warm:

```bash
systemctl daemon-reload
systemctl disable coherex-llm
systemctl restart coherex-vllm
```

## 3. Install secrets and systemd units

```bash
install -m 0600 deploy/coherex-minutes.env.example /etc/coherex-minutes.env
# Edit every <...> value before starting.
editor /etc/coherex-minutes.env

install -m 0644 deploy/coherex-minutes-api.service /etc/systemd/system/
install -m 0644 deploy/coherex-minutes-worker.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now coherex-minutes-api coherex-minutes-worker
```

Set `COHEREX_VIDEO_ALLOWED_HOSTS` to the exact storage/CDN hostnames used by
signed URLs, including any redirect destination. Downloads are denied when this
allowlist is empty or the hostname is absent.

Check free disk before going live and set `COHEREX_MINUTES_MIN_FREE_BYTES`
accordingly:

```bash
df -h /var/lib
```

The reserve must exceed one meeting's video, or every download defers and the
queue stalls. A deferred download logs a `Deferring download for ...` warning
each sweep — alert on it; it means the disk needs attention, and meetings are
piling up queued rather than failing.

There must be exactly **one** API uvicorn worker and one minutes worker.
Multiple API workers would each create an ingest pool; multiple minutes workers
would violate the one-ASR-job hardware limit.

Check:

```bash
curl http://127.0.0.1:8080/health
journalctl -u coherex-minutes-api -u coherex-minutes-worker -f
```

## 4. TLS and firewall

Point the chosen DNS name at the VM, install Caddy, replace
`minutes.example.com` in `Caddyfile.example`, and install it:

```bash
install -m 0644 deploy/Caddyfile.example /etc/caddy/Caddyfile
systemctl reload caddy

ufw allow 22/tcp
ufw allow 443/tcp
ufw enable
```

Before enabling the firewall, confirm the SSH allow rule is present from a
second session. Do **not** allow 8000, 8001, or 8080 publicly.

## 5. Smoke test

Preferred: run the scripted check, which drives all three endpoints and
validates the payload against the contract (see
[TESTING.md](../docs/TESTING.md)):

```bash
cd /opt/coherex && source /opt/coherex-venv/bin/activate
scripts/smoke_test_api.py --video-url https://storage.example.com/signed/meeting.mp4
```

Or by hand:


```bash
curl https://minutes.example.com/v1/meeting-minutes \
  -H "Authorization: Bearer $AI_SERVICE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "meetingId": "deployment-smoke-001",
    "videoUrl": "https://storage.example.com/signed/meeting.mp4",
    "language": "ar"
  }'
```

Poll the returned id at:

```text
GET /v1/meeting-minutes/deployment-smoke-001/status
GET /v1/meeting-minutes/deployment-smoke-001
```

An id is intentionally permanent. Use a new id for every smoke test; failed
ids are sticky until an administrator removes them from SQLite.
