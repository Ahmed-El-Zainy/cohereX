# Meeting minutes API — agreed mapping (review)

This is the reviewed design for exposing this repo’s SSH/CPU deployment as the
three endpoints in [`required_intergration.md`](../required_intergration.md).
The first implementation is in `coherex_minutes/`, with deployment templates in
`deploy/`; it is **not deployed to the server until the steps in
[`deploy/README.md`](../deploy/README.md) are applied**.

**Share with the platform team:** [`PLATFORM_TEAM_GUIDE.md`](PLATFORM_TEAM_GUIDE.md)
(client contract + current “not live” status).

Read first:

- [`required_intergration.md`](../required_intergration.md) — the platform contract
- [`DEPLOYMENT.md`](DEPLOYMENT.md) — ASR (vLLM) on the CPU box
- [`LLM_DEPLOYMENT.md`](LLM_DEPLOYMENT.md) — LLM (llama.cpp) and why ASR/LLM cannot run together
- [`PIPELINE.md`](PIPELINE.md) — today’s laptop pipeline (`pipeline.py` + SSH toggle)

---

## 1. Purpose of this mapping

The platform needs an **AI service** it can call with a `meetingId` and a
**video URL**, then poll until minutes and typed decisions exist.

What we have today is a **lab**: a small CPU VM, two mutually exclusive
systemd services, and clients on a laptop that SSH-toggle the box. That cannot
be `{AI_SERVICE_BASE_URL}` without a new always-on HTTP layer, a job store, and
a worker that runs **on the box**.

**Success for v1** is not “fast” or “GPU-like throughput.” It is:

- Match the three routes, JSON shapes, statuses, and error envelope in the
  integration guide.
- Survive **long mixed Arabic/English board recordings** without dropping the
  end of the meeting or inventing names/acronyms/owners.
- Stay within **8GB RAM** (ASR and LLM still take turns).

---

## 2. What is already deployed (facts)

| Piece | Role today | Why it exists |
| --- | --- | --- |
| CPU VM (2 vCPU, 8GB, no GPU, Ubuntu 24.04) | The only inference machine | Hardware we have; GPU would be better (see DEPLOYMENT.md). |
| `coherex-vllm` `:8000` | Arabic-07 ASR via vLLM CPU | Transcription. Uses ~7GB RAM + swap; ~1–2.5 min per ~30s of audio; **one job at a time**. |
| `coherex-llm` `:8001` | Qwen2.5-3B GGUF via llama.cpp | Minutes/extraction from **text**. ~2GB RAM. Cannot be warm with vLLM. |
| `scripts/toggle_server.sh` | SSH `systemctl` flip | Frees RAM by running **only one** heavy service. Default rest state: ASR. |
| `main.py` / `pipeline.py` / `llm_client.py` | Run **on a laptop** | VAD and audio load were **off the box** so vLLM did not share RAM with Silero/alignment. CLI tasks (`minutes_draft`, etc.) are **not** the platform schema. |

**Implication:** we do **not** wrap `pipeline.py` as-is. The integration API must
own ingest, queue, progress, and the spec JSON. The heavy models stay as
localhost backends.

---

## 3. Target architecture (purpose of each part)

```
Platform
  │  HTTPS :443  Authorization: Bearer {AI_SERVICE_API_KEY}
  ▼
Caddy/nginx (TLS)
  │  proxy to 127.0.0.1:8080
  ▼
coherex-minutes-api          ← always on; cheap; never toggles GPU-like RAM
  │  write/read SQLite
  ▼
Job store (SQLite) + files under /var/lib/coherex/jobs/{meetingId}/
  ▲
  │  poll / update
coherex-minutes-worker       ← always on; owns queue, ffmpeg, toggles, LLM assemble
  │
  ├─ systemctl start/stop coherex-vllm / coherex-llm   (local, not SSH)
  ├─ 127.0.0.1:8000  /v1/audio/transcriptions
  └─ 127.0.0.1:8001  /v1/chat/completions
```

| Part | Purpose |
| --- | --- |
| **Minutes API process** | Be the spec: accept POST, idempotency, status GET, minutes GET. Must stay up while a job runs for hours so polling works. Must **not** load models. |
| **Minutes worker process** | Drain the FIFO queue. Download (if not done), ffmpeg, ASR, toggle, LLM map-reduce, write result, then delete media. Isolated so an API restart does not kill ASR. |
| **SQLite** | Durable jobs: `meetingId` is the primary key, statuses, progress, error, timestamps. Survives reboot with systemd restart. |
| **Job directory** | Video (temporary), chunk list, **ASR checkpoint** (resume), transcript, LLM slice notes, final JSON. |
| **vLLM (localhost)** | Only ASR. Bound to **127.0.0.1** so the internet cannot hit it. |
| **llama.cpp (localhost)** | Only minutes/decisions generation. Same bind rule. |
| **Local `systemctl` toggle** | Same RAM constraint as today, but the worker is **on the box**, so SSH toggle is the wrong tool. |
| **Caddy/nginx :443** | `{AI_SERVICE_BASE_URL}` as `https://…`. Bearer token is not sent in cleartext. |
| **Firewall 22 + 443** | SSH for ops; TLS for the platform. Nothing else public. |

**Why not in-process FastAPI + background thread?** Polling must survive a
transcription that lasts hours and a vLLM stop that can take 1–2 minutes.
A dedicated worker matches the existing systemd model (`journalctl`,
`Restart=on-failure`).

**Why not keep `:8000`/`:8001` public?** The guide defines **one** credential and
**three** paths. ASR/LLM are implementation details. Leaving them on `0.0.0.0`
would be a second accidental product (today: no firewall).

---

## 4. API mapping (contract vs this service)

Base URL: `{AI_SERVICE_BASE_URL}` after TLS, e.g. `https://minutes.example.com`.

Auth: `Authorization: Bearer {AI_SERVICE_API_KEY}` (minutes API only; vLLM/LLM
keys stay on the box for localhost).

Content type: `application/json`. Timestamps: ISO 8601 UTC.

`meetingId` is supplied by the platform. This service **does not** mint a second
job id. On this VM, `meetingId` is **globally unique** (the guide’s “within an
organization” is the platform’s problem; our store has no org column in v1).

### 4.1 `POST /v1/meeting-minutes`

**Purpose:** start (or reuse) one generation job.

- **202** body matches the guide: `success`, `data.meetingId`, `data.status: "QUEUED"`, `data.createdAt`.
- **Same `meetingId` again:** return the **existing** job and current status. Do
  **not** enqueue a duplicate, including when status is `FAILED`.
- **New `meetingId` while another job runs:** **enqueue** (`QUEUED`), FIFO, one
  worker. Purpose: the spec already has `QUEUED`; failing with “busy” would
  drop a submit the platform already assembled.
- **`language`:** v1 accepts missing, `"ar"`, or `"auto"` -- all three resolve
  to `ar`. The guide documents `auto` as the **default**, so rejecting it would
  fail a client that follows the spec exactly. `"en"` is still rejected
  (`UNSUPPORTED_LANGUAGE`): it needs the 14-language base model, which cannot be
  warm alongside Arabic-07 on this box. Mixed Arabic/English audio is handled
  **by the Arabic-07 model**, not by switching models. GET always reports
  `"language": "ar"` (the language actually used).

  Known limitation of resolving `auto` to `ar`: a genuinely English-only
  recording submitted as `auto` is transcribed by the Arabic model and yields
  poor minutes rather than a clear error. v1 is for Arabic-primary board
  meetings; if English-only recordings become real traffic, that needs the
  second model, not a prompt change.
- POST stays **fast**: do **not** download the video inside the HTTP request.

### 4.2 `GET /v1/meeting-minutes/{meetingId}/status`

**Purpose:** poll until `COMPLETED` or `FAILED`. Always **200** when the job
exists (including `FAILED`), as in the guide.

Allowed `status` values only: `QUEUED` | `PROCESSING` | `COMPLETED` | `FAILED`.

`progress`: integer 0–100.

`stage` values **published to the platform** (only these styles, so clients can
match the guide):

| When | `status` | `stage` | `progress` (approx.) |
| --- | --- | --- | --- |
| Waiting for the single worker | `QUEUED` | none on POST; on poll use `QUEUED` | 0 |
| Download, ffmpeg, ASR chunks (incl. resume) | `PROCESSING` | `TRANSCRIBING` | ~0–85 from ingest + `chunks_done / chunks_total` |
| LLM map-reduce + assemble | `PROCESSING` | `GENERATING_MINUTES` | ~85–99 from slice index |
| Result written | `COMPLETED` | `COMPLETED` | 100 |
| Terminal error | `FAILED` | stage **where it died** (`TRANSCRIBING` or `GENERATING_MINUTES`) | last value; plus `error` |

Chunk numbers (`40/120`) stay in **worker logs**, not in `stage`. Purpose:
the guide’s examples use `TRANSCRIBING` / `GENERATING_MINUTES` / `COMPLETED`;
free-form stages would break a client that treats `stage` as an enum.

### 4.3 `GET /v1/meeting-minutes/{meetingId}`

**Purpose:** return minutes **once**, then forever the **same** payload
(repeated reads, no new job).

- `COMPLETED`: **200** `success: true` and the spec `data` object (`content.sections`, `decisions`, `generatedAt`, `language`).
- `QUEUED` / `PROCESSING`: **200** `success: false`, `error.code: "MINUTES_NOT_READY"`.
- `FAILED`: **200** `success: false`, `error.code: "GENERATION_FAILED"` (details already on the status route).
- Unknown `meetingId`: **404** `success: false`, `error.code: "NOT_FOUND"`.

**Purpose of not returning empty `success: true`:** the platform must not persist
blank minutes as if generation succeeded.

The GET body **does not** include the raw transcript. The guide has no such
field; transcript is an ops artifact.

### 4.4 Error envelope

Every failure uses the guide’s shape:

```json
{
  "success": false,
  "error": {
    "code": "INVALID_VIDEO_URL",
    "message": "The video URL is invalid or cannot be accessed."
  }
}
```

Planned codes (v1):

| Code | Purpose |
| --- | --- |
| `INVALID_VIDEO_URL` | URL missing, not http(s), or download/HTTP failure. |
| `VIDEO_HOST_NOT_ALLOWED` | URL/redirect hostname is absent from the configured storage allowlist. |
| `VIDEO_TOO_LARGE` | Over **2 GB** — disk/DoS guard, not “meetings must be short.” |
| `VIDEO_TOO_LONG` | Audio longer than **~3 hours** after ffmpeg probe — runaway-file guard only. Real board meetings **under** that run to completion even if they take many hours. |
| `UNSUPPORTED_LANGUAGE` | `en`, or any value outside `ar`/`auto`. |
| `MINUTES_NOT_READY` | GET minutes before `COMPLETED`. |
| `GENERATION_FAILED` | GET minutes after `FAILED`. |
| `NOT_FOUND` | Unknown `meetingId`. |
| `UNAUTHORIZED` | Missing/wrong Bearer token. |
| `TRANSCRIPTION_FAILED` | ASR path (also on status `error` when `FAILED` during `TRANSCRIBING`). |
| `MINUTES_GENERATION_FAILED` | LLM/assemble path. |

---

## 5. Job lifecycle (why each step)

```
POST 202
  → persist job QUEUED
  → start download in background (does not wait for ASR RAM)
  → worker (when free): ffmpeg → ASR chunks (checkpoint) → toggle LLM
  → map-reduce minutes/decisions → validate JSON
  → COMPLETED + delete video/WAV chunks
  → toggle back to ASR (resting state)
```

| Step | Purpose |
| --- | --- |
| **Download immediately after accept** | FIFO wait can last hours. Signed `videoUrl` will expire if we fetch only when ASR starts. POST stays 202. Overlap: download of job B while job A transcribes is OK (network/disk, not vLLM). |
| **ffmpeg extract + ~30s splits** | This box survived ASR as **small chunks**. No Silero/pyannote/alignment **on the box** (those ran on the laptop; stacking them on 8GB next to vLLM OOMs). Slightly worse cut points than Silero; acceptable vs crashing. |
| **One chunk at a time to vLLM** | Stability and accuracy under RAM pressure. Concurrent chunks contended and timed out in the original deployment. |
| **Checkpoint after every chunk** | Long meetings must **not** FAIL and reset (POST cannot retry `FAILED`). Worker/vLLM crash → resume same `meetingId`, stay `PROCESSING`. |
| **Bound consecutive retries** | Transient failures back off and resume checkpoints. After 10 consecutive failures without progress, mark the job `FAILED` so one poisoned job cannot block FIFO forever. Any successful progress resets the counter. |
| **No short duration cap** | A 2-hour board is a **long job**, not an error. Slow is OK; aborting for length is not. |
| **2 GB / ~3 h guards** | Protect disk and reject garbage files, not typical meetings. |
| **Toggle to LLM only after ASR** | RAM. While LLM is up, **several** chat calls are cheap (no extra toggle). |
| **Two LLM products, then assemble in the worker** | Spec is (1) five Markdown sections and (2) typed `decisions[]`. One giant JSON from 3B is brittle. Old `llm_client.py` tasks stay **CLI-only**; they do not match the spec (votes, different headings). |
| **Map-reduce over the transcript** | llama.cpp context is small (`-c ~8192`). Truncating drops late resolutions. Per-slice decisions + merge/dedupe; per-slice notes then a **final** pass into the five keys. Worker **never invents** owners, dates, Latin acronyms, or attendance. |
| **Delete video and decoded WAV chunks when terminal** | GET only needs the result JSON. Media is large and sensitive. Keep transcript and small checkpoints for ops/debug. |
| **FAILED is sticky** | Guide: regeneration is a **later explicit** feature. POST must not start a new job. Ops reset is SSH/admin, not public. |

---

## 6. Language and mixed Arabic/English

**Purpose:** real meetings mix Arabic and English (loanwords, acronyms). The
deployed model
[`CohereLabs/cohere-transcribe-arabic-07-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026)
is already documented in this repo as supporting **`ar`, `en`, and
Arabic–English code-switching**. We keep that single checkpoint.

| Rule | Purpose |
| --- | --- |
| ASR: Arabic-07, treat as Arabic-primary mixed | Do not swap to the 14-language model for “there is English in the room.” |
| Minutes prose: Modern Standard Arabic | Matches board-secretary deliverable and current LLM default. |
| Keep Latin **only if ASR wrote Latin** | Spec: do not invent names. If ASR wrote `BIC`, keep `BIC`. If it only wrote `البي اي سي`, **do not** invent `BIC`. |
| LLM prompts must state that rule | A 3B model already missed “BIC” when the transcript was phonetic Arabic (`LLM_DEPLOYMENT.md`). Silence in the prompt repeats that class of miss even when Latin **is** in the transcript. |
| GET `language`: `"ar"` | Field is primary/output language, not “this file is English-only.” |

---

## 7. Minutes and decisions shape

Must match the guide (and existing platform `POST /api/minutes` content type):

- `content` is an **object**, not a string. `sections` is always an array.
- Section `content` is Markdown (GFM tables).
- Keys **in this order:** `meeting_info`, `attendance`, `introduction`, `agenda`, `main_items`.
- **No** decisions inside `sections`.
- `decisions[]`: `title` required; `kind` `RESOLUTION` | `ASSIGNMENT`; `type` `FOR_EXECUTION`; optional `agendaItemOrder`, `responsiblePersonName`, `completionDuration` / `Unit` (`DAYS` | `WEEKS` | `MONTHS`) only for assignments.
- No voting results, no platform user ids, no DB ids.
- Unsupported facts → `null` or omit optional text — **do not invent**.

Worker post-processing **purpose:** drop illegal fields, coerce enums, empty
`decisions` if parse fails after retries rather than publishing garbage JSON.

---

## 8. Security and operations

| Measure | Purpose |
| --- | --- |
| TLS on 443, API on `127.0.0.1:8080` | Platform-safe `AI_SERVICE_BASE_URL`; token not on HTTP. |
| vLLM/LLM on `127.0.0.1` | Only the worker talks to models. |
| Reject private/loopback video URLs (including redirects) | Prevent the download endpoint being used for SSRF into the VM or cloud metadata services. |
| One public Bearer secret | Matches the integration guide. |
| `ufw`: 22 and 443 | Close the current “everything listening is public” gap in DEPLOYMENT.md. |
| systemd: `coherex-minutes-api`, `coherex-minutes-worker` | Enable on boot. **Do not** enable both vLLM and LLM at once (existing rule). Worker starts ASR as the idle backend when idle. |
| Laptop `main.py` against public `:8000` | **Goes away** unless SSH tunnel / localhost on the server. Purpose of Q7: the box becomes a minutes product, not a public ASR API. |

---

## 9. Explicitly out of v1

| Item | Why later |
| --- | --- |
| Public regenerate / POST-retry of `FAILED` | Guide postponed it; idempotency would break. |
| English-only, or real language detection behind `auto` | Extra models/ML on an 8GB box; mixed AR/EN is already Arabic-07, and `auto` resolves to `ar`. |
| Silero/pyannote/alignment on the server | RAM; not required for the spec (no word timestamps in GET). |
| Concurrent ASR jobs | Hardware cannot. |
| Warm ASR + LLM together | Needs more RAM or a second machine. |
| Org id in the API | Not in the three endpoints. |
| Transcript in GET | Not in the guide. |
| Reuse of `TASK_PROMPTS` as the integration mapper | Different document (votes, secretary sections). |

---

## 10. How this differs from `pipeline.py`

| `pipeline.py` (today) | Minutes API (this design) |
| --- | --- |
| You run it on a laptop | Platform HTTP |
| SSH toggle | Local systemd from worker |
| Local Silero VAD | ffmpeg splits on the server |
| Tasks: summary / minutes_draft / … | Spec sections + `decisions[]` |
| No `meetingId`, no poll | Job store, 202 + status + GET |
| Audio file path | `videoUrl` download |
| No TLS product API | HTTPS + Bearer |

Keep `pipeline.py` for **manual** lab runs until the API exists; do not teach
the platform to SSH.

---

## 11. Review checklist

Use this to approve or send back the design:

- [ ] FIFO + sticky `FAILED` is acceptable to the platform (no POST retry).
- [ ] Multi-hour wall clock for a long video is acceptable (accuracy over speed).
- [ ] 2 GB / ~3 h guards are acceptable.
- [ ] Mixed meetings: Latin only when present in ASR output is acceptable (no invented `BIC`).
- [ ] DNS name exists (or will exist) for Let’s Encrypt on this VM.
- [ ] Platform will send `"language": "ar"` or omit it.
- [ ] Breaking public `:8000`/`:8001` is acceptable.

When this document is approved, implementation can follow it without
re-litigating these forks.
