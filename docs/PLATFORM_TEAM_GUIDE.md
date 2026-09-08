# Meeting minutes AI service — teammate integration guide

**Status (2026-09-08): not live on the server.**

The three APIs from `required_intergration.md` are **implemented in this repo**
(`coherex_minutes/`) and covered by automated tests. They are **not** installed
on the Ubuntu VM yet (no public `{AI_SERVICE_BASE_URL}`, no systemd
`coherex-minutes-api` / worker). Platform work can start against this contract;
end-to-end calls will fail until deploy finishes (`deploy/README.md`).

When the box is live, fill these in and treat this file as the client spec:

| Env | Value |
| --- | --- |
| `AI_SERVICE_BASE_URL` | `https://<dns-name>` (HTTPS only; no `:8000` / `:8001`) |
| `AI_SERVICE_API_KEY` | Bearer secret shared with the platform |

---

## What you integrate

Generate minutes from a **meeting video URL**, using the **platform’s existing
`meetingId`**. Poll until done, then read minutes + decisions once.

```
POST /v1/meeting-minutes
        → 202 { meetingId, status, createdAt }
GET  /v1/meeting-minutes/{meetingId}/status
        → poll until COMPLETED or FAILED
GET  /v1/meeting-minutes/{meetingId}
        → only after COMPLETED
```

Shared rules:

- `Content-Type: application/json`
- `Authorization: Bearer {AI_SERVICE_API_KEY}`
- Timestamps: ISO 8601 UTC (`2026-09-07T10:30:00Z`)
- You own `meetingId`. The AI service does **not** mint a second job id.
- Same `meetingId` is **one job forever**. POST again returns that job; it does
  **not** regenerate. `FAILED` stays failed until ops reset (not a public API).

---

## 1. Submit — `POST /v1/meeting-minutes`

```http
POST /v1/meeting-minutes
Authorization: Bearer {AI_SERVICE_API_KEY}
Content-Type: application/json
```

```json
{
  "meetingId": "cm123platformMeetingId",
  "videoUrl": "https://storage.example.com/meetings/board-meeting.mp4",
  "language": "ar"
}
```

| Field | Required | Client notes |
| --- | --- | --- |
| `meetingId` | yes | `A–Z a–z 0–9 _ -`, max 128. Same id on status and GET. |
| `videoUrl` | yes | HTTPS URL the AI box can download. If signed, it must stay valid until **download starts** (accepted immediately; download is background). Hostname must be on the AI allowlist (storage/CDN). |
| `language` | no | **v1: send `"ar"` or omit.** Mixed Arabic/English audio is supported. `"en"` and `"auto"` return `UNSUPPORTED_LANGUAGE`. |

**202 Accepted** (new or existing job):

```json
{
  "success": true,
  "data": {
    "meetingId": "cm123platformMeetingId",
    "status": "QUEUED",
    "createdAt": "2026-09-07T10:30:00Z"
  }
}
```

If that `meetingId` already exists, `status` is whatever it is now (`QUEUED`,
`PROCESSING`, `COMPLETED`, or `FAILED`) — not a second job.

⚠️ **A re-POST ignores every field except `meetingId`.** Submitting the same id
with a *corrected* `videoUrl` does **not** update the job and does **not** retry
it — you get the original job, still bound to the original URL, and the response
looks like a success. Fixing a bad URL therefore requires a **new `meetingId`**
(or an ops reset). Do not build a "resubmit to repair" path.

**Do not** wait for minutes in this request. It returns as soon as the job is
recorded.

---

## 2. Status — `GET /v1/meeting-minutes/{meetingId}/status`

Poll until `COMPLETED` or `FAILED`. HTTP **200** when the job exists, including
failures (lookup succeeded).

```json
{
  "success": true,
  "data": {
    "meetingId": "cm123platformMeetingId",
    "status": "PROCESSING",
    "progress": 65,
    "stage": "GENERATING_MINUTES",
    "updatedAt": "2026-09-07T10:34:20Z"
  }
}
```

| `status` | Meaning |
| --- | --- |
| `QUEUED` | Waiting behind another meeting (one worker), or waiting for disk space to download the video. `progress` 0. Both clear on their own — keep polling, do not resubmit. |
| `PROCESSING` | Download/ffmpeg/ASR or LLM. |
| `COMPLETED` | Safe to GET minutes. |
| `FAILED` | Terminal. See `error`. Do not POST the same id to retry. |

| `stage` (when processing) | Meaning |
| --- | --- |
| `QUEUED` | Waiting. |
| `TRANSCRIBING` | Ingest + audio split + ASR. |
| `GENERATING_MINUTES` | LLM minutes/decisions. |
| `COMPLETED` | Done. |

`progress` is 0–100. Suggested poll: every **5–15 s**. A long board video on this
CPU box can take **hours**, not minutes. One meeting at a time.

Failed status still **200**:

```json
{
  "success": true,
  "data": {
    "meetingId": "cm123platformMeetingId",
    "status": "FAILED",
    "progress": 42,
    "stage": "TRANSCRIBING",
    "error": {
      "code": "TRANSCRIPTION_FAILED",
      "message": "Audio could not be transcribed."
    },
    "updatedAt": "2026-09-07T10:33:00Z"
  }
}
```

Unknown id: **404** `{ "success": false, "error": { "code": "NOT_FOUND", ... } }`.

---

## 3. Minutes — `GET /v1/meeting-minutes/{meetingId}`

Call **only** after status is `COMPLETED`. Repeat GET returns the **same**
payload, byte for byte, with no expiry in v1 — results are stored indefinitely
and are never regenerated. Persist on your side anyway; do not treat this
endpoint as your system of record.

Success **200**. Generated text is **Arabic** — the English examples in
`required_intergration.md` illustrate the *shape* only:

```json
{
  "success": true,
  "data": {
    "meetingId": "cm123platformMeetingId",
    "language": "ar",
    "content": {
      "sections": [
        {
          "key": "meeting_info",
          "title": "بيانات الاجتماع",
          "content": "| اليوم والتاريخ | المكان | الوقت |\n| --- | --- | --- |\n| غير مذكور في التسجيل | غير مذكور في التسجيل | غير مذكور في التسجيل |"
        },
        { "key": "attendance", "title": "الحضور", "content": "" },
        {
          "key": "introduction",
          "title": "المقدمة",
          "content": "افتتح رئيس المجلس الاجتماع ورحّب بالحضور."
        },
        {
          "key": "agenda",
          "title": "جدول الأعمال",
          "content": "| # | البند |\n| --- | --- |\n| 1 | اعتماد الميزانية السنوية |"
        },
        {
          "key": "main_items",
          "title": "البنود الرئيسية",
          "content": "**1. اعتماد الميزانية السنوية**\n\nاستعرض المجلس الميزانية المقترحة وناقش التكاليف التشغيلية المتوقعة."
        }
      ]
    },
    "decisions": [
      {
        "title": "اعتماد الميزانية السنوية 2027",
        "description": "اعتماد الميزانية السنوية المعروضة خلال الاجتماع.",
        "kind": "RESOLUTION",
        "type": "FOR_EXECUTION",
        "agendaItemOrder": 1,
        "responsiblePersonName": null
      },
      {
        "title": "إعداد خطة التنفيذ النهائية",
        "kind": "ASSIGNMENT",
        "type": "FOR_EXECUTION",
        "responsiblePersonName": "أحمد علي",
        "completionDuration": 14,
        "completionDurationUnit": "DAYS"
      }
    ],
    "generatedAt": "2026-09-07T10:36:10Z"
  }
}
```

`content` matches platform `POST /api/minutes`:

- `content` is an **object**, not a string.
- `sections` is **always exactly these five, in this order**: `meeting_info`,
  `attendance`, `introduction`, `agenda`, `main_items`. Nothing is dropped and
  nothing else is added, so you can render slots without null-checking the list.
- `title` is **model-generated Arabic prose and may be `""`**. Key your layout,
  translations, and storage off `key`, never off `title`.
- Section `content` is Markdown (GFM tables) and may be `""`.
- Decisions are **only** in `decisions[]`, never inside sections.
- Missing facts → empty string, `null`, or an Arabic "not stated" line. None of
  these are transport errors; persist them as-is.

**Optional decision keys are omitted, not `null`.** This differs from the
examples in `required_intergration.md`. Read them with `??`/optional chaining,
not `=== null`:

| Key | When present |
| --- | --- |
| `title`, `kind`, `type`, `responsiblePersonName` | always (`responsiblePersonName` may be `null`) |
| `description` | only when the model produced one |
| `agendaItemOrder` | only when a positive integer was inferred |
| `completionDuration`, `completionDurationUnit` | **only when `kind` is `ASSIGNMENT`** (may be `null` there); absent entirely on `RESOLUTION` |

```ts
type MinutesContent = {
  sections: Array<{ key: string; title?: string; content?: string }>;
};

type GeneratedDecision = {
  title: string;
  description?: string;
  kind: "RESOLUTION" | "ASSIGNMENT";
  type: "FOR_EXECUTION";
  agendaItemOrder?: number;
  responsiblePersonName?: string | null;
  completionDuration?: number | null;
  completionDurationUnit?: "DAYS" | "WEEKS" | "MONTHS" | null;
};
```

If you GET too early (**200**, not 409):

| Job state | `error.code` |
| --- | --- |
| `QUEUED` / `PROCESSING` | `MINUTES_NOT_READY` |
| `FAILED` | `GENERATION_FAILED` |
| Never submitted | `NOT_FOUND` (404) |

There is **no transcript** in this response.

---

## Errors (all endpoints)

```json
{
  "success": false,
  "error": {
    "code": "INVALID_VIDEO_URL",
    "message": "The video URL is invalid or cannot be accessed."
  }
}
```

Every failure on every route uses this envelope — including a mistyped path or
a wrong HTTP method — so `error.code` is always safe to read.

| Code | HTTP | When |
| --- | --- | --- |
| `UNAUTHORIZED` | 401 | Missing/wrong Bearer |
| `SERVICE_UNAVAILABLE` | 503 | API restarting or misconfigured. **Retryable** — back off and repeat the same call; it is not a job failure |
| `INVALID_VIDEO_URL` | 400 | Bad URL |
| `UNSUPPORTED_LANGUAGE` | 400 | `en` / `auto` |
| `INVALID_REQUEST` | 422 | Other body validation (e.g. illegal `meetingId`) |
| `REQUEST_FAILED` | 404 / 405 | Wrong **path** or method — a client bug, not an unknown meeting. Unknown `meetingId` is `NOT_FOUND`; check this first if every call fails |
| `NOT_FOUND` | 404 | Unknown `meetingId` |
| `MINUTES_NOT_READY` | 200 | GET minutes before complete |
| `GENERATION_FAILED` | 200 | GET minutes after failed job |
| `VIDEO_HOST_NOT_ALLOWED` | (job `FAILED`) | URL host not on AI allowlist |
| `VIDEO_TOO_LARGE` | (job `FAILED`) | Over 2 GB |
| `VIDEO_TOO_LONG` | (job `FAILED`) | Audio over ~3 hours |
| `TRANSCRIPTION_FAILED` / `MINUTES_GENERATION_FAILED` | (job `FAILED`) | Pipeline error |

---

## Suggested client flow

1. Ensure the video URL is HTTPS, signed long enough for **queue wait + download**,
   and hosted on the agreed storage host.
2. `POST` with `meetingId` + `videoUrl` + `"language": "ar"`.
3. If `202` and status already `COMPLETED`, skip to step 5 (idempotent replay).
4. Poll **status** until `COMPLETED` or `FAILED`. Do not busy-loop; expect long jobs.
5. On `COMPLETED`, `GET` minutes and persist. Safe to GET again later.
6. On `FAILED`, surface `error` to the user. **New `meetingId` or ops reset** —
   do not POST the same id again expecting a retry.

---

## Capacity (set expectations)

This v1 runs on a **small CPU VM**. ASR and the LLM cannot run at the same time.
Jobs are **FIFO, one at a time**. Mixed Arabic/English is supported; minutes are
Arabic (`language: "ar"` on GET). English words stay Latin **only if ASR wrote
them**. The model must not invent names, votes, or user ids.

---

## Curl (after go-live)

```bash
export AI_SERVICE_BASE_URL=https://minutes.example.com
export AI_SERVICE_API_KEY=...

curl -sS "$AI_SERVICE_BASE_URL/v1/meeting-minutes" \
  -H "Authorization: Bearer $AI_SERVICE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"meetingId":"cm123platformMeetingId","videoUrl":"https://storage.example.com/meetings/board.mp4","language":"ar"}'

curl -sS "$AI_SERVICE_BASE_URL/v1/meeting-minutes/cm123platformMeetingId/status" \
  -H "Authorization: Bearer $AI_SERVICE_API_KEY"

curl -sS "$AI_SERVICE_BASE_URL/v1/meeting-minutes/cm123platformMeetingId" \
  -H "Authorization: Bearer $AI_SERVICE_API_KEY"
```

---

## For platform vs AI ops

| Audience | Doc |
| --- | --- |
| **This file** — platform / frontend / backend clients | Contract and client behavior |
| [`required_intergration.md`](../required_intergration.md) | Original product spec |
| [`MEETING_MINUTES_API.md`](MEETING_MINUTES_API.md) | Design decisions on the CPU box |
| [`deploy/README.md`](../deploy/README.md) | How to put the API on the VM |
| [`TESTING.md`](TESTING.md) | Pytest (not a live meeting) |

**Blocker for real calls:** someone with SSH must apply `deploy/README.md`, set
`AI_SERVICE_API_KEY`, `COHEREX_VIDEO_ALLOWED_HOSTS`, DNS/TLS, then share
`AI_SERVICE_BASE_URL` and the key. Until then, clients can implement the flow
above against mocks or wait.
