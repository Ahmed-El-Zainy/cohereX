
# AI Meeting Minutes API Integration Guide

This document describes the three endpoints required to generate meeting minutes from a video:

1. Submit the `meetingId` and video URL.
2. Check the processing status using the same `meetingId`.
3. Retrieve generated minutes content and extracted decisions.

## Shared API details

- Base URL: `{AI_SERVICE_BASE_URL}`
- Content type: `application/json`
- Authentication: `Authorization: Bearer {AI_SERVICE_API_KEY}`
- All timestamps use ISO 8601 UTC, for example `2026-09-07T10:30:00Z`.
- `meetingId` is supplied in the submission request and must be used by all three endpoints. The AI service does not create a second meeting or job ID.
- A `meetingId` identifies one generation job within an organization.
- The final result must remain available for repeated reads. A successful retry must return the same result and must not start a new job.

## 1. Submit meeting video

Starts asynchronous transcription and minutes generation.

### Request

`POST /v1/meeting-minutes`

```json
{
  "meetingId": "cm123platformMeetingId",
  "videoUrl": "https://storage.example.com/meetings/board-meeting.mp4",
  "language": "ar"
}
```

| Field         | Type                        | Required | Description                                                                                        |
| ------------- | --------------------------- | -------- | -------------------------------------------------------------------------------------------------- |
| `meetingId` | string                      | Yes      | Existing meeting ID supplied by the caller.                                                        |
| `videoUrl`  | string                      | Yes      | A URL that the AI service can download. If signed, it must remain valid long enough for ingestion. |
| `language`  | `ar`, `en`, or `auto` | No       | Spoken/output language. Defaults to`auto`.                                                       |

### Accepted response

HTTP `202 Accepted`

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

Submitting the same `meetingId` again must not create another job. The AI service should return the existing job and its current status. Regeneration can be added later as a separate, explicit feature.

## 2. Check generation status

Call this endpoint until the status is `COMPLETED` or `FAILED`.

### Request

`GET /v1/meeting-minutes/{meetingId}/status`

No request body.

### Processing response

HTTP `200 OK`

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

### Completed response

```json
{
  "success": true,
  "data": {
    "meetingId": "cm123platformMeetingId",
    "status": "COMPLETED",
    "progress": 100,
    "stage": "COMPLETED",
    "updatedAt": "2026-09-07T10:36:10Z"
  }
}
```

### Failed response

The endpoint still returns HTTP `200` because the status lookup succeeded.

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

Allowed `status` values:

- `QUEUED`
- `PROCESSING`
- `COMPLETED`
- `FAILED`

`progress` is an integer from `0` to `100`.

## 3. Get generated minutes

Call this endpoint only after the status is `COMPLETED`.

### Request

`GET /v1/meeting-minutes/{meetingId}`

No request body.

### Success response

HTTP `200 OK`

```json
{
  "success": true,
  "data": {
    "meetingId": "cm123platformMeetingId",
    "language": "en",
    "content": {
      "sections": [
        {
          "key": "meeting_info",
          "title": "Meeting details",
          "content": "| Day & date | Place | Time |\n| --- | --- | --- |\n| Monday, 07/09/2026 | Main Boardroom | From 10:00 to 11:30 |"
        },
        {
          "key": "attendance",
          "title": "Attendance",
          "content": "| # | Name | Position | Attendance |\n| --- | --- | --- | --- |\n| 1 | Ahmed Ali | Chair | Present |"
        },
        {
          "key": "introduction",
          "title": "Introduction",
          "content": "The chair opened the meeting and welcomed the attendees."
        },
        {
          "key": "agenda",
          "title": "Agenda",
          "content": "| # | Item |\n| --- | --- |\n| 1 | Approve the annual budget |"
        },
        {
          "key": "main_items",
          "title": "Main items",
          "content": "**1. Approve the annual budget**\n\nThe board reviewed the proposed annual budget and discussed the projected operating costs."
        }
      ]
    },
    "decisions": [
      {
        "title": "Approve the 2027 annual budget",
        "description": "Approve the annual budget presented during the meeting.",
        "kind": "RESOLUTION",
        "type": "FOR_EXECUTION",
        "agendaItemOrder": 1,
        "responsiblePersonName": null,
        "completionDuration": null,
        "completionDurationUnit": null
      },
      {
        "title": "Prepare the final implementation plan",
        "description": "Prepare and circulate the final implementation plan.",
        "kind": "ASSIGNMENT",
        "type": "FOR_EXECUTION",
        "agendaItemOrder": 1,
        "responsiblePersonName": "Ahmed Ali",
        "completionDuration": 14,
        "completionDurationUnit": "DAYS"
      }
    ],
    "generatedAt": "2026-09-07T10:36:10Z"
  }
}
```

### Minutes content requirements

The `content` field intentionally matches the existing `POST /api/minutes` request format:

```ts
type MinutesContent = {
  sections: Array<{
    key: string;
    title?: string;
    content?: string;
  }>;
};
```

- `content` must be a JSON object, not a JSON-encoded string.
- `sections` must always be an array, even when no content was detected.
- Section `content` is Markdown text. Tables use GitHub-Flavored Markdown.
- Use the standard keys, in this order: `meeting_info`, `attendance`, `introduction`, `agenda`, `main_items`.
- Do not place `decisions` inside `content.sections`; return them in the top-level `decisions` array.
- Do not invent names, dates, attendance, decisions, owners, or deadlines. Use `null` or omit optional text when it cannot be supported by the recording.

### Decision requirements

```ts
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

- `title` is required and must contain the agreed action or resolution, not discussion text.
- Use `RESOLUTION` for an approved resolution and `ASSIGNMENT` for an action assigned to a person.
- `completionDuration` and `completionDurationUnit` apply only to `ASSIGNMENT`.
- Return the responsible person's name in `responsiblePersonName`. Do not return a platform user ID.
- `agendaItemOrder` refers to the one-based agenda item number.
- The AI must not generate voting results, status, internal user IDs, or database IDs.

## Error handling

Use the following error format for every endpoint:

```json
{
  "success": false,
  "error": {
    "code": "INVALID_VIDEO_URL",
    "message": "The video URL is invalid or cannot be accessed."
  }
}
```
