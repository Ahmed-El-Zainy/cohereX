"""Resumable ffmpeg → ASR → LLM map/reduce processing pipeline."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import httpx

from .config import Settings
from .store import Job, JobStore, utc_now

logger = logging.getLogger(__name__)

SECTION_KEYS = ["meeting_info", "attendance", "introduction", "agenda", "main_items"]

# Qwen2.5 is a Chinese-trained multilingual model and leaks CJK tokens into
# Arabic prose under pressure -- a real run produced "الم发言人" for "the
# previous speaker". Three layers guard against it, because a prompt alone only
# discourages the behaviour:
#   1. these GBNF grammars make non-Arabic scripts structurally unreachable,
#   2. _language_rule() tells the model the same thing in words,
#   3. _strip_foreign_scripts() catches anything that still slips through.
# llama.cpp honours `grammar` on /v1/chat/completions; verified on the box.
_ARABIC_RANGES = (
    "[\\u0600-\\u06FF] | [\\u0750-\\u077F] | [\\u08A0-\\u08FF] | "
    "[\\uFB50-\\uFDFF] | [\\uFE70-\\uFEFF]"
)
PROSE_GRAMMAR = (
    "root ::= char+\n"
    f"char ::= {_ARABIC_RANGES} | [a-zA-Z0-9] | [ \\t\\r\\n] | "
    "[\\x21-\\x2F] | [\\x3A-\\x40] | [\\x5B-\\x60] | [\\x7B-\\x7E]\n"
)
# Same character set, but wrapped in a JSON-array structure -- so a malformed
# reply is impossible as well as an off-script one. Excludes the raw " and \
# that would break a JSON string; escapes are allowed explicitly.
JSON_ARRAY_GRAMMAR = (
    'root ::= "[" ws (obj (ws "," ws obj)*)? ws "]"\n'
    'obj  ::= "{" ws pair (ws "," ws pair)* ws "}"\n'
    'pair ::= str ws ":" ws val\n'
    'val  ::= str | num | "true" | "false" | "null"\n'
    'str  ::= "\\"" ch* "\\""\n'
    f'ch   ::= {_ARABIC_RANGES} | [a-zA-Z0-9] | [ ] | [\\x21] | [\\x23-\\x2F] | '
    '[\\x3A-\\x40] | [\\x5B] | [\\x5D-\\x60] | [\\x7B-\\x7E] | "\\\\" ["\\\\/bfnrt]\n'
    'num  ::= "-"? [0-9]+\n'
    'ws   ::= [ \\t\\n]*\n'
)

# CJK, kana, hangul, and their fullwidth/punctuation blocks. Latin is kept:
# board meetings legitimately mix in English terms.
_FOREIGN_SCRIPTS = re.compile(
    "[\u3000-\u303F\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF"
    "\uAC00-\uD7AF\uFF00-\uFFEF]+"
)


def _strip_foreign_scripts(text: str) -> str:
    """Last-resort net under the grammar. Mangles the word rather than shipping
    a Chinese one -- but it firing at all means layer 1 failed, so it warns."""
    if not text:
        return text
    cleaned = _FOREIGN_SCRIPTS.sub("", text)
    if cleaned != text:
        logger.warning(
            "Stripped non-Arabic script from model output; the grammar should "
            "have prevented this. Removed: %r",
            "".join(_FOREIGN_SCRIPTS.findall(text))[:80],
        )
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned
VALID_DECISION_KINDS = {"RESOLUTION", "ASSIGNMENT"}
VALID_DURATION_UNITS = {"DAYS", "WEEKS", "MONTHS"}


class FatalJobError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _extract_json(text: str) -> Any:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        starts = [index for index in (cleaned.find("{"), cleaned.find("[")) if index >= 0]
        if not starts:
            raise
        start = min(starts)
        closing = "}" if cleaned[start] == "{" else "]"
        end = cleaned.rfind(closing)
        if end < start:
            raise
        return json.loads(cleaned[start : end + 1])


def _write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def _write_json_atomic(path: Path, value: Any) -> None:
    _write_text_atomic(path, json.dumps(value, ensure_ascii=False, indent=2))


class ServiceManager:
    def __init__(self, settings: Settings):
        self.settings = settings

    def switch_to(self, target: str) -> None:
        if not self.settings.manage_services:
            return
        if target == "asr":
            stop, start, health_url, api_key = (
                self.settings.llm_service,
                self.settings.asr_service,
                f"{self.settings.vllm_url.rstrip('/')}/health",
                self.settings.vllm_api_key,
            )
        elif target == "llm":
            stop, start, health_url, api_key = (
                self.settings.asr_service,
                self.settings.llm_service,
                f"{self.settings.llm_url.rstrip('/')}/health",
                self.settings.llm_api_key,
            )
        else:
            raise ValueError(f"Unknown service target: {target}")

        subprocess.run(["systemctl", "stop", stop], check=True)
        subprocess.run(["systemctl", "start", start], check=True)
        self._wait_for_health(health_url, api_key)

    def _wait_for_health(self, url: str, api_key: str) -> None:
        deadline = time.monotonic() + self.settings.service_timeout_seconds
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        while time.monotonic() < deadline:
            try:
                if httpx.get(url, headers=headers, timeout=5).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(5)
        raise TimeoutError(f"Service did not become healthy: {url}")


class MeetingProcessor:
    def __init__(
        self,
        settings: Settings,
        store: JobStore,
        services: ServiceManager | None = None,
    ):
        self.settings = settings
        self.store = store
        self.services = services or ServiceManager(settings)

    def process(self, job: Job) -> dict[str, Any]:
        job_dir = self.settings.jobs_dir / job.meeting_id
        source = job_dir / "source.video"
        if not source.is_file():
            raise FatalJobError("INVALID_VIDEO_URL", "Downloaded video file is missing.")

        chunks = self._prepare_chunks(job, source, job_dir)
        self.services.switch_to("asr")
        transcript = self._transcribe(job, chunks, job_dir)

        self.store.update_progress(job.meeting_id, "GENERATING_MINUTES", 85)
        if not transcript.strip():
            return {
                "meetingId": job.meeting_id,
                "language": "ar",
                "content": {
                    "sections": [
                        {"key": key, "title": "", "content": ""}
                        for key in SECTION_KEYS
                    ]
                },
                "decisions": [],
                "generatedAt": utc_now(),
            }
        self.services.switch_to("llm")
        return self._generate(job, transcript, job_dir)

    def _prepare_chunks(self, job: Job, source: Path, job_dir: Path) -> list[Path]:
        chunks_dir = job_dir / "chunks"
        complete_marker = chunks_dir / ".complete"
        if complete_marker.is_file():
            chunks = sorted(chunks_dir.glob("*.wav"))
            if chunks:
                return chunks

        duration = self._probe_duration(source)
        if duration > self.settings.max_duration_seconds:
            raise FatalJobError(
                "VIDEO_TOO_LONG",
                f"Meeting audio exceeds the {self.settings.max_duration_seconds:g} second limit.",
            )

        chunks_dir.mkdir(parents=True, exist_ok=True)
        for stale in chunks_dir.glob("*.wav"):
            stale.unlink()
        # Chunk indexes are the checkpoint identity. If chunks are regenerated,
        # every downstream checkpoint must be discarded to avoid pairing old
        # text with different audio boundaries.
        for stale in (
            job_dir / "asr-checkpoint.jsonl",
            job_dir / "transcript.txt",
            job_dir / "sections.json",
        ):
            stale.unlink(missing_ok=True)
        shutil.rmtree(job_dir / "llm-slices", ignore_errors=True)
        command = [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "segment",
            "-segment_time",
            str(self.settings.chunk_seconds),
            "-reset_timestamps",
            "1",
            str(chunks_dir / "%06d.wav"),
        ]
        try:
            subprocess.run(command, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise FatalJobError("INVALID_VIDEO_URL", "Video could not be decoded by ffmpeg.") from exc

        chunks = sorted(chunks_dir.glob("*.wav"))
        if not chunks:
            raise FatalJobError("INVALID_VIDEO_URL", "Video contains no decodable audio.")
        complete_marker.write_text("ok\n", encoding="ascii")
        return chunks

    @staticmethod
    def _probe_duration(source: Path) -> float:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(source),
        ]
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            return float(result.stdout.strip())
        except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as exc:
            raise FatalJobError("INVALID_VIDEO_URL", "Video duration could not be read.") from exc

    def _transcribe(self, job: Job, chunks: list[Path], job_dir: Path) -> str:
        checkpoint = job_dir / "asr-checkpoint.jsonl"
        completed = self._read_asr_checkpoint(checkpoint)
        total = len(chunks)
        headers = (
            {"Authorization": f"Bearer {self.settings.vllm_api_key}"}
            if self.settings.vllm_api_key
            else {}
        )
        url = f"{self.settings.vllm_url.rstrip('/')}/v1/audio/transcriptions"

        with httpx.Client(timeout=self.settings.asr_timeout_seconds) as client:
            for index, chunk in enumerate(chunks):
                if index in completed:
                    continue
                with chunk.open("rb") as audio:
                    response = client.post(
                        url,
                        headers=headers,
                        files={"file": (chunk.name, audio, "audio/wav")},
                        data={"model": self.settings.vllm_model, "language": "ar"},
                    )
                response.raise_for_status()
                text = response.json().get("text", "").strip()
                record = {"index": index, "text": text}
                with checkpoint.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    os.fsync(output.fileno())
                completed[index] = text
                progress = 10 + int(75 * (index + 1) / total)
                self.store.update_progress(job.meeting_id, "TRANSCRIBING", progress)

        transcript = "\n".join(completed.get(index, "") for index in range(total)).strip()
        transcript_path = job_dir / "transcript.txt"
        _write_text_atomic(transcript_path, transcript)
        return transcript

    @staticmethod
    def _read_asr_checkpoint(path: Path) -> dict[int, str]:
        completed: dict[int, str] = {}
        if not path.is_file():
            return completed
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                completed[int(record["index"])] = str(record.get("text", ""))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                logger.warning("Ignoring malformed ASR checkpoint line in %s", path)
        return completed

    def _generate(self, job: Job, transcript: str, job_dir: Path) -> dict[str, Any]:
        slices = self._slice_transcript(transcript)
        notes_dir = job_dir / "llm-slices"
        notes_dir.mkdir(exist_ok=True)
        notes: list[str] = []
        all_decisions: list[dict[str, Any]] = []

        total_calls = max(1, len(slices) * 2 + 1)
        done_calls = 0
        for index, text_slice in enumerate(slices):
            note_path = notes_dir / f"{index:04d}-notes.txt"
            decision_path = notes_dir / f"{index:04d}-decisions.json"

            if note_path.is_file():
                note = note_path.read_text(encoding="utf-8")
            else:
                note = self._ask(self._notes_prompt(text_slice), max_tokens=700,
                                 grammar=PROSE_GRAMMAR)
                _write_text_atomic(note_path, note)
            notes.append(note)
            done_calls += 1
            self._llm_progress(job.meeting_id, done_calls, total_calls)

            if decision_path.is_file():
                decisions = json.loads(decision_path.read_text(encoding="utf-8"))
            else:
                raw = self._ask(self._decisions_prompt(text_slice),
                                grammar=JSON_ARRAY_GRAMMAR)
                parsed = _extract_json(raw)
                decisions = self._validate_decisions(parsed)
                _write_json_atomic(decision_path, decisions)
            all_decisions.extend(decisions)
            done_calls += 1
            self._llm_progress(job.meeting_id, done_calls, total_calls)

        sections_path = job_dir / "sections.json"
        if sections_path.is_file():
            sections = json.loads(sections_path.read_text(encoding="utf-8"))
        else:
            reduced_notes = self._reduce_notes(notes, notes_dir)
            raw_sections = self._ask(self._sections_prompt(reduced_notes),
                                     grammar=JSON_ARRAY_GRAMMAR)
            sections = self._validate_sections(_extract_json(raw_sections))
            _write_json_atomic(sections_path, sections)
        self._llm_progress(job.meeting_id, total_calls, total_calls)

        return {
            "meetingId": job.meeting_id,
            "language": "ar",
            "content": {"sections": sections},
            "decisions": self._deduplicate_decisions(all_decisions),
            "generatedAt": utc_now(),
        }

    def _ask(self, prompt: str, max_tokens: int | None = None,
             grammar: str | None = None) -> str:
        headers = (
            {"Authorization": f"Bearer {self.settings.llm_api_key}"}
            if self.settings.llm_api_key
            else {}
        )
        payload: dict[str, Any] = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or self.settings.llm_max_tokens,
            "temperature": 0.1,
        }
        if grammar:
            payload["grammar"] = grammar
        response = httpx.post(
            f"{self.settings.llm_url.rstrip('/')}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.settings.llm_timeout_seconds,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _reduce_notes(self, notes: list[str], notes_dir: Path) -> list[str]:
        """Hierarchically compact notes so the final prompt never drops the tail."""
        limit = max(1000, self.settings.transcript_slice_chars)
        current = notes
        round_number = 0
        while sum(len(note) + 20 for note in current) > limit:
            groups: list[list[str]] = []
            group: list[str] = []
            size = 0
            for note in current:
                if group and size + len(note) + 20 > limit:
                    groups.append(group)
                    group, size = [], 0
                group.append(note)
                size += len(note) + 20
            if group:
                groups.append(group)

            reduced: list[str] = []
            for index, grouped_notes in enumerate(groups):
                path = notes_dir / f"reduce-{round_number:02d}-{index:04d}.txt"
                if path.is_file():
                    compacted = path.read_text(encoding="utf-8")
                else:
                    compacted = self._ask(
                        self._reduce_prompt(grouped_notes),
                        max_tokens=900,
                        grammar=PROSE_GRAMMAR,
                    )
                    _write_text_atomic(path, compacted)
                reduced.append(compacted)
            current = reduced
            round_number += 1
            if round_number > 8:
                raise ValueError("Could not compact meeting notes into the LLM context window")
        return current

    def _slice_transcript(self, transcript: str) -> list[str]:
        limit = max(1000, self.settings.transcript_slice_chars)
        paragraphs = transcript.splitlines() or [transcript]
        slices: list[str] = []
        current: list[str] = []
        size = 0
        for paragraph in paragraphs:
            if current and size + len(paragraph) + 1 > limit:
                slices.append("\n".join(current))
                current, size = [], 0
            while len(paragraph) > limit:
                if current:
                    slices.append("\n".join(current))
                    current, size = [], 0
                slices.append(paragraph[:limit])
                paragraph = paragraph[limit:]
            current.append(paragraph)
            size += len(paragraph) + 1
        if current:
            slices.append("\n".join(current))
        return slices or [""]

    def _llm_progress(self, meeting_id: str, completed: int, total: int) -> None:
        self.store.update_progress(
            meeting_id, "GENERATING_MINUTES", 85 + int(14 * completed / total)
        )

    @staticmethod
    def _language_rule() -> str:
        return (
            "اكتب السرد بالعربية الفصحى. الاجتماع قد يمزج العربية والإنجليزية. "
            "احتفظ بالمصطلح اللاتيني فقط إذا ظهر بالحروف اللاتينية في النص. "
            "إذا ظهر منقولاً صوتياً بالعربية فاحتفظ به كما هو ولا تخمّن اختصاراً لاتينياً. "
            "اكتب بالحروف العربية فقط، ويجوز إبقاء المصطلحات الإنجليزية بالحروف "
            "اللاتينية. لا تستخدم الحروف الصينية أو اليابانية أو الكورية إطلاقاً. "
            "لا تخترع أسماء أو تواريخ أو حضوراً أو ملاك إجراءات أو مواعيد."
        )

    def _notes_prompt(self, text: str) -> str:
        return (
            "استخرج ملاحظات واقعية موجزة من هذا الجزء من نص اجتماع مجلس إدارة. "
            "غطِّ معلومات الاجتماع والحضور والمقدمة وبنود جدول الأعمال والنقاشات الرئيسية. "
            "لا تضف قرارات في هذه الملاحظات؛ ستُستخرج منفصلة. "
            f"{self._language_rule()}\n\nالنص:\n{text}"
        )

    def _decisions_prompt(self, text: str) -> str:
        return (
            "استخرج فقط القرارات المعتمدة والتكليفات الصريحة من هذا الجزء. "
            "أعد JSON array فقط، بلا Markdown. كل عنصر: "
            '{"title":"...","description":"... أو احذفها","kind":"RESOLUTION أو ASSIGNMENT",'
            '"type":"FOR_EXECUTION","agendaItemOrder":1 أو احذفها,'
            '"responsiblePersonName":"الاسم" أو null,'
            '"completionDuration":عدد أو null,"completionDurationUnit":"DAYS أو WEEKS أو MONTHS" أو null}. '
            "إذا لا توجد قرارات أعد []. لا تعدّ النقاش قراراً. حقول المدة للتكليف فقط. "
            f"{self._language_rule()}\n\nالنص:\n{text}"
        )

    def _sections_prompt(self, notes: list[str]) -> str:
        joined = "\n\n--- جزء ---\n".join(notes)
        return (
            "حوّل الملاحظات التالية إلى JSON array فقط لمحضر اجتماع. "
            "يجب أن تكون العناصر بالترتيب والمفاتيح التالية فقط: "
            "meeting_info, attendance, introduction, agenda, main_items. "
            'شكل العنصر {"key":"...","title":"...","content":"Markdown"}. '
            "استخدم Markdown وجداول GFM عند الحاجة. لا تضع القرارات في الأقسام. "
            "إذا لا توجد معلومة مدعومة استخدم محتوى فارغاً. "
            f"{self._language_rule()}\n\nالملاحظات:\n{joined}"
        )

    def _reduce_prompt(self, notes: list[str]) -> str:
        joined = "\n\n--- جزء ---\n".join(notes)
        return (
            "ادمج الملاحظات التالية في سجل واقعي مضغوط. حافظ على كل بند جدول أعمال "
            "واسم وتاريخ وحضور ونقطة نقاش مذكورة؛ احذف التكرار فقط ولا تستنتج معلومات. "
            "لا تضف القرارات، فهي تُعالج منفصلة. اجعل الناتج موجزاً ليسع مرحلة لاحقة. "
            f"{self._language_rule()}\n\nالملاحظات:\n{joined}"
        )

    @staticmethod
    def _validate_sections(value: Any) -> list[dict[str, str]]:
        if isinstance(value, dict):
            value = value.get("sections")
        if not isinstance(value, list):
            raise ValueError("LLM sections response is not an array")
        by_key: dict[str, dict[str, str]] = {}
        for item in value:
            if not isinstance(item, dict) or item.get("key") not in SECTION_KEYS:
                continue
            key = item["key"]
            by_key[key] = {
                "key": key,
                "title": _strip_foreign_scripts(str(item.get("title") or "")),
                "content": _strip_foreign_scripts(str(item.get("content") or "")),
            }
        return [
            by_key.get(key, {"key": key, "title": "", "content": ""})
            for key in SECTION_KEYS
        ]

    @staticmethod
    def _validate_decisions(value: Any) -> list[dict[str, Any]]:
        if isinstance(value, dict):
            value = value.get("decisions")
        if not isinstance(value, list):
            raise ValueError("LLM decisions response is not an array")
        valid: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            title = _strip_foreign_scripts(str(item.get("title") or "")).strip()
            kind = item.get("kind")
            if not title or kind not in VALID_DECISION_KINDS:
                continue
            decision: dict[str, Any] = {
                "title": title,
                "kind": kind,
                "type": "FOR_EXECUTION",
            }
            description = _strip_foreign_scripts(
                str(item.get("description") or "")
            ).strip()
            if description:
                decision["description"] = description
            order = item.get("agendaItemOrder")
            if isinstance(order, int) and order > 0:
                decision["agendaItemOrder"] = order
            owner = item.get("responsiblePersonName")
            decision["responsiblePersonName"] = (
                _strip_foreign_scripts(str(owner)).strip() or None if owner else None
            )
            if kind == "ASSIGNMENT":
                duration = item.get("completionDuration")
                unit = item.get("completionDurationUnit")
                decision["completionDuration"] = (
                    duration if isinstance(duration, int) and duration > 0 else None
                )
                decision["completionDurationUnit"] = (
                    unit if unit in VALID_DURATION_UNITS else None
                )
            valid.append(decision)
        return valid

    @staticmethod
    def _deduplicate_decisions(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[tuple[str, str]] = set()
        result: list[dict[str, Any]] = []
        for decision in decisions:
            normalized = re.sub(r"\W+", "", decision["title"].casefold())
            key = (decision["kind"], normalized)
            if key not in seen:
                seen.add(key)
                result.append(decision)
        return result

    @staticmethod
    def delete_video(job_dir: Path) -> None:
        for name in ("source.video", "source.video.part"):
            (job_dir / name).unlink(missing_ok=True)
