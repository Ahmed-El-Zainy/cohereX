# Session log: deploying CohereX + a board-secretary LLM

A chronological account of one working session that took CohereX from "just
a local CLI tool" to a deployed, tested, two-service API (ASR + LLM) with a
board-secretary-scoped task set. This is a narrative record, not a
how-to — for the how-to, see the docs it links to throughout.

## 1. Starting point

Ask: deploy the CohereX app to a specific server (`root@<SERVER_IP>`).
CohereX itself ([README.md](../README.md)) is a Python CLI/library for
speech transcription (word-level timestamps, diarization, SRT/VTT/etc.
output) built on the Cohere Transcribe ASR model — not a web app, no
existing Dockerfile or deploy script.

**Reality check first**: SSH'd in and found the target box has **no GPU** —
a 2-vCPU / 8GB RAM VM (Cirrus Logic virtual VGA, Skylake CPU with AVX-512).
That ruled out the obvious path and set the shape of everything that
followed: every subsequent piece of this deployment had to be re-fought for
CPU-only, memory-constrained hardware.

## 2. ASR server: vLLM on CPU

Full account: [DEPLOYMENT.md](DEPLOYMENT.md).

Built vLLM from source with `VLLM_TARGET_DEVICE=cpu` (the PyPI wheel is
GPU-only) — a ~1 hour build compiling oneDNN plus vLLM's own CPU kernels,
with individual compile units spiking to 2.5-3.6GB RSS on an 8GB box.

Then installed CohereX itself, which triggered the first real bug: `pip
install coherex[all]` re-resolved vLLM's own dependency and — because the
shallow git clone had no version tags — decided the freshly-built CPU vLLM
(`0.1.dev1+...`) didn't satisfy `vllm>=0.19.0`, silently replaced it with
the GPU-oriented PyPI wheel, and dragged in a pile of CUDA packages.
Fixed by rebuilding vLLM with `--no-deps` after removing the wrong wheel.

Then hit three more failures getting the systemd service to actually start:

1. **torchaudio ABI break** — PyPI's torchaudio stopped at 2.11.0, incompatible
   with the required torch 2.13.0+cpu. Fixed two places that needed it but
   didn't need its *compiled extension*: made `coherex/alignment.py`'s
   import lazy (Arabic alignment doesn't need it anyway), and ported vLLM's
   own `melscale_fbanks` helper (pure math, no compiled code) out of
   torchaudio entirely into vLLM's local source tree.
2. **`--gpu-memory-utilization` misunderstanding** — on the CPU backend this
   flag is a ceiling on weights+KV-cache+overhead *combined*, not KV-cache
   alone. Needed `0.65`, not `0.1`/`0.3` (which were below what the 3.85GB
   checkpoint alone required).
3. **OOM during compile warmup** — fixed with `--enforce-eager` (skips
   torch.compile) plus an 8GB swapfile as a safety margin.

End state: `coherex-vllm` systemd service, `CohereLabs/cohere-transcribe-arabic-07-2026`
served on port 8000, API-key-gated, `0.0.0.0` (reachable from anywhere, per
the stated use case of calling it "from any project").

## 3. CLI defaults changed to match the deployment's actual use case

Since the model/server are Arabic-focused, changed `coherex/__main__.py` and
`coherex/asr.py` defaults: model → `cohere-transcribe-arabic-07-2026`,
`--language` → `ar` (was required with no default), `--vad_method` → `silero`,
`--max_line_width`/`--max_line_count` → `42`/`2`, `--output_dir` → `out-ar/`.
README updated to match.

## 4. main.py: testing the real API

Full account: [TESTING.md](TESTING.md).

A smoke-test script, evolved in stages across the session:
- Started as a `/health` + `/v1/models` check plus a basic transcription run
- Added a **live progress bar** — the ASR box takes 1-2.5 minutes *per audio
  chunk*, so a naive progress indicator would sit at 0% for minutes and look
  hung. Solved with a background-thread ticker that keeps the elapsed timer
  visibly moving between coherex's own (coarse) progress callbacks.
- Added a `.env`-based config loader (no external dependency) so it runs
  with zero flags once `.env` is filled in from `.env.example`
- Extended to accept arbitrary audio file paths as positional args, not
  just scanning `samples/`
- Extended to write **all six** CohereX output formats
  (`.txt/.srt/.vtt/.tsv/.json/.aud`) via the project's real writer classes —
  which surfaced a real bug in `coherex/utils.py`: `get_writer("all", ...)`
  was silently excluding `.aud` from "all". Fixed at the source
  (`OUTPUT_FORMATS` is now one canonical list), benefiting the real
  `coherex` CLI too, not just this script.

Also tuned `coherex/vllm_backend.py`'s `VLLMBackend` defaults down
(`max_workers` 8→2, `timeout` 120s→300s) after the first real transcription
attempt timed out — 8 concurrent requests against a 2-vCPU server was pure
contention, not throughput.

## 5. LLM server: a second service, same box, different constraints

Full account: [LLM_DEPLOYMENT.md](LLM_DEPLOYMENT.md).

Asked what LLM(s) could post-process transcript output. Flagged upfront that
the ASR server alone already runs the box at ~7GB/7.8GB RAM with swap
active — no room for a second warm model. User chose: self-host (not a
hosted API), accept a **toggle** (one service warm at a time, free) over
resizing the box or provisioning a second one.

Chose **llama.cpp** over vLLM for the LLM itself — vLLM is GPU-throughput
oriented and its CPU build was the whole fight in §2; llama.cpp is built
around quantized models from the start. Build took ~9 minutes (vs. ~1 hour
for vLLM), peak memory under 1GB (vs. 2.5-3.6GB spikes).

Model: **Qwen2.5-3B-Instruct**, Q4_K_M GGUF quantization (2.0GB), chosen for
Arabic coverage relative to size. Loads in ~6 seconds, ~2GB RAM resident.

Built `scripts/toggle_server.sh` (`asr`/`llm`/`status`) to flip between the
two systemd services over SSH, and `llm_client.py` mirroring `main.py`'s
`.env`-based pattern, with `ask()`/`TASK_PROMPTS` for turning a transcript
into a summary/analysis via the LLM's OpenAI-compatible `/v1/chat/completions`.

**Real limitation found in testing**: asked the model whether the transcript
mentioned "BIC" (Board Investment Committee) — it does, but as a phonetic
Arabic transliteration ("البي اي سي") rather than the Latin acronym, and the
3B model missed the connection. Documented as a known limitation, not
silently smoothed over.

**Real operational quirk found**: stopping `coherex-vllm` during a toggle
sat in `deactivating (stop-sigterm)` for over a minute in testing — vLLM's
multiprocessing workers don't all react to SIGTERM immediately. Not a bug,
but budget for it.

## 6. pipeline.py: the two services as one sequential command

Full account: [PIPELINE.md](PIPELINE.md).

Glued `main.py`'s `transcribe_files()` and `llm_client.py`'s `ask()`/
`TASK_PROMPTS` together with the toggle script in between:
`toggle→asr → transcribe → toggle→llm → run task(s) → toggle→asr`. No
duplicated logic — it imports and calls the existing functions.

**Verified with a real, full run**: toggle (no-op, already warm) →
transcribed the 3-minute sample (9:04) → toggled to LLM → generated a
summary → toggled back to ASR → confirmed healthy again. ~11 minutes total,
now documented as the realistic expectation rather than something to be
surprised by.

## 7. Scoping the LLM's job: the Board Secretary document

Asked to review `assets/مهام أمين سر مجلس الإدارة.docx` — a real Saudi
corporate-governance job description for a board secretary (أمين سر مجلس
الإدارة), spanning pre-meeting prep, in-meeting documentation, post-meeting
follow-up, decision tracking, governance records, and board↔management
coordination, across 8 sections.

Extracted its text (no `python-docx` installed; read the raw
`word/document.xml` inside the `.docx` zip directly instead of adding a
dependency for a one-time read), translated and mapped it against what an
agent that only ever sees a **transcript** — no calendar, no email, no
document archive, no cross-meeting persistence — can actually do.

Redesigned `llm_client.py`'s `TASK_PROMPTS` around that mapping:
- `summary` (kept)
- **`minutes_draft`** (new) — 7 structured sections: attendees, meeting
  time, discussion, decisions (exact wording), votes, objections,
  conflict-of-interest disclosures, each explicitly `"Not stated in
  transcript"` if absent rather than invented
- **`decisions_log`** (new) — every decision, quoted/closely paraphrased
- `action_items` (tightened) — now requires `Action` / `Owner` /
  `Target date` fields explicitly, matching the docx's own phrasing
- **`conflicts_of_interest`** (new)
- `qa` (kept)

Documented explicitly, in [LLM_DEPLOYMENT.md's Board Secretary task
scope](LLM_DEPLOYMENT.md#board-secretary-task-scope), what's *out* of scope
and why: scheduling, invitations, document collection, quorum checks
against real bylaws, signatures, and any registry that needs to persist
across multiple meetings (a real Board Action Tracker) — these need
integration work beyond "ask an LLM about one transcript," not a bigger
model.

**Verified against the real transcript**: `minutes_draft`, `decisions_log`,
and `conflicts_of_interest` all correctly reported "not stated" / "none
found" for fields this particular clip doesn't cover, rather than
fabricating content. Caught one cosmetic quirk (`decisions_log` repeating
its answer with a stray code-fence) and documented it rather than hiding it.

## Final architecture

```
                    ┌─────────────────────────────────────┐
                    │   Server (<SERVER_IP>, 2 vCPU/8GB)    │
                    │                                       │
  audio file ──────▶│  coherex-vllm (port 8000)             │──▶ transcript
                    │  CohereLabs/cohere-transcribe-arabic  │
                    │  [warm by default]                    │
                    │                                       │
  transcript ───────│  coherex-llm (port 8001)               │──▶ minutes /
                    │  Qwen2.5-3B-Instruct (llama.cpp)      │    decisions /
                    │  [off by default — toggle to use]     │    action items
                    └─────────────────────────────────────┘
                       ▲ only one of these two is ever warm ▲
                         scripts/toggle_server.sh flips it
```

**Repo additions this session**: `main.py`, `llm_client.py`, `pipeline.py`,
`scripts/toggle_server.sh`, `.env.example`, and
`docs/{DEPLOYMENT,TESTING,LLM_DEPLOYMENT,PIPELINE,SESSION_LOG}.md`. Real
credentials live only in the gitignored `.env` — every doc uses
`<SERVER_IP>`/`<...KEY>` placeholders.

## Where to start, depending on what you need

- **Deploy this from scratch on a new box**: [DEPLOYMENT.md](DEPLOYMENT.md), then [LLM_DEPLOYMENT.md](LLM_DEPLOYMENT.md)
- **Just call the already-deployed API**: [TESTING.md](TESTING.md) (`main.py`) and [LLM_DEPLOYMENT.md §5](LLM_DEPLOYMENT.md#5-using-it) (`llm_client.py`)
- **Transcribe-then-analyze in one command**: [PIPELINE.md](PIPELINE.md) (`pipeline.py`)
- **What the LLM is actually scoped to do and why**: [LLM_DEPLOYMENT.md's Board Secretary task scope](LLM_DEPLOYMENT.md#board-secretary-task-scope)
