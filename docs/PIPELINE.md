# The full pipeline: transcribe, then analyze

`pipeline.py` (repo root) runs both halves of this deployment as one
sequential command — audio in, a board-secretary-style deliverable out
(minutes draft, decisions log, action items, etc. — see
[LLM_DEPLOYMENT.md's Board Secretary task scope](LLM_DEPLOYMENT.md#board-secretary-task-scope))
— instead of running `main.py` and `llm_client.py` (and the toggle between
them) by hand every time. This documents how it works, how to run it, and
how to fall back to the manual steps if something goes wrong.

Read these first if you haven't:
- [DEPLOYMENT.md](DEPLOYMENT.md) — the ASR (vLLM) server
- [TESTING.md](TESTING.md) — `main.py`, the ASR API test client
- [LLM_DEPLOYMENT.md](LLM_DEPLOYMENT.md) — the LLM (llama.cpp) server, the
  toggle, `llm_client.py`

`pipeline.py` doesn't reimplement any of those — it imports `main.py`'s
`transcribe_files()` and `llm_client.py`'s `ask()`/`TASK_PROMPTS` directly,
and shells out to `scripts/toggle_server.sh` for the switch. Same code
paths as doing each step by hand, just sequenced with the toggle wired in
between.

## Why "sequential parts," not "both at once"

The two servers can't run warm at the same time on this box — not enough
RAM (see LLM_DEPLOYMENT.md's "Why this needed its own design decision"
section). So the pipeline is genuinely sequential, not just organized that
way for convenience:

```
toggle → asr    (stop LLM if running, start ASR — ~1-2 min if cold)
  ↓
transcribe audio                                  (main.py's transcribe_files)
  ↓
toggle → llm    (stop ASR, start LLM — a few seconds)
  ↓
run LLM task(s) on the transcript                 (llm_client.py's ask)
  ↓
toggle → asr    (stop LLM, restart ASR — back to the default resting state)
```

Each toggle actually stops the other service — this is real infrastructure
switching over SSH, not a simulated wait. Budget the timing notes below,
don't assume it's instant.

## 1. Prerequisites

Everything TESTING.md and LLM_DEPLOYMENT.md already require:

- Both servers deployed (`coherex-vllm` and `coherex-llm` systemd units
  exist on the box)
- `.env` filled in with all of: `COHEREX_VLLM_URL`, `COHEREX_VLLM_API_KEY`,
  `COHEREX_LLM_URL`, `COHEREX_LLM_API_KEY`, `COHEREX_SERVER_SSH`
- SSH access to the box working key-based (no interactive password prompt —
  `scripts/toggle_server.sh` calls `ssh` directly, so it needs to connect
  non-interactively)
- CohereX installed locally with the ML stack (same requirement as
  `main.py` — VAD runs on your machine, not the server)

## 2. Running it

```bash
# Default: transcribe, then summarize
python pipeline.py samples/saudi_business_03min.mp3

# Draft structured board minutes instead — attendees, decisions (exact
# wording), votes, objections, conflicts of interest, one section each
python pipeline.py samples/saudi_business_03min.mp3 --task minutes_draft --max_tokens 1200

# Extract just the decisions, or just the action items (owner + target date each)
python pipeline.py samples/saudi_business_03min.mp3 --task decisions_log
python pipeline.py samples/saudi_business_03min.mp3 --task action_items

# Flag any conflict-of-interest mentions or recusals
python pipeline.py samples/saudi_business_03min.mp3 --task conflicts_of_interest

# Multiple tasks in one run — transcribes once, then runs each LLM task
# against that same transcript (no need to re-transcribe per task)
python pipeline.py samples/saudi_business_03min.mp3 --task decisions_log --task action_items

# Ask a specific question about the transcript
python pipeline.py samples/saudi_business_03min.mp3 --task qa \
  --question "Which committee approves the largest investments?"

# Force English + the base model instead of this fork's Arabic defaults
python pipeline.py samples/saudi_business_03min.mp3 \
  --language en --model CohereLabs/cohere-transcribe-03-2026

# Leave the LLM server warm afterward instead of switching back to ASR
# (e.g. if you're about to run several llm_client.py calls by hand)
python pipeline.py samples/saudi_business_03min.mp3 --no_restore

# You're managing server state yourself (e.g. already toggled manually) —
# skip the automatic toggle calls, just wait for whatever's already running
python pipeline.py samples/saudi_business_03min.mp3 --skip_toggle
```

Every flag:

| Flag | Default | Purpose |
|---|---|---|
| `audio` (positional) | — (required) | one audio file to transcribe and analyze |
| `--task` | `summary` | repeatable — `summary`, `minutes_draft`, `decisions_log`, `action_items`, `conflicts_of_interest`, and/or `qa`; each runs against the same transcript. See [LLM_DEPLOYMENT.md's Board Secretary task scope](LLM_DEPLOYMENT.md#board-secretary-task-scope) for what each one does and why these six |
| `--question` | — | required if `qa` is one of the tasks |
| `--language` | `ar` | passed through to transcription |
| `--vad_method` | `silero` | passed through to transcription |
| `--model` | `CohereLabs/cohere-transcribe-arabic-07-2026` | ASR model name — must match what the server actually loaded |
| `--vllm_url` / `--vllm_api_key` | `.env` | ASR server |
| `--llm_url` / `--llm_api_key` | `.env` | LLM server |
| `--max_tokens` | `800` | max tokens per LLM task response — raise for `minutes_draft` |
| `--output_dir` | `pipeline-out/` | where the transcript and each task's output land |
| `--skip_toggle` | off | don't call `toggle_server.sh` — assume the right service is already up at each step |
| `--no_restore` | off | skip the final toggle back to ASR — leaves the LLM server warm |

## 3. Reading the output

For `name.ext` with `--task decisions_log --task action_items`,
`--output_dir` (`pipeline-out/` by default) gets:

- `name.txt` — the plain-text transcript (same writer `main.py` uses,
  `--output_format txt` only — this pipeline doesn't produce srt/vtt/json/etc.;
  run `main.py` directly first if you want every format, then
  `llm_client.py --transcript <path>` against that output)
- `name.decisions_log.txt` — the decisions-log task's response
- `name.action_items.txt` — the action-items task's response
- one `name.<task>.txt` per `--task` you passed (e.g. `name.minutes_draft.txt`,
  `name.summary.txt`, `name.conflicts_of_interest.txt`, `name.qa.txt`)

Console output shows both steps as they happen — the transcription
progress bar (see TESTING.md's [Progress bar](TESTING.md#progress-bar)
section), then each LLM task's response printed as it completes.

## 4. Timing expectations (measured on the reference box)

Don't run this expecting a fast turnaround — it's two multi-minute
operations back to back, plus toggle overhead:

- **Toggle to ASR** (if the LLM was warm): ~1-2 minutes. `coherex-vllm`
  loading a ~4GB checkpoint takes time even from local disk cache — see
  DEPLOYMENT.md §9.
- **Transcription**: ~1-2.5 minutes *per 30-second audio chunk* on this
  CPU-only box (DEPLOYMENT.md §9) — a 3-minute clip took about 7 minutes
  end to end in testing.
- **Toggle to LLM**: a few seconds to start, but **stopping `coherex-vllm`
  itself can take over a minute** — it doesn't always react to SIGTERM
  promptly (LLM_DEPLOYMENT.md §4). This is the single most unpredictable
  step timing-wise.
- **Each LLM task**: well under a minute for a transcript this size
  (LLM_DEPLOYMENT.md's sizing notes) — this part is fast; the ASR side is
  the bottleneck.
- **Final toggle back to ASR**: another ~1-2 minutes.

**Total for one run with one task, cold start: roughly 10-12 minutes.**
Running multiple `--task` values in one invocation only adds the (fast) LLM
step time per extra task, not another full transcription — that's the
whole point of batching tasks in one call instead of running `pipeline.py`
once per task.

## 5. Falling back to manual steps

If `pipeline.py` fails partway (network blip, a toggle timing out, etc.),
nothing about the servers is left in a broken state that the manual tools
can't recover — `--skip_toggle` plus `scripts/toggle_server.sh status` to
see what's actually running, then continue by hand:

```bash
scripts/toggle_server.sh status
scripts/toggle_server.sh asr      # or llm, depending on what you need next
python main.py samples/saudi_business_03min.mp3 --output_format txt      # if transcription didn't finish
python llm_client.py --transcript pipeline-out/saudi_business_03min.txt --task summary   # if the LLM step didn't finish
scripts/toggle_server.sh asr      # restore the default resting state when done
```

## 6. Troubleshooting

- **`ASR server URL/key required` / `LLM server URL/key required`**: `.env`
  is missing one of the four `COHEREX_*_URL`/`COHEREX_*_API_KEY` variables —
  check `cat .env` has all four (see [§1](#1-prerequisites)).
  `COHEREX_SERVER_SSH` is separate and only needed for the toggle itself.
- **Hangs at `Waiting for ASR server (...) to become healthy...`**: normal
  for up to a couple minutes on a cold start (see [§4](#4-timing-expectations-measured-on-the-reference-box)).
  Past ~4 minutes with no change, check the server directly:
  `ssh $COHEREX_SERVER_SSH journalctl -u coherex-vllm -f` for a crash or
  OOM kill (DEPLOYMENT.md §8).
- **`subprocess.CalledProcessError` from `toggle_server.sh`**: the SSH
  command itself failed — check `COHEREX_SERVER_SSH` in `.env` is right and
  that `ssh $COHEREX_SERVER_SSH true` works non-interactively on its own,
  outside this script.
- **The LLM step starts before the ASR server is actually gone**: shouldn't
  happen — `wait_for_health` only returns once `/health` returns `200`, and
  the toggle's `systemctl stop` blocks until the old service is actually
  down before `systemctl start` runs the new one. If you see stale
  transcription output at the LLM step, it's more likely you're looking at
  `pipeline-out/` files from a previous run — check the modification time.
- **Only got the LLM's answer for one `--task` when you passed several**:
  each task's response is still saved to its own `name.<task>.txt`, even
  though the console only shows them one after another — check
  `--output_dir` for all of them, not just what scrolled past.
