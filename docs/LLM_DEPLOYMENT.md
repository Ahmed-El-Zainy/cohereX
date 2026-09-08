# Deploying a local LLM alongside the ASR server

This documents adding a second, small LLM server to the same box that runs
the CohereX ASR deployment ([DEPLOYMENT.md](DEPLOYMENT.md)) — for turning a
board-meeting transcript into what a board secretary
(أمين سر مجلس الإدارة) actually needs from it. Read DEPLOYMENT.md first if
you haven't; this assumes that setup already exists.

Platform HTTP mapping (minutes + decisions, implemented in the checkout but
not yet deployed):
[MEETING_MINUTES_API.md](MEETING_MINUTES_API.md).

## Board Secretary task scope

The task list this LLM performs is scoped to a real job description:
[`assets/مهام أمين سر مجلس الإدارة.docx`](../assets/مهام%20أمين%20سر%20مجلس%20الإدارة.docx),
a Saudi corporate-governance document defining a board secretary's
responsibilities across 8 areas — pre-meeting prep, in-meeting
documentation, post-meeting follow-up, decision tracking, governance
records, board↔management coordination, and committee/assembly work.

That job spans a lot more than this agent can do — it only ever sees a
**text transcript**, with no calendar, no email, no document archive, and
no persistence across meetings (yet). Splitting the full task list by that
constraint:

**Implemented — feasible from a transcript alone** (`llm_client.py`'s
`TASK_PROMPTS`, one task per bullet):

- `summary` — a quick 3-5 sentence overview
- `minutes_draft` — structured draft minutes: attendees, meeting
  start/end time, discussion summary, decisions (exact wording), votes,
  objections/reservations, conflict-of-interest disclosures — one section
  each, `"Not stated in transcript"` for anything not covered
- `decisions_log` — every decision/recommendation, quoted or closely
  paraphrased as stated, not summarized loosely
- `action_items` — each with `Action` / `Owner` / `Target date` fields,
  matching the docx's "تحديد الإجراءات المطلوبة، والمسؤول عن كل إجراء،
  والموعد المستهدف للتنفيذ"
- `conflicts_of_interest` — flags any disclosure or recusal mentioned
- `qa` — free-form questions about the transcript's content

**Explicitly out of scope** — these need systems this agent doesn't have
access to, not a bigger model: pre-meeting scheduling/invitations/agenda
prep (the transcript only exists *after* the meeting happens), collecting
documents from executive management, checking quorum against actual bylaws,
confirming attendance ahead of time, obtaining minute signatures,
maintaining governance registries *over time* (a real Board Action Tracker
needs to persist across meetings — this agent processes one transcript per
run), being a live point of contact, and all committee/general-assembly
administration. If any of these become worth building, they need
integration work beyond "ask an LLM about a transcript" — a persistence
layer for the Action Tracker most of all.

## Why this needed its own design decision, not just "add a model"

The ASR server alone already runs this 8GB-RAM box close to its limit —
~7GB RAM and 2-3GB of swap in active use just serving one request at a time
(DEPLOYMENT.md §9). There is no room to also keep an LLM loaded at the same
time. Two real options exist:

1. **Bigger box / second server** — the only way to have both warm
   simultaneously. Not done here (would need to be provisioned separately).
2. **Toggle: one service warm at a time** — what this deployment does. Free,
   works on the existing box, but the ASR and LLM APIs are mutually
   exclusive — starting one stops the other.

This also ruled out **vLLM** for the LLM itself. vLLM is built for GPU
throughput and its CPU backend (what the ASR server uses) is heavy — see
DEPLOYMENT.md §3 for the ~1 hour build and the memory-tuning fight that took.
**llama.cpp** is a much better fit for a small CPU box: its build takes
~9 minutes (vs. ~1 hour for vLLM+oneDNN), it's designed around quantized
models from the start, and a 3B-parameter model at 4-bit quantization loads
in seconds and uses ~2GB RAM — well inside what's left after the ASR server
is stopped.

## Model choice: Qwen2.5-3B-Instruct (Q4_K_M GGUF)

- **Multilingual, including Arabic** — this project's transcripts are
  Arabic business audio; Qwen2.5 has meaningfully better Arabic coverage
  than similarly-sized Llama or Gemma models.
- **Small enough to matter on this box**: `qwen2.5-3b-instruct-q4_k_m.gguf`
  is 2.0GB on disk, ~2GB resident once loaded — a 1.5B variant exists and is
  smaller still, but 3B gave noticeably better instruction-following for
  structured extraction in testing.
- Downloaded from the official `Qwen/Qwen2.5-3B-Instruct-GGUF` repo on
  Hugging Face — no gating, no token needed (unlike the ASR model).

**Known accuracy limitation, found in testing**: a Q&A test asked whether
the transcript mentioned "BIC" (Board Investment Committee) — the actual
transcript does mention it, but as a phonetic Arabic transliteration
("البي اي سي") rather than the Latin acronym. The model answered "no BIC
mentioned," missing the connection. This is a real limitation of a 3B model
reasoning across a phonetically-transliterated acronym in ASR output, not a
deployment bug — expect this class of miss on technical/English loanword
terms rendered phonetically in Arabic ASR transcripts. A larger model or a
prompt that explicitly asks it to consider phonetic transliterations of
English terms would likely help; neither is done here.

## 1. Build llama.cpp

```bash
cd /opt
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
cmake --build build --config Release -j2 --target llama-server llama-cli llama-quantize
```

`-DGGML_NATIVE=ON` lets it auto-detect and use this CPU's AVX-512 (`-march=native`
in the CMake output confirms it). `-DLLAMA_CURL=OFF` skips llama.cpp's
built-in model-downloading support — the model is fetched separately via
`hf download` below, so this avoids an OpenSSL dependency the box didn't
have. `-j2` matches `nproc`, same reasoning as the vLLM build in
DEPLOYMENT.md §3. Unlike that build, this one doesn't need careful memory
babysitting — peak RSS during compilation stayed well under 1GB per
translation unit, nothing like the 2.5-3.6GB spikes vLLM's kernels hit.

## 2. Download the model

```bash
source /opt/coherex-venv/bin/activate   # reuse the venv's huggingface_hub
mkdir -p /opt/models
hf download Qwen/Qwen2.5-3B-Instruct-GGUF qwen2.5-3b-instruct-q4_k_m.gguf --local-dir /opt/models
```

## 3. systemd service (disabled by default)

```ini
# /etc/systemd/system/coherex-llm.service
[Unit]
Description=CohereX local LLM server (llama.cpp, Qwen2.5-3B-Instruct)
After=network.target

[Service]
Type=simple
User=root
ExecStart=/opt/llama.cpp/build/bin/llama-server -m /opt/models/qwen2.5-3b-instruct-q4_k_m.gguf --host 127.0.0.1 --port 8001 -c 8192 -t 2 --api-key <LLM_API_KEY>
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

```bash
systemctl daemon-reload
# deliberately NOT `systemctl enable` — stays off unless you start it, so a
# reboot doesn't bring up both services and OOM the box
```

Port `8001` (the ASR server owns `8000`) is localhost-only. `-c 8192` is the
context window used by the meeting-minutes worker's map-reduce slices. `-t 2`
matches `nproc`, same reasoning as `VLLM_CPU_OMP_THREADS_BIND` in
DEPLOYMENT.md.

## 4. The toggle

Since both services can't be warm together, `scripts/toggle_server.sh` (repo
root) flips between them over SSH, reading the target from `.env`'s
`COHEREX_SERVER_SSH`:

```bash
scripts/toggle_server.sh llm      # stop coherex-vllm, start coherex-llm
scripts/toggle_server.sh asr      # stop coherex-llm, start coherex-vllm (back to default)
scripts/toggle_server.sh status   # which one is up right now
```

**Stopping `coherex-vllm` can be slow** — in testing it sat in
`deactivating (stop-sigterm)` for over a minute before actually exiting
(vLLM's multiprocessing workers don't all react to SIGTERM immediately).
The script's `systemctl stop` waits for this; it's not a hang, just budget
1-2 minutes for a `toggle_server.sh asr` call the first time you see it.
Starting `coherex-llm` is fast by comparison — a few seconds to load.

The **default/at-rest state is `coherex-vllm` running**, matching the
existing ASR deployment — `coherex-llm` is only ever running when you've
explicitly toggled to it, and toggling back to `asr` is the way to leave
things when you're done with the LLM.

## 5. Using it

Raw API (OpenAI-compatible `/v1/chat/completions`, same shape vLLM exposes).
Run this on the server or through an SSH tunnel; port 8001 is not public:

```bash
curl http://127.0.0.1:8001/v1/chat/completions \
  -H "Authorization: Bearer <LLM_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Summarize: ..."}], "max_tokens": 500}'
```

Or `llm_client.py` (repo root, mirrors `main.py`'s pattern — `.env`-based
config, no setup beyond what `main.py` already needs):

```bash
python llm_client.py --transcript out-ar/saudi_business_03min.txt --task summary
python llm_client.py --transcript out-ar/saudi_business_03min.txt --task minutes_draft --max_tokens 1200
python llm_client.py --transcript out-ar/saudi_business_03min.txt --task decisions_log
python llm_client.py --transcript api-test-out/x.txt --task action_items
python llm_client.py --transcript out-ar/x.txt --task conflicts_of_interest
python llm_client.py --transcript out-ar/x.txt --task qa \
  --question "Which committee approves the largest investments?"
python llm_client.py --prompt "Translate this to English: ..."   # raw prompt, no transcript
```

`--output <path>` saves the response text to a file (also always printed).
`COHEREX_LLM_URL` / `COHEREX_LLM_API_KEY` in `.env` (see `.env.example`) mean
no flags are needed for the common case. `minutes_draft` produces the most
content (7 sections) — raise `--max_tokens` past the 800 default for it,
especially on a longer meeting.

**All six tasks respond in Arabic**, not English — the source transcript,
the [Board Secretary task scope](#board-secretary-task-scope) document, and
the deliverable itself (real minutes for a Saudi company) are all Arabic,
so an English default would have meant translating at every step for no
reason and risking exactly the kind of transliteration error noted above.
The `qa` question can still be asked in English or Arabic — the *answer*
comes back in Arabic regardless. Use `--prompt` with your own instructions
(as in the `Translate this to English: ...` example above) if you need a
different output language for a one-off.

**Tested against the real transcript** (`out-ar/saudi_business_03min.txt`,
a clip describing PIF governance structure with no live decisions in it):
`minutes_draft` produced correct, professionally-worded Modern Standard
Arabic, following the 7-section structure exactly and writing
`"غير مذكور في النص"` ("not mentioned in the text") for attendees,
decisions, votes, objections, and conflicts — none of those are actually in
this particular clip, and it didn't invent any. `decisions_log` and
`conflicts_of_interest` both correctly reported none found rather than
fabricating something. **Known quirk**: `decisions_log`'s response (tested
before the Arabic rewrite, not yet re-verified after) repeated its one-line
conclusion twice and appended a stray ` ```plaintext ` code block — a minor
formatting tic of the 3B model, not a correctness issue; the actual content
was still right both times.

### Full workflow: transcribe, then draft minutes

```bash
scripts/toggle_server.sh asr                                    # make sure ASR is up
python main.py samples/saudi_business_03min.mp3 --output_format txt
scripts/toggle_server.sh llm                                    # switch to the LLM
python llm_client.py --transcript api-test-out/saudi_business_03min.txt --task minutes_draft --max_tokens 1200
scripts/toggle_server.sh asr                                    # switch back
```

This exact sequence — including the toggles — is also available as one
command: `python pipeline.py samples/saudi_business_03min.mp3`. See
[PIPELINE.md](PIPELINE.md) for the full walkthrough and timing
expectations.

## 6. For speaker-attributed answers ("who decided what")

Neither `main.py` nor the raw transcripts used above carry speaker labels —
diarization has been off by default this whole deployment. To get
speaker-attributed transcripts to feed the LLM, run the actual `coherex` CLI
(not `main.py`) with `--diarize`:

```bash
coherex samples/saudi_business_03min.mp3 --backend vllm \
  --vllm_url "$COHEREX_VLLM_URL" --vllm_api_key "$COHEREX_VLLM_API_KEY" \
  --diarize -o out-diarized/
```

This needs a Hugging Face token with access to
`pyannote/speaker-diarization-community-1` (gated — see the main
[README](../README.md#requirements)) and runs locally, not on either
server. The resulting `.txt`/`.json` will have `[SPEAKER_00]:`-style
prefixes per segment, which `llm_client.py --task qa` (or a custom prompt)
can then use to attribute statements to specific speakers. This hasn't been
tested end-to-end as part of this deployment — diarization itself is a
separate, existing CohereX feature, not something built for this LLM
addition.

## Sizing notes

- Build: ~9 minutes on 2 vCPUs (vs. ~1 hour for vLLM). Peak memory during
  build stayed under 1GB — no OOM risk, unlike the vLLM build.
- Model: 2.0GB on disk (`qwen2.5-3b-instruct-q4_k_m.gguf`), ~2GB RAM
  resident once `coherex-llm` is running.
- A `summary`/`decisions_log`/`action_items`/`conflicts_of_interest`/`qa`
  request over a ~20-line transcript completed in well under a minute; the
  longer `minutes_draft` (7 sections, `--max_tokens 1200`) took closer to a
  minute. Either way this is a completely different performance profile from
  the ASR server's 1-2.5 minutes *per 30-second audio chunk*. LLM text
  generation on this hardware is comfortable; ASR was the hard part.
