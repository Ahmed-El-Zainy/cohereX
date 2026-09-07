# Testing a deployed API with main.py

`main.py` (repo root) is a smoke-test script for a running CohereX vLLM
server — it hits the real HTTP API, not a mock, and saves what it gets back
so you can review it. This documents every step to run it, all its flags,
and what to expect. For how the server itself was built and deployed, see
[DEPLOYMENT.md](DEPLOYMENT.md).

## What it actually does

1. `GET /health` and `GET /v1/models` against the server, printing the
   result — a fast way to confirm the server is up before waiting on a real
   transcription.
2. For each audio file: calls `coherex.load_model(backend="vllm")`, which
   does local VAD chunking (splits audio into ≤30s segments) and sends each
   chunk to the remote server's `/v1/audio/transcriptions` endpoint — the
   exact same path the CLI's `--backend vllm` flag uses.
3. Saves the transcript to `api-test-out/` in **every** output format
   CohereX supports by default — `.txt`, `.srt`, `.vtt`, `.tsv`, `.json`,
   `.aud` — using the project's own writers (`coherex.utils.get_writer`), the
   same code the CLI itself uses. Narrow it to one format with
   `--output_format`.
4. Shows a live progress bar while it waits (see [Progress bar](#progress-bar) below).

## 1. Prerequisites

- A deployed CohereX vLLM server (see [DEPLOYMENT.md](DEPLOYMENT.md)) — you
  need its URL and API key.
- CohereX installed locally with the ML stack (`pip install -e .` or
  `pip install coherex`) — `main.py` imports `coherex` directly, so this is
  not optional even though transcription itself runs on the remote server;
  VAD and audio loading happen on your machine.
- `httpx` and `tqdm` available (both already pulled in transitively by
  CohereX's own dependencies — nothing extra to install).

## 2. One-time setup: `.env`

`main.py` reads the server URL and API key from a local `.env` file so you
don't have to export environment variables every run.

```bash
cp .env.example .env
```

Edit `.env`:

```bash
COHEREX_VLLM_URL=http://<server-ip>:8000
COHEREX_VLLM_API_KEY=<key>
```

`.env` is gitignored (`.gitignore` already lists it) — **never commit real
values**. `.env.example` stays in the repo with placeholders only, as a
template for anyone else who clones it.

If you'd rather not use a file, environment variables or CLI flags both
override it: `COHEREX_VLLM_URL` / `COHEREX_VLLM_API_KEY`, or
`--vllm_url` / `--vllm_api_key`. Precedence: CLI flag > env var > `.env` >
error ("`--vllm_url is required`").

## 3. Running it

```bash
# every audio file in samples/ (the default)
python main.py

# just these files, from anywhere on disk
python main.py path/to/one.mp3 other.wav

# override language/model per run (defaults come from coherex/__main__.py:
# arabic-07-2026 model, language "ar")
python main.py --language ar --model CohereLabs/cohere-transcribe-03-2026 clip.mp3

# only check the server is alive, skip transcription entirely
python main.py --skip_transcribe
```

Every flag:

| Flag                         | Default                                         | Purpose                                                                                                                                                                                                                       |
| ---------------------------- | ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `audio` (positional)       | every file in`--samples_dir`                  | one or more audio files to transcribe                                                                                                                                                                                         |
| `--vllm_url`               | `$COHEREX_VLLM_URL` / `.env`                | server URL, e.g.`http://1.2.3.4:8000`                                                                                                                                                                                       |
| `--vllm_api_key`           | `$COHEREX_VLLM_API_KEY` / `.env`            | server API key                                                                                                                                                                                                                |
| `--model`                  | `CohereLabs/cohere-transcribe-arabic-07-2026` | model name as served by vLLM — must match what the server actually loaded (`GET /v1/models` shows this)                                                                                                                    |
| `--language`               | `ar`                                          | language code passed to the ASR call                                                                                                                                                                                          |
| `--vad_method`             | `silero`                                      | `silero` or `pyannote` for local VAD chunking                                                                                                                                                                             |
| `--batch_size`             | `2`                                           | VAD chunks grouped per progress-bar update; matches the vLLM backend's`max_workers=2` default (see [DEPLOYMENT.md §9](DEPLOYMENT.md#9-testing-the-deployment)) — lower means more frequent, finer-grained progress updates |
| `--samples_dir`            | `samples/`                                    | folder scanned when no files are given positionally                                                                                                                                                                           |
| `--output_dir`             | `api-test-out/`                               | where results are saved                                                                                                                                                                                                       |
| `--output_format` / `-f` | `all`                                         | `all` (every format below), or one of `txt`, `srt`, `vtt`, `tsv`, `json`, `aud`                                                                                                                                 |
| `--max_line_width`         | `42`                                          | max characters per subtitle line (`srt`/`vtt` only)                                                                                                                                                                       |
| `--max_line_count`         | `2`                                           | max lines per subtitle cue (`srt`/`vtt` only)                                                                                                                                                                             |
| `--highlight_words`        | `false`                                       | underline each word as spoken in`srt`/`vtt` — needs word-level alignment, which this script doesn't do (see [§4](#4-reading-the-output)), so it's a no-op here; kept for CLI-flag parity                                 |
| `--skip_transcribe`        | off                                             | only run the health/models check                                                                                                                                                                                              |

## Progress bar

Because this box (see DEPLOYMENT.md) takes roughly 1–2.5 minutes *per chunk*,
a naive progress indicator would sit at 0% for minutes at a time and look
hung. `main.py` handles this two ways:

- **Live elapsed-time ticker**: a background thread refreshes the bar every
  second regardless of whether any real progress has happened, so the
  `[MM:SS<...]` elapsed counter is always visibly moving.
- **Real percentage jumps**: wired to coherex's `progress_callback`, firing
  every `--batch_size` chunks (default 2) rather than only once at the very
  end.

Example of what this looked like in a real run (7 chunks, `--batch_size 2`):
sits at 0% ticking for ~3 minutes, jumps to 29% (2/7) the moment the first
batch finishes, then continues — rather than looking frozen for the entire
~7 minute run.

## 4. Reading the output

For each input file `name.ext`, by default (`--output_format all`) **six**
files land in `--output_dir` (`api-test-out/`), one per format CohereX
supports, all written by the project's real writers
(`coherex/utils.py`'s `get_writer`) — not hand-rolled formatting in
`main.py`:

| File          | Format               | Notes                                                                                 |
| ------------- | -------------------- | ------------------------------------------------------------------------------------- |
| `name.txt`  | plain text           | one line per segment, no timestamps                                                   |
| `name.srt`  | SubRip subtitles     | segment-level cue timing (`HH:MM:SS,mmm`)                                           |
| `name.vtt`  | WebVTT subtitles     | same cues, WebVTT timestamp format (`MM:SS.mmm`)                                    |
| `name.tsv`  | tab-separated        | columns`start_ms`, `end_ms`, `text` (integer milliseconds), one row per segment |
| `name.json` | full result          | `{"segments": [{"text", "start", "end"}, ...], "language": "ar"}`                   |
| `name.aud`  | Audacity label track | importable directly as labels in Audacity                                             |

**No word-level alignment here** — this script only exercises the raw ASR
API (`coherex.load_model(backend="vllm")` + `.transcribe()`), not the full
CLI pipeline's separate alignment step. That means `.srt`/`.vtt` cues are
one per VAD segment (up to ~30s each) rather than tightly split by
`--max_line_width`/`--max_line_count` at the word level — those two flags
still apply (segments longer than `--max_line_width` get soft-wrapped) but
without word timestamps to snap to, so don't expect the same subtitle
polish as the full CLI. For word-aligned subtitles, run the main
`coherex` CLI itself with `--backend vllm` pointing at the same server
(see the main [README](../README.md#serving-with-vllm)) — that runs
alignment locally after the same remote ASR call this script makes.

Pass `--output_format <fmt>` to write just one file instead of all six —
see [§5](#5-example-commands-config-recipes) for examples.

This "all six, `.aud` included" behavior lives in `coherex/utils.py`'s
`get_writer` itself (its `OUTPUT_FORMATS` list), not in `main.py` — so
`coherex audio.mp3` (the real CLI, default `-f all`) writes the same six
files, not just five.

The console also prints each segment as `[start --> end] text` after the
file finishes, a `Done in <N>s (<M> segments)` summary, and the full list of
paths written for that file.

## 5. Example commands (config recipes)

Every command below is copy-pasteable as-is and targets the file actually in
this repo's `samples/` folder, `samples/saudi_business_03min.mp3` (Arabic
business audio, ~3 minutes). Add more files to `samples/` and the no-argument
form picks them all up automatically.

```bash
# Everything, every format — the default. Writes .txt .srt .vtt .tsv .json .aud
python main.py samples/saudi_business_03min.mp3

# Same thing, with the default made explicit (useful in scripts, so the
# behavior doesn't silently change if main.py's default ever does)
python main.py samples/saudi_business_03min.mp3 --output_format all

# Same, but scan the whole samples/ folder instead of naming one file
python main.py

# Only subtitles
python main.py samples/saudi_business_03min.mp3 --output_format srt
python main.py samples/saudi_business_03min.mp3 --output_format vtt

# Only plain text / only the full JSON (segments + timing, easiest to post-process)
python main.py samples/saudi_business_03min.mp3 --output_format txt
python main.py samples/saudi_business_03min.mp3 --output_format json

# Only an Audacity label track
python main.py samples/saudi_business_03min.mp3 --output_format aud

# Tighter subtitle lines, e.g. for a narrow video player
python main.py samples/saudi_business_03min.mp3 --output_format srt \
  --max_line_width 32 --max_line_count 1

# Force English + the upstream 14-language base model instead of this fork's
# Arabic default (see the CLI defaults table in DEPLOYMENT.md)
python main.py samples/saudi_business_03min.mp3 \
  --language en --model CohereLabs/cohere-transcribe-03-2026

# pyannote VAD instead of silero (needs an HF token available locally — see
# the main README's Requirements section)
python main.py samples/saudi_business_03min.mp3 --vad_method pyannote

# Keep results out of api-test-out/, e.g. to compare two runs side by side
python main.py samples/saudi_business_03min.mp3 --output_dir out-test-baseline/
python main.py samples/saudi_business_03min.mp3 --output_dir out-test-english/ --language en

# Finer-grained progress-bar updates: tick after every single chunk instead
# of every 2 (costs nothing extra server-side, just reports more often)
python main.py samples/saudi_business_03min.mp3 --batch_size 1

# Point at a different server/key for one run without touching .env
python main.py samples/saudi_business_03min.mp3 \
  --vllm_url http://<other-server-ip>:8000 --vllm_api_key <other-key>

# Underline each word as spoken in srt/vtt — a no-op here (see the flag
# table above: this script doesn't run alignment, so there's no word-level
# timing to underline), kept only for flag parity with the real CLI.
# Note: str2bool (coherex/utils.py) only accepts "True"/"False", capitalized.
python main.py samples/saudi_business_03min.mp3 --output_format srt --highlight_words True

# Just confirm the server is up and which model it's serving — no transcription
python main.py --skip_transcribe
```

### The real CLI, for comparison

`get_writer`'s "all means every format, `.aud` included" behavior lives in
`coherex/utils.py`, not in `main.py` — so the actual `coherex` CLI, run
against the same server with `--backend vllm`, writes the same six files
plus real word-level alignment (which `main.py` intentionally skips — see
[§4](#4-reading-the-output)):

```bash
# .env is only auto-loaded by main.py's own loader, not by your shell — load
# it into the shell too so these vars are actually set:
set -a && source .env && set +a

coherex samples/saudi_business_03min.mp3 --backend vllm \
  --vllm_url "$COHEREX_VLLM_URL" --vllm_api_key "$COHEREX_VLLM_API_KEY" \
  -o out-ar/
# -> out-ar/saudi_business_03min.{txt,srt,vtt,tsv,json,aud}, word-aligned
```

## 6. Troubleshooting

- **`--vllm_url is required` / `--vllm_api_key is required`**: `.env` is
  missing, empty, or not in the repo root you're running from. Check
  `cat .env` prints both variables.
- **`GET /health -> 000` or a connection error**: the server process isn't
  running or isn't reachable from where you are. `ssh` in and check
  `systemctl status coherex-vllm` (see DEPLOYMENT.md §7–8).
  `httpx.ConnectTimeout`/`ReadTimeout` here usually means a firewall or the
  server crash-looping, not a slow request — those show up later, inside
  `transcribe_files`, not in `check_endpoints`.
- **Hangs (progress bar ticking, but never advancing)**: expected if a
  single chunk is genuinely still processing — see the timing notes in
  DEPLOYMENT.md §9. If it's stuck well past ~5 minutes on one chunk with no
  jump at all, check the server side (`journalctl -u coherex-vllm -f`) for
  an OOM kill or crash — the client will eventually hit its own timeout
  (`coherex/vllm_backend.py`'s `VLLMBackend` defaults to `timeout=300.0`)
  and raise rather than hang forever.
- **`FileNotFoundError: Audio file(s) not found: ...`**: a positional path
  doesn't exist — checked up front, before any model loading, so this fails
  fast rather than after a slow VAD/backend-load step.
- **Model name mismatch**: if `--model` doesn't match what the server
  actually loaded, vLLM will typically 400 on the actual request even though
  `/health` and `/v1/models` both look fine — check `python main.py --skip_transcribe` prints the model you expect under "served model:".
