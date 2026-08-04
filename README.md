# CohereX

Speech transcription with word-level timestamps and speaker diarization, built on
the [Cohere Transcribe](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
ASR model.

Cohere Transcribe produces accurate text but no timestamps, no speaker labels, and no
language detection. CohereX adds those around it, following the same pipeline design as
[WhisperX](https://github.com/m-bain/whisperX):

```
audio → VAD → Cohere Transcribe → wav2vec2 forced alignment → diarization → subtitles
```

- Voice activity detection (pyannote or silero) splits speech into chunks and drops silence.
- Cohere Transcribe transcribes each chunk.
- A wav2vec2 model force-aligns the transcript to the audio for per-word timestamps.
- pyannote assigns a speaker to every word and segment.
- Results are written as SRT, VTT, TXT, TSV, or JSON.

## Requirements

- Python 3.10–3.13
- [FFmpeg](https://ffmpeg.org/) on your PATH (used for audio decoding)
- A Hugging Face account with access to the gated models:
  - [`CohereLabs/cohere-transcribe-03-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
  - [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) (only if you use `--diarize`)

Accept the model terms on their Hugging Face pages, then log in:

```bash
hf auth login
```

## Install

```bash
git clone https://github.com/bakrianoo/cohereX.git
cd cohereX
pip install -e .
```

Optional automatic language detection needs one extra package:

```bash
pip install -e ".[langid]"
```

GPU is strongly recommended. On CPU the model runs but is slow.

## Quick start

Transcribe a file and write all output formats to `out/`:

```bash
coherex audio.mp3 --language en -o out/
```

Add speaker labels:

```bash
coherex audio.mp3 --language en --diarize -o out/
```

Let CohereX detect the language (needs the `langid` extra):

```bash
coherex audio.mp3 --language auto -o out/
```

Produce only an SRT with two lines per cue:

```bash
coherex audio.mp3 --language en -f srt --max_line_width 42 --max_line_count 2 -o out/
```

`--language` is required. Cohere Transcribe has no built-in language detection, and passing
the wrong language produces a fluent but wrong transcription rather than an error. Use `auto`
if you are unsure.

## Supported languages

`en`, `fr`, `de`, `es`, `it`, `pt`, `nl`, `pl`, `el`, `ar`, `ja`, `zh`, `vi`, `ko`.

Automatic detection (`--language auto`) chooses from this set only.

## Common options

| Option | Default | Description |
| --- | --- | --- |
| `--language` | — | Language code or `auto`. Required. |
| `--diarize` | off | Assign speaker labels (needs the pyannote model + token). |
| `--no_align` | off | Skip forced alignment (segment-level timestamps only). |
| `--device` | `cuda` if available | `cpu` or `cuda`. |
| `--compute_type` | `default` | `bfloat16`, `float16`, `float32`, or `default` (bfloat16 on GPU, float32 on CPU). |
| `--batch_size` | `8` | VAD chunks per forward pass. Helps on GPU; use `1` on CPU. |
| `--vad_method` | `pyannote` | `pyannote` or `silero`. |
| `--chunk_size` | `30` | Max seconds per VAD chunk. Keep below 35. |
| `--output_format` / `-f` | `all` | `srt`, `vtt`, `txt`, `tsv`, `json`, `aud`, or `all`. |
| `--output_dir` / `-o` | `.` | Where to write outputs. |
| `--punctuation` | `true` | Set `false` for lower-cased output without punctuation. |
| `--max_line_width` | none | Max characters per subtitle line. |
| `--max_line_count` | none | Max lines per subtitle cue. |
| `--min_speakers` / `--max_speakers` | none | Constrain the speaker count for diarization. |
| `--hf_token` | none | Hugging Face token (or use `hf auth login`). |

Run `coherex --help` for the full list.

## Python API

```python
import coherex

model = coherex.load_model(device="cuda", compute_type="bfloat16", vad_method="pyannote")

# 1. Transcribe (segment-level timestamps from VAD)
result = model.transcribe("audio.mp3", language="en", batch_size=8)

# 2. Word-level timestamps
align_model, metadata = coherex.load_align_model("en", device="cuda")
result = coherex.align(result["segments"], align_model, metadata, "audio.mp3", "cuda")

# 3. Speaker labels
from coherex.diarize import DiarizationPipeline
diarizer = DiarizationPipeline(device="cuda")
speakers = diarizer("audio.mp3")
result = coherex.assign_word_speakers(speakers, result)

for seg in result["segments"]:
    print(seg["start"], seg["end"], seg.get("speaker"), seg["text"])
```

To detect the language from audio:

```python
model = coherex.load_model(device="cuda")
language = coherex.detect_language(model, "audio.mp3")
result = model.transcribe("audio.mp3", language=language)
```

## Output

JSON contains segments and a flat `word_segments` list, each word carrying `start`, `end`,
`score`, and (with `--diarize`) `speaker`:

```json
{
  "segments": [
    {
      "start": 0.83,
      "end": 6.33,
      "text": "This week, I traveled to Chicago...",
      "speaker": "SPEAKER_00",
      "words": [
        {"word": "This", "start": 0.83, "end": 1.01, "score": 0.98, "speaker": "SPEAKER_00"}
      ]
    }
  ],
  "word_segments": [
    {"word": "This", "start": 0.83, "end": 1.01, "score": 0.98, "speaker": "SPEAKER_00"}
  ],
  "language": "en"
}
```

## Notes and limitations

- **Language is required.** There is no reliable failure mode for the wrong language — the
  model will transcribe confidently in whatever language you specify.
- **VAD matters.** Cohere Transcribe transcribes non-speech audio as hallucinated text, so
  the VAD step is on by default and should stay on for noisy input.
- **14 languages only**, listed above.
- **Alignment coverage.** Word timestamps come from a per-language wav2vec2 model. If none is
  available for a language, run with `--no_align` to get segment-level timestamps instead.
- **`auto` detection cost.** It probes the model once per candidate language, so it is slower
  than passing a code directly. Prefer an explicit `--language` when you know it.

## How it fits together

Only the ASR-specific pieces are unique to CohereX:

- `asr.py` — loads Cohere Transcribe and transcribes VAD chunks.
- `transcribe.py` — the end-to-end pipeline.
- `langid.py` — optional language detection.
- `__main__.py` — the command-line interface.

Alignment (`alignment.py`), diarization (`diarize.py`), VAD (`vads/`), subtitle formatting
(`SubtitlesProcessor.py`), and the output writers (`utils.py`) follow WhisperX.

## Credits

- [Cohere Transcribe](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) — the ASR model.
- [WhisperX](https://github.com/m-bain/whisperX) — the pipeline design and the alignment,
  diarization, VAD, and subtitle components.
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) — VAD and diarization.

## License

Apache 2.0.
