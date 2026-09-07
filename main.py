"""
Smoke test for a deployed CohereX vLLM API server.

Checks /health and /v1/models on the server, then runs one or more audio
files through the real deployed API — coherex.load_model(backend="vllm")
does local VAD chunking and sends each chunk to the remote server for ASR,
exactly like the CLI's --backend vllm path. Results are saved to
api-test-out/ for review.

Usage:
    python main.py                              # every file in samples/
    python main.py path/to/one.mp3 other.wav    # just these files, anywhere
    python main.py --language en --model CohereLabs/cohere-transcribe-03-2026 clip.mp3

Server URL/key come from a local .env file (see .env.example — copy it to
.env and fill in real values; .env is gitignored, never commit it) or the
COHEREX_VLLM_URL / COHEREX_VLLM_API_KEY env vars, or --vllm_url/--vllm_api_key.

See docs/DEPLOYMENT.md for how the server was set up and its known limits
(single request at a time, ~1-2.5 min per 30s chunk on the reference box).
"""
import argparse
import os
import threading
import time
from pathlib import Path

import httpx
from tqdm import tqdm

from coherex.asr import DEFAULT_MODEL
from coherex.utils import OUTPUT_FORMATS, get_writer, str2bool

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SAMPLES_DIR = REPO_ROOT / "samples"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "api-test-out"
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".webm"}
# Matches VLLMBackend's default max_workers=2 (coherex/vllm_backend.py) — a
# smaller --batch_size than the model's own default (8) makes the progress
# bar tick after every 1-2 chunks instead of jumping straight from 0% to
# 100% once the whole file is done.
DEFAULT_BATCH_SIZE = 2


def load_dotenv(path: Path = REPO_ROOT / ".env") -> None:
    """Minimal .env loader: KEY=VALUE lines, no external dependency. Existing
    environment variables always win over the file."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def check_endpoints(base_url: str, api_key: str) -> None:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    with httpx.Client(timeout=10.0) as client:
        health = client.get(f"{base_url}/health")
        print(f"GET /health -> {health.status_code}")
        health.raise_for_status()

        models = client.get(f"{base_url}/v1/models", headers=headers)
        print(f"GET /v1/models -> {models.status_code}")
        if models.status_code == 200:
            for served in models.json().get("data", []):
                print(f"  served model: {served.get('id')}")


class _LiveProgress:
    """A tqdm bar that keeps ticking (elapsed time visibly moving) even
    between the coarse updates coherex's progress_callback gives us, so a
    multi-minute wait on a single chunk still looks alive, not hung."""

    def __init__(self, desc: str):
        self.bar = tqdm(total=100, desc=desc, unit="%")
        self._stop = threading.Event()
        self._ticker = threading.Thread(target=self._tick, daemon=True)
        self._ticker.start()

    def _tick(self):
        while not self._stop.wait(1.0):
            self.bar.refresh()

    def __call__(self, percent_complete: float) -> None:
        self.bar.n = round(percent_complete, 1)
        self.bar.refresh()

    def close(self) -> None:
        self._stop.set()
        self._ticker.join()
        self.bar.n = 100
        self.bar.refresh()
        self.bar.close()


def transcribe_files(
    audio_files: list[Path],
    output_dir: Path,
    vllm_url: str,
    vllm_api_key: str,
    model: str,
    language: str,
    vad_method: str,
    batch_size: int,
    output_format: str,
    max_line_width: int,
    max_line_count: int,
    highlight_words: bool,
) -> None:
    import coherex

    missing = [p for p in audio_files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Audio file(s) not found: {', '.join(str(p) for p in missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading vLLM backend at {vllm_url} (model={model})...")
    asr_model = coherex.load_model(
        model_name=model,
        backend="vllm",
        vllm_url=vllm_url,
        vllm_api_key=vllm_api_key,
        vad_method=vad_method,
    )

    try:
        for audio_path in audio_files:
            print(f"\nTranscribing {audio_path.name} ...")
            progress = _LiveProgress(desc=audio_path.name)
            started = time.monotonic()
            try:
                result = asr_model.transcribe(
                    str(audio_path),
                    language=language,
                    batch_size=batch_size,
                    print_progress=False,
                    progress_callback=progress,
                )
            finally:
                progress.close()
            elapsed = time.monotonic() - started

            for seg in result["segments"]:
                print(f"  [{seg['start']:>7.2f} --> {seg['end']:>7.2f}] {seg['text']}")
            print(f"Done in {elapsed:.1f}s ({len(result['segments'])} segments)")

            write_options = {
                "max_line_width": max_line_width,
                "max_line_count": max_line_count,
                "highlight_words": highlight_words,
            }
            get_writer(output_format, str(output_dir))(result, str(audio_path), write_options)
            formats = OUTPUT_FORMATS if output_format == "all" else [output_format]
            saved_paths = [output_dir / f"{audio_path.stem}.{fmt}" for fmt in formats]
            print("Saved: " + ", ".join(str(p) for p in saved_paths))
    finally:
        asr_model.shutdown()


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", nargs="*", type=Path, help="audio file(s) to transcribe; defaults to every file in --samples_dir")
    parser.add_argument("--vllm_url", default=os.environ.get("COHEREX_VLLM_URL"), help="vLLM server URL (or set COHEREX_VLLM_URL / .env)")
    parser.add_argument("--vllm_api_key", default=os.environ.get("COHEREX_VLLM_API_KEY"), help="vLLM API key (or set COHEREX_VLLM_API_KEY / .env)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model name as served by vLLM")
    parser.add_argument("--language", default="ar", help="language code to transcribe with")
    parser.add_argument("--vad_method", default="silero", choices=["silero", "pyannote"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="VAD chunks per progress-bar tick (smaller = more frequent updates)")
    parser.add_argument("--samples_dir", type=Path, default=DEFAULT_SAMPLES_DIR, help="used when no audio files are given positionally")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_format", "-f", default="all", choices=["all"] + OUTPUT_FORMATS, help="which output file(s) to write; 'all' writes every format, aud included")
    parser.add_argument("--max_line_width", type=int, default=42, help="max characters per subtitle line (srt/vtt)")
    parser.add_argument("--max_line_count", type=int, default=2, help="max lines per subtitle cue (srt/vtt)")
    parser.add_argument("--highlight_words", type=str2bool, default=False, help="underline each word as it's spoken in srt/vtt (needs word-level alignment; no-op here since this script doesn't align)")
    parser.add_argument("--skip_transcribe", action="store_true", help="only check /health and /v1/models, skip running audio")
    args = parser.parse_args()

    if not args.vllm_url:
        parser.error("--vllm_url is required (set it in .env, export COHEREX_VLLM_URL, or pass --vllm_url)")
    if not args.vllm_api_key:
        parser.error("--vllm_api_key is required (set it in .env, export COHEREX_VLLM_API_KEY, or pass --vllm_api_key)")

    check_endpoints(args.vllm_url, args.vllm_api_key)

    if args.skip_transcribe:
        return

    audio_files = args.audio or sorted(
        p for p in args.samples_dir.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not audio_files:
        print(f"No audio files given and none found in {args.samples_dir}")
        return

    transcribe_files(
        audio_files,
        args.output_dir,
        args.vllm_url,
        args.vllm_api_key,
        args.model,
        args.language,
        args.vad_method,
        args.batch_size,
        args.output_format,
        args.max_line_width,
        args.max_line_count,
        args.highlight_words,
    )


if __name__ == "__main__":
    main()
