"""
End-to-end pipeline: transcribe audio on the ASR server, then post-process
the transcript with the local LLM — automatically switching the remote box
between the two services, since they can't run warm at the same time (not
enough RAM; see docs/LLM_DEPLOYMENT.md).

This is glue, not new logic: it calls main.py's transcribe_files() and
llm_client.py's ask()/TASK_PROMPTS directly, and shells out to
scripts/toggle_server.sh for the switch — same code paths as running each
piece by hand, just sequenced.

Usage:
    python pipeline.py samples/saudi_business_03min.mp3
    python pipeline.py samples/saudi_business_03min.mp3 --task minutes_draft --max_tokens 1200
    python pipeline.py samples/saudi_business_03min.mp3 --task decisions_log --task action_items
    python pipeline.py samples/saudi_business_03min.mp3 --task conflicts_of_interest
    python pipeline.py samples/saudi_business_03min.mp3 --task qa \\
        --question "Which committee approves the largest investments?"
    python pipeline.py samples/saudi_business_03min.mp3 --no_restore   # leave the LLM warm after
    python pipeline.py samples/saudi_business_03min.mp3 --skip_toggle  # you manage server state yourself

See docs/PIPELINE.md for the full walkthrough, timing expectations, and
troubleshooting.
"""
import argparse
import os
import subprocess
import time
from pathlib import Path

import httpx

import llm_client
import main as coherex_main

REPO_ROOT = Path(__file__).resolve().parent
TOGGLE_SCRIPT = REPO_ROOT / "scripts" / "toggle_server.sh"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "pipeline-out"


def toggle(mode: str) -> None:
    print(f"\n--- scripts/toggle_server.sh {mode} ---")
    subprocess.run([str(TOGGLE_SCRIPT), mode], check=True)


def wait_for_health(url: str, label: str, timeout: float = 240.0) -> None:
    print(f"Waiting for {label} ({url}) to become healthy...")
    deadline = time.monotonic() + timeout
    with httpx.Client(timeout=5.0) as client:
        while time.monotonic() < deadline:
            try:
                if client.get(f"{url}/health").status_code == 200:
                    print(f"{label} is up.")
                    return
            except httpx.HTTPError:
                pass
            time.sleep(5)
    raise TimeoutError(f"{label} at {url} did not become healthy within {timeout:.0f}s")


def main() -> None:
    coherex_main.load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path, help="one audio file to transcribe and analyze")
    parser.add_argument("--task", action="append", choices=list(llm_client.TASK_PROMPTS), help="repeatable; which LLM task(s) to run on the transcript (default: summary)")
    parser.add_argument("--question", help="required if --task qa is used")
    parser.add_argument("--language", default="ar")
    parser.add_argument("--vad_method", default="silero", choices=["silero", "pyannote"])
    parser.add_argument("--model", default=coherex_main.DEFAULT_MODEL, help="ASR model name as served by vLLM")
    parser.add_argument("--vllm_url", default=os.environ.get("COHEREX_VLLM_URL"), help="ASR server URL (or set COHEREX_VLLM_URL / .env)")
    parser.add_argument("--vllm_api_key", default=os.environ.get("COHEREX_VLLM_API_KEY"), help="ASR server API key (or set COHEREX_VLLM_API_KEY / .env)")
    parser.add_argument("--llm_url", default=os.environ.get("COHEREX_LLM_URL"), help="LLM server URL (or set COHEREX_LLM_URL / .env)")
    parser.add_argument("--llm_api_key", default=os.environ.get("COHEREX_LLM_API_KEY"), help="LLM server API key (or set COHEREX_LLM_API_KEY / .env)")
    parser.add_argument("--max_tokens", type=int, default=800, help="max tokens per LLM task response; raise for minutes_draft on longer meetings")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip_toggle", action="store_true", help="assume the right service is already running at each step; don't call toggle_server.sh")
    parser.add_argument("--no_restore", action="store_true", help="leave the LLM server warm afterward instead of switching back to the ASR server")
    args = parser.parse_args()

    tasks = args.task or ["summary"]
    if "qa" in tasks and not args.question:
        parser.error("--task qa requires --question")
    if not args.vllm_url or not args.vllm_api_key:
        parser.error("ASR server URL/key required (.env, or --vllm_url/--vllm_api_key)")
    if not args.llm_url or not args.llm_api_key:
        parser.error("LLM server URL/key required (.env, or --llm_url/--llm_api_key)")
    if not args.audio.is_file():
        parser.error(f"audio file not found: {args.audio}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1/2: transcribe ---
    if not args.skip_toggle:
        toggle("asr")
    wait_for_health(args.vllm_url, "ASR server")
    print(f"\n=== Step 1/2: Transcribing {args.audio.name} ===")
    coherex_main.transcribe_files(
        [args.audio],
        args.output_dir,
        args.vllm_url,
        args.vllm_api_key,
        args.model,
        args.language,
        args.vad_method,
        batch_size=2,
        output_format="txt",
        max_line_width=42,
        max_line_count=2,
        highlight_words=False,
    )
    transcript_path = args.output_dir / f"{args.audio.stem}.txt"
    transcript_text = transcript_path.read_text(encoding="utf-8")

    # --- Step 2/2: LLM post-processing ---
    if not args.skip_toggle:
        toggle("llm")
    wait_for_health(args.llm_url, "LLM server")
    print(f"\n=== Step 2/2: Running {len(tasks)} LLM task(s) on {transcript_path.name} ===")
    for task in tasks:
        print(f"\n--- Task: {task} ---")
        prompt = llm_client.TASK_PROMPTS[task].format(text=transcript_text, question=args.question)
        result = llm_client.ask(args.llm_url, args.llm_api_key, prompt, args.max_tokens, 0.3)
        print(result)
        out_path = args.output_dir / f"{args.audio.stem}.{task}.txt"
        out_path.write_text(result, encoding="utf-8")
        print(f"Saved {out_path}")

    if not args.skip_toggle and not args.no_restore:
        toggle("asr")
        wait_for_health(args.vllm_url, "ASR server")

    print("\nDone.")


if __name__ == "__main__":
    main()
