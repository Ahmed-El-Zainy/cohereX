import argparse
import importlib.metadata
import platform

import torch

from coherex.utils import optional_float, optional_int, str2bool
from coherex.log_utils import setup_logging

# Cohere Transcribe supports exactly these 14 languages (no auto-detection).
SUPPORTED_LANGUAGES = [
    "en", "fr", "de", "es", "it", "pt", "nl", "pl", "el", "ar", "ja", "zh", "vi", "ko",
]

DEFAULT_MODEL = "CohereLabs/cohere-transcribe-03-2026"
DEFAULT_DIARIZE_MODEL = "pyannote/speaker-diarization-community-1"

try:
    __version__ = importlib.metadata.version("coherex")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0"


def cli():
    # fmt: off
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio", nargs="+", type=str, help="audio file(s) to transcribe")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="name of the Cohere ASR model to use")
    parser.add_argument("--model_cache_only", type=str2bool, default=False, help="If True, will not attempt to download models, instead using cached models from --model_dir")
    parser.add_argument("--model_dir", type=str, default=None, help="the path to save model files; uses the HuggingFace cache by default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="device type to use for PyTorch inference (e.g. cpu, cuda)")
    parser.add_argument("--device_index", default=0, type=int, help="device index to use for inference")
    parser.add_argument("--batch_size", default=8, type=int, help="the number of VAD chunks to transcribe per generate() call")
    parser.add_argument("--compute_type", default="default", type=str, choices=["default", "bfloat16", "float16", "float32"], help="compute type for computation; 'default' uses bfloat16 on GPU, float32 on CPU")

    # backend / vLLM serving
    parser.add_argument("--backend", type=str, default="local", choices=["local", "vllm"], help="ASR backend: 'local' runs the model in-process; 'vllm' offloads transcription to a vLLM server")
    parser.add_argument("--vllm_url", type=str, default=None, help="URL of a running vLLM server (e.g. http://localhost:8000). If omitted with --backend vllm, CohereX starts its own vLLM server and stops it when finished")
    parser.add_argument("--vllm_api_key", type=str, default=None, help="API key for the vLLM server, if it requires one (defaults to the VLLM_API_KEY env var)")
    parser.add_argument("--vllm_args", type=str, default=None, help="extra arguments passed to 'vllm serve' when CohereX launches its own server, e.g. \"--max-model-len 448 --gpu-memory-utilization 0.8\"")

    parser.add_argument("--output_dir", "-o", type=str, default=".", help="directory to save the outputs")
    parser.add_argument("--output_format", "-f", type=str, default="all", choices=["all", "srt", "vtt", "txt", "tsv", "json", "aud"], help="format of the output file; if not specified, all available formats will be produced")
    parser.add_argument("--verbose", type=str2bool, default=True, help="whether to print out the progress and debug messages")
    parser.add_argument("--log-level", type=str, default=None, choices=["debug", "info", "warning", "error", "critical"], help="logging level (overrides --verbose if set)")

    parser.add_argument("--language", type=str, default=None, choices=SUPPORTED_LANGUAGES + ["auto"], help="language spoken in the audio (REQUIRED — use a code, or 'auto' to detect by probing the model)")

    # ASR params
    parser.add_argument("--punctuation", type=str2bool, default=True, help="whether to produce punctuation and casing; if False, output is lower-cased with no punctuation")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="maximum number of tokens to generate per audio chunk")

    # alignment params
    parser.add_argument("--align_model", default=None, help="Name of phoneme-level ASR model to do alignment")
    parser.add_argument("--interpolate_method", default="nearest", choices=["nearest", "linear", "ignore"], help="For word .srt, method to assign timestamps to non-aligned words, or merge them into neighbouring.")
    parser.add_argument("--no_align", action='store_true', help="Do not perform phoneme alignment")
    parser.add_argument("--return_char_alignments", action='store_true', help="Return character-level alignments in the output json file")

    # vad params
    parser.add_argument("--vad_method", type=str, default="pyannote", choices=["pyannote", "silero"], help="VAD method to be used")
    parser.add_argument("--vad_onset", type=float, default=0.500, help="Onset threshold for VAD (see pyannote.audio), reduce this if speech is not being detected")
    parser.add_argument("--vad_offset", type=float, default=0.363, help="Offset threshold for VAD (see pyannote.audio), reduce this if speech is not being detected.")
    parser.add_argument("--chunk_size", type=int, default=30, help="Chunk size for merging VAD segments. Default is 30; must stay below Cohere's 35s limit.")

    # diarization params
    parser.add_argument("--diarize", action="store_true", help="Apply diarization to assign speaker labels to each segment/word")
    parser.add_argument("--min_speakers", default=None, type=int, help="Minimum number of speakers to in audio file")
    parser.add_argument("--max_speakers", default=None, type=int, help="Maximum number of speakers to in audio file")
    parser.add_argument("--diarize_model", default=DEFAULT_DIARIZE_MODEL, type=str, help="Name of the speaker diarization model to use")
    parser.add_argument("--speaker_embeddings", action="store_true", help="Include speaker embeddings in JSON output (only works with --diarize)")

    # subtitle formatting
    parser.add_argument("--max_line_width", type=optional_int, default=None, help="(not possible with --no_align) the maximum number of characters in a line before breaking the line")
    parser.add_argument("--max_line_count", type=optional_int, default=None, help="(not possible with --no_align) the maximum number of lines in a segment")
    parser.add_argument("--highlight_words", type=str2bool, default=False, help="(not possible with --no_align) underline each word as it is spoken in srt and vtt")

    parser.add_argument("--threads", type=optional_int, default=0, help="number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS")

    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace Access Token to access the gated Cohere and PyAnnote models")

    parser.add_argument("--print_progress", type=str2bool, default=False, help="if True, progress will be printed in transcribe() and align() methods.")
    parser.add_argument("--version", "-V", action="version", version=f"%(prog)s {__version__}", help="Show coherex version information and exit")
    parser.add_argument("--python-version", "-P", action="version", version=f"Python {platform.python_version()} ({platform.python_implementation()})", help="Show python version information and exit")
    # fmt: on

    args = parser.parse_args().__dict__

    log_level = args.get("log_level")
    verbose = args.get("verbose")

    if log_level is not None:
        setup_logging(level=log_level)
    elif verbose:
        setup_logging(level="info")
    else:
        setup_logging(level="warning")

    from coherex.transcribe import transcribe_task

    transcribe_task(args, parser)


if __name__ == "__main__":
    cli()
