import argparse
import gc
import os
import shlex

import torch

from coherex.alignment import align, load_align_model
from coherex.asr import DEFAULT_MODEL, load_model
from coherex.audio import load_audio
from coherex.diarize import DiarizationPipeline, assign_word_speakers
from coherex.langid import detect_language
from coherex.schema import AlignedTranscriptionResult, TranscriptionResult
from coherex.utils import get_writer
from coherex.log_utils import get_logger

logger = get_logger(__name__)


def transcribe_task(args: dict, parser: argparse.ArgumentParser):
    """Transcription task to be called from CLI.

    Args:
        args: Dictionary of command-line arguments.
        parser: argparse.ArgumentParser object.
    """
    model_name: str = args.pop("model")
    batch_size: int = args.pop("batch_size")
    model_dir: str = args.pop("model_dir")
    model_cache_only: bool = args.pop("model_cache_only")
    output_dir: str = args.pop("output_dir")
    output_format: str = args.pop("output_format")
    device: str = args.pop("device")
    device_index: int = args.pop("device_index")
    compute_type: str = args.pop("compute_type")
    verbose: bool = args.pop("verbose")

    backend: str = args.pop("backend")
    vllm_url: str = args.pop("vllm_url")
    vllm_api_key: str = args.pop("vllm_api_key") or os.environ.get("VLLM_API_KEY")
    vllm_args_raw: str = args.pop("vllm_args")

    os.makedirs(output_dir, exist_ok=True)

    align_model_name: str = args.pop("align_model")
    interpolate_method: str = args.pop("interpolate_method")
    no_align: bool = args.pop("no_align")
    return_char_alignments: bool = args.pop("return_char_alignments")

    hf_token: str = args.pop("hf_token")
    vad_method: str = args.pop("vad_method")
    vad_onset: float = args.pop("vad_onset")
    vad_offset: float = args.pop("vad_offset")
    chunk_size: int = args.pop("chunk_size")

    diarize: bool = args.pop("diarize")
    min_speakers: int = args.pop("min_speakers")
    max_speakers: int = args.pop("max_speakers")
    diarize_model_name: str = args.pop("diarize_model")
    print_progress: bool = args.pop("print_progress")
    return_speaker_embeddings: bool = args.pop("speaker_embeddings")

    if return_speaker_embeddings and not diarize:
        parser.error("--speaker_embeddings requires --diarize")

    # Cohere Transcribe has no language detection; require an explicit code or 'auto'.
    language: str = args.pop("language")
    auto_detect = language == "auto"
    if language is None:
        parser.error(
            "--language is required: pass a language code or 'auto' to detect it."
        )

    if (threads := args.pop("threads")) > 0:
        torch.set_num_threads(threads)

    asr_options = {
        "punctuation": args.pop("punctuation"),
        "max_new_tokens": args.pop("max_new_tokens"),
    }

    writer = get_writer(output_format, output_dir)
    word_options = ["highlight_words", "max_line_count", "max_line_width"]
    if no_align:
        for option in word_options:
            if args[option]:
                parser.error(f"--{option} not possible with --no_align")
    if args["max_line_count"] and not args["max_line_width"]:
        logger.warning("--max_line_count has no effect without --max_line_width")
    writer_args = {arg: args.pop(arg) for arg in word_options}

    # Part 1: VAD & ASR
    model = load_model(
        model_name,
        device=device,
        device_index=device_index,
        compute_type=compute_type,
        language=None if auto_detect else language,
        batch_size=batch_size,
        asr_options=asr_options,
        vad_method=vad_method,
        vad_options={
            "chunk_size": chunk_size,
            "vad_onset": vad_onset,
            "vad_offset": vad_offset,
        },
        download_root=model_dir,
        local_files_only=model_cache_only,
        use_auth_token=hf_token,
        backend=backend,
        vllm_url=vllm_url,
        vllm_api_key=vllm_api_key,
        vllm_args=shlex.split(vllm_args_raw) if vllm_args_raw else None,
    )

    results = []
    audio = None
    for audio_path in args.pop("audio"):
        audio = load_audio(audio_path)
        if auto_detect:
            logger.info("Detecting language...")
            detected = detect_language(model, audio, verbose=verbose)
        else:
            detected = language
        logger.info(f"Performing transcription ({detected})...")
        result: TranscriptionResult = model.transcribe(
            audio,
            language=detected,
            batch_size=batch_size,
            chunk_size=chunk_size,
            print_progress=print_progress,
            verbose=verbose,
        )
        results.append((result, audio_path, detected))

    # Unload ASR model / VAD, and stop the vLLM server if CohereX started one.
    model.shutdown()
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # Part 2: Align
    if not no_align:
        tmp_results = results
        results = []
        align_model = align_metadata = None
        for result, audio_path, lang in tmp_results:
            input_audio = audio_path if len(tmp_results) > 1 else audio

            # (Re)load the alignment model when the language changes between files.
            if align_metadata is None or align_metadata["language"] != lang:
                align_model, align_metadata = load_align_model(
                    lang, device, model_name=align_model_name, model_dir=model_dir, model_cache_only=model_cache_only
                )

            if align_model is not None and len(result["segments"]) > 0:
                logger.info("Performing alignment...")
                result: AlignedTranscriptionResult = align(
                    result["segments"],
                    align_model,
                    align_metadata,
                    input_audio,
                    device,
                    interpolate_method=interpolate_method,
                    return_char_alignments=return_char_alignments,
                    print_progress=print_progress,
                )

            results.append((result, audio_path, lang))

        del align_model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # Part 3: Diarize
    if diarize:
        if hf_token is None:
            logger.warning(
                "No --hf_token provided; the diarization model is gated and will fail to load without it."
            )
        tmp_results = results
        logger.info(f"Performing diarization using model: {diarize_model_name}")
        results = []
        diarize_model = DiarizationPipeline(
            model_name=diarize_model_name, token=hf_token, device=device, cache_dir=model_dir
        )
        for result, input_audio_path, lang in tmp_results:
            diarize_result = diarize_model(
                input_audio_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                return_embeddings=return_speaker_embeddings,
            )

            if return_speaker_embeddings:
                diarize_segments, speaker_embeddings = diarize_result
            else:
                diarize_segments = diarize_result
                speaker_embeddings = None

            result = assign_word_speakers(diarize_segments, result, speaker_embeddings)
            results.append((result, input_audio_path, lang))

    # Part 4: Write
    for result, audio_path, lang in results:
        result["language"] = lang
        writer(result, audio_path, writer_args)
