import os
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
from transformers import AutoProcessor, CohereAsrForConditionalGeneration

from coherex.audio import SAMPLE_RATE, load_audio
from coherex.schema import SingleSegment, TranscriptionResult, ProgressCallback
from coherex.vads import Vad, Silero, Pyannote
from coherex.log_utils import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL = "CohereLabs/cohere-transcribe-03-2026"

# Fallback language set used by the vLLM backend, which cannot read the model
# config locally. Matches the base Cohere Transcribe model.
DEFAULT_SUPPORTED_LANGUAGES = [
    "en", "fr", "de", "es", "it", "pt", "nl", "pl", "el", "ar", "ja", "zh", "vi", "ko",
]


@dataclass
class CohereAsrOptions:
    """Decoding options for the Cohere Transcribe model."""
    punctuation: bool = True
    max_new_tokens: int = 256


class CohereAsrPipeline:
    """
    VAD-chunked transcription pipeline for the Cohere Transcribe model.

    Mirrors the control flow of WhisperX's FasterWhisperPipeline: run VAD,
    merge speech into chunks no longer than ``chunk_size`` seconds, then
    transcribe each chunk. Segment start/end times come from the VAD, which
    is exactly what the downstream forced-alignment stage expects.
    """

    def __init__(
        self,
        model: CohereAsrForConditionalGeneration,
        processor: AutoProcessor,
        vad,
        vad_params: dict,
        options: CohereAsrOptions,
        device: Union[str, torch.device],
        language: Optional[str] = None,
        batch_size: int = 8,
        backend=None,
    ):
        self.model = model
        self.processor = processor
        self.vad_model = vad
        self._vad_params = vad_params
        self.options = options
        self.preset_language = language
        self.batch_size = batch_size
        self.backend = backend
        if isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device(device)
        if backend is not None:
            self.supported_languages = list(backend.supported_languages)
        else:
            self.supported_languages = list(getattr(model.config, "supported_languages", []))

    def _validate_language(self, language: Optional[str]) -> str:
        if language is None:
            raise ValueError(
                "Cohere Transcribe does not perform language detection. "
                "Pass an explicit language code (one of: "
                f"{', '.join(self.supported_languages)})."
            )
        if self.supported_languages and language not in self.supported_languages:
            raise ValueError(
                f"Unsupported language '{language}'. Cohere Transcribe supports: "
                f"{', '.join(self.supported_languages)}."
            )
        return language

    def transcribe_batch(self, waveforms: List[np.ndarray], language: str) -> List[str]:
        """Transcribe a batch of ≤chunk_size-second waveform slices in one forward pass."""
        if self.backend is not None:
            return self.backend.transcribe_batch(waveforms, language)
        inputs = self.processor(
            waveforms,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            language=language,
            punctuation=self.options.punctuation,
        )
        # Each slice is ≤chunk_size (< model's 35s limit) so it maps to a single
        # internal audio chunk; audio_chunk_index lets decode() regroup per slice.
        audio_chunk_index = inputs.get("audio_chunk_index")
        inputs = inputs.to(self.device, dtype=self.model.dtype)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs, max_new_tokens=self.options.max_new_tokens
            )
        texts = self.processor.decode(
            outputs,
            skip_special_tokens=True,
            audio_chunk_index=audio_chunk_index,
            language=language,
        )
        if isinstance(texts, str):
            texts = [texts]
        return [t.strip() for t in texts]

    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        language: Optional[str] = None,
        batch_size: Optional[int] = None,
        chunk_size: int = 30,
        print_progress: bool = False,
        combined_progress: bool = False,
        verbose: bool = False,
        progress_callback: ProgressCallback = None,
    ) -> TranscriptionResult:
        if isinstance(audio, str):
            audio = load_audio(audio)

        language = self._validate_language(language or self.preset_language)
        batch_size = batch_size or self.batch_size

        # Pre-process audio and merge chunks as defined by the VAD backend.
        if issubclass(type(self.vad_model), Vad):
            waveform = self.vad_model.preprocess_audio(audio)
            merge_chunks = self.vad_model.merge_chunks
        else:
            waveform = Pyannote.preprocess_audio(audio)
            merge_chunks = Pyannote.merge_chunks

        vad_segments = self.vad_model({"waveform": waveform, "sample_rate": SAMPLE_RATE})
        vad_segments = merge_chunks(
            vad_segments,
            chunk_size,
            onset=self._vad_params["vad_onset"],
            offset=self._vad_params["vad_offset"],
        )

        segments: List[SingleSegment] = []
        total_segments = len(vad_segments)
        for batch_start in range(0, total_segments, batch_size):
            batch = vad_segments[batch_start:batch_start + batch_size]
            waveforms = [
                audio[int(c["start"] * SAMPLE_RATE):int(c["end"] * SAMPLE_RATE)]
                for c in batch
            ]
            texts = self.transcribe_batch(waveforms, language)

            for chunk, text in zip(batch, texts):
                if verbose:
                    print(f"Transcript: [{round(chunk['start'], 3)} --> {round(chunk['end'], 3)}] {text}")
                segments.append(
                    {
                        "text": text,
                        "start": round(chunk["start"], 3),
                        "end": round(chunk["end"], 3),
                    }
                )

            processed = min(batch_start + batch_size, total_segments)
            if print_progress:
                base_progress = (processed / total_segments) * 100
                percent_complete = base_progress / 2 if combined_progress else base_progress
                print(f"Progress: {percent_complete:.2f}%...")
            if progress_callback is not None:
                progress_callback((processed / total_segments) * 100)

        return {"segments": segments, "language": language}

    def shutdown(self):
        """Release the ASR backend. Stops a vLLM server if CohereX started one."""
        if self.backend is not None:
            self.backend.shutdown()


def _resolve_dtype(compute_type: str, device: str) -> torch.dtype:
    if compute_type == "default":
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        logger.info(f"Compute type not specified, defaulting to {dtype} for device {device}")
        return dtype
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if compute_type not in mapping:
        raise ValueError(f"Invalid compute_type: {compute_type}. Choose from {list(mapping)} or 'default'.")
    return mapping[compute_type]


def load_model(
    model_name: str = DEFAULT_MODEL,
    device: str = "cpu",
    device_index: int = 0,
    compute_type: str = "default",
    language: Optional[str] = None,
    batch_size: int = 8,
    asr_options: Optional[dict] = None,
    vad_model: Optional[Vad] = None,
    vad_method: Optional[str] = "pyannote",
    vad_options: Optional[dict] = None,
    download_root: Optional[str] = None,
    local_files_only: bool = False,
    use_auth_token: Optional[Union[str, bool]] = None,
    backend: str = "local",
    vllm_url: Optional[str] = None,
    vllm_api_key: Optional[str] = None,
    vllm_args: Optional[List[str]] = None,
) -> CohereAsrPipeline:
    """Load the Cohere Transcribe model for inference.

    Args:
        model_name: HuggingFace repo id of the Cohere ASR model.
        device: 'cpu' or 'cuda'.
        device_index: CUDA device index.
        compute_type: 'default', 'bfloat16', 'float16', or 'float32'.
        language: Optional preset language code (required at transcribe time).
        batch_size: Number of VAD chunks to transcribe per generate() call.
        asr_options: Overrides for CohereAsrOptions (e.g. punctuation, max_new_tokens).
        vad_model: A manually assigned VAD model (takes priority over vad_method).
        vad_method: 'pyannote' or 'silero'.
        vad_options: Overrides for VAD parameters.
        download_root: Cache directory for model files.
        local_files_only: Use only cached files, do not download.
        use_auth_token: HuggingFace token for the gated Cohere model.
        backend: 'local' runs the model in-process; 'vllm' offloads transcription
            to a vLLM server.
        vllm_url: URL of a running vLLM server. If None with backend='vllm',
            CohereX starts (and later stops) its own vLLM server.
        vllm_api_key: API key for the vLLM server, if it requires one.
        vllm_args: Extra 'vllm serve' arguments used when CohereX launches its
            own server (e.g. ['--max-model-len', '448']).

    Returns:
        A CohereAsrPipeline.
    """
    remote_backend = None
    if backend == "vllm":
        remote_backend = _build_vllm_backend(model_name, vllm_url, vllm_api_key, vllm_args)
        model = None
        processor = None
    elif backend == "local":
        dtype = _resolve_dtype(compute_type, device)
        target_device = f"cuda:{device_index}" if device == "cuda" else device

        processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=download_root,
            local_files_only=local_files_only,
            token=use_auth_token,
        )
        model = CohereAsrForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            cache_dir=download_root,
            local_files_only=local_files_only,
            token=use_auth_token,
        ).to(target_device)
        model.eval()
    else:
        raise ValueError(f"Invalid backend: {backend!r}. Choose 'local' or 'vllm'.")

    target_device = f"cuda:{device_index}" if device == "cuda" else device

    default_asr_options = {"punctuation": True, "max_new_tokens": 256}
    if asr_options is not None:
        default_asr_options.update(asr_options)
    options = CohereAsrOptions(**default_asr_options)

    default_vad_options = {
        "chunk_size": 30,  # kept below Cohere's 35s hard limit
        "vad_onset": 0.500,
        "vad_offset": 0.363,
    }
    if vad_options is not None:
        default_vad_options.update(vad_options)

    # Manually assigned vad_model has higher priority than vad_method.
    if vad_model is not None:
        logger.info("Using manually assigned vad_model. vad_method is ignored.")
    else:
        if vad_method == "silero":
            vad_model = Silero(**default_vad_options)
        elif vad_method == "pyannote":
            device_vad = target_device if device == "cuda" else device
            vad_model = Pyannote(torch.device(device_vad), token=use_auth_token, **default_vad_options)
        else:
            raise ValueError(f"Invalid vad_method: {vad_method}")

    return CohereAsrPipeline(
        model=model,
        processor=processor,
        vad=vad_model,
        vad_params=default_vad_options,
        options=options,
        device=target_device,
        language=language,
        batch_size=batch_size,
        backend=remote_backend,
    )


def _build_vllm_backend(model_name, vllm_url, vllm_api_key, vllm_args):
    """Connect to a vLLM server, or start one if no URL is given."""
    from coherex.vllm_backend import ManagedVLLMServer, VLLMBackend

    if vllm_url:
        logger.info(f"Using vLLM backend at {vllm_url}")
        return VLLMBackend(
            base_url=vllm_url,
            model=model_name,
            api_key=vllm_api_key,
            supported_languages=DEFAULT_SUPPORTED_LANGUAGES,
        )

    server = ManagedVLLMServer(model_name, api_key=vllm_api_key, extra_args=vllm_args).start()
    return VLLMBackend(
        base_url=server.base_url,
        model=model_name,
        api_key=vllm_api_key,
        supported_languages=DEFAULT_SUPPORTED_LANGUAGES,
        server=server,
    )
