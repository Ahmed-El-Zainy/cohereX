"""
Optional spoken-language detection for CohereX.

Cohere Transcribe has no built-in language detection, so this module probes the
model itself: it transcribes a short clip in each candidate language and keeps
the language whose output is longest and most self-consistent (verified with a
lightweight text language detector). Non-Latin scripts are separated cleanly by
`langdetect`; among Latin-script languages, output length + detector agreement
provide the signal.

This is an optional convenience — passing an explicit ``--language`` is always
faster and more reliable.
"""
from typing import List, Optional, Union

import numpy as np

from coherex.audio import SAMPLE_RATE, load_audio
from coherex.log_utils import get_logger

logger = get_logger(__name__)

# Normalize langdetect codes to the codes Cohere uses.
_LANGDETECT_MAP = {"zh-cn": "zh", "zh-tw": "zh"}

# Languages written without word spaces — measure diversity over characters.
_NO_SPACE_LANGS = {"zh", "ja", "ko"}


def _diversity(text: str, lang: str) -> float:
    """Fraction of distinct tokens — low values flag repetitive hallucinations."""
    tokens = list(text) if lang in _NO_SPACE_LANGS else text.split()
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def detect_language(
    pipeline,
    audio: Union[str, np.ndarray],
    candidates: Optional[List[str]] = None,
    probe_seconds: float = 20.0,
    max_new_tokens: int = 48,
    verbose: bool = False,
) -> str:
    """Detect the spoken language by probing the Cohere model.

    Args:
        pipeline: A loaded CohereAsrPipeline.
        audio: Path or waveform array.
        candidates: Language codes to consider (defaults to all supported).
        probe_seconds: Length of the leading clip used for probing.
        max_new_tokens: Cap on tokens generated per probe.

    Returns:
        The detected language code.
    """
    if isinstance(audio, str):
        audio = load_audio(audio)

    candidates = candidates or pipeline.supported_languages
    if not candidates:
        raise ValueError("No candidate languages available for detection.")

    probe = audio[: int(probe_seconds * SAMPLE_RATE)]

    try:
        from langdetect import DetectorFactory, detect_langs
        DetectorFactory.seed = 0
        have_langdetect = True
    except ImportError:
        have_langdetect = False
        logger.warning(
            "langdetect not installed; falling back to output-length heuristic "
            "(weak among Latin-script languages). Install with: pip install langdetect"
        )

    # Temporarily cap generation length for cheap probing.
    original_max = pipeline.options.max_new_tokens
    pipeline.options.max_new_tokens = max_new_tokens
    try:
        scores = {}
        for lang in candidates:
            text = pipeline.transcribe_batch([probe], lang)[0]
            if not text:
                scores[lang] = 0.0
                continue

            agree = 0.0
            if have_langdetect:
                try:
                    for result in detect_langs(text):
                        code = _LANGDETECT_MAP.get(result.lang, result.lang)
                        if code == lang:
                            agree = result.prob
                            break
                except Exception:
                    pass

            # Length rewards coherent output; detector agreement is a strong
            # booster; diversity (squared) suppresses repetitive hallucinations
            # that fluent wrong-language decoding tends to produce.
            diversity = _diversity(text, lang)
            scores[lang] = len(text) * (1.0 + 3.0 * agree) * (diversity ** 2)
            if verbose:
                logger.info(
                    f"[langid] {lang}: len={len(text)} agree={agree:.2f} "
                    f"div={diversity:.2f} score={scores[lang]:.0f} :: {text[:60]}"
                )
    finally:
        pipeline.options.max_new_tokens = original_max

    best = max(scores, key=scores.get)
    logger.info(f"Detected language: {best}")
    return best
