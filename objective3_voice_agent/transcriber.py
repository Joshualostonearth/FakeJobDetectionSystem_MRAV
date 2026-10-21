"""
transcriber.py — Step 5: Transcribe the call recording using faster-whisper.

The WhisperModel is loaded once at module level and reused across calls.
Never reload the model per call — it's expensive.
"""

import logging
from typing import Optional

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Singleton model instance
# ──────────────────────────────────────────────────────────
_model: Optional[WhisperModel] = None


def get_model(config: dict) -> WhisperModel:
    """
    Return the shared WhisperModel, initialising it on first call.

    Args:
        config: voice_agent_config dict containing whisper_model_size,
                whisper_device, and whisper_compute_type.
    """
    global _model
    if _model is None:
        model_size = config.get("whisper_model_size", "base")
        device = config.get("whisper_device", "cpu")
        compute_type = config.get("whisper_compute_type", "int8")

        logger.info(
            f"[Transcriber] Loading Whisper model '{model_size}' "
            f"on {device} ({compute_type})"
        )
        _model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logger.info("[Transcriber] Whisper model loaded.")
    return _model


def transcribe_recording(wav_path: str, config: dict) -> dict:
    """
    Transcribe a WAV recording and return structured metadata.

    Args:
        wav_path: Local path to the WAV file
        config:   voice_agent_config dict

    Returns:
        dict with keys:
            transcript       (str)   — full transcribed text
            language         (str)   — detected language code
            duration_seconds (float) — audio duration
            word_count       (int)   — number of words in transcript
            confidence       (float) — average log-probability across segments
    """
    model = get_model(config)

    logger.info(f"[Transcriber] Transcribing: {wav_path}")
    segments, info = model.transcribe(wav_path, beam_size=5)

    all_text: list[str] = []
    all_probs: list[float] = []

    for seg in segments:
        text = seg.text.strip()
        if text:
            all_text.append(text)
            all_probs.append(seg.avg_logprob)

    transcript = " ".join(all_text).strip()
    avg_confidence = round(sum(all_probs) / len(all_probs), 4) if all_probs else 0.0

    result = {
        "transcript": transcript,
        "language": info.language,
        "duration_seconds": round(info.duration, 2),
        "word_count": len(transcript.split()) if transcript else 0,
        "confidence": avg_confidence,
    }

    logger.info(
        f"[Transcriber] Done — {result['word_count']} words, "
        f"lang={result['language']}, conf={result['confidence']}"
    )
    return result
