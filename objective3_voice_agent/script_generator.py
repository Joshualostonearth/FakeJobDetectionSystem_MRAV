"""
script_generator.py — Step 2: Generate verification call script and synthesise audio.

Uses a LangChain PromptTemplate to build the script dynamically, then
synthesises it to MP3 using edge-tts (Microsoft Edge TTS, no API key needed).
"""

import asyncio
import logging
import os
from pathlib import Path

import edge_tts
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# Script template (LangChain PromptTemplate)
# ──────────────────────────────────────────────────────────

SCRIPT_TEMPLATE = PromptTemplate.from_template(
    "Hello. This is an automated verification call on behalf of a job applicant. "
    "This call may be recorded for verification purposes.\n\n"
    "We are calling to confirm whether {company_name} currently has an active opening "
    "for a {role} position based in {location}.\n\n"
    "{recruiter_line}\n\n"
    "You do not need to share any confidential information. A simple yes or no is "
    "sufficient. Thank you for your time."
)


# ──────────────────────────────────────────────────────────
# Voice selection
# ──────────────────────────────────────────────────────────

def select_voice(location: str, config: dict) -> str:
    """
    Select the appropriate edge-tts voice based on the job location string.

    Falls back to voice_fallback if no region is detected.
    """
    loc = location.lower()
    if any(x in loc for x in ["india", "bangalore", "mumbai", "delhi", "hyderabad", "pune", "chennai"]):
        return config.get("voice_india", "en-IN-NeerjaNeural")
    elif any(x in loc for x in ["uk", "london", "england", "manchester", "birmingham"]):
        return config.get("voice_uk", "en-GB-SoniaNeural")
    elif any(x in loc for x in ["us", "usa", "new york", "california", "san francisco", "seattle", "chicago"]):
        return config.get("voice_us", "en-US-AriaNeural")
    return config.get("voice_fallback", "en-US-JennyNeural")


# ──────────────────────────────────────────────────────────
# Script builder
# ──────────────────────────────────────────────────────────

def build_script(entity: dict) -> str:
    """
    Build the call script from entity fields using the LangChain PromptTemplate.

    The recruiter_line is included only when recruiter_name is a non-empty string.
    The mandatory disclosure line always precedes any question.
    """
    recruiter_name = entity.get("recruiter_name") or ""
    recruiter_line = (
        f"Could you confirm whether {recruiter_name} is part of your recruitment team?"
        if recruiter_name.strip()
        else ""
    )

    script = SCRIPT_TEMPLATE.format(
        company_name=entity.get("company_name", "the company"),
        role=entity.get("role", entity.get("job_role", "the advertised role")),
        location=entity.get("location", "your location"),
        recruiter_line=recruiter_line,
    )

    # Collapse consecutive blank lines and strip leading/trailing whitespace
    import re as _re
    script = _re.sub(r'\n{3,}', '\n\n', script).strip()

    word_count = len(script.split())
    if word_count > 80:
        logger.warning(
            f"[ScriptGenerator] Script is {word_count} words (target ≤ 80). "
            "Consider shortening entity fields."
        )

    logger.debug(f"[ScriptGenerator] Script ({word_count} words):\n{script}")
    return script


# ──────────────────────────────────────────────────────────
# Audio synthesis
# ──────────────────────────────────────────────────────────

async def _synthesise_audio(script: str, voice: str, output_path: str) -> None:
    """Async synthesis using edge-tts. Saves MP3 to output_path."""
    communicate = edge_tts.Communicate(script, voice)
    await communicate.save(output_path)
    logger.info(f"[ScriptGenerator] Audio saved to {output_path} (voice={voice})")


async def generate_verification_audio(
    entity: dict,
    output_path: str,
    config: dict,
) -> str:
    """
    Build call script, select voice, and synthesise to MP3.

    Args:
        entity:      Job offer entity dict (company_name, role, recruiter_name, location)
        output_path: Destination path for the MP3 file
        config:      voice_agent_config dict (voice names)

    Returns:
        The output_path where the MP3 was saved.
    """
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    script = build_script(entity)
    voice = select_voice(entity.get("location", ""), config)

    logger.info(f"[ScriptGenerator] Synthesising with voice: {voice}")
    await _synthesise_audio(script, voice, output_path)

    return output_path
