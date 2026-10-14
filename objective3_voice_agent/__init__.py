# objective3_voice_agent — Voice Verification Agent
# Public API re-exported at package level for cross-objective imports.

import sys
from pathlib import Path

# Add this package's directory to sys.path so that internal imports
# (e.g. `from voice_schemas import ...` inside voice_agent.py) resolve correctly
# when the package is imported from the project root.
_pkg_dir = str(Path(__file__).parent)
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from objective3_voice_agent.voice_agent import run_voice_verification, load_config
from objective3_voice_agent.voice_to_mass import voice_transcript_to_mass, voice_result_to_analysis
from voice_schemas import PipelineResult, VoiceVerificationResult, SourcedNumber

__all__ = [
    "run_voice_verification",
    "load_config",
    "voice_transcript_to_mass",
    "voice_result_to_analysis",
    "PipelineResult",
    "VoiceVerificationResult",
    "SourcedNumber",
]
