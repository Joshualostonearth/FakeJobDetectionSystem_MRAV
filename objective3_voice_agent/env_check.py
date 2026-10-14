"""
env_check.py — Step 0: Environment validation for Objective 3.

Imported by voice_agent.py before anything else runs.
Handles .env creation/reading and Ollama health check.
"""

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


# ──────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────

ENV_TEMPLATE = """\
# ============================================================
# OBJECTIVE 3 — VOICE VERIFICATION AGENT (WebRTC)
# Runs locally; the only setting is the Ollama model.
# ============================================================

OLLAMA_MODEL="ministral-3"
"""

DEFAULT_OLLAMA_MODEL = "ministral-3"


# ──────────────────────────────────────────────────────────
# check_env()
# ──────────────────────────────────────────────────────────

def check_env() -> None:
    """
    Load .env, creating it from the template if it does not exist.

    No API keys are required — the call is simulated over WebRTC and all
    models run locally through Ollama.
    """
    env_path = Path(".env")

    if not env_path.exists():
        env_path.write_text(ENV_TEMPLATE, encoding="utf-8")
        print("✅ .env file created from template.")

    load_dotenv(dotenv_path=env_path, override=True)

    ollama_model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    print(f"  {'OLLAMA_MODEL':<30}: {ollama_model}\n")


# ──────────────────────────────────────────────────────────
# check_ollama()
# ──────────────────────────────────────────────────────────

def check_ollama(model: str = None) -> None:
    """
    Verify Ollama is installed, running, and the target model is available.

    Args:
        model: Model name to check (e.g. "ministral-3"). Reads OLLAMA_MODEL from env if None.
    """
    load_dotenv(override=False)
    model = model or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)

    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if model not in result.stdout:
            print(f"ERROR: Model '{model}' not found in Ollama.")
            print(f"Fix:   ollama pull {model}")
            sys.exit(1)
        print(f"✅ Ollama running. Model '{model}' available.")

    except FileNotFoundError:
        print("ERROR: Ollama is not installed.")
        print("Fix (Linux/Mac): curl https://ollama.ai/install.sh | sh")
        print("Fix (Windows):   winget install Ollama.Ollama")
        sys.exit(1)

    except subprocess.TimeoutExpired:
        print("ERROR: Ollama process timed out. Is the server running?")
        print("Fix:   ollama serve")
        sys.exit(1)
