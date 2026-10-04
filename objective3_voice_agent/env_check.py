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
# OBJECTIVE 3 — VOICE VERIFICATION AGENT
# Fill in all values before running the pipeline
# ============================================================

TWILIO_ACCOUNT_SID=""
TWILIO_AUTH_TOKEN=""
TWILIO_PHONE_NUMBER=""
GOOGLE_PLACES_API_KEY=""
SERPAPI_KEY=""
OPENCORPORATES_API_KEY=""
NGROK_AUTHTOKEN=""
OLLAMA_MODEL="llama3.2"
"""

REQUIRED_KEYS = ["TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_PHONE_NUMBER"]
OPTIONAL_KEYS = [
    "GOOGLE_PLACES_API_KEY",
    "SERPAPI_KEY",
    "OPENCORPORATES_API_KEY",
    "NGROK_AUTHTOKEN",
]


# ──────────────────────────────────────────────────────────
# check_env()
# ──────────────────────────────────────────────────────────

def check_env() -> None:
    """
    Validate .env file and required keys.

    - If .env does not exist: create the template and exit (developer must fill it in).
    - If .env exists: load it, print per-key status, hard-exit if any required key is empty.
    """
    env_path = Path(".env")

    if not env_path.exists():
        env_path.write_text(ENV_TEMPLATE, encoding="utf-8")
        print("✅ .env file created from template.")
        print("   Fill in your API keys then run again:")
        print("   Required: TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER")
        sys.exit(0)

    load_dotenv(dotenv_path=env_path, override=True)

    print(".env found. Key status:")

    missing_required: list[str] = []

    for key in REQUIRED_KEYS:
        val = os.getenv(key, "")
        status = "✅ set" if val else "❌ MISSING"
        print(f"  {key:<30}: {status}")
        if not val:
            missing_required.append(key)

    optional_labels = {
        "GOOGLE_PLACES_API_KEY":  "empty — Google Places disabled",
        "SERPAPI_KEY":            "empty — SerpAPI fallback disabled",
        "OPENCORPORATES_API_KEY": "empty — OpenCorporates disabled (optional)",
        "NGROK_AUTHTOKEN":        "empty — Ngrok disabled (set use_ngrok: false in config)",
    }

    for key in OPTIONAL_KEYS:
        val = os.getenv(key, "")
        if val:
            status = "✅ set"
        else:
            status = f"⚠️  {optional_labels.get(key, 'empty')}"
        print(f"  {key:<30}: {status}")

    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
    print(f"  {'OLLAMA_MODEL':<30}: ✅ set ({ollama_model})")

    if missing_required:
        print(f"\nERROR: Required keys missing: {', '.join(missing_required)}")
        print("Fill them in .env and try again.")
        sys.exit(1)

    print("\n✅ All required keys present.\n")


# ──────────────────────────────────────────────────────────
# check_ollama()
# ──────────────────────────────────────────────────────────

def check_ollama(model: str = None) -> None:
    """
    Verify Ollama is installed, running, and the target model is available.

    Args:
        model: Model name to check (e.g. "llama3.2"). Reads OLLAMA_MODEL from env if None.
    """
    load_dotenv(override=False)
    model = model or os.getenv("OLLAMA_MODEL", "llama3.2")

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
