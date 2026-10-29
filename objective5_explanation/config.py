"""
config.py — LLM provider settings for the Objective 5 explanation module.

API keys are intentionally left empty. Either paste a key below for local use
(do NOT commit it), or set the GROQ_API_KEY / OPENAI_API_KEY environment
variables (e.g. in the git-ignored .env file). When no key is available the
module falls back to a deterministic, template-based explanation.
"""

# ── Groq (primary) ─────────────────────────────────────────
GROQ_API_KEY = ""
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"

# ── OpenAI GPT (fallback) ──────────────────────────────────
OPENAI_API_KEY = ""
OPENAI_MODEL = "gpt-4o-mini"

# ── Generation settings ────────────────────────────────────
TEMPERATURE = 0.2
MAX_TOKENS = 900
REQUEST_TIMEOUT_SECONDS = 60.0
