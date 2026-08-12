import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class KeyPool:
    """Simple wrapper for local Ollama inference. No API keys needed."""

    def __init__(self):
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        self.model = os.getenv("OLLAMA_MODEL", "ministral-3")
        self.client = OpenAI(
            base_url=base_url,
            api_key="ollama",  # Ollama doesn't need a real key
            timeout=300.0,  # 5 minutes — vision inference on CPU is slow
        )
        print(f"[KeyPool] Connected to local Ollama server at {base_url}")

    def execute(self, api_call_fn, max_total_retries=3):
        """Executes api_call_fn(client) with simple retry logic."""
        for attempt in range(max_total_retries):
            try:
                return api_call_fn(self.client)
            except Exception as e:
                print(f"[KeyPool] Error (attempt {attempt + 1}/{max_total_retries}): {e}")
                if attempt == max_total_retries - 1:
                    raise


# Shared global instance
key_pool = KeyPool()
