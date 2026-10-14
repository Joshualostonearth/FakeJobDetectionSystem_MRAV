"""
audio_server.py — Step 3a: FastAPI web server, SSE stream, and dashboard.

Responsibilities:
  - Serves the live test-call dashboard (GET /)
  - Accepts test-call form submissions (POST /test-call) — runs pipeline in background thread
  - Streams live pipeline step events to browser via SSE (GET /stream/{call_id})
  - Checks WebRTC bridge health (GET /bridge-health)

Note: The actual calling infrastructure uses WebRTC (webrtc-call/server) +
      call_bridge (whisper_Test/call_bridge.py). This server handles the
      Objective 3 pipeline orchestration and dashboard only.
"""

import asyncio
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

load_dotenv(override=False)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────

app = FastAPI(title="Objective 3 — Voice Verification Agent (WebRTC)")

# Ensure directories exist before mounting
Path("temp/audio").mkdir(parents=True, exist_ok=True)

app.mount("/audio", StaticFiles(directory="temp/audio"), name="audio")

# Resolve template/static paths relative to this file so the server works
# from any working directory
_HERE = Path(__file__).parent
templates = Jinja2Templates(directory=str(_HERE / "templates"))

# Serve static CSS/JS
app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")

# ──────────────────────────────────────────────────────────
# Shared state
# ──────────────────────────────────────────────────────────

# Per-call SSE queues: call_id → asyncio.Queue of event dicts
call_queues: dict[str, asyncio.Queue] = {}

# Thread pool for running the synchronous pipeline
_executor = ThreadPoolExecutor(max_workers=4)

# Runtime config — injected by voice_agent.py at startup
_config: dict = {}


def set_config(config: dict) -> None:
    """Called by voice_agent.py to inject runtime config into this module."""
    global _config
    _config = config


# ──────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────

@app.get("/")
async def dashboard(request: Request):
    """Serve the live test-call dashboard."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/test-call")
async def test_call(request: Request):
    """
    Accept form data from dashboard, assign a call_id, start the pipeline
    in a background thread, return call_id immediately so the browser can
    open the SSE stream.
    """
    form = await request.form()

    entity = {
        "company_name":  form.get("company_name", ""),
        "role":          form.get("role", ""),
        "recruiter_name": form.get("recruiter_name") or None,
        "location":      form.get("location", ""),
    }
    call_id = f"test_{int(time.time())}"

    # Create SSE queue for this call
    call_queues[call_id] = asyncio.Queue()

    # Run pipeline in thread pool (it's synchronous)
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        _executor,
        _run_pipeline_sync,
        call_id,
        entity,
        loop,
    )

    return JSONResponse({"call_id": call_id})


@app.get("/stream/{call_id}")
async def stream_events(call_id: str):
    """
    SSE endpoint — browser connects here to receive live pipeline step events
    and the final result payload.
    """
    if call_id not in call_queues:
        # Create queue in case client connects before POST returns
        call_queues[call_id] = asyncio.Queue()

    async def event_generator():
        q = call_queues[call_id]
        try:
            while True:
                msg = await asyncio.wait_for(q.get(), timeout=120)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("type") == "result":
                    break
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'type': 'error', 'text': 'Stream timed out'})}\n\n"
        finally:
            call_queues.pop(call_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/bridge-health")
async def bridge_health():
    """
    Check if the WebRTC call bridge (call_bridge.py) is reachable.
    """
    import requests as req

    bridge_url = _config.get("webrtc_bridge_url", "http://localhost:5000")

    try:
        resp = req.get(f"{bridge_url}/bridge/status", timeout=5)
        if resp.ok:
            return JSONResponse({
                "bridge": "connected",
                "bridge_url": bridge_url,
                "session": resp.json(),
            })
        else:
            return JSONResponse(
                {"bridge": "error", "bridge_url": bridge_url, "status_code": resp.status_code},
                status_code=502,
            )
    except req.ConnectionError:
        return JSONResponse(
            {"bridge": "disconnected", "bridge_url": bridge_url,
             "hint": "Start the bridge: python whisper_Test/call_bridge.py"},
            status_code=503,
        )
    except Exception as e:
        return JSONResponse(
            {"bridge": "error", "bridge_url": bridge_url, "error": str(e)},
            status_code=500,
        )


# ──────────────────────────────────────────────────────────
# Pipeline runner (synchronous, runs in thread pool)
# ──────────────────────────────────────────────────────────

def _push_event(loop: asyncio.AbstractEventLoop, call_id: str, event: dict) -> None:
    """Thread-safe SSE event push — bridges sync thread → async queue."""
    if call_id in call_queues:
        asyncio.run_coroutine_threadsafe(
            call_queues[call_id].put(event), loop
        )


def _run_pipeline_sync(
    call_id: str,
    entity: dict,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """
    Synchronous wrapper that runs the pipeline and pushes SSE events.
    Executed in a ThreadPoolExecutor worker thread.
    """
    def push(text: str, step: int = None):
        prefix = f"[{step}] " if step else ""
        _push_event(loop, call_id, {"type": "step", "text": f"{prefix}{text}"})

    try:
        from voice_agent import run_voice_verification

        result = run_voice_verification(
            claim_id=call_id,
            entity=entity,
            config=_config,
            event_push=push,
        )
        _push_event(loop, call_id, {"type": "result", "data": result})

    except Exception as e:
        logger.error(f"[AudioServer] Pipeline error for {call_id}: {e}", exc_info=True)
        _push_event(loop, call_id, {
            "type": "result",
            "data": {
                "claim_id": call_id,
                "pipeline_completed": False,
                "error": str(e),
            },
        })


# ──────────────────────────────────────────────────────────
# Server startup
# ──────────────────────────────────────────────────────────

def start_server(port: int) -> None:
    """
    Start the FastAPI/uvicorn server in a background daemon thread.

    Args:
        port: Local port to bind to
    """
    import threading

    def _run():
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="warning",
        )

    thread = threading.Thread(target=_run, daemon=True, name="uvicorn-server")
    thread.start()
    time.sleep(1.5)  # Give uvicorn time to bind
    logger.info(f"[AudioServer] Uvicorn listening on port {port}")
