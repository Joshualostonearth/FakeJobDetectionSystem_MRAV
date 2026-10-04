"""
audio_server.py — Step 3a: FastAPI web server, Twilio webhook, SSE stream, and dashboard.

Responsibilities:
  - Serves static MP3 files to Twilio (/audio/)
  - Receives Twilio recording webhook (POST /webhook/call-complete)
  - Serves the live test-call dashboard (GET /)
  - Accepts test-call form submissions (POST /test-call) — runs pipeline in background thread
  - Streams live pipeline step events to browser via SSE (GET /stream/{call_id})
  - Manages Ngrok tunnel startup
"""

import asyncio
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

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

app = FastAPI(title="Objective 3 — Voice Verification Agent")

# Ensure directories exist before mounting
Path("temp/audio").mkdir(parents=True, exist_ok=True)
Path("temp/recordings").mkdir(parents=True, exist_ok=True)

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

# Recording URLs delivered by Twilio webhook, keyed by call_sid
pending_recordings: dict[str, str] = {}

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
        "phone":         None,  # NEVER from the posting — number sourcing handles this
    }
    override_phone: Optional[str] = form.get("override_phone") or None
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
        override_phone,
        loop,
    )

    return JSONResponse({"call_id": call_id})


@app.post("/test-twilio-only")
async def test_twilio_only(request: Request):
    """
    Test endpoint: dials the override_phone, says a test message, and hangs up.
    Bypasses the entire objective 3 pipeline.
    """
    from twilio.rest import Client
    form = await request.form()
    override_phone = form.get("override_phone")

    if not override_phone:
        return JSONResponse({"status": "error", "message": "Phone number is required."}, status_code=400)

    try:
        client = Client(
            os.getenv("TWILIO_ACCOUNT_SID"),
            os.getenv("TWILIO_AUTH_TOKEN")
        )

        call = client.calls.create(
            url="http://demo.twilio.com/docs/voice.xml",
            to=override_phone,
            from_=os.getenv("TWILIO_PHONE_NUMBER")
        )
        logger.info(f"[TestTwilio] Ping call placed to {override_phone}, SID={call.sid}")
        return JSONResponse({"status": "success", "message": f"Calling {override_phone} now!", "call_sid": call.sid})

    except Exception as e:
        logger.error(f"[TestTwilio] Failed to call {override_phone}: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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


@app.get("/twiml/{claim_id}")
async def get_twiml(claim_id: str, request: Request):
    """
    Twilio fetches this URL to get the TwiML instructions for the call.
    Using a URL bypasses the trial account block on inline 'twiml' parameters.
    """
    from fastapi.responses import Response
    public_base = _config.get("public_base_url", f"https://{request.url.netloc}")
    audio_url = f"{public_base}/audio/{claim_id}_verification.mp3"
    webhook_url = f"{public_base}/webhook/call-complete"
    
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Play>{audio_url}</Play>
    <Record
        maxLength="60"
        timeout="5"
        recordingStatusCallback="{webhook_url}"
        recordingStatusCallbackMethod="POST"
    />
    <Say>Thank you for your response. Goodbye.</Say>
    <Hangup/>
</Response>"""
    return Response(content=twiml, media_type="application/xml")


@app.post("/webhook/call-complete")
async def call_complete_webhook(request: Request):
    """
    Twilio posts recording details here when a recording is ready.
    Stores the recording URL in pending_recordings keyed by CallSid.
    """
    form = await request.form()
    recording_url = form.get("RecordingUrl")
    call_sid = form.get("CallSid")
    call_status = form.get("CallStatus")

    logger.info(
        f"[Webhook] call-complete received — "
        f"CallSid={call_sid}, status={call_status}, "
        f"recording={'yes' if recording_url else 'no'}"
    )

    if recording_url and call_sid:
        pending_recordings[call_sid] = str(recording_url)

    return JSONResponse({"status": "received"})


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
    override_phone: Optional[str],
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
            override_phone=override_phone,
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
# Ngrok
# ──────────────────────────────────────────────────────────

def start_ngrok(port: int, config: dict) -> str:
    """
    Start an Ngrok tunnel and return the public HTTPS URL.

    Args:
        port:   Local port to tunnel
        config: voice_agent_config dict (use_ngrok, public_base_url)

    Returns:
        Public base URL string (https://xxxx.ngrok-free.app)
    """
    if not config.get("use_ngrok", True):
        url = config.get("public_base_url", f"http://localhost:{port}")
        logger.info(f"[AudioServer] Ngrok disabled. Using: {url}")
        return url

    try:
        from pyngrok import ngrok

        auth_token = os.getenv("NGROK_AUTHTOKEN", "")
        if auth_token:
            ngrok.set_auth_token(auth_token)

        tunnel = ngrok.connect(port, "http")
        public_url = tunnel.public_url

        # Ensure HTTPS (Twilio requires it)
        if public_url.startswith("http://"):
            public_url = public_url.replace("http://", "https://", 1)

        config["public_base_url"] = public_url
        logger.info(f"[AudioServer] ✅ Ngrok tunnel: {public_url}")
        return public_url

    except Exception as e:
        logger.error(f"[AudioServer] Ngrok failed: {e}")
        fallback = f"http://localhost:{port}"
        config["public_base_url"] = fallback
        return fallback


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
