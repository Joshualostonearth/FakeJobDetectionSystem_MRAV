"""
Call Bridge — Flask HTTP Sidecar for WebRTC Integration
========================================================

Wraps the caller.py AI logic into REST endpoints that the Node.js
WebRTC server calls. Handles:
  - Session creation (question generation via Ollama)
  - Audio processing (STT via WhisperLive → answer processing)
  - TTS synthesis (Edge-TTS → MP3 bytes)

Endpoints:
  POST /bridge/start-session   Start a new verification session
  POST /bridge/process-audio   Transcribe audio + process answer + return TTS
  POST /bridge/synthesize      Generate TTS audio for given text
  POST /bridge/speak-first     Get Q1 greeting + TTS audio
  GET  /bridge/status          Current session state

Usage:
    cd whisper_Test
    source .venv/bin/activate
    pip install flask
    python call_bridge.py
    # → http://localhost:5000
"""

import base64
import json
import os
import sys
import traceback

from flask import Flask, request, jsonify, Response

# Import from our existing modules
from voice_tools import EdgeTTSFile, WhisperFileTranscriber
from scraper_service import scrape_website_for_contact
from caller import (
    QuestionGenerator,
    WebRTCCallSession,
    OllamaChat,
    build_response_filename,
    get_time,
    write_file,
    OLLAMA_MODEL,
    QUESTION_GEN_PROMPT,
    RESPONSES_DIR,
    TTS_VOICE,
)

# ──────────────────────────────────────────────
# Flask App
# ──────────────────────────────────────────────

app = Flask(__name__)

# ── Shared state (single session) ─────────────
_session: WebRTCCallSession | None = None
_tts: EdgeTTSFile | None = None
_stt: WhisperFileTranscriber | None = None
_session_meta: dict = {}


def _get_tts() -> EdgeTTSFile:
    """Lazy-init the TTS engine."""
    global _tts
    if _tts is None:
        _tts = EdgeTTSFile(voice=TTS_VOICE)
        print("[Bridge] EdgeTTSFile initialised.")
    return _tts


def _get_stt() -> WhisperFileTranscriber:
    """Lazy-init the STT engine."""
    global _stt
    if _stt is None:
        _stt = WhisperFileTranscriber()
        print("[Bridge] WhisperFileTranscriber initialised.")
    return _stt


# ──────────────────────────────────────────────
# POST /bridge/scrape-url
# ──────────────────────────────────────────────

@app.route("/bridge/scrape-url", methods=["POST"])
def scrape_url():
    """Scrape web page to extract company details and contact phone numbers."""
    try:
        data = request.get_json() or {}
        url = data.get("url", "").strip()
        if not url:
            return jsonify({"success": False, "error": "No URL provided"}), 400

        print(f"\n[Bridge] === Scraping URL for contact number: {url} ===")
        result = scrape_website_for_contact(url)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ──────────────────────────────────────────────
# POST /bridge/start-session
# ──────────────────────────────────────────────

@app.route("/bridge/start-session", methods=["POST"])
def start_session():
    """Start a new verification session.

    Accepts the input JSON (company data), generates 10 questions
    via Ollama LLM, and returns the session state.

    Request body: same format as whisper_Test/input/*.json
    {
        "company_name": "Acme Tech Solutions",
        "job_title": "Data Analyst Intern",
        "job_type": "internship",
        "offer_details": { ... },
        "suspicion_reasons": [ ... ]
    }

    Response:
    {
        "success": true,
        "session_id": "acme_tech_solutions_data_analyst_intern",
        "questions": [ { "id": 1, "question": "...", ... }, ... ],
        "meta": { "company": "...", "job": "...", ... }
    }
    """
    global _session, _session_meta

    try:
        input_data = request.get_json()
        if not input_data:
            return jsonify({"success": False, "error": "No JSON body provided."}), 400

        company = input_data.get("company_name", "")
        job_title = input_data.get("job_title", "")

        if not company or not job_title:
            return jsonify({"success": False, "error": "Missing company_name or job_title."}), 400

        print(f"\n[Bridge] ═══ Starting session: {company} — {job_title} ═══")

        # ── Phase 1: Load ──
        response_filename = build_response_filename(company, job_title)
        response_path = os.path.join(RESPONSES_DIR, response_filename)

        # ── Phase 2: Generate questions ──
        print("[Bridge] Generating questions via Ollama...")
        llm = OllamaChat(
            model=OLLAMA_MODEL,
            system_prompt=QUESTION_GEN_PROMPT,
        )
        generator = QuestionGenerator(llm)
        questions = generator.generate(input_data)

        # Build the response structure
        response_data = {
            "meta": {
                "company": company,
                "job": job_title,
                "caller_type": "automated AI assistant (WebRTC)",
                "created_at": get_time(),
            },
            "responses": [
                {
                    "id": q["id"],
                    "question": q["question"],
                    "purpose": q["purpose"],
                    "targets_reason_ids": q["targets_reason_ids"],
                    "legit_answer_hint": q["legit_answer_hint"],
                    "answer": None,
                    "status": "pending",
                    "asked_at": None,
                    "answered_at": None,
                }
                for q in questions
            ],
        }

        # Save initial response file
        write_file(response_path, json.dumps(response_data, indent=2, ensure_ascii=False))

        # Create the session
        _session = WebRTCCallSession(response_data, response_path)
        _session_meta = {
            "session_id": response_filename.replace(".json", ""),
            "company": company,
            "job": job_title,
            "response_path": response_path,
        }

        print(f"[Bridge] ✓ Session created with {len(questions)} questions.")

        return jsonify({
            "success": True,
            "session_id": _session_meta["session_id"],
            "questions": questions,
            "meta": response_data["meta"],
            "total_questions": len(questions),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ──────────────────────────────────────────────
# POST /bridge/speak-first
# ──────────────────────────────────────────────

@app.route("/bridge/speak-first", methods=["POST"])
def speak_first():
    """Generate TTS for Q1 greeting (called after phone user says 'Hello?').

    Response:
    {
        "success": true,
        "agent_text": "Hello, this is an automated AI assistant...",
        "tts_audio_b64": "<base64-encoded MP3>",
        "question_index": 0,
        "conversation_complete": false
    }
    """
    global _session

    if _session is None:
        return jsonify({"success": False, "error": "No active session. Call /bridge/start-session first."}), 400

    try:
        greeting = _session.get_greeting_text()
        print(f"[Bridge] Speaking Q1: \"{greeting[:60]}...\"")

        # Mark Q1 as asked
        q1 = _session.get_current_question()
        if q1:
            q1["asked_at"] = get_time()
            _session._save()

        # Generate TTS audio
        tts = _get_tts()
        mp3_bytes = tts.synthesize(greeting)

        if not mp3_bytes:
            return jsonify({"success": False, "error": "TTS synthesis returned empty audio."}), 500

        print(f"[Bridge] ✓ Q1 TTS generated: {len(mp3_bytes)} bytes")

        return jsonify({
            "success": True,
            "agent_text": greeting,
            "tts_audio_b64": base64.b64encode(mp3_bytes).decode("ascii"),
            "tts_mime_type": "audio/mpeg",
            "question_index": 0,
            "conversation_complete": False,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ──────────────────────────────────────────────
# POST /bridge/process-audio
# ──────────────────────────────────────────────

@app.route("/bridge/process-audio", methods=["POST"])
def process_audio():
    """Transcribe uploaded audio, process the answer, generate TTS for next question.

    Request: multipart/form-data with 'audio' file field
    Optional query param: ?format=webm (audio format hint)

    Response:
    {
        "success": true,
        "transcript": "Yes, the internship is currently open.",
        "agent_text": "Could you share your official company website?",
        "tts_audio_b64": "<base64-encoded MP3>",
        "answer_status": "answered",
        "conversation_complete": false,
        "question_index": 2,
        "total_questions": 10
    }
    """
    global _session

    if _session is None:
        return jsonify({"success": False, "error": "No active session."}), 400

    if _session.is_complete:
        return jsonify({
            "success": True,
            "transcript": "",
            "agent_text": "Thank you very much for your time. Have a great day.",
            "conversation_complete": True,
            "question_index": _session.current_index,
            "total_questions": _session.total_questions,
        })

    try:
        # ── Get audio from request ──
        audio_file = request.files.get("audio")
        if not audio_file:
            return jsonify({"success": False, "error": "No 'audio' file in request."}), 400

        audio_bytes = audio_file.read()
        audio_format = request.args.get("format", "webm")

        print(f"\n[Bridge] ── Processing audio: {len(audio_bytes)} bytes ({audio_format}) ──")

        # ── STT: Transcribe ──
        stt = _get_stt()
        transcript = stt.transcribe_bytes(audio_bytes, format=audio_format)

        print(f"[Bridge] Transcript: \"{transcript[:80]}{'...' if len(transcript) > 80 else ''}\"")

        # ── Process answer through CallSession ──
        result = _session.process_answer(transcript)

        # ── Generate TTS for the response ──
        agent_text = None
        tts_audio_b64 = None
        tts_mime_type = None

        # Handle special cases
        if result.get("bot_disclosure"):
            # Speak bot disclosure + re-ask question
            agent_text = result["bot_disclosure"] + " " + (result.get("next_question") or "")
        elif result.get("sensitive_refusal"):
            # Speak refusal + re-ask question
            agent_text = result["sensitive_refusal"] + " " + (result.get("next_question") or "")
        elif result.get("answer_status") == "wait_request":
            # Don't speak — just wait for the next audio
            agent_text = None
        elif result.get("next_question"):
            agent_text = result["next_question"]
        elif result.get("conversation_complete"):
            agent_text = "Thank you very much for your time. Have a great day."

        if agent_text:
            tts = _get_tts()
            mp3_bytes = tts.synthesize(agent_text)
            if mp3_bytes:
                tts_audio_b64 = base64.b64encode(mp3_bytes).decode("ascii")
                tts_mime_type = "audio/mpeg"
                print(f"[Bridge] ✓ TTS generated: {len(mp3_bytes)} bytes")

        return jsonify({
            "success": True,
            "transcript": transcript,
            "agent_text": agent_text,
            "tts_audio_b64": tts_audio_b64,
            "tts_mime_type": tts_mime_type,
            "answer_status": result.get("answer_status"),
            "answer_recorded": result.get("answer_recorded"),
            "conversation_complete": result.get("conversation_complete", False),
            "question_index": result.get("question_index", 0),
            "total_questions": result.get("total_questions", 10),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ──────────────────────────────────────────────
# POST /bridge/synthesize
# ──────────────────────────────────────────────

@app.route("/bridge/synthesize", methods=["POST"])
def synthesize():
    """Generate TTS audio for arbitrary text (used by /api/test-tts).

    Request body: { "text": "Hello world" }

    Response: audio/mpeg binary (MP3 bytes)
    """
    try:
        data = request.get_json()
        text = data.get("text", "") if data else ""

        if not text:
            return jsonify({"success": False, "error": "Missing 'text' in request body."}), 400

        print(f"[Bridge] Synthesizing: \"{text[:60]}...\"")

        tts = _get_tts()
        mp3_bytes = tts.synthesize(text)

        if not mp3_bytes:
            return jsonify({"success": False, "error": "TTS returned empty audio."}), 500

        return Response(
            mp3_bytes,
            mimetype="audio/mpeg",
            headers={"Content-Length": str(len(mp3_bytes))},
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ──────────────────────────────────────────────
# GET /bridge/status
# ──────────────────────────────────────────────

@app.route("/bridge/status", methods=["GET"])
def status():
    """Return current session state.

    Response:
    {
        "bridge": "ok",
        "session_active": true,
        "session_id": "acme_tech_solutions_data_analyst_intern",
        "state": "active",
        "question_index": 3,
        "total_questions": 10
    }
    """
    if _session is None:
        return jsonify({
            "bridge": "ok",
            "session_active": False,
        })

    state = _session.get_state()
    return jsonify({
        "bridge": "ok",
        "session_active": True,
        "session_id": _session_meta.get("session_id", ""),
        **state,
    })


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Call Bridge — Flask HTTP Sidecar")
    print("  Endpoints:")
    print("    POST /bridge/start-session")
    print("    POST /bridge/speak-first")
    print("    POST /bridge/process-audio")
    print("    POST /bridge/synthesize")
    print("    GET  /bridge/status")
    print("=" * 60)

    # Ensure responses directory exists
    os.makedirs(RESPONSES_DIR, exist_ok=True)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,  # No auto-reload (WhisperLive client doesn't like it)
    ) 
