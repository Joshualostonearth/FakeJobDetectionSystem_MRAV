<<<<<<< HEAD
const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const path = require('path');
const { execFile, exec } = require('child_process');
=======
'use strict';
>>>>>>> eacde25df052357f220a1f2821fed36a3d589623

// ── Environment ───────────────────────────────────────────────────────────────
require('dotenv').config();

const express  = require('express');
const http     = require('http');
const { Server } = require('socket.io');
const path     = require('path');
const multer   = require('multer');
const fs       = require('fs');

// ── Person 1 modules ──────────────────────────────────────────────────────────
const { transcribeAudio }                   = require('./audio/stt');
const { synthesizeSpeech }                  = require('./audio/tts');
const { mimeToExt, cleanupFile }            = require('./audio/audio-utils');
const { registerTTSFile, getTTSFile, removeTTSFile } = require('./audio/audio-store');
const { sendTranscript, handleAgentResponse } = require('./interfaces/ai-interface');

// ── Express setup ─────────────────────────────────────────────────────────────
const app = express();
app.use(express.json());
<<<<<<< HEAD

// Serve static files from the public directory
=======
>>>>>>> eacde25df052357f220a1f2821fed36a3d589623
app.use(express.static(path.join(__dirname, '../public')));

const server = http.createServer(app);

// ── Socket.IO ─────────────────────────────────────────────────────────────────
const io = new Server(server, {
    cors: { origin: '*' },
});

<<<<<<< HEAD
const PORT = process.env.PORT || 3000;
=======
// ── Multer — audio upload storage ─────────────────────────────────────────────
// Store uploaded audio in server/uploads/ with original extension preserved.
// The uploads/ directory is created by audio-utils.js on module load.

const UPLOADS_DIR = path.join(__dirname, 'uploads');

const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        // Ensure uploads dir exists (audio-utils creates it, but belt-and-suspenders)
        if (!fs.existsSync(UPLOADS_DIR)) fs.mkdirSync(UPLOADS_DIR, { recursive: true });
        cb(null, UPLOADS_DIR);
    },
    filename: (req, file, cb) => {
        const ext  = mimeToExt(file.mimetype);
        const name = `upload_${Date.now()}.${ext}`;
        cb(null, name);
    },
});

const upload = multer({
    storage,
    limits: { fileSize: 50 * 1024 * 1024 }, // 50 MB max
    fileFilter: (req, file, cb) => {
        if (file.mimetype.startsWith('audio/')) {
            cb(null, true);
        } else {
            cb(new Error('[Audio] Rejected non-audio file upload'));
        }
    },
});

// ═════════════════════════════════════════════════════════════════════════════
// REST API ROUTES
// ═════════════════════════════════════════════════════════════════════════════

/**
 * POST /api/audio
 * 
 * Receives a recorded audio blob from the browser, runs it through the full
 * pipeline: STT → AI Interface → TTS → responds with TTS audio URL.
 * 
 * Request:  multipart/form-data  { audio: <blob> }
 * Response: application/json
 * {
 *   success:         true,
 *   transcript:      "The spoken text",
 *   duration:        4.2,
 *   agentResponse:   "The AI agent's reply text",
 *   ttsId:           "uuid-for-tts-audio",
 *   conversationComplete: false
 * }
 */
app.post('/api/audio', upload.single('audio'), async (req, res) => {
    const audioPath = req.file?.path;

    // ── Validate ──────────────────────────────────────────────────────────────
    if (!req.file || !audioPath) {
        console.error('[Audio] POST /api/audio — no audio file in request');
        return res.status(400).json({ success: false, error: 'No audio file received.' });
    }

    console.log(`[Audio] Received upload: ${req.file.filename} (${req.file.size} bytes, ${req.file.mimetype})`);

    try {
        // ── STT ───────────────────────────────────────────────────────────────
        let sttResult;
        try {
            sttResult = await transcribeAudio(audioPath);
        } catch (sttErr) {
            console.error('[STT] Transcription failed:', sttErr.message);
            cleanupFile(audioPath);
            return res.status(500).json({ success: false, error: 'STT transcription failed.', detail: sttErr.message });
        }

        const transcript = {
            type:      'transcript',
            text:      sttResult.transcript,
            timestamp: Date.now(),
        };

        // Emit transcript to any browser listeners via Socket.IO
        io.emit('transcript', transcript);
        console.log('[Audio] Emitted transcript event via Socket.IO');

        // ── AI Interface ──────────────────────────────────────────────────────
        let agentResponse;
        try {
            agentResponse = await sendTranscript(transcript);
        } catch (aiErr) {
            console.error('[AI Interface] Failed:', aiErr.message);
            cleanupFile(audioPath);
            return res.status(500).json({ success: false, error: 'AI interface failed.', detail: aiErr.message });
        }

        // ── TTS ───────────────────────────────────────────────────────────────
        let ttsResult;
        try {
            ttsResult = await synthesizeSpeech(agentResponse.text);
        } catch (ttsErr) {
            console.error('[TTS] Synthesis failed:', ttsErr.message);
            cleanupFile(audioPath);
            return res.status(500).json({ success: false, error: 'TTS synthesis failed.', detail: ttsErr.message });
        }

        // Register TTS file and get an ID for the browser to fetch it
        const ttsId = registerTTSFile(ttsResult.audioPath, ttsResult.mimeType);

        // Notify browser via Socket.IO so the TTS can be played through WebRTC
        io.emit('agent-response', {
            type:                'agent_response',
            text:                agentResponse.text,
            ttsUrl:              `/api/tts/${ttsId}`,
            conversation_complete: agentResponse.conversation_complete,
        });

        console.log(`[Audio] Pipeline complete. TTS ID: ${ttsId}`);

        // ── Cleanup uploaded audio ────────────────────────────────────────────
        cleanupFile(audioPath);

        return res.json({
            success:              true,
            transcript:           sttResult.transcript,
            duration:             sttResult.duration,
            agentResponse:        agentResponse.text,
            ttsId,
            conversationComplete: agentResponse.conversation_complete,
        });

    } catch (err) {
        console.error('[Audio] Unexpected error in /api/audio:', err);
        cleanupFile(audioPath);
        return res.status(500).json({ success: false, error: 'Internal server error.' });
    }
});

/**
 * GET /api/tts/:id
 * 
 * Serves a generated TTS audio file by ID.
 * The browser fetches this, decodes it, and plays it through the WebRTC track.
 * The file is deleted after serving (one-time use).
 */
app.get('/api/tts/:id', (req, res) => {
    const { id } = req.params;
    const entry  = getTTSFile(id);

    if (!entry) {
        console.error(`[TTS] File not found for ID: ${id}`);
        return res.status(404).json({ error: 'TTS audio not found or already consumed.' });
    }

    const { filePath, mimeType } = entry;

    if (!fs.existsSync(filePath)) {
        removeTTSFile(id);
        console.error(`[TTS] File missing on disk: ${filePath}`);
        return res.status(404).json({ error: 'TTS audio file missing.' });
    }

    console.log(`[TTS] Serving TTS audio: ${id}`);

    res.setHeader('Content-Type', mimeType);
    res.setHeader('Cache-Control', 'no-store');

    const readStream = fs.createReadStream(filePath);

    readStream.on('end', () => {
        // Delete after served — one-time consumption
        removeTTSFile(id);
    });

    readStream.on('error', (err) => {
        console.error('[TTS] Error streaming file:', err.message);
        removeTTSFile(id);
        if (!res.headersSent) {
            res.status(500).json({ error: 'Failed to stream TTS audio.' });
        }
    });

    readStream.pipe(res);
});

/**
 * POST /api/test-tts
 * 
 * Developer test endpoint — trigger TTS directly without going through STT.
 * 
 * Request:  application/json  { text: "Hello world" }
 * Response: application/json  { success: true, ttsId: "..." }
 */
app.post('/api/test-tts', async (req, res) => {
    const text = req.body?.text;
    if (!text) {
        return res.status(400).json({ success: false, error: 'Missing "text" in request body.' });
    }

    console.log(`[TTS] Test synthesis: "${text}"`);

    try {
        const ttsResult = await synthesizeSpeech(text);
        const ttsId     = registerTTSFile(ttsResult.audioPath, ttsResult.mimeType);

        io.emit('agent-response', {
            type:                 'agent_response',
            text,
            ttsUrl:               `/api/tts/${ttsId}`,
            conversation_complete: false,
        });

        return res.json({ success: true, ttsId, ttsUrl: `/api/tts/${ttsId}` });
    } catch (err) {
        console.error('[TTS] Test synthesis failed:', err.message);
        return res.status(500).json({ success: false, error: err.message });
    }
});

/**
 * GET /api/status
 * Health check — confirms server and modules are running.
 */
app.get('/api/status', (req, res) => {
    res.json({
        status:      'ok',
        sttProvider: process.env.STT_PROVIDER || 'mock',
        ttsProvider: process.env.TTS_PROVIDER || 'mock',
    });
});

// ═════════════════════════════════════════════════════════════════════════════
// SOCKET.IO SIGNALING  (PRESERVED EXACTLY — DO NOT MODIFY BELOW THIS LINE)
// ═════════════════════════════════════════════════════════════════════════════

>>>>>>> eacde25df052357f220a1f2821fed36a3d589623
const ROOM = 'voice-room';
const PROJECT_ROOT = path.join(__dirname, '../..');

function extractJSON(str) {
  const start = str.indexOf('{');
  const end = str.lastIndexOf('}');
  if (start !== -1 && end > start) {
    try {
      return JSON.parse(str.substring(start, end + 1));
    } catch (e) {}
  }
  return { raw: str.trim() };
}

// Helper to run python script commands
function runPython(args, cwd = PROJECT_ROOT) {
  return new Promise((resolve, reject) => {
    execFile('python', args, { cwd, maxBuffer: 1024 * 1024 * 10 }, (error, stdout, stderr) => {
      if (error && !stdout) {
        console.error(`[Python Exec Error]`, stderr || error.message);
        return reject(error);
      }
      resolve(extractJSON(stdout));
    });
  });
}

// Helper to run python inline code
function runPythonInline(code, args = [], cwd = PROJECT_ROOT) {
  return new Promise((resolve, reject) => {
    const fullArgs = ['-c', code, ...args];
    execFile('python', fullArgs, { cwd, maxBuffer: 1024 * 1024 * 10 }, (error, stdout, stderr) => {
      if (error && !stdout) {
        console.error(`[Python Inline Error]`, stderr || error.message);
        return reject(error);
      }
      resolve(extractJSON(stdout));
    });
  });
}

// ──────────────────────────────────────────────
// API Endpoints
// ──────────────────────────────────────────────

// 1. Web Scraper Endpoint (testscraper.py integration)
app.post('/api/scrape', async (req, res) => {
  const { url, prompt } = req.body;
  if (!url) {
    return res.status(400).json({ error: 'URL is required for scraping.' });
  }

  console.log(`[API /api/scrape] Triggering testscraper.py for URL: ${url}`);
  try {
    const args = ['testscraper.py', '--url', url, '--json-only'];
    if (prompt) {
      args.push('--prompt', prompt);
    }
    const result = await runPython(args);
    console.log(`[API /api/scrape] Successfully scraped phone: ${result.phone_number || result.email_or_phone}`);
    return res.json({ success: true, data: result });
  } catch (err) {
    console.error(`[API /api/scrape Failed]`, err.message);
    return res.status(500).json({ error: 'Scraping failed', details: err.message });
  }
});

// 2. Initialize Voice Verification Session
app.post('/api/voice/init', async (req, res) => {
  const jobEntities = req.body || {};
  console.log(`[API /api/voice/init] Initializing call session for: ${jobEntities.company_name || 'Target Company'}`);

  const code = `
import json, sys
sys.path.insert(0, '.')
from integration.voice_agent_api import init_verification_session
entities = json.loads(sys.argv[1])
result = init_verification_session(entities)
print(json.dumps(result))
`;

  try {
    const result = await runPythonInline(code, [JSON.stringify(jobEntities)]);
    return res.json({ success: true, ...result });
  } catch (err) {
    return res.status(500).json({ error: 'Failed to initialize voice session', details: err.message });
  }
});

// 3. Process Transcript Turn (Person 1 STT -> Person 2 Voice Agent)
app.post('/api/voice/turn', async (req, res) => {
  const { session_id, text, timestamp } = req.body;
  if (!session_id || !text) {
    return res.status(400).json({ error: 'session_id and text are required' });
  }

  const code = `
import json, sys
sys.path.insert(0, '.')
from integration.voice_agent_api import process_transcript_turn
session_id = sys.argv[1]
turn_data = json.loads(sys.argv[2])
result = process_transcript_turn(session_id, turn_data)
print(json.dumps(result))
`;

  const turnInput = { type: 'transcript', text, timestamp: timestamp || 0.0 };

  try {
    const result = await runPythonInline(code, [session_id, JSON.stringify(turnInput)]);
    return res.json({ success: true, response: result });
  } catch (err) {
    return res.status(500).json({ error: 'Failed to process turn', details: err.message });
  }
});

// 4. Finalize Call Session & Run Consensus Credibility Filter
app.post('/api/voice/finalize', async (req, res) => {
  const { session_id } = req.body;
  if (!session_id) {
    return res.status(400).json({ error: 'session_id is required' });
  }

  const code = `
import json, sys
sys.path.insert(0, '.')
from integration.voice_agent_api import finalize_session, hand_off_to_objective_2
session_id = sys.argv[1]
session_data = finalize_session(session_id)
state = session_data.get('verification_state', {})
signals = session_data.get('voice_signals', {})
obj1_context = state.get('context', {})
report = hand_off_to_objective_2(obj1_context, [], signals)
print(json.dumps({'signals': signals, 'report': report}))
`;

  try {
    const result = await runPythonInline(code, [session_id]);
    return res.json({ success: true, data: result });
  } catch (err) {
    return res.status(500).json({ error: 'Failed to finalize session', details: err.message });
  }
});

// Socket.IO WebRTC Signaling Setup
io.on('connection', (socket) => {
    console.log(`[Socket] User connected: ${socket.id}`);

<<<<<<< HEAD
  socket.on('join', () => {
    socket.join(ROOM);
    console.log(`[Socket] User ${socket.id} joined room: ${ROOM}`);
    socket.to(ROOM).emit('user-joined', socket.id);
  });

  socket.on('offer', (data) => {
    socket.to(ROOM).emit('offer', data);
  });

  socket.on('answer', (data) => {
    socket.to(ROOM).emit('answer', data);
  });

  socket.on('ice-candidate', (data) => {
    socket.to(ROOM).emit('ice-candidate', data);
  });

  socket.on('hangup', () => {
    socket.to(ROOM).emit('hangup', socket.id);
  });

  socket.on('disconnect', () => {
    socket.to(ROOM).emit('hangup', socket.id);
  });
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`[Server] Voice Agent Server running on http://0.0.0.0:${PORT}`);
=======
    // When a user joins the room
    socket.on('join', () => {
        socket.join(ROOM);
        console.log(`[Socket] User ${socket.id} joined room: ${ROOM}`);
        socket.to(ROOM).emit('user-joined', socket.id);
    });

    // Relay WebRTC Offer
    socket.on('offer', (data) => {
        console.log(`[Signaling] Relaying offer from ${socket.id}`);
        socket.to(ROOM).emit('offer', data);
    });

    // Relay WebRTC Answer
    socket.on('answer', (data) => {
        console.log(`[Signaling] Relaying answer from ${socket.id}`);
        socket.to(ROOM).emit('answer', data);
    });

    // Relay ICE candidates
    socket.on('ice-candidate', (data) => {
        console.log(`[Signaling] Relaying ICE candidate from ${socket.id}`);
        socket.to(ROOM).emit('ice-candidate', data);
    });

    // Relay Hangup
    socket.on('hangup', () => {
        console.log(`[Signaling] User hung up: ${socket.id}`);
        socket.to(ROOM).emit('hangup', socket.id);
    });

    // Handle Disconnect
    socket.on('disconnect', () => {
        console.log(`[Socket] User disconnected: ${socket.id}`);
        socket.to(ROOM).emit('hangup', socket.id);
    });
});

// ═════════════════════════════════════════════════════════════════════════════
// START SERVER
// ═════════════════════════════════════════════════════════════════════════════

const PORT = process.env.PORT || 3000;

server.listen(PORT, '0.0.0.0', () => {
    console.log(`[Server] Signaling + API server running on http://0.0.0.0:${PORT}`);
    console.log(`[Server] STT provider: ${process.env.STT_PROVIDER || 'mock'}`);
    console.log(`[Server] TTS provider: ${process.env.TTS_PROVIDER || 'mock'}`);
>>>>>>> eacde25df052357f220a1f2821fed36a3d589623
});

