'use strict';

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
const { mimeToExt, cleanupFile, getTempFilePath } = require('./audio/audio-utils');
const { registerTTSFile, getTTSFile, removeTTSFile } = require('./audio/audio-store');
const { sendTranscript, handleAgentResponse } = require('./interfaces/ai-interface');

// ── Express setup ─────────────────────────────────────────────────────────────
const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, '../public')));

const server = http.createServer(app);

// ── Socket.IO ─────────────────────────────────────────────────────────────────
const io = new Server(server, {
    cors: { origin: '*' },
});

// ── Multer — audio upload storage ─────────────────────────────────────────────
// Store uploaded audio in server/uploads/ with original extension preserved.
// The uploads/ directory is created by audio-utils.js on module load.

const upload = multer({
    storage: multer.memoryStorage(),
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
    // ── Validate ──────────────────────────────────────────────────────────────
    if (!req.file || !req.file.buffer) {
        console.error('[Audio] POST /api/audio — no audio file in request');
        return res.status(400).json({ success: false, error: 'No audio file received.' });
    }

    console.log(`[Audio] Received upload: (${req.file.size} bytes, ${req.file.mimetype})`);

    try {
        // Forward the audio buffer to Python bridge
        const ext = mimeToExt(req.file.mimetype);
        const blob = new Blob([req.file.buffer], { type: req.file.mimetype });
        const formData = new FormData();
        formData.append('audio', blob, `recording.${ext}`);
        
        const bridgeRes = await fetch(`http://localhost:5000/bridge/process-audio?format=${ext}`, {
            method: 'POST',
            body: formData,
        });
        
        if (!bridgeRes.ok) {
            throw new Error(`Bridge returned HTTP ${bridgeRes.status}: ${await bridgeRes.text()}`);
        }
        
        const result = await bridgeRes.json();
        if (!result.success) {
            throw new Error(result.error || 'Bridge error');
        }

        // Emit transcript to any browser listeners via Socket.IO
        if (result.transcript) {
            io.emit('transcript', {
                type: 'transcript',
                text: result.transcript,
                timestamp: Date.now(),
            });
        }

        let ttsId = null;

        // TTS audio is returned separately — register it
        if (result.tts_audio_b64) {
            const audioBuf = Buffer.from(result.tts_audio_b64, 'base64');
            const audioPath = getTempFilePath('mp3');
            fs.writeFileSync(audioPath, audioBuf);
            ttsId = registerTTSFile(audioPath, result.tts_mime_type || 'audio/mpeg');
            
            io.emit('agent-response', {
                type: 'agent_response',
                text: result.agent_text,
                ttsUrl: `/api/tts/${ttsId}`,
                conversation_complete: result.conversation_complete,
                questionIndex: result.question_index,
            });
            console.log(`[Audio] Pipeline complete. TTS ID: ${ttsId}`);
        } else if (result.agent_text) {
             io.emit('agent-response', {
                type: 'agent_response',
                text: result.agent_text,
                conversation_complete: result.conversation_complete,
                questionIndex: result.question_index,
            });
            console.log(`[Audio] Pipeline complete. No TTS generated.`);
        } else {
             console.log(`[Audio] Pipeline complete. No agent response.`);
        }
        
        return res.json({ 
            success: true, 
            transcript: result.transcript,
            agentResponse: result.agent_text,
            ttsId,
            conversationComplete: result.conversation_complete,
            questionIndex: result.question_index,
            answerStatus: result.answer_status
        });

    } catch (err) {
        console.error('[Audio] Unexpected error in /api/audio:', err);
        return res.status(500).json({ success: false, error: 'Internal server error.', detail: err.message });
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

app.get('/api/status', async (req, res) => {
    let bridgeStatus = 'unknown';
    try {
        const bridgeRes = await fetch('http://localhost:5000/bridge/status');
        if (bridgeRes.ok) {
            bridgeStatus = 'connected';
        } else {
            bridgeStatus = 'error';
        }
    } catch {
        bridgeStatus = 'disconnected';
    }

    res.json({
        status:      'ok',
        sttProvider: process.env.STT_PROVIDER || 'bridge',
        ttsProvider: process.env.TTS_PROVIDER || 'edgetts',
        bridgeStatus
    });
});

/**
 * POST /api/scrape-number
 * Scrapes contact phone numbers and company details from a target URL via Python bridge.
 */
app.post('/api/scrape-number', async (req, res) => {
    try {
        console.log(`[Scraper API] Scraping URL: ${req.body?.url}`);
        const bridgeRes = await fetch('http://localhost:5000/bridge/scrape-url', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(req.body),
        });
        const data = await bridgeRes.json();
        return res.status(bridgeRes.status).json(data);
    } catch (err) {
        console.error('[Scraper API] Error connecting to bridge scraper:', err.message);
        // Fallback response if python bridge is offline
        const targetUrl = req.body?.url || 'https://example.com';
        return res.json({
            success: true,
            url: targetUrl,
            company_name: "Extracted Target Company",
            primary_phone: "+1 (800) 555-0199",
            phone_numbers: ["+1 (800) 555-0199", "+1 (650) 253-0000"],
            emails: ["contact@targetcompany.com"],
            description: "Fallback contact info (Bridge offline)",
            is_fallback: true
        });
    }
});

/**
 * POST /api/start-session
 * Starts a new verification session via the Python bridge.
 */
app.post('/api/start-session', async (req, res) => {
    try {
        const bridgeRes = await fetch('http://localhost:5000/bridge/start-session', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(req.body),
        });
        const data = await bridgeRes.json();
        return res.status(bridgeRes.status).json(data);
    } catch (err) {
        console.error('[Session] Error starting session:', err);
        return res.status(500).json({ success: false, error: err.message });
    }
});

/**
 * GET /api/session-status
 * Checks the status of the current verification session.
 */
app.get('/api/session-status', async (req, res) => {
    try {
        const bridgeRes = await fetch('http://localhost:5000/bridge/status');
        const data = await bridgeRes.json();
        return res.status(bridgeRes.status).json(data);
    } catch (err) {
        console.error('[Session] Error checking session status:', err);
        return res.status(500).json({ success: false, error: err.message });
    }
});

/**
 * POST /api/speak-first
 * Triggers the Q1 greeting after the phone user says "Hello?".
 */
app.post('/api/speak-first', async (req, res) => {
    try {
        const bridgeRes = await fetch('http://localhost:5000/bridge/speak-first', {
            method: 'POST',
        });
        const result = await bridgeRes.json();
        
        if (result.success && result.tts_audio_b64) {
            const audioBuf = Buffer.from(result.tts_audio_b64, 'base64');
            const audioPath = getTempFilePath('mp3');
            fs.writeFileSync(audioPath, audioBuf);
            const ttsId = registerTTSFile(audioPath, result.tts_mime_type || 'audio/mpeg');
            
            io.emit('agent-response', {
                type: 'agent_response',
                text: result.agent_text,
                ttsUrl: `/api/tts/${ttsId}`,
                conversation_complete: result.conversation_complete,
                questionIndex: result.question_index,
            });
            
            return res.json({ success: true, ttsId, text: result.agent_text });
        }
        return res.status(bridgeRes.status).json(result);
    } catch (err) {
        console.error('[Session] Error in speak-first:', err);
        return res.status(500).json({ success: false, error: err.message });
    }
});

// ═════════════════════════════════════════════════════════════════════════════
// SOCKET.IO SIGNALING  (PRESERVED EXACTLY — DO NOT MODIFY BELOW THIS LINE)
// ═════════════════════════════════════════════════════════════════════════════

const ROOM = 'voice-room';

io.on('connection', (socket) => {
    console.log(`[Socket] User connected: ${socket.id}`);

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
});
