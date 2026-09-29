const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const path = require('path');
const { execFile, exec } = require('child_process');

const app = express();
app.use(express.json());

// Serve static files from the public directory
app.use(express.static(path.join(__dirname, '../public')));

const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: '*',
  }
});

const PORT = process.env.PORT || 3000;
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
});

