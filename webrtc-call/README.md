# WebRTC Voice Agent — Person 1 Infrastructure

> **Person 1** owns all audio/communication infrastructure.
> **Person 2** will plug in the LLM / conversation manager later.

---

## Architecture

```
                       PERSON 2 (future)
              ┌─────────────────────────────┐
              │     AI / Conversation Mgr   │
              │  LLM · Verification State   │
              │  Question Selection · etc.  │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │          PERSON 1            │
              │   Audio / Communication      │
              ├──────────────────────────────┤
              │  WebRTC   · Socket.IO        │
              │  Audio Capture (MediaRec.)   │
              │  STT Abstraction (stt.js)    │
              │  TTS Abstraction (tts.js)    │
              │  AI Interface Boundary       │
              │  TTS → WebRTC Injection      │
              └──────────────┬──────────────┘
                             │  WebRTC
                             ▼
                       Receiver (phone)
```

---

## File Structure

```
webrtc-call/
├── server/
│   ├── server.js                    Main server: signaling + REST API
│   ├── package.json
│   ├── .env                         API keys (never commit)
│   ├── .gitignore
│   │
│   ├── audio/
│   │   ├── stt.js                   STT abstraction (WhisperFlow plug-in here)
│   │   ├── tts.js                   TTS abstraction (real TTS plug-in here)
│   │   ├── audio-utils.js           File path helpers, MIME utils
│   │   └── audio-store.js           Temp file lifecycle management
│   │
│   └── interfaces/
│       └── ai-interface.js          PERSON 1 ↔ PERSON 2 boundary
│
└── public/
    ├── index.html
    ├── style.css
    ├── app.js                       WebRTC + signaling + UI wiring
    └── audio/
        └── audio-manager.js         Browser-side audio class
```

---

## Data Flow

```
Phone speaks
   │
   ▼  (WebRTC)
remoteStream  (browser)
   │
   ▼  MediaRecorder  [audio-manager.js]
audio Blob
   │
   ▼  POST /api/audio
server/server.js
   │
   ▼  transcribeAudio()  [audio/stt.js]
{ transcript, duration }
   │
   ▼  sendTranscript()  [interfaces/ai-interface.js]
   │       ↑
   │   PERSON 2 IMPLEMENTS LLM HERE
   │
   ▼  { text, conversation_complete }
   │
   ▼  synthesizeSpeech()  [audio/tts.js]
audio file
   │
   ▼  Socket.IO 'agent-response' event
browser [audio-manager.js]
   │
   ▼  AudioContext → MediaStreamDestination → RTCPeerConnection
   │
   ▼  (WebRTC)
Phone hears AI voice
```

---

## Person 1 ↔ Person 2 Interface

### What Person 2 needs to do

Open **`server/interfaces/ai-interface.js`** and replace the body of `sendTranscript()`:

```js
async function sendTranscript(transcript) {
    // PERSON 2: Replace this with your LLM call
    const llmResponse = await myConversationManager.process(transcript.text);
    return {
        type: 'agent_response',
        text: llmResponse.nextQuestion,
        conversation_complete: llmResponse.isDone,
    };
}
```

### Transcript object (Person 1 → Person 2)

```json
{
  "type":      "transcript",
  "text":      "Yes, the internship is currently open.",
  "timestamp": 1723456789000
}
```

### Agent response object (Person 2 → Person 1)

```json
{
  "type":                "agent_response",
  "text":                "Could you tell me the official careers page?",
  "conversation_complete": false
}
```

---

## WhisperFlow Integration Point

When WhisperFlow API details are available:

1. Open **`server/audio/stt.js`**
2. Implement `_transcribeWithWhisperFlow(audioPath)`
3. Set in `.env`:

```
STT_PROVIDER=whisperflow
STT_API_KEY=<your key>
STT_API_URL=<endpoint>
```

No other files need to change.

---

## Real TTS Integration Point

1. Open **`server/audio/tts.js`**
2. Add a new `_synthesizeWithRealProvider()` function
3. Update the switch in `synthesizeSpeech()` to call it
4. Set in `.env`:

```
TTS_PROVIDER=<provider>
TTS_API_KEY=<your key>
TTS_API_URL=<endpoint>
```

---

## How to Run

```bash
cd webrtc-call/server
npm install
node server.js
```

Server starts on `http://0.0.0.0:3000`.

Open from laptop: `http://localhost:3000`
Open from phone (same WiFi): `http://<your-laptop-ip>:3000`

---

## Test Procedures

### TEST 1 — WebRTC Call (must always pass)

1. Open `http://localhost:3000` on laptop
2. Open `http://<ip>:3000` on phone
3. Both click **Join Room**
4. Laptop clicks **Call**, phone clicks **Accept Call**
5. **Expected:** Two-way audio works

### TEST 2 — Remote Audio Capture

1. Establish call (TEST 1)
2. Laptop clicks **▶ Start Capture**
3. Phone user speaks
4. Laptop clicks **■ Stop & Process**
5. **Expected:** Console shows `[Audio] Blob ready: XXXX bytes`

### TEST 3 — Audio Upload

Same as TEST 2.
**Expected (server console):**
```
[Audio] Received upload: upload_XXXX.webm (XXXX bytes)
```

### TEST 4 — STT

Same as TEST 2 + 3.
**Expected (server console):**
```
[STT] Transcribing: upload_XXXX.webm
[STT] Transcript: "[MOCK STT] Hello, this is a test transcription..."
```

### TEST 5 — AI Interface

**Expected (server console):**
```
[AI Interface] Transcript received:
[AI Interface]   text: "[MOCK STT] Hello, this is a test..."
[AI Interface] Agent response (MOCK):
[AI Interface]   text: "Thank you for that information..."
```

### TEST 6 — TTS Generation

**Expected (server console):**
```
[TTS] Synthesizing: "Thank you for that information..."
[TTS] Mock audio generated: XXXX.wav (X.Xs)
[Audio Store] Registered TTS file: <uuid>
```

### TEST 7 — TTS → WebRTC

**Expected (browser console):**
```
[Audio] Agent response received via Socket.IO: Thank you for...
[Audio] Fetching TTS audio: /api/tts/<uuid>
[Audio] TTS playing through WebRTC. Duration: X.XXs
```
**Expected:** Phone hears the AI voice through the WebRTC call.

### TEST 8 — Direct TTS Test (no STT needed)

1. During an active call, click **🔊 Test TTS → WebRTC**
2. Type any text in the prompt
3. **Expected:** Phone hears the typed text spoken back through WebRTC

### TEST 9 — API Status

```bash
curl http://localhost:3000/api/status
```
```json
{ "status": "ok", "sttProvider": "mock", "ttsProvider": "mock" }
```

---

## Audio Modes

| Mode  | Description |
|-------|-------------|
| HUMAN | Normal two-way call. Microphone ↔ microphone. |
| AGENT | Remote audio captured → STT → AI → TTS → WebRTC back to receiver |

Switch via `audioManager.setMode('human' | 'agent')` in browser console.

---

## Security

- All API keys stored in `server/.env` — never in browser code
- `.env` in `.gitignore`
- Audio files deleted from disk after processing
- TTS files deleted after serving (one-time use, 5-min TTL)
- No secrets exposed via Socket.IO

---

## Current Limitations

1. **STT is a mock** — returns hardcoded text until WhisperFlow is integrated
2. **TTS is a mock** — generates silent WAV until real TTS is integrated (phone won't hear a voice, just silence; the WebRTC pipeline is real)
3. **Single room** — only one pair of peers supported (college demo scope)
4. **No TURN server** — may fail on restrictive networks (Google STUN only)
5. **No authentication** — anyone with the URL can join

---

## Dependencies

| Package   | Purpose |
|-----------|---------|
| express   | HTTP server + static files |
| socket.io | WebRTC signaling + event delivery |
| multer    | Multipart audio upload handling |
| dotenv    | .env loading |
| uuid      | Unique temp file names |
