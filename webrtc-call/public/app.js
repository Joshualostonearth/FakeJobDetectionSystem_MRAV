'use strict';

// ═════════════════════════════════════════════════════════════════════════════
// PERSON 1 — Browser Main: WebRTC + Signaling + AudioManager wiring
//
// All original WebRTC code is preserved below.
// AudioManager is instantiated and wired in the "AUDIO MANAGER SETUP" section.
// ═════════════════════════════════════════════════════════════════════════════

// === UI Elements ===
const micStatusEl       = document.getElementById('mic-status');
const signalingStatusEl = document.getElementById('signaling-status');
const callStatusEl      = document.getElementById('call-status');
const audioModeEl       = document.getElementById('audio-mode');

const btnJoin      = document.getElementById('btn-join');
const btnCall      = document.getElementById('btn-call');
const btnAccept    = document.getElementById('btn-accept');
const btnHangup    = document.getElementById('btn-hangup');

// AI Pipeline controls
const btnStartAI   = document.getElementById('btn-start-ai');
const btnStopAI    = document.getElementById('btn-stop-ai');
const btnTestTTS   = document.getElementById('btn-test-tts');

// Session controls
const btnStartSession = document.getElementById('btn-start-session');
const inputFileSelect = document.getElementById('input-file');
const questionsListEl = document.getElementById('questions-list');

// Transcript / Agent panel
const transcriptEl = document.getElementById('transcript-text');
const agentEl      = document.getElementById('agent-text');
const aiStatusEl   = document.getElementById('ai-status');

const remoteAudio  = document.getElementById('remoteAudio');

// === Globals (WebRTC — unchanged) ===
let socket;
let localStream;
let peerConnection;
let incomingOffer  = null;
let remoteStream   = null;

// STUN server configuration for NAT traversal (unchanged)
const iceServers = {
    iceServers: [
        { urls: 'stun:stun.l.google.com:19302' }
    ]
};

// Session State
let sessionActive = false;
let currentQuestionIndex = 0;

// ═════════════════════════════════════════════════════════════════════════════
// AUDIO MANAGER SETUP
// ═════════════════════════════════════════════════════════════════════════════

const audioManager = new AudioManager();

// ── Person 2 transcript callback ──────────────────────────────────────────────
// Person 2 can replace or extend this callback.
audioManager.onTranscript = (transcript) => {
    console.log('[App] Transcript received via fetch:', transcript.text);
    // UI update is handled by the socket.on('transcript') listener to keep things centralized
};

// ── Auto-loop trigger ─────────────────────────────────────────────────────────
audioManager.onTTSComplete = () => {
    if (sessionActive) {
        setTimeout(() => autoRecordLoop(), 500); // Small pause before next question
    }
};

// ── Error callback ────────────────────────────────────────────────────────────
audioManager.onError = (err) => {
    console.error('[App] AudioManager error:', err.message);
    updateAIStatus('Error: ' + err.message, 'red');
};

// ── State change callback ─────────────────────────────────────────────────────
audioManager.onStateChange = (state) => {
    console.log('[App] AudioManager state:', state);

    switch (state) {
        case 'recording:started':
            updateAIStatus('Recording remote audio...', '#f0a500');
            btnStartAI.disabled = true;
            btnStopAI.disabled  = false;
            break;
        case 'recording:stopped':
            updateAIStatus('Processing...', '#888');
            break;
        case 'upload:started':
            updateAIStatus('Uploading audio...', '#888');
            break;
        case 'upload:complete':
            updateAIStatus('Processing complete. Awaiting AI response...', '#2ecc71');
            break;
        case 'tts:playing':
            updateAIStatus('AI speaking through WebRTC...', '#2ecc71');
            break;
        case 'tts:ended':
            updateAIStatus('Ready', '#2ecc71');
            btnStartAI.disabled = false;
            btnStopAI.disabled  = true;
            break;
    }
};

// ═════════════════════════════════════════════════════════════════════════════
// 1. MICROPHONE ACCESS  (unchanged)
// ═════════════════════════════════════════════════════════════════════════════

async function getMicrophone() {
    try {
        updateStatus(micStatusEl, 'Requesting microphone permission...');
        localStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
        updateStatus(micStatusEl, 'Microphone ready (Permission granted)', 'green');
        return true;
    } catch (err) {
        console.error('[WebRTC] Error accessing microphone:', err);
        updateStatus(micStatusEl, 'Microphone access denied or error', 'red');
        alert('Microphone access is required. Are you on HTTPS or localhost?');
        return false;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// 2. SIGNALING SETUP  (original handlers preserved; new events added)
// ═════════════════════════════════════════════════════════════════════════════

function connectSignaling() {
    socket = io();

    socket.on('connect', () => {
        updateStatus(signalingStatusEl, 'Connected', 'green');
        console.log('[Signaling] Connected to server.');
        socket.emit('join');
        updateStatus(callStatusEl, 'Waiting for another user...');
        btnJoin.disabled = true;
        btnCall.disabled = false;
    });

    socket.on('user-joined', (userId) => {
        console.log(`[Signaling] User joined: ${userId}`);
        updateStatus(callStatusEl, 'Another user joined. Ready to call.');
    });

    // Handle incoming offer (unchanged)
    socket.on('offer', async (offer) => {
        console.log('[Signaling] Received Offer', offer);
        incomingOffer = offer;
        updateStatus(callStatusEl, 'Incoming call...');
        btnCall.style.display   = 'none';
        btnAccept.style.display = 'block';
        
        // Start playing the ringing sound
        startRinging();
    });

    // Handle incoming answer (unchanged)
    socket.on('answer', async (answer) => {
        console.log('[Signaling] Received Answer', answer);
        if (peerConnection) {
            await peerConnection.setRemoteDescription(new RTCSessionDescription(answer));
            updateStatus(callStatusEl, 'Connected (Answer received)', 'green');
        }
    });

    // Handle incoming ICE candidates (unchanged)
    socket.on('ice-candidate', async (candidate) => {
        console.log('[Signaling] Received ICE Candidate', candidate);
        try {
            if (peerConnection) {
                await peerConnection.addIceCandidate(new RTCIceCandidate(candidate));
            }
        } catch (e) {
            console.error('[WebRTC] Error adding ICE candidate:', e);
        }
    });

    // Handle remote hangup (unchanged)
    socket.on('hangup', () => {
        console.log('[Signaling] Remote user hung up.');
        cleanupCall('Call ended remotely');
    });

    // ── NEW: Agent response from server (TTS ready) ───────────────────────────
    socket.on('agent-response', (response) => {
        console.log('[App] Agent response received via Socket.IO:', response.text?.substring(0, 60));
        updateAgentPanel(response.text);
        updateAIStatus('Agent responding...', '#2ecc71');
        
        if (response.questionIndex !== undefined) {
            updateQuestionStatus(response.questionIndex, response.conversation_complete);
        }

        // Route TTS audio through WebRTC — only on the operator device (Person 1)
        // The other device (Person 2 / phone) hears it through the WebRTC audio stream
        if (response.ttsUrl && sessionActive) {
            audioManager.handleAgentResponse(response);

            // If this is the final response, end session AFTER TTS finishes playing
            if (response.conversation_complete) {
                const origCallback = audioManager.onTTSComplete;
                audioManager.onTTSComplete = () => {
                    sessionActive = false;
                    updateAIStatus('Verification Complete', '#2ecc71');
                    btnStartSession.disabled = false;
                    // Restore the normal callback
                    audioManager.onTTSComplete = origCallback;
                };
            }
        } else if (response.conversation_complete) {
            // No TTS but conversation is done
            sessionActive = false;
            updateAIStatus('Verification Complete', '#2ecc71');
            btnStartSession.disabled = false;
        } else if (sessionActive) {
            // No TTS (e.g. wait request), just start listening again immediately
            setTimeout(() => autoRecordLoop(), 500);
        }
    });

    // ── NEW: Transcript echoed back from server ────────────────────────────────
    socket.on('transcript', (transcript) => {
        console.log('[App] Transcript from server:', transcript.text);
        updateTranscriptPanel(transcript.text);
    });

    socket.on('disconnect', () => {
        updateStatus(signalingStatusEl, 'Disconnected', 'red');
        console.log('[Signaling] Disconnected from server.');
    });

    socket.on('connect_error', (err) => {
        console.error('[Signaling] Connection error:', err.message);
        updateStatus(signalingStatusEl, 'Connection error', 'red');
    });
}

// ═════════════════════════════════════════════════════════════════════════════
// 3. WEBRTC PEERCONNECTION  (unchanged core; AudioManager wired in ontrack)
// ═════════════════════════════════════════════════════════════════════════════

function createPeerConnection() {
    peerConnection = new RTCPeerConnection(iceServers);

    // Add local audio tracks (unchanged)
    localStream.getTracks().forEach(track => {
        peerConnection.addTrack(track, localStream);
    });

    // Remote track received (unchanged + AudioManager wire)
    peerConnection.ontrack = (event) => {
        console.log('[WebRTC] Received remote track.');
        remoteStream = event.streams[0];

        if (remoteAudio.srcObject !== remoteStream) {
            remoteAudio.srcObject = remoteStream;
            updateStatus(callStatusEl, 'Connected (Audio playing)', 'green');
        }

        // Wire remote stream into AudioManager
        audioManager.setRemoteStream(remoteStream);
        audioManager.setPeerConnection(peerConnection);

        // Enable AI pipeline controls
        btnStartAI.disabled = false;
        btnTestTTS.disabled = false;
        if (btnStartSession) btnStartSession.disabled = false;

        console.log('[WebRTC] Remote stream wired to AudioManager.');
    };

    // ICE candidate generation (unchanged)
    peerConnection.onicecandidate = (event) => {
        if (event.candidate) {
            console.log('[WebRTC] Generated ICE Candidate. Sending...');
            socket.emit('ice-candidate', event.candidate);
        }
    };

    // Connection state monitoring (unchanged + improved logging)
    peerConnection.oniceconnectionstatechange = () => {
        const state = peerConnection.iceConnectionState;
        console.log(`[WebRTC] ICE state: ${state}`);

        if (state === 'disconnected' || state === 'failed') {
            console.error('[WebRTC] Connection failed or disconnected.');
            cleanupCall('Connection failed/disconnected');
        }
        if (state === 'connected') {
            updateStatus(callStatusEl, 'WebRTC connected', 'green');
        }
    };

    peerConnection.onconnectionstatechange = () => {
        console.log(`[WebRTC] Connection state: ${peerConnection.connectionState}`);
    };
}

// ═════════════════════════════════════════════════════════════════════════════
// 4. CALL LOGIC  (unchanged)
// ═════════════════════════════════════════════════════════════════════════════

async function startCall() {
    btnCall.disabled = true;
    updateStatus(callStatusEl, 'Calling...');
    createPeerConnection();

    try {
        const offer = await peerConnection.createOffer();
        await peerConnection.setLocalDescription(offer);
        console.log('[Signaling] Sending Offer.');
        socket.emit('offer', offer);
        btnHangup.disabled = false;
    } catch (e) {
        console.error('[WebRTC] Error creating offer:', e);
        cleanupCall('Call failed');
    }
}

async function acceptCall() {
    stopRinging();
    btnAccept.style.display = 'none';
    btnCall.style.display   = 'block';
    btnCall.disabled        = true;
    btnHangup.disabled      = false;
    updateStatus(callStatusEl, 'Connecting...');
    createPeerConnection();

    try {
        await peerConnection.setRemoteDescription(new RTCSessionDescription(incomingOffer));
        const answer = await peerConnection.createAnswer();
        await peerConnection.setLocalDescription(answer);
        console.log('[Signaling] Sending Answer.');
        socket.emit('answer', answer);
    } catch (e) {
        console.error('[WebRTC] Error accepting call:', e);
        cleanupCall('Call failed');
    }
}

function cleanupCall(reason) {
    stopRinging();

    if (peerConnection) {
        peerConnection.close();
        peerConnection = null;
    }

    remoteAudio.srcObject = null;
    remoteStream          = null;
    incomingOffer         = null;

    // Cleanup AudioManager
    audioManager.destroy();

    updateStatus(callStatusEl, reason || 'Call ended');
    btnCall.disabled   = false;
    btnHangup.disabled = true;
    btnAccept.style.display = 'none';
    btnCall.style.display   = 'block';
    btnStartAI.disabled     = true;
    btnStopAI.disabled      = true;
    btnTestTTS.disabled     = true;
    if (btnStartSession) btnStartSession.disabled = true;
    sessionActive           = false;
    updateAIStatus('Idle', '#888');
}

function hangUp() {
    socket.emit('hangup');
    cleanupCall('You hung up');
}

// ═════════════════════════════════════════════════════════════════════════════
// 5. AI PIPELINE CONTROLS (new — Person 1 wires AudioManager to buttons)
// ═════════════════════════════════════════════════════════════════════════════

function startAIPipeline() {
    if (!remoteStream) {
        alert('No active call. Please connect to a call first.');
        return;
    }
    audioManager.setMode('agent');
    audioManager.startRemoteAudioCapture();
    console.log('[App] AI pipeline started — recording remote audio.');
}

function stopAIPipeline() {
    audioManager.stopRemoteAudioCapture();
    console.log('[App] AI pipeline stopped.');
}

async function testTTS() {
    const text = prompt('Enter text for TTS test:', 'Hello, this is a test of the text to speech system.');
    if (!text) return;

    updateAIStatus('Sending TTS test...', '#888');

    try {
        const res = await fetch('/api/test-tts', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ text }),
        });
        const data = await res.json();
        if (data.success) {
            console.log('[App] TTS test triggered. ID:', data.ttsId);
            updateAIStatus('TTS audio sent to receiver via WebRTC', '#2ecc71');
        } else {
            throw new Error(data.error);
        }
    } catch (err) {
        console.error('[App] TTS test failed:', err.message);
        updateAIStatus('TTS test failed: ' + err.message, 'red');
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// 6. VERIFICATION SESSION CONTROLS
// ═════════════════════════════════════════════════════════════════════════════

async function startVerificationSession() {
    if (!remoteStream) {
        alert('No active call. Please connect to a call first.');
        return;
    }
    
    btnStartSession.disabled = true;
    updateAIStatus('Starting session...', '#f0a500');
    
    try {
        // Fetch the local JSON file (input data with company info + suspicion reasons)
        const fileName = inputFileSelect.value;
        const resJson = await fetch(`/${fileName}.json`);
        const inputData = await resJson.json();
        
        // Step 1: Create the session (generates questions via Ministral)
        const res = await fetch('/api/start-session', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(inputData),
        });
        
        const data = await res.json();
        if (!data.success) throw new Error(data.error);
        
        sessionActive = true;
        updateQuestionsPanel(data.questions);
        audioManager.setMode('agent');
        
        // Step 2: Speak the greeting + Q1 (generates TTS audio)
        updateAIStatus('AI introducing itself...', '#f0a500');
        const speakRes = await fetch('/api/speak-first', { method: 'POST' });
        const speakData = await speakRes.json();
        
        if (speakData.success && speakData.tts_audio_b64) {
            // The server already emits 'agent-response' via Socket.IO,
            // which triggers TTS playback and then autoRecordLoop via onTTSComplete.
            updateAgentPanel(speakData.agent_text);
            updateAIStatus('AI speaking Q1...', '#2ecc71');
        } else {
            // Fallback: no TTS, just start listening
            console.warn('[Session] speak-first returned no audio, starting listen loop.');
            autoRecordLoop();
        }
        
    } catch (err) {
        console.error('[Session] Failed to start:', err);
        updateAIStatus('Failed to start session: ' + err.message, 'red');
        btnStartSession.disabled = false;
    }
}

function autoRecordLoop() {
    if (!sessionActive || !remoteStream) return;
    updateAIStatus('Listening...', '#f0a500');
    audioManager.startRemoteAudioCaptureWithSilenceDetection(2000);
}

function updateQuestionsPanel(questions) {
    if (!questionsListEl) return;
    questionsListEl.innerHTML = '';
    
    questions.forEach((q, idx) => {
        const div = document.createElement('div');
        div.className = 'question-item';
        div.id = `question-${idx}`;
        div.innerHTML = `
            <span class="status-dot"></span>
            <span class="question-text">Q${q.id}: ${q.purpose}</span>
        `;
        questionsListEl.appendChild(div);
    });
}

function updateQuestionStatus(index, isComplete) {
    if (!questionsListEl) return;
    
    // Reset all
    document.querySelectorAll('.question-item').forEach((el, i) => {
        el.classList.remove('active');
        if (i < index) el.classList.add('completed');
    });
    
    if (!isComplete && index >= 0) {
        const activeEl = document.getElementById(`question-${index}`);
        if (activeEl) {
            activeEl.classList.add('active');
            activeEl.classList.remove('completed');
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// UTILITY
// ═════════════════════════════════════════════════════════════════════════════

function updateStatus(element, text, color = '#555') {
    if (element) {
        element.textContent = text;
        element.style.color = color;
    }
}

function updateAIStatus(text, color = '#888') {
    if (aiStatusEl) {
        aiStatusEl.textContent = text;
        aiStatusEl.style.color = color;
    }
}

function updateTranscriptPanel(text) {
    if (transcriptEl) {
        if (transcriptEl.textContent.trim() === 'Transcript will appear here after recording...') {
            transcriptEl.textContent = '';
        }
        const p = document.createElement('div');
        p.style.marginBottom = '8px';
        p.textContent = `> ${text}`;
        transcriptEl.appendChild(p);
        transcriptEl.scrollTop = transcriptEl.scrollHeight;
    }
}

function updateAgentPanel(text) {
    if (agentEl) {
        if (agentEl.textContent.trim() === 'Agent response will appear here...') {
            agentEl.textContent = '';
        }
        const p = document.createElement('div');
        p.style.marginBottom = '8px';
        p.textContent = `> ${text}`;
        agentEl.appendChild(p);
        agentEl.scrollTop = agentEl.scrollHeight;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// RINGTONE LOGIC
// ═════════════════════════════════════════════════════════════════════════════

let ringtoneInterval;
let ringtoneAudioCtx;

function startRinging() {
    if (ringtoneAudioCtx) return; // already ringing
    
    // Create audio context only when needed
    ringtoneAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
    
    function playRingBeep() {
        if (!ringtoneAudioCtx || ringtoneAudioCtx.state === 'closed') return;
        
        const osc1 = ringtoneAudioCtx.createOscillator();
        const osc2 = ringtoneAudioCtx.createOscillator();
        const gainNode = ringtoneAudioCtx.createGain();
        
        // Classic US/UK dual-tone ring frequencies
        osc1.frequency.value = 440;
        osc2.frequency.value = 480;
        
        osc1.connect(gainNode);
        osc2.connect(gainNode);
        gainNode.connect(ringtoneAudioCtx.destination);
        
        const now = ringtoneAudioCtx.currentTime;
        gainNode.gain.setValueAtTime(0, now);
        gainNode.gain.linearRampToValueAtTime(0.3, now + 0.05); // Fade in
        gainNode.gain.setValueAtTime(0.3, now + 1.95);
        gainNode.gain.linearRampToValueAtTime(0, now + 2.0);    // Fade out
        
        osc1.start(now);
        osc2.start(now);
        osc1.stop(now + 2.0);
        osc2.stop(now + 2.0);
    }
    
    // Play immediately, then repeat every 4 seconds (2s ring, 2s silence)
    playRingBeep();
    ringtoneInterval = setInterval(playRingBeep, 4000);
}

function stopRinging() {
    if (ringtoneInterval) {
        clearInterval(ringtoneInterval);
        ringtoneInterval = null;
    }
    if (ringtoneAudioCtx) {
        ringtoneAudioCtx.close();
        ringtoneAudioCtx = null;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// EVENT LISTENERS
// ═════════════════════════════════════════════════════════════════════════════

btnJoin.addEventListener('click', async () => {
    const micGranted = await getMicrophone();
    if (micGranted) {
        connectSignaling();
    }
});

btnCall.addEventListener('click',   startCall);
btnAccept.addEventListener('click', acceptCall);
btnHangup.addEventListener('click', hangUp);

btnStartAI.addEventListener('click', startAIPipeline);
btnStopAI.addEventListener('click',  stopAIPipeline);
btnTestTTS.addEventListener('click', testTTS);

if (btnStartSession) {
    btnStartSession.addEventListener('click', startVerificationSession);
}
