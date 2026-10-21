'use strict';

/**
 * audio-manager.js — Person 1 Browser-Side Audio Module
 * 
 * Handles everything audio on the browser:
 *   - Remote audio capture (MediaRecorder on remoteStream)
 *   - Audio upload to /api/audio
 *   - TTS audio playback through the WebRTC connection
 *   - Mode switching between human call and AI agent modes
 * 
 * ════════════════════════════════════════════════════════════════════
 * PERSON 2 INTERFACE (browser side)
 * 
 * After constructing AudioManager, Person 2 can set:
 *
 *   audioManager.onTranscript = (transcript) => {
 *       // transcript: { type, text, timestamp }
 *   };
 *
 * Person 2 can also call:
 *
 *   audioManager.handleAgentResponse({
 *       type: 'agent_response',
 *       text: '...',
 *       ttsUrl: '/api/tts/<id>',
 *       conversation_complete: false
 *   });
 * ════════════════════════════════════════════════════════════════════
 */

class AudioManager {

    /**
     * @param {Object} opts
     * @param {RTCPeerConnection} [opts.peerConnection]  - Set after WebRTC connection is established
     * @param {MediaStream}       [opts.remoteStream]    - The remote WebRTC stream
     */
    constructor(opts = {}) {
        this.peerConnection  = opts.peerConnection  || null;
        this.remoteStream    = opts.remoteStream    || null;

        // ── Internal state ────────────────────────────────────────────────────
        this._mediaRecorder   = null;
        this._recordedChunks  = [];
        this._isRecording     = false;
        this._audioContext    = null;
        this._ttsTrackSender  = null; // RTCRtpSender for the TTS audio track
        this._originalMicTrack = null; // saved mic track to restore after TTS
        this._mode            = 'human'; // 'human' | 'agent'
        this._mimeType        = this._detectSupportedMimeType();
        this._autoRecord      = false;  // when true, auto-starts recording when remote track arrives
        this._recordInterval  = null;   // for continuous chunked recording

        // ── Callbacks for Person 2 ────────────────────────────────────────────
        // Set these after construction:
        //   audioManager.onTranscript = fn
        //   audioManager.onError      = fn
        //   audioManager.onStateChange = fn

        /** @type {function({ type: string, text: string, timestamp: number }): void} */
        this.onTranscript   = null;

        /** @type {function(Error): void} */
        this.onError        = null;

        /** @type {function(string): void} */
        this.onStateChange  = null;
        
        /** @type {function(): void} */
        this.onTTSComplete  = null;

        console.log('[Audio] AudioManager initialised. MIME type:', this._mimeType || 'unknown');
    }

    // ── Configuration ─────────────────────────────────────────────────────────

    /**
     * Attach or update the RTCPeerConnection.
     * Must be called before TTS playback through WebRTC can work.
     * @param {RTCPeerConnection} pc
     */
    setPeerConnection(pc) {
        this.peerConnection = pc;
        console.log('[Audio] PeerConnection attached to AudioManager.');
    }

    /**
     * Attach or update the remote MediaStream.
     * Must be called before recording can start.
     * @param {MediaStream} stream
     */
    setRemoteStream(stream) {
        this.remoteStream = stream;
        console.log('[Audio] Remote stream attached to AudioManager.');

        if (this._autoRecord && this._mode === 'agent') {
            console.log('[Audio] Auto-starting remote capture (agent mode).');
            this.startRemoteAudioCapture();
        }
    }

    /**
     * Switch between 'human' and 'agent' modes.
     * - 'human': Normal two-way call. No recording, no STT, no TTS injection.
     * - 'agent': Remote audio is captured → STT → AI → TTS → WebRTC.
     * 
     * @param {'human'|'agent'} mode
     */
    setMode(mode) {
        if (mode !== 'human' && mode !== 'agent') {
            console.error('[Audio] Invalid mode. Use "human" or "agent".');
            return;
        }
        console.log(`[Audio] Mode: ${this._mode} → ${mode}`);
        this._mode = mode;
        this._emitState(`mode:${mode}`);
    }

    // ── Remote Audio Capture ──────────────────────────────────────────────────

    /**
     * Start recording the remote WebRTC audio stream.
     * When stopRemoteAudioCapture() is called, the recording is uploaded to /api/audio.
     */
    startRemoteAudioCapture() {
        if (!this.remoteStream) {
            this._emitError('[Audio] Cannot start capture — remote stream not available.');
            return;
        }
        if (this._isRecording) {
            console.warn('[Audio] Already recording.');
            return;
        }
        if (!this._mimeType) {
            this._emitError('[Audio] No supported MediaRecorder MIME type found in this browser.');
            return;
        }

        this._recordedChunks = [];

        try {
            this._mediaRecorder = new MediaRecorder(this.remoteStream, {
                mimeType: this._mimeType,
            });
        } catch (err) {
            this._emitError(`[Audio] MediaRecorder failed to start: ${err.message}`);
            return;
        }

        this._mediaRecorder.ondataavailable = (event) => {
            if (event.data && event.data.size > 0) {
                this._recordedChunks.push(event.data);
            }
        };

        this._mediaRecorder.onerror = (event) => {
            this._emitError(`[Audio] MediaRecorder error: ${event.error?.message || 'unknown'}`);
        };

        this._mediaRecorder.onstop = () => {
            console.log('[Audio] MediaRecorder stopped. Chunks:', this._recordedChunks.length);
            if (this._recordedChunks.length > 0) {
                const blob = new Blob(this._recordedChunks, { type: this._mimeType });
                console.log('[Audio] Blob ready:', blob.size, 'bytes');
                this._uploadAudioBlob(blob);
            } else {
                console.warn('[Audio] No audio data recorded.');
            }
            this._recordedChunks = [];
            this._isRecording    = false;
        };

        this._mediaRecorder.start();
        this._isRecording = true;
        console.log('[Audio] Started recording remote audio. MIME:', this._mimeType);
        this._emitState('recording:started');
    }
    
    /**
     * Start recording and automatically stop after a period of silence.
     * @param {number} silenceMs - Milliseconds of silence before stopping.
     */
    startRemoteAudioCaptureWithSilenceDetection(silenceMs = 2000) {
        this.startRemoteAudioCapture();
        
        if (!this._audioContext || this._audioContext.state === 'closed') {
            this._audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        
        const analyser = this._audioContext.createAnalyser();
        const source = this._audioContext.createMediaStreamSource(this.remoteStream);
        source.connect(analyser);
        
        let silenceStart = null;
        const recordingStartTime = Date.now();
        const minRecordingMs = 3000; // Wait at least 3s before allowing silence-stop
        
        const checkSilence = () => {
            if (!this._isRecording) return; // Stop checking if recording stopped
            
            const data = new Uint8Array(analyser.frequencyBinCount);
            analyser.getByteFrequencyData(data);
            const volume = data.reduce((a, b) => a + b, 0) / data.length;
            
            const elapsed = Date.now() - recordingStartTime;
            
            // Don't check for silence until minimum recording time has passed
            if (elapsed < minRecordingMs) {
                requestAnimationFrame(checkSilence);
                return;
            }
            
            if (volume < 5) { // silence threshold
                if (!silenceStart) {
                    silenceStart = Date.now();
                } else if (Date.now() - silenceStart > silenceMs) {
                    console.log(`[Audio] Silence detected for ${silenceMs}ms after ${(elapsed/1000).toFixed(1)}s recording. Stopping capture.`);
                    this.stopRemoteAudioCapture();
                    return;
                }
            } else {
                silenceStart = null;
            }
            
            requestAnimationFrame(checkSilence);
        };
        
        checkSilence();
    }

    /**
     * Stop the current recording.
     * Triggers upload to /api/audio automatically.
     */
    stopRemoteAudioCapture() {
        if (!this._mediaRecorder || this._mediaRecorder.state === 'inactive') {
            console.warn('[Audio] No active recording to stop.');
            return;
        }
        console.log('[Audio] Stopping recording...');
        this._mediaRecorder.stop();
        this._emitState('recording:stopped');
    }

    /**
     * Convenience — record for a fixed duration then auto-stop.
     * Useful for chunked capture in AI agent mode.
     * @param {number} durationMs
     */
    captureChunk(durationMs = 5000) {
        this.startRemoteAudioCapture();
        setTimeout(() => this.stopRemoteAudioCapture(), durationMs);
    }

    // ── Audio Upload ──────────────────────────────────────────────────────────

    /**
     * Upload a recorded audio Blob to POST /api/audio.
     * The server handles STT → AI → TTS and emits the result back via Socket.IO.
     * 
     * @private
     * @param {Blob} blob
     */
    async _uploadAudioBlob(blob) {
        console.log('[Audio] Uploading blob:', blob.size, 'bytes,', blob.type);
        this._emitState('upload:started');

        const formData = new FormData();
        formData.append('audio', blob, `recording.${this._mimeToExt(blob.type)}`);

        try {
            const response = await fetch('/api/audio', {
                method: 'POST',
                body:   formData,
            });

            if (!response.ok) {
                const errBody = await response.json().catch(() => ({}));
                throw new Error(errBody.error || `HTTP ${response.status}`);
            }

            const data = await response.json();
            console.log('[Audio] Upload successful. Transcript:', data.transcript);

            // Notify Person 2's callback if set
            if (typeof this.onTranscript === 'function') {
                this.onTranscript({
                    type:      'transcript',
                    text:      data.transcript,
                    timestamp: Date.now(),
                });
            }

            this._emitState('upload:complete');

        } catch (err) {
            this._emitError(`[Audio] Upload failed: ${err.message}`);
        }
    }

    // ── TTS Playback through WebRTC ───────────────────────────────────────────

    /**
     * Fetch TTS audio from the server and play it through the WebRTC connection.
     * 
     * Flow:
     *   /api/tts/:id  →  ArrayBuffer  →  AudioContext  →  MediaStreamDestination
     *   → MediaStreamTrack  →  RTCPeerConnection.addTrack / replaceTrack
     * 
     * The receiver (phone) hears the AI voice through the existing WebRTC connection.
     * The local microphone track is preserved; only the outgoing track is replaced.
     * 
     * @param {string} ttsUrl - e.g. '/api/tts/some-uuid'
     */
    async playTTSAudio(ttsUrl) {
        if (!this.peerConnection) {
            this._emitError('[Audio] Cannot play TTS — no RTCPeerConnection.');
            return;
        }

        console.log('[Audio] Fetching TTS audio:', ttsUrl);

        let arrayBuffer;
        try {
            const response = await fetch(ttsUrl);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            arrayBuffer = await response.arrayBuffer();
        } catch (err) {
            this._emitError(`[Audio] Failed to fetch TTS audio: ${err.message}`);
            return;
        }

        try {
            // Re-use or create AudioContext
            if (!this._audioContext || this._audioContext.state === 'closed') {
                this._audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }
            if (this._audioContext.state === 'suspended') {
                await this._audioContext.resume();
            }

            const audioBuffer = await this._audioContext.decodeAudioData(arrayBuffer);

            // Create a destination stream — this is what we inject into WebRTC
            const destination = this._audioContext.createMediaStreamDestination();

            // Source node plays the decoded TTS audio
            const source = this._audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(destination);

            const [ttsTrack] = destination.stream.getAudioTracks();

            if (!ttsTrack) {
                this._emitError('[Audio] Failed to get audio track from MediaStreamDestination.');
                return;
            }

            // Inject into WebRTC ─────────────────────────────────────────────
            await this._injectTTSTrack(ttsTrack);

            // Start playback — audio flows through WebRTC to the receiver
            source.start(0);

            console.log(`[Audio] TTS playing through WebRTC. Duration: ${audioBuffer.duration.toFixed(2)}s`);
            this._emitState('tts:playing');

            source.onended = () => {
                console.log('[Audio] TTS playback complete.');
                this._emitState('tts:ended');
                // Restore mic track so phone user audio comes through again
                this._restoreMicTrack();
                if (typeof this.onTTSComplete === 'function') {
                    this.onTTSComplete();
                }
            };

        } catch (err) {
            this._emitError(`[Audio] TTS playback through WebRTC failed: ${err.message}`);
        }
    }

    /**
     * Inject a TTS MediaStreamTrack into the RTCPeerConnection.
     * 
     * Strategy:
     *   - Find the existing audio sender (the mic track) and replace it
     *     with the TTS track. This avoids renegotiation entirely.
     *   - Save the original mic track so it can be restored after TTS ends.
     * 
     * @private
     * @param {MediaStreamTrack} ttsTrack
     */
    async _injectTTSTrack(ttsTrack) {
        // Find the existing audio sender (mic track)
        const senders = this.peerConnection.getSenders();
        const audioSender = senders.find(s => s.track && s.track.kind === 'audio');

        if (audioSender) {
            // Save original mic track for restoration after TTS
            if (!this._originalMicTrack) {
                this._originalMicTrack = audioSender.track;
            }
            await audioSender.replaceTrack(ttsTrack);
            this._ttsTrackSender = audioSender;
            console.log('[WebRTC] Replaced mic track with TTS track.');
        } else {
            console.warn('[WebRTC] No audio sender found. Adding TTS track as new sender.');
            this._ttsTrackSender = this.peerConnection.addTrack(
                ttsTrack,
                new MediaStream([ttsTrack])
            );
        }
    }

    /**
     * Restore the original microphone track after TTS playback ends.
     * @private
     */
    async _restoreMicTrack() {
        if (this._originalMicTrack && this._ttsTrackSender) {
            try {
                await this._ttsTrackSender.replaceTrack(this._originalMicTrack);
                console.log('[WebRTC] Restored original mic track.');
            } catch (err) {
                console.error('[WebRTC] Failed to restore mic track:', err);
            }
        }
    }

    // ── Person 2 Interface ────────────────────────────────────────────────────

    /**
     * Called when the server sends an agent-response event (via Socket.IO).
     * This is the incoming direction of the Person 1 ↔ Person 2 interface
     * on the browser side.
     * 
     * Person 2 can also call this directly if they want to trigger TTS
     * from the browser side.
     * 
     * @param {{ type: string, text: string, ttsUrl: string, conversation_complete: boolean }} response
     */
    handleAgentResponse(response) {
        if (!response || !response.ttsUrl) {
            console.error('[Audio] handleAgentResponse: invalid response or missing ttsUrl', response);
            return;
        }
        console.log('[Audio] Agent response received:', response.text?.substring(0, 60));
        this.playTTSAudio(response.ttsUrl);
    }

    // ── MIME detection ────────────────────────────────────────────────────────

    /**
     * Detect the best supported MIME type for MediaRecorder in this browser.
     * Prefers opus/webm for smallest file size and best compatibility.
     * @returns {string|null}
     */
    _detectSupportedMimeType() {
        const candidates = [
            'audio/webm;codecs=opus',
            'audio/webm',
            'audio/ogg;codecs=opus',
            'audio/ogg',
            'audio/mp4',
            '',  // let browser choose
        ];

        for (const type of candidates) {
            if (type === '' || MediaRecorder.isTypeSupported(type)) {
                console.log('[Audio] Selected MIME type:', type || '(browser default)');
                return type || null;
            }
        }
        return null;
    }

    _mimeToExt(mimeType) {
        if (!mimeType) return 'bin';
        if (mimeType.includes('webm')) return 'webm';
        if (mimeType.includes('ogg'))  return 'ogg';
        if (mimeType.includes('mp4'))  return 'mp4';
        if (mimeType.includes('wav'))  return 'wav';
        return 'bin';
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    _emitError(message) {
        console.error(message);
        if (typeof this.onError === 'function') {
            this.onError(new Error(message));
        }
    }

    _emitState(state) {
        if (typeof this.onStateChange === 'function') {
            this.onStateChange(state);
        }
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────

    /**
     * Release all resources. Call when the call ends.
     */
    destroy() {
        if (this._mediaRecorder && this._mediaRecorder.state !== 'inactive') {
            this._mediaRecorder.stop();
        }
        if (this._audioContext && this._audioContext.state !== 'closed') {
            this._audioContext.close();
        }
        clearInterval(this._recordInterval);
        this._ttsTrackSender = null;
        this._audioContext   = null;
        this._mediaRecorder  = null;
        this.peerConnection  = null;
        this.remoteStream    = null;
        console.log('[Audio] AudioManager destroyed.');
    }
}

// Export as a global (vanilla JS, no bundler)
window.AudioManager = AudioManager;
