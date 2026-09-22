'use strict';

/**
 * stt.js — Speech-to-Text Abstraction Layer
 * 
 * This module is the ONLY place in the codebase that knows which STT
 * provider is being used. All other modules call transcribeAudio() and
 * receive a plain transcript string back.
 * 
 * ════════════════════════════════════════════════════════════════════
 * PERSON 2 / WHISPERFLOW PLUG-IN POINT
 * 
 * When WhisperFlow API details become available:
 *   1. Set in .env:
 *        STT_PROVIDER=whisperflow
 *        STT_API_KEY=<key>
 *        STT_API_URL=<endpoint>
 *   2. Implement _transcribeWithWhisperFlow() below.
 *   3. The rest of the application changes zero lines.
 * ════════════════════════════════════════════════════════════════════
 */

require('dotenv').config({ path: require('path').join(__dirname, '..', '.env') });

const fs   = require('fs');
const path = require('path');

const STT_PROVIDER = process.env.STT_PROVIDER || 'mock';

console.log(`[STT] Provider: ${STT_PROVIDER}`);

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Transcribe an audio file to text.
 * 
 * This is the single function the rest of the application depends on.
 * Person 2 does NOT call this directly — it is invoked by the /api/audio route
 * and the result is forwarded to ai-interface.js.
 * 
 * @param {string} audioPath - Absolute path to the audio file on disk
 * @returns {Promise<{ transcript: string, duration: number }>}
 * @throws Will throw if transcription fails — caller must handle.
 */
async function transcribeAudio(audioPath) {
    console.log(`[STT] Transcribing: ${path.basename(audioPath)}`);

    switch (STT_PROVIDER) {
        case 'whisperflow':
            return await _transcribeWithWhisperFlow(audioPath);

        case 'bridge':
            return await _transcribeWithBridge(audioPath);

        case 'mock':
        default:
            return await _transcribeWithMock(audioPath);
    }
}

// ── Provider: Mock ─────────────────────────────────────────────────────────────
// Temporary mock implementation for testing the full pipeline before a real
// STT service is available. Returns a hardcoded transcript.

async function _transcribeWithMock(audioPath) {
    // Simulate a short network/processing delay
    await _delay(600);

    // Estimate duration from file size (rough)
    let duration = 0;
    try {
        const stats = fs.statSync(audioPath);
        duration = parseFloat(((stats.size * 8) / (64 * 1000)).toFixed(2));
    } catch { /* ignore */ }

    const transcript = '[MOCK STT] Hello, this is a test transcription from the mock STT provider.';

    console.log(`[STT] Transcript: "${transcript}"`);
    console.log(`[STT] Estimated duration: ${duration}s`);

    return { transcript, duration };
}

// ── Provider: WhisperFlow ──────────────────────────────────────────────────────
// ════════════════════════════════════════════════════════════════════
// WHISPERFLOW IMPLEMENTATION GOES HERE
// 
// When the API details are available, implement this function.
// Expected shape of return value:
//   { transcript: string, duration: number }
//
// Example skeleton:
//   const response = await fetch(process.env.STT_API_URL, {
//       method: 'POST',
//       headers: { Authorization: `Bearer ${process.env.STT_API_KEY}` },
//       body: formDataWithAudioFile,
//   });
//   const data = await response.json();
//   return { transcript: data.text, duration: data.duration };
// ════════════════════════════════════════════════════════════════════

async function _transcribeWithWhisperFlow(audioPath) {
    throw new Error(
        '[STT] WhisperFlow provider is not yet implemented. ' +
        'Set STT_PROVIDER=mock in .env to use the mock provider, ' +
        'or implement _transcribeWithWhisperFlow() in server/audio/stt.js.'
    );
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function _delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ── Provider: Bridge ──────────────────────────────────────────────────────────
// When the Python bridge handles audio processing end-to-end, this is a no-op.
// The actual transcription happens in call_bridge.py's /bridge/process-audio.

async function _transcribeWithBridge(audioPath) {
    console.log('[STT] Bridge mode — STT handled by Python sidecar.');
    // Return a placeholder; the bridge does the real transcription.
    // In bridge mode, server.js sends raw audio to the bridge directly
    // and bypasses this module's transcription.
    let duration = 0;
    try {
        const stats = fs.statSync(audioPath);
        duration = parseFloat(((stats.size * 8) / (64 * 1000)).toFixed(2));
    } catch { /* ignore */ }

    return { transcript: '[Bridge mode]', duration };
}

// ─────────────────────────────────────────────────────────────────────────────

module.exports = { transcribeAudio };
