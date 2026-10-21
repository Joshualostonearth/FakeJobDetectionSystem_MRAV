'use strict';

/**
 * tts.js — Text-to-Speech Abstraction Layer
 * 
 * This module is the ONLY place in the codebase that knows which TTS
 * provider is being used. All other modules call synthesizeSpeech() and
 * receive an audio buffer / file path back.
 * 
 * ════════════════════════════════════════════════════════════════════
 * PERSON 2 / TTS PLUG-IN POINT
 * 
 * When a real TTS provider becomes available:
 *   1. Set in .env:
 *        TTS_PROVIDER=<provider>
 *        TTS_API_KEY=<key>
 *        TTS_API_URL=<endpoint>
 *   2. Implement _synthesizeWithRealProvider() below.
 *   3. The rest of the application changes zero lines.
 * ════════════════════════════════════════════════════════════════════
 */

require('dotenv').config({ path: require('path').join(__dirname, '..', '.env') });

const fs   = require('fs');
const path = require('path');
const { getTempFilePath } = require('./audio-utils');

const TTS_PROVIDER = process.env.TTS_PROVIDER || 'mock';

console.log(`[TTS] Provider: ${TTS_PROVIDER}`);

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Synthesize text into speech audio.
 * 
 * This is the single function the rest of the application depends on.
 * Called by the /api/audio route after the AI interface returns a response.
 * 
 * @param {string} text - The text to synthesize
 * @returns {Promise<{ audioPath: string, mimeType: string, durationEstimate: number }>}
 * @throws Will throw if synthesis fails — caller must handle.
 */
async function synthesizeSpeech(text) {
    console.log(`[TTS] Synthesizing: "${text.substring(0, 80)}${text.length > 80 ? '...' : ''}"`);

    switch (TTS_PROVIDER) {
        case 'edgetts':
            return await _synthesizeWithEdgeTTS(text);

        case 'mock':
        default:
            return await _synthesizeWithMock(text);
    }
}

// ── Provider: Mock ─────────────────────────────────────────────────────────────
// Generates a minimal valid WAV file (silence) so the entire pipeline from
// text → audio file → browser → WebRTC track is real and testable.
// Replace this with a real TTS API call when available.

async function _synthesizeWithMock(text) {
    await _delay(300); // simulate processing

    // Estimate ~200ms of audio per word (rough)
    const wordCount      = text.split(/\s+/).length;
    const durationSec    = Math.max(1, wordCount * 0.35);
    const sampleRate     = 16000;
    const numSamples     = Math.floor(sampleRate * durationSec);
    const wavBuffer      = _generateSilentWAV(numSamples, sampleRate);

    const audioPath = getTempFilePath('wav');
    fs.writeFileSync(audioPath, wavBuffer);

    console.log(`[TTS] Mock audio generated: ${path.basename(audioPath)} (${durationSec.toFixed(1)}s)`);

    return {
        audioPath,
        mimeType: 'audio/wav',
        durationEstimate: durationSec,
    };
}

// ── Provider: Real TTS ─────────────────────────────────────────────────────────
// ════════════════════════════════════════════════════════════════════
// REAL TTS IMPLEMENTATION GOES HERE
//
// When TTS API details are available, implement this function and
// update synthesizeSpeech() to call it based on TTS_PROVIDER env var.
//
// Expected return shape:
//   { audioPath: string, mimeType: string, durationEstimate: number }
//
// Example skeleton:
//   const response = await fetch(process.env.TTS_API_URL, {
//       method: 'POST',
//       headers: { Authorization: `Bearer ${process.env.TTS_API_KEY}`, 'Content-Type': 'application/json' },
//       body: JSON.stringify({ text, voice: 'en-US-standard' }),
//   });
//   const audioBuffer = Buffer.from(await response.arrayBuffer());
//   const audioPath = getTempFilePath('wav');
//   fs.writeFileSync(audioPath, audioBuffer);
//   return { audioPath, mimeType: 'audio/wav', durationEstimate: ... };
// ════════════════════════════════════════════════════════════════════

// ── WAV file generator ─────────────────────────────────────────────────────────

/**
 * Generate a minimal valid WAV file containing silence.
 * This makes the mock TTS produce a real, playable audio file
 * so the full pipeline (TTS → WebRTC → phone) can be tested.
 * 
 * @param {number} numSamples  - Number of 16-bit samples
 * @param {number} sampleRate  - Sample rate in Hz (e.g. 16000)
 * @returns {Buffer}
 */
function _generateSilentWAV(numSamples, sampleRate = 16000) {
    const numChannels  = 1;
    const bitsPerSample = 16;
    const byteRate     = sampleRate * numChannels * (bitsPerSample / 8);
    const blockAlign   = numChannels * (bitsPerSample / 8);
    const dataSize     = numSamples * blockAlign;
    const fileSize     = 44 + dataSize;

    const buf = Buffer.alloc(fileSize, 0);

    // RIFF header
    buf.write('RIFF', 0);
    buf.writeUInt32LE(fileSize - 8, 4);
    buf.write('WAVE', 8);

    // fmt chunk
    buf.write('fmt ', 12);
    buf.writeUInt32LE(16, 16);          // chunk size
    buf.writeUInt16LE(1, 20);           // PCM = 1
    buf.writeUInt16LE(numChannels, 22);
    buf.writeUInt32LE(sampleRate, 24);
    buf.writeUInt32LE(byteRate, 28);
    buf.writeUInt16LE(blockAlign, 32);
    buf.writeUInt16LE(bitsPerSample, 34);

    // data chunk
    buf.write('data', 36);
    buf.writeUInt32LE(dataSize, 40);
    // Samples are zero (silence) — already zeroed by Buffer.alloc

    return buf;
}

// ── Provider: Edge-TTS (via Python Bridge) ────────────────────────────────────

async function _synthesizeWithEdgeTTS(text) {
    console.log(`[TTS] Synthesizing with Edge-TTS (via bridge): "${text.substring(0, 50)}..."`);
    
    // We use the global fetch API (available in Node 18+)
    const res = await fetch('http://localhost:5000/bridge/synthesize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
    });

    if (!res.ok) {
        const errorText = await res.text();
        throw new Error(`Bridge returned HTTP ${res.status}: ${errorText}`);
    }

    const arrayBuffer = await res.arrayBuffer();
    const audioBuffer = Buffer.from(arrayBuffer);
    
    const audioPath = getTempFilePath('mp3');
    fs.writeFileSync(audioPath, audioBuffer);
    
    // Estimate duration: ~0.35s per word for English
    const wordCount = text.split(/\s+/).length;
    const durationSec = Math.max(1, wordCount * 0.35);

    console.log(`[TTS] Edge-TTS audio generated: ${path.basename(audioPath)} (${durationSec.toFixed(1)}s estimate)`);

    return {
        audioPath,
        mimeType: 'audio/mpeg',
        durationEstimate: durationSec,
    };
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function _delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ─────────────────────────────────────────────────────────────────────────────

module.exports = { synthesizeSpeech };
