'use strict';

/**
 * audio-store.js
 * 
 * Manages the lifecycle of temporary audio files:
 *   - Uploaded audio blobs (for STT processing)
 *   - Generated TTS audio files (served back to browser)
 * 
 * Files are stored under server/uploads/ and cleaned up after use.
 */

const fs   = require('fs');
const path = require('path');
const { v4: uuidv4 } = require('uuid');
const { UPLOADS_DIR, getTTSFilePath, cleanupFile } = require('./audio-utils');

// ── TTS file registry ─────────────────────────────────────────────────────────
// Maps TTS ID → { filePath, mimeType, createdAt }
// Person 2 does not need to interact with this directly.

const _ttsRegistry = new Map();

const TTS_TTL_MS = 5 * 60 * 1000; // 5 minutes — clean up old TTS files

// ── Uploaded audio ────────────────────────────────────────────────────────────

/**
 * Save a raw audio buffer to a temp file.
 * @param {Buffer} buffer    - Raw audio data
 * @param {string} ext       - File extension, e.g. 'webm'
 * @returns {string}         - Absolute path of saved file
 */
function saveUploadedAudio(buffer, ext = 'webm') {
    const filePath = path.join(UPLOADS_DIR, `upload_${uuidv4()}.${ext}`);
    fs.writeFileSync(filePath, buffer);
    console.log('[Audio Store] Saved upload:', path.basename(filePath), `(${buffer.length} bytes)`);
    return filePath;
}

// ── TTS audio ─────────────────────────────────────────────────────────────────

/**
 * Register a TTS audio file and return its ID for retrieval.
 * @param {string} filePath  - Absolute path to the generated audio file
 * @param {string} mimeType  - e.g. 'audio/wav'
 * @returns {string}         - TTS ID for use in /api/tts/:id
 */
function registerTTSFile(filePath, mimeType = 'audio/wav') {
    const id = uuidv4();
    _ttsRegistry.set(id, { filePath, mimeType, createdAt: Date.now() });
    console.log('[Audio Store] Registered TTS file:', id, path.basename(filePath));
    return id;
}

/**
 * Retrieve TTS file info by ID.
 * @param {string} id
 * @returns {{ filePath: string, mimeType: string } | null}
 */
function getTTSFile(id) {
    return _ttsRegistry.get(id) || null;
}

/**
 * Delete a TTS file and remove it from the registry.
 * Call after the browser has received/played the audio.
 * @param {string} id
 */
function removeTTSFile(id) {
    const entry = _ttsRegistry.get(id);
    if (entry) {
        cleanupFile(entry.filePath);
        _ttsRegistry.delete(id);
    }
}

// ── TTL cleanup ───────────────────────────────────────────────────────────────

/**
 * Remove TTS files older than TTS_TTL_MS.
 * Called periodically so stale files don't accumulate.
 */
function cleanupStaleTTSFiles() {
    const now = Date.now();
    for (const [id, entry] of _ttsRegistry.entries()) {
        if (now - entry.createdAt > TTS_TTL_MS) {
            console.log('[Audio Store] TTL expired, cleaning up TTS:', id);
            removeTTSFile(id);
        }
    }
}

// Run cleanup every 2 minutes
setInterval(cleanupStaleTTSFiles, 2 * 60 * 1000);

// ─────────────────────────────────────────────────────────────────────────────

module.exports = {
    saveUploadedAudio,
    registerTTSFile,
    getTTSFile,
    removeTTSFile,
};
