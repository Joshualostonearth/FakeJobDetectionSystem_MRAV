'use strict';

/**
 * audio-utils.js
 * 
 * Utility helpers for temporary file management and audio metadata.
 * All file paths are scoped to server/uploads/ to keep temp files isolated.
 */

const path = require('path');
const fs   = require('fs');
const { v4: uuidv4 } = require('uuid');

// ── Uploads directory ────────────────────────────────────────────────────────

const UPLOADS_DIR = path.join(__dirname, '..', 'uploads');

/**
 * Ensure the uploads/ directory exists.
 * Called once at module load — safe to call multiple times.
 */
function ensureUploadsDir() {
    if (!fs.existsSync(UPLOADS_DIR)) {
        fs.mkdirSync(UPLOADS_DIR, { recursive: true });
        console.log('[Audio] Created uploads directory:', UPLOADS_DIR);
    }
}

ensureUploadsDir();

// ── File path helpers ────────────────────────────────────────────────────────

/**
 * Generate a unique temporary file path inside uploads/.
 * @param {string} ext - File extension without dot, e.g. 'webm' or 'wav'
 * @returns {string} Absolute file path
 */
function getTempFilePath(ext) {
    return path.join(UPLOADS_DIR, `${uuidv4()}.${ext}`);
}

/**
 * Get the path for a TTS output file by ID.
 * @param {string} id  - UUID
 * @param {string} ext - e.g. 'wav'
 * @returns {string}
 */
function getTTSFilePath(id, ext = 'wav') {
    return path.join(UPLOADS_DIR, `tts_${id}.${ext}`);
}

// ── Cleanup ──────────────────────────────────────────────────────────────────

/**
 * Delete a temporary file. Errors are logged but not thrown.
 * @param {string} filePath
 */
function cleanupFile(filePath) {
    if (!filePath) return;
    try {
        if (fs.existsSync(filePath)) {
            fs.unlinkSync(filePath);
            console.log('[Audio] Cleaned up temp file:', path.basename(filePath));
        }
    } catch (err) {
        console.error('[Audio] Failed to clean up file:', filePath, err.message);
    }
}

// ── Audio metadata ───────────────────────────────────────────────────────────

/**
 * Estimate audio duration from file size (rough approximation).
 * This is a fallback — replace with ffprobe for accuracy if needed.
 * 
 * @param {string} filePath
 * @param {number} [bitrateKbps=64] - Assumed bitrate in kbps (opus webm ≈ 64)
 * @returns {number} Estimated duration in seconds
 */
function estimateAudioDuration(filePath, bitrateKbps = 64) {
    try {
        const stats  = fs.statSync(filePath);
        const bytes  = stats.size;
        const bitsPerSec = bitrateKbps * 1000;
        return parseFloat(((bytes * 8) / bitsPerSec).toFixed(2));
    } catch {
        return 0;
    }
}

/**
 * Map a MIME type string to a file extension.
 * @param {string} mimeType - e.g. 'audio/webm;codecs=opus'
 * @returns {string} e.g. 'webm'
 */
function mimeToExt(mimeType) {
    if (!mimeType) return 'bin';
    if (mimeType.includes('webm'))  return 'webm';
    if (mimeType.includes('ogg'))   return 'ogg';
    if (mimeType.includes('mp4'))   return 'mp4';
    if (mimeType.includes('mpeg'))  return 'mp3';
    if (mimeType.includes('wav'))   return 'wav';
    return 'bin';
}

// ────────────────────────────────────────────────────────────────────────────

module.exports = {
    UPLOADS_DIR,
    getTempFilePath,
    getTTSFilePath,
    cleanupFile,
    estimateAudioDuration,
    mimeToExt,
};
