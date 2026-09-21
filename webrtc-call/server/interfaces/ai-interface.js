'use strict';

/**
 * ai-interface.js — Person 1 ↔ Person 2 Integration Boundary
 * 
 * ════════════════════════════════════════════════════════════════════════════
 * THIS IS THE MOST IMPORTANT FILE FOR PERSON 2 TO READ
 * 
 * Person 1 owns:  sendTranscript()  — delivers a transcript to Person 2
 * Person 2 owns:  the function body of sendTranscript() (LLM call goes here)
 * 
 * Person 1 also owns: handleAgentResponse() — receives Person 2's text
 *                     and routes it through TTS → WebRTC
 * 
 * Person 2 does NOT call handleAgentResponse() directly —
 * Person 2 returns a response object FROM sendTranscript(), which
 * Person 1 then passes to handleAgentResponse().
 * ════════════════════════════════════════════════════════════════════════════
 * 
 * DATA CONTRACTS
 * ──────────────
 * Person 1 → Person 2  (transcript object):
 * {
 *   type:      "transcript",
 *   text:      "Yes, the software engineering internship is currently open.",
 *   timestamp: 1723456789      // Unix timestamp (ms)
 * }
 * 
 * Person 2 → Person 1  (agent response object):
 * {
 *   type:                "agent_response",
 *   text:                "Could you tell me the official careers page?",
 *   conversation_complete: false
 * }
 * ════════════════════════════════════════════════════════════════════════════
 */

const EventEmitter = require('events');

// ── Event emitter for decoupled transcript delivery ───────────────────────────
// Person 2 can also listen to 'transcript' events on this emitter
// if they prefer an event-based integration instead of replacing sendTranscript().

const aiEvents = new EventEmitter();

// ── Transcript → AI (Person 2 replaces this) ─────────────────────────────────

/**
 * Called by Person 1 when a new transcript is ready.
 * 
 * @param {{ type: string, text: string, timestamp: number }} transcript
 * @returns {Promise<{ type: string, text: string, conversation_complete: boolean }>}
 * 
 * ════════════════════════════════════════════════════════════════════════════
 * PERSON 2: REPLACE THE BODY OF THIS FUNCTION WITH YOUR LLM CALL.
 * 
 * Steps:
 *   1. Receive the transcript
 *   2. Call your LLM / conversation manager
 *   3. Return the agent response object
 * 
 * Example integration:
 *   const llmResponse = await myConversationManager.process(transcript.text);
 *   return {
 *       type: 'agent_response',
 *       text: llmResponse.nextQuestion,
 *       conversation_complete: llmResponse.isDone,
 *   };
 * ════════════════════════════════════════════════════════════════════════════
 */
async function sendTranscript(transcript) {
    console.log('[AI Interface] Transcript received:');
    console.log(`[AI Interface]   text: "${transcript.text}"`);
    console.log(`[AI Interface]   timestamp: ${transcript.timestamp}`);

    // Emit as an event (Person 2 can also listen here instead of replacing this function)
    aiEvents.emit('transcript', transcript);

    // ════════════════════════════════════════════
    // MOCK RESPONSE — Person 2 replaces this block
    // ════════════════════════════════════════════
    const mockResponse = _getMockResponse(transcript);

    console.log('[AI Interface] Agent response (MOCK):');
    console.log(`[AI Interface]   text: "${mockResponse.text}"`);
    console.log(`[AI Interface]   conversation_complete: ${mockResponse.conversation_complete}`);

    return mockResponse;
    // ════════════════════════════════════════════
}

// ── Agent response handler (Person 1 owns this) ───────────────────────────────

/**
 * Called by Person 1 after receiving an agent response from sendTranscript().
 * Routes the response text through TTS → WebRTC audio track.
 * 
 * Person 2 does NOT call this. Person 1 calls this internally.
 * Person 2's text arrives via the return value of sendTranscript().
 * 
 * @param {{ type: string, text: string, conversation_complete: boolean }} response
 * @param {Function} onTTSReady - callback(ttsId: string) called when TTS is ready
 */
async function handleAgentResponse(response, onTTSReady) {
    console.log('[AI Interface] Handling agent response → TTS pipeline');

    if (!response || !response.text) {
        console.error('[AI Interface] Invalid agent response: missing text');
        return;
    }

    // Emit for any external listeners
    aiEvents.emit('agent-response', response);

    // Signal TTS readiness to the caller (server.js route handler)
    if (typeof onTTSReady === 'function') {
        onTTSReady(response);
    }
}

// ── Mock response generator ───────────────────────────────────────────────────
// This is a simple rotating set of generic responses so the pipeline
// can be tested from end to end before Person 2 integrates the LLM.
// REMOVE or REPLACE when Person 2 implements sendTranscript().

const _mockResponses = [
    'Thank you for that information. Could you please tell me more?',
    'I understand. Can you elaborate on that point?',
    'That is helpful. What else can you tell me?',
    'Interesting. Could you provide any additional details?',
    'I see. Is there anything else you would like to add?',
];

let _mockIndex = 0;

function _getMockResponse(transcript) {
    const text = _mockResponses[_mockIndex % _mockResponses.length];
    _mockIndex++;

    return {
        type: 'agent_response',
        text,
        conversation_complete: false,
    };
}

// ─────────────────────────────────────────────────────────────────────────────

module.exports = {
    sendTranscript,
    handleAgentResponse,
    aiEvents,          // Export the event emitter for Person 2's convenience
};
