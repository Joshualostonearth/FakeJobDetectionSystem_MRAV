"""
AI Phone-Calling Assistant — Fake Job Detection
=================================================

Calls a company on behalf of a candidate to verify whether a job/internship
offer is genuine. Asks 10 scripted questions and records responses.

Pipeline:
  Input JSON → Ollama LLM (generate questions) → Edge-TTS (speak) →
  WhisperLive (listen) → Response JSON

Usage:
    1. Start the WhisperLive server:   python server.py
    2. Make sure Ollama is running:    ollama serve
    3. Run:                            python caller.py input/company_role.json
       Dry-run (no call):              python caller.py input/company_role.json --dry-run

Press Ctrl+C to stop the call at any time.
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timezone

from voice_tools import EdgeTTSSpeaker, OllamaChat, WhisperListener


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

OLLAMA_MODEL = "qwen3-vl:latest"
TTS_VOICE = "en-US-AriaNeural"

INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input")
RESPONSES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "responses")

FORBIDDEN_WORDS = {"fake", "scam", "fraud", "suspicious", "fraudulent", "illegitimate"}

# Question generation system prompt for the LLM
QUESTION_GEN_PROMPT = """\
You are a question writer for an automated AI phone assistant.
The assistant calls a company to confirm details of a job offer that a candidate received.
You must write EXACTLY 10 questions that will be spoken aloud via text-to-speech.

RULES:
1. Tone: Polite, calm, friendly, neutral. NEVER accuse. NEVER use the words "fake", "scam", "fraud", or "suspicious".
2. NEVER reveal the purpose is checking for fraud. Say only that you are confirming details of an offer.
3. Simple everyday English. No jargon.
4. One sentence per question, under 20 words (hard max 25). Question 1 may have two short sentences (hard max 30 words).
5. Ask ONE thing per question. Never join two questions with "and also".
6. Prefer open questions (what, who, when, how). Use yes/no only for simple confirmation.
7. No bullets, emojis, symbols, URLs, or abbreviations.
8. NEVER use placeholder brackets like [Candidate's Name]. Always say "the candidate" or "a candidate".
9. Do NOT lead the person toward an answer. Ask open questions to discover information, do not reveal specific details and ask them to confirm.

CONTENT ORDER:
- Question 1: Disclose you are an automated AI assistant calling on behalf of a candidate, then confirm you reached the right company.
- Questions 2-3: Company identity (website, office location, registration).
- Questions 4-6: Role details (job title, responsibilities, start date, salary).
- Questions 7-8: Selection process (interview, documents, fees, payments).
- Question 9: Communication and documents (official email, offer letter).
- Question 10: Polite closing — ask for the official contact channel or next steps and thank them.

EVERY suspicion reason ID must appear in at least one question's targets_reason_ids.
No duplicate or near-duplicate questions.

OUTPUT FORMAT — respond with ONLY a JSON array, no other text:
[
  {
    "id": 1,
    "question": "...",
    "purpose": "... (max 12 words)",
    "targets_reason_ids": ["R1", ...],
    "legit_answer_hint": "... (max 15 words)"
  },
  ...
]
"""


# ──────────────────────────────────────────────
# Tool functions (spec-compliant wrappers)
# ──────────────────────────────────────────────

def read_file(path: str) -> str:
    """Read and return file contents."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(path: str, text: str):
    """Overwrite a file with the given text."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def get_time() -> str:
    """Return current time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


# ──────────────────────────────────────────────
# Question Validator
# ──────────────────────────────────────────────

class QuestionValidator:
    """Validates generated questions against all spec rules."""

    def validate(self, questions: list[dict], reason_ids: list[str]) -> list[str]:
        """Return a list of error strings. Empty list = valid."""
        errors = []

        # Check count
        if len(questions) != 10:
            errors.append(f"Expected exactly 10 questions, got {len(questions)}.")

        # Track which reason IDs are covered
        covered_ids = set()

        for i, q in enumerate(questions, 1):
            qtext = q.get("question", "")
            word_count = len(qtext.split())

            # Word limit
            if i == 1:
                if word_count > 30:
                    errors.append(f"Q{i}: {word_count} words exceeds 30-word max for Q1.")
            else:
                if word_count > 25:
                    errors.append(f"Q{i}: {word_count} words exceeds 25-word max.")

            # Forbidden words
            lower_q = qtext.lower()
            for fw in FORBIDDEN_WORDS:
                if fw in lower_q:
                    errors.append(f"Q{i}: contains forbidden word '{fw}'.")

            # Purpose word limit (max 12 words)
            purpose = q.get("purpose", "")
            if len(purpose.split()) > 12:
                errors.append(f"Q{i}: purpose exceeds 12-word limit.")

            # Legit answer hint word limit (max 15 words)
            hint = q.get("legit_answer_hint", "")
            if len(hint.split()) > 15:
                errors.append(f"Q{i}: legit_answer_hint exceeds 15-word limit.")

            # Collect covered reason IDs
            for rid in q.get("targets_reason_ids", []):
                covered_ids.add(rid)

            # Check for bracket placeholders like [Candidate's Name]
            if re.search(r'\[.*?\]', qtext):
                errors.append(f"Q{i}: contains bracket placeholder. Use 'the candidate' instead.")

        # Q10 should be a closing question
        if len(questions) >= 10:
            q10_lower = questions[9].get("question", "").lower()
            closing_keywords = ["thank", "contact", "next step", "reach", "follow up", "channel"]
            if not any(kw in q10_lower for kw in closing_keywords):
                errors.append("Q10: should be a polite closing question (contact/next steps/thank).")

        # All reason IDs must be covered
        missing_ids = set(reason_ids) - covered_ids
        if missing_ids:
            errors.append(f"Missing coverage for reason IDs: {', '.join(sorted(missing_ids))}.")

        return errors


# ──────────────────────────────────────────────
# Question Generator (Hybrid: LLM + Validation)
# ──────────────────────────────────────────────

class QuestionGenerator:
    """
    Generates 10 questions using the hybrid method:
    1. Builds a detailed prompt with input data + all rules
    2. Sends to Ollama LLM for creative generation
    3. Validates output against all rules
    4. On failure, sends errors back to LLM for correction (max 3 retries)
    """

    def __init__(self, llm: OllamaChat):
        self.llm = llm
        self.validator = QuestionValidator()

    def generate(self, input_data: dict) -> list[dict]:
        """Generate and validate 10 questions. Returns the validated list."""
        company = input_data["company_name"]
        job_title = input_data["job_title"]
        job_type = input_data.get("job_type", "job")
        offer_details = input_data.get("offer_details", {})
        suspicion_reasons = input_data.get("suspicion_reasons", [])
        reason_ids = [r["id"] for r in suspicion_reasons]

        # Build the context for the LLM
        context = self._build_context(company, job_title, job_type, offer_details, suspicion_reasons)

        max_retries = 3
        last_errors = []

        for attempt in range(1, max_retries + 1):
            print(f"\n  [GENERATOR] Attempt {attempt}/{max_retries}...")

            if attempt == 1:
                prompt = context
            else:
                # Feed back the errors for correction
                error_text = "\n".join(f"- {e}" for e in last_errors)
                prompt = (
                    f"The previous questions had these validation errors:\n{error_text}\n\n"
                    f"Please fix ALL errors and regenerate the complete set of 10 questions.\n"
                    f"Remember the original context:\n{context}"
                )

            # Get LLM response
            raw_response = self.llm.generate(prompt)
            print(f"  [GENERATOR] Got LLM response ({len(raw_response)} chars).")

            # Parse JSON from response
            questions = self._parse_json(raw_response)
            if questions is None:
                last_errors = ["Failed to parse JSON from LLM response. Ensure output is ONLY a valid JSON array."]
                print(f"  [GENERATOR] JSON parse failed.")
                continue

            # Validate
            errors = self.validator.validate(questions, reason_ids)
            if not errors:
                print(f"  [GENERATOR] ✓ All validations passed on attempt {attempt}.")
                return questions

            print(f"  [GENERATOR] ✗ {len(errors)} validation error(s):")
            for e in errors:
                print(f"    - {e}")
            last_errors = errors

        # If all retries failed, use template fallback
        print(f"\n  [GENERATOR] LLM retries exhausted. Falling back to template questions.")
        return self._template_fallback(input_data, reason_ids)

    def _build_context(self, company, job_title, job_type, offer_details, suspicion_reasons) -> str:
        """Build the full prompt context for the LLM."""
        details_str = "\n".join(f"  - {k}: {v}" for k, v in offer_details.items())
        reasons_str = "\n".join(f"  - {r['id']}: {r['reason']}" for r in suspicion_reasons)

        return (
            f"Company: {company}\n"
            f"Job Title: {job_title}\n"
            f"Job Type: {job_type}\n"
            f"Offer Details:\n{details_str}\n\n"
            f"Suspicion Reason IDs and reasons (EACH must be targeted by at least one question):\n"
            f"{reasons_str}\n\n"
            f"Generate exactly 10 questions following all the rules above."
        )

    def _parse_json(self, raw: str) -> list[dict] | None:
        """Extract a JSON array from the LLM response."""
        # Try to find JSON array in the response
        # First try: the whole response is JSON
        try:
            result = json.loads(raw.strip())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

        # Second try: extract JSON array from markdown code block or mixed text
        match = re.search(r'\[[\s\S]*\]', raw)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

        return None

    def _template_fallback(self, input_data: dict, reason_ids: list[str]) -> list[dict]:
        """
        Deterministic template-based fallback if the LLM fails all retries.
        Fills in company/job specifics from input data.
        """
        company = input_data["company_name"]
        job_title = input_data["job_title"]
        offer_details = input_data.get("offer_details", {})
        location = offer_details.get("location", "the listed location")
        salary = offer_details.get("salary", "the offered compensation")
        contact_email = offer_details.get("contact_email", "an email address")

        templates = [
            {
                "id": 1,
                "question": f"Hello, this is an automated AI assistant calling on behalf of a candidate. Am I speaking with someone at {company}?",
                "purpose": "Disclose AI identity and confirm company",
                "targets_reason_ids": [rid for rid in reason_ids if rid == "R1"] or [reason_ids[0]],
                "legit_answer_hint": "Yes, confirms the company name and greets back",
            },
            {
                "id": 2,
                "question": f"Could you share your official company website address?",
                "purpose": "Verify company has a real web presence",
                "targets_reason_ids": ["R1"] if "R1" in reason_ids else [reason_ids[0]],
                "legit_answer_hint": "Provides a working company website URL",
            },
            {
                "id": 3,
                "question": f"Where is your main office or headquarters located?",
                "purpose": "Confirm physical presence and legitimacy",
                "targets_reason_ids": ["R1"] if "R1" in reason_ids else [reason_ids[0]],
                "legit_answer_hint": "Gives a specific verifiable office address",
            },
            {
                "id": 4,
                "question": f"The candidate received an offer for {job_title}. Could you confirm this role exists?",
                "purpose": "Verify the role is real and currently open",
                "targets_reason_ids": ["R2"] if "R2" in reason_ids else [reason_ids[min(1, len(reason_ids)-1)]],
                "legit_answer_hint": "Confirms the role and describes it briefly",
            },
            {
                "id": 5,
                "question": f"What does the interview or selection process look like for this role?",
                "purpose": "Check if a proper hiring process exists",
                "targets_reason_ids": ["R2"] if "R2" in reason_ids else [reason_ids[min(1, len(reason_ids)-1)]],
                "legit_answer_hint": "Describes interview rounds and evaluation steps",
            },
            {
                "id": 6,
                "question": f"Could you confirm the compensation details for this position?",
                "purpose": "Verify offered pay matches company records",
                "targets_reason_ids": ["R3"] if "R3" in reason_ids else [reason_ids[min(2, len(reason_ids)-1)]],
                "legit_answer_hint": "Confirms salary range consistent with the offer",
            },
            {
                "id": 7,
                "question": f"Are there any fees or payments required from the candidate before starting?",
                "purpose": "Check for upfront payment demands",
                "targets_reason_ids": ["R3"] if "R3" in reason_ids else [reason_ids[min(2, len(reason_ids)-1)]],
                "legit_answer_hint": "Says no fees are required from candidates",
            },
            {
                "id": 8,
                "question": f"What documents will the candidate need to provide for onboarding?",
                "purpose": "Identify if unusual documents are requested",
                "targets_reason_ids": ["R3", "R4"] if "R3" in reason_ids and "R4" in reason_ids else [reason_ids[-1]],
                "legit_answer_hint": "Lists standard documents like ID and tax forms",
            },
            {
                "id": 9,
                "question": f"What is the official email address for communicating about this offer?",
                "purpose": "Verify official communication channel exists",
                "targets_reason_ids": ["R4"] if "R4" in reason_ids else [reason_ids[-1]],
                "legit_answer_hint": "Provides a corporate domain email address",
            },
            {
                "id": 10,
                "question": f"What would be the best way for the candidate to follow up with your team? Thank you for your time.",
                "purpose": "Get official contact and close politely",
                "targets_reason_ids": ["R4"] if "R4" in reason_ids else [reason_ids[-1]],
                "legit_answer_hint": "Provides official phone or email for follow-up",
            },
        ]

        return templates


# ──────────────────────────────────────────────
# Call Session — Phase 3: Ask, Listen, Record
# ──────────────────────────────────────────────

class CallSession:
    """
    Manages the actual call: iterates through questions, speaks them,
    listens for answers, and records everything to the response file.
    """

    def __init__(self, speaker: EdgeTTSSpeaker, listener: WhisperListener, response_path: str):
        self.speaker = speaker
        self.listener = listener
        self.response_path = response_path
        self._call_ended = False

    def run(self, response_data: dict):
        """Execute Phase 3: ask all 10 questions, one at a time."""
        responses = response_data["responses"]
        total = len(responses)
        prev_question = None  # Track previous question for cross-bleed cleanup

        print(f"\n{'=' * 60}")
        print(f"  Starting call — {total} questions to ask")
        print(f"{'=' * 60}\n")

        for i, entry in enumerate(responses):
            if self._call_ended:
                # Mark remaining questions as no_answer
                entry["status"] = "no_answer"
                entry["answer"] = None
                self._save(response_data)
                continue

            qnum = entry["id"]
            question = entry["question"]

            print(f"\n  ── Question {qnum}/{total} ──")
            print(f"  Q: {question}")

            # 1. Record asked_at
            entry["asked_at"] = get_time()

            # 2. Speak the question
            print(f"  [SPEAKING]...", end="", flush=True)
            self.speaker.speak(question)
            print(f"\r\033[K  [SPEAKING] Done.")

            # 2b. Flush mic-captured TTS segments before listening
            self.listener.flush()

            # 3. Listen for reply
            print(f"  [LISTENING]...")
            reply = self.listener.listen(silence_seconds=2, timeout_seconds=10)

            # 4. If empty, retry once
            if not reply:
                retry_text = "Sorry, I did not catch that. " + question
                print(f"  [RETRY] No reply, asking again...")
                self.speaker.speak(retry_text)
                self.listener.flush()
                reply = self.listener.listen(silence_seconds=2, timeout_seconds=10)

            # 5. Handle the reply
            if reply:
                # Check for special situations
                reply_lower = reply.lower()

                # They asked if we're a bot
                if self._is_bot_question(reply_lower):
                    self.speaker.speak(
                        "Yes, I am an automated AI assistant calling on behalf of a candidate."
                    )
                    # Re-ask the current question (this exchange is not the answer)
                    self.speaker.speak(question)
                    self.listener.flush()
                    reply = self.listener.listen(silence_seconds=2, timeout_seconds=10)

                # They asked to wait
                if reply and self._is_wait_request(reply.lower()):
                    print(f"  [WAITING] They asked to wait...")
                    reply = self.listener.listen(silence_seconds=2, timeout_seconds=20)

                # They asked for sensitive data
                if reply and self._is_sensitive_request(reply.lower()):
                    self.speaker.speak(
                        "I am sorry, I cannot share personal details. "
                        + question
                    )
                    self.listener.flush()
                    reply = self.listener.listen(silence_seconds=2, timeout_seconds=10)

                # They refused or ended the call
                if reply and self._is_refusal(reply.lower()):
                    print(f"  [CALL ENDED] They refused to continue.")
                    entry["answer"] = self._clean_answer(reply, question, prev_question)
                    entry["status"] = "answered"
                    entry["answered_at"] = get_time()
                    self._call_ended = True
                    self._save(response_data)
                    prev_question = question
                    continue

            # Record the result
            if reply:
                # Clean question bleed from transcript before saving
                cleaned_reply = self._clean_answer(reply, question, prev_question)
                entry["answered_at"] = get_time()
                entry["answer"] = cleaned_reply
                entry["status"] = "answered"
                print(f"  A: {cleaned_reply}")
            else:
                entry["status"] = "no_answer"
                entry["answer"] = None
                entry["answered_at"] = None
                print(f"  A: (no answer)")

            # 6. Save after every question
            self._save(response_data)
            prev_question = question

        return response_data

    def _save(self, data: dict):
        """Write the response file immediately."""
        write_file(self.response_path, json.dumps(data, indent=2, ensure_ascii=False))

    def _is_bot_question(self, text: str) -> bool:
        """Check if they're asking whether we're a bot/AI."""
        bot_phrases = ["are you a bot", "are you a robot", "are you ai", "are you an ai",
                       "is this a bot", "is this ai", "are you human", "are you a real person"]
        return any(p in text for p in bot_phrases)

    def _is_wait_request(self, text: str) -> bool:
        """Check if they asked us to hold/wait."""
        wait_phrases = ["hold on", "one moment", "please wait", "give me a moment",
                        "just a second", "hold please", "one second"]
        return any(p in text for p in wait_phrases)

    def _is_sensitive_request(self, text: str) -> bool:
        """Check if they're asking for sensitive data."""
        sensitive_phrases = ["bank account", "bank details", "credit card", "social security",
                             "password", "otp", "id number", "aadhaar", "passport number",
                             "pin number"]
        return any(p in text for p in sensitive_phrases)

    def _is_refusal(self, text: str) -> bool:
        """Check if they refused to continue or ended the call."""
        refusal_phrases = ["goodbye", "i'm hanging up", "don't call again", "stop calling",
                           "i refuse", "no comment", "end this call", "not interested"]
        return any(p in text for p in refusal_phrases)

    @staticmethod
    def _clean_answer(transcript: str, question: str, prev_question: str | None = None) -> str:
        """Remove question text from the transcript if Whisper captured it.

        When the mic picks up TTS playback, the question (and sometimes the
        previous question) gets transcribed along with the actual answer.
        This method strips those artifacts using exact and fuzzy matching.

        Args:
            transcript: The raw Whisper transcript (may contain question text).
            question: The current question that was just spoken.
            prev_question: The previous question (may bleed into the answer).

        Returns:
            The cleaned transcript with question text removed.
        """
        if not transcript:
            return transcript

        result = transcript

        # Clean against previous question first (it appears earlier in transcript)
        if prev_question:
            result = CallSession._strip_question(result, prev_question)

        # Clean against current question
        result = CallSession._strip_question(result, question)

        # If cleaning left us with nothing, return original (better than empty)
        return result.strip() if result.strip() else transcript

    @staticmethod
    def _strip_question(transcript: str, question: str) -> str:
        """Remove a single question's text from a transcript.

        Handles three scenarios:
        1. Exact substring match (question appears verbatim)
        2. Fuzzy match at start/end of transcript (Whisper garbles slightly)
        3. Sliding-window fuzzy match (question appears mid-transcript)
        """
        if not transcript or not question:
            return transcript

        # Normalize for comparison
        t_lower = transcript.lower().strip()
        q_lower = question.lower().strip()
        q_clean = q_lower.rstrip('?.,!').strip()

        # --- Strategy 1: Exact substring match (case-insensitive) ---
        idx = t_lower.find(q_clean)
        if idx != -1:
            cleaned = transcript[:idx] + transcript[idx + len(q_clean):]
            # Clean up lone punctuation marks left over from removal
            # (e.g., a trailing "?" where the question was stripped)
            # Only target punctuation surrounded by whitespace or at edges
            cleaned = re.sub(r'(?:^|\s+)[?!]+(?:\s+|$)', ' ', cleaned)
            cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
            if cleaned:
                return cleaned
            return transcript  # Don't return empty

        # --- Strategy 2 & 3: Word-level fuzzy matching ---
        q_words = [w.rstrip('?.,!').lower() for w in question.split()]
        t_words = transcript.split()

        if len(q_words) < 3 or len(t_words) <= len(q_words):
            return transcript

        # Strategy 2a: Check question words at the BEGINNING of transcript
        overlap_start = 0
        for qw, tw in zip(q_words, t_words):
            if qw == tw.lower().rstrip('?.,!'):
                overlap_start += 1
            else:
                break
        if overlap_start / len(q_words) > 0.6:
            cut_at = overlap_start
            # Extend past trailing words that are also in the question
            q_word_set = {w.rstrip('?.,!').lower() for w in question.split()}
            while cut_at < len(t_words) and t_words[cut_at].lower().rstrip('?.,!') in q_word_set:
                cut_at += 1
            cleaned = ' '.join(t_words[cut_at:]).strip(' .,?!')
            if cleaned:
                return cleaned

        # Strategy 2b: Check question words at the END of transcript
        overlap_end = 0
        for qw, tw in zip(reversed(q_words), reversed(t_words)):
            if qw == tw.lower().rstrip('?.,!'):
                overlap_end += 1
            else:
                break
        if overlap_end / len(q_words) > 0.6:
            cleaned = ' '.join(t_words[:len(t_words) - overlap_end]).strip(' .,?!')
            if cleaned:
                return cleaned

        # Strategy 3: Sliding window — find a span of words in the middle
        # that matches >60% of question words (Whisper may garble the question)
        window_size = len(q_words)
        best_overlap = 0
        best_start = -1
        best_end = -1

        for start_idx in range(len(t_words) - window_size + 1):
            window = t_words[start_idx:start_idx + window_size]
            overlap = sum(
                1 for qw, tw in zip(q_words, window)
                if qw == tw.lower().rstrip('?.,!')
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_start = start_idx
                best_end = start_idx + window_size

        if best_overlap / len(q_words) > 0.6:
            before = ' '.join(t_words[:best_start]).strip(' .,?!')
            after = ' '.join(t_words[best_end:]).strip(' .,?!')
            cleaned = f"{before} {after}".strip()
            cleaned = re.sub(r'\s{2,}', ' ', cleaned)
            if cleaned:
                return cleaned

        return transcript


# ──────────────────────────────────────────────
# WebRTC Call Session — Step-by-step mode
# ──────────────────────────────────────────────

class WebRTCCallSession:
    """Non-blocking version of CallSession for HTTP bridge use.

    Instead of running a blocking loop over all 10 questions, this class
    lets an external caller advance one question at a time via HTTP.

    State machine:
        READY → WAITING_FOR_ANSWER → (process_answer) → WAITING_FOR_ANSWER → ... → COMPLETE

    Usage:
        session = WebRTCCallSession(response_data, response_path)
        q = session.get_current_question()   # returns Q1
        result = session.process_answer("Yes, this is Acme Tech.")
        # result = { answer_recorded, next_question, conversation_complete, ... }
    """

    def __init__(self, response_data: dict, response_path: str):
        self.response_data = response_data
        self.response_path = response_path
        self.current_index = 0
        self._call_ended = False
        self._prev_question = None

    @property
    def total_questions(self) -> int:
        return len(self.response_data.get("responses", []))

    @property
    def is_complete(self) -> bool:
        return self._call_ended or self.current_index >= self.total_questions

    def get_state(self) -> dict:
        """Return current session state for the /bridge/status endpoint."""
        return {
            "state": "complete" if self.is_complete else "active",
            "question_index": self.current_index,
            "total_questions": self.total_questions,
            "call_ended": self._call_ended,
        }

    def get_current_question(self) -> dict | None:
        """Return the current question entry, or None if done."""
        if self.is_complete:
            return None
        return self.response_data["responses"][self.current_index]

    def get_greeting_text(self) -> str:
        """Return the greeting: a standalone introduction statement + Q1.

        The introduction is always a statement (not a question) that discloses
        the AI identity. The actual Q1 from the response data is appended as
        the first verification question.
        """
        company = self.response_data.get("meta", {}).get("company", "your company")
        intro = (
            f"Hello, this is an automated AI assistant calling on behalf of a "
            f"candidate who received a job offer from {company}."
        )
        if self.total_questions > 0:
            q = self.response_data["responses"][0]
            return f"{intro} {q['question']}"
        return intro

    def process_answer(self, transcript: str) -> dict:
        """Process one answer from the phone user, advance to next question.

        This is the core method the bridge calls for each audio upload.
        It replicates the inner loop logic from CallSession.run().

        Args:
            transcript: The STT transcript of the phone user's speech.

        Returns:
            {
                "answer_recorded": str | None,
                "answer_status": "answered" | "no_answer",
                "next_question": str | None,
                "conversation_complete": bool,
                "question_index": int,
                "total_questions": int,
            }
        """
        if self.is_complete:
            return {
                "answer_recorded": None,
                "answer_status": "complete",
                "next_question": None,
                "conversation_complete": True,
                "question_index": self.current_index,
                "total_questions": self.total_questions,
            }

        entry = self.response_data["responses"][self.current_index]
        question = entry["question"]

        print(f"\n  ── Processing answer for Q{entry['id']}/{self.total_questions} ──")

        # Record asked_at if not already set
        if not entry.get("asked_at"):
            entry["asked_at"] = get_time()

        reply = transcript.strip() if transcript else ""

        if reply:
            reply_lower = reply.lower()

            # Check for special situations (reusing CallSession's static methods)

            # They asked if we're a bot — note this for the bridge
            # (In the WebRTC flow, the bridge will speak the bot-disclosure
            # and re-ask the question via a separate response)
            if self._is_bot_question(reply_lower):
                print(f"  [SESSION] Bot question detected. Will re-ask after disclosure.")
                # Don't advance — return special response
                return {
                    "answer_recorded": None,
                    "answer_status": "bot_question",
                    "next_question": question,  # re-ask same question
                    "bot_disclosure": "Yes, I am an automated AI assistant calling on behalf of a candidate.",
                    "conversation_complete": False,
                    "question_index": self.current_index,
                    "total_questions": self.total_questions,
                }

            # They asked to wait — tell the bridge to listen again with longer timeout
            if self._is_wait_request(reply_lower):
                print(f"  [SESSION] Wait request detected.")
                return {
                    "answer_recorded": None,
                    "answer_status": "wait_request",
                    "next_question": None,  # don't re-ask, just listen again
                    "conversation_complete": False,
                    "question_index": self.current_index,
                    "total_questions": self.total_questions,
                }

            # They asked for sensitive data
            if self._is_sensitive_request(reply_lower):
                print(f"  [SESSION] Sensitive data request detected.")
                return {
                    "answer_recorded": None,
                    "answer_status": "sensitive_request",
                    "next_question": question,  # re-ask
                    "sensitive_refusal": "I am sorry, I cannot share personal details.",
                    "conversation_complete": False,
                    "question_index": self.current_index,
                    "total_questions": self.total_questions,
                }

            # They refused or ended the call
            if self._is_refusal(reply_lower):
                print(f"  [SESSION] Call refusal detected.")
                cleaned = CallSession._clean_answer(reply, question, self._prev_question)
                entry["answer"] = cleaned
                entry["status"] = "answered"
                entry["answered_at"] = get_time()
                self._call_ended = True
                self._save()

                # Mark remaining questions as no_answer
                for remaining in self.response_data["responses"][self.current_index + 1:]:
                    remaining["status"] = "no_answer"
                    remaining["answer"] = None
                self._save()

                return {
                    "answer_recorded": cleaned,
                    "answer_status": "refusal",
                    "next_question": None,
                    "conversation_complete": True,
                    "question_index": self.current_index,
                    "total_questions": self.total_questions,
                }

            # Normal answer — clean and record
            cleaned = CallSession._clean_answer(reply, question, self._prev_question)
            entry["answer"] = cleaned
            entry["status"] = "answered"
            entry["answered_at"] = get_time()
            print(f"  A: {cleaned}")

        else:
            # No speech detected
            entry["status"] = "no_answer"
            entry["answer"] = None
            entry["answered_at"] = None
            print(f"  A: (no answer)")

        # Save after every question
        self._save()
        self._prev_question = question
        self.current_index += 1

        # Get next question (if any)
        next_q = self.get_current_question()
        next_text = next_q["question"] if next_q else None
        is_complete = self.is_complete

        # If we just finished the last question, mark session complete
        if is_complete:
            self.response_data["meta"]["completed_at"] = get_time()
            self._save()

        return {
            "answer_recorded": entry.get("answer"),
            "answer_status": entry.get("status", "no_answer"),
            "next_question": next_text,
            "conversation_complete": is_complete,
            "question_index": self.current_index,
            "total_questions": self.total_questions,
        }

    def _save(self):
        """Write the response file immediately."""
        write_file(self.response_path, json.dumps(self.response_data, indent=2, ensure_ascii=False))

    # ── Reuse detection methods from CallSession ──

    @staticmethod
    def _is_bot_question(text: str) -> bool:
        bot_phrases = ["are you a bot", "are you a robot", "are you ai", "are you an ai",
                       "is this a bot", "is this ai", "are you human", "are you a real person"]
        return any(p in text for p in bot_phrases)

    @staticmethod
    def _is_wait_request(text: str) -> bool:
        wait_phrases = ["hold on", "one moment", "please wait", "give me a moment",
                        "just a second", "hold please", "one second"]
        return any(p in text for p in wait_phrases)

    @staticmethod
    def _is_sensitive_request(text: str) -> bool:
        sensitive_phrases = ["bank account", "bank details", "credit card", "social security",
                             "password", "otp", "id number", "aadhaar", "passport number",
                             "pin number"]
        return any(p in text for p in sensitive_phrases)

    @staticmethod
    def _is_refusal(text: str) -> bool:
        refusal_phrases = ["goodbye", "i'm hanging up", "don't call again", "stop calling",
                           "i refuse", "no comment", "end this call", "not interested"]
        return any(p in text for p in refusal_phrases)


# ──────────────────────────────────────────────
# Main Orchestrator — Phases 1-4
# ──────────────────────────────────────────────

def build_response_filename(company_name: str, job_title: str) -> str:
    """Build response filename: company_role.json (lowercase, underscores)."""
    combined = f"{company_name}_{job_title}"
    # Lowercase, replace non-alphanumeric with underscore, collapse multiples
    sanitized = re.sub(r'[^a-z0-9]+', '_', combined.lower()).strip('_')
    return f"{sanitized}.json"


def main():
    """Entry point: run all 4 phases."""

    # ── Parse CLI args ──
    if len(sys.argv) < 2:
        print("Usage: python caller.py <input_json_path> [--dry-run]")
        print("       python caller.py input/acme_tech_solutions_data_analyst_intern.json")
        sys.exit(1)

    input_path = sys.argv[1]
    dry_run = "--dry-run" in sys.argv

    # ══════════════════════════════════════════
    # PHASE 1: LOAD
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  PHASE 1: LOAD")
    print("=" * 60)

    raw = read_file(input_path)
    input_data = json.loads(raw)

    company = input_data["company_name"]
    job_title = input_data["job_title"]

    response_filename = build_response_filename(company, job_title)
    response_path = os.path.join(RESPONSES_DIR, response_filename)

    print(f"  Company  : {company}")
    print(f"  Job Title: {job_title}")
    print(f"  Output   : {response_path}")

    # ══════════════════════════════════════════
    # PHASE 2: GENERATE QUESTIONS (Hybrid)
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  PHASE 2: GENERATE QUESTIONS")
    print("=" * 60)

    llm = OllamaChat(
        model=OLLAMA_MODEL,
        system_prompt=QUESTION_GEN_PROMPT,
    )

    generator = QuestionGenerator(llm)
    questions = generator.generate(input_data)

    # Build the response structure
    response_data = {
        "meta": {
            "company": company,
            "job": job_title,
            "caller_type": "automated AI assistant",
            "created_at": get_time(),
        },
        "responses": [
            {
                "id": q["id"],
                "question": q["question"],
                "purpose": q["purpose"],
                "targets_reason_ids": q["targets_reason_ids"],
                "legit_answer_hint": q["legit_answer_hint"],
                "answer": None,
                "status": "pending",
                "asked_at": None,
                "answered_at": None,
            }
            for q in questions
        ],
    }

    # Save initial response file (answers empty)
    write_file(response_path, json.dumps(response_data, indent=2, ensure_ascii=False))
    print(f"\n  ✓ Saved {len(questions)} questions to {response_path}")

    # Print the questions for review
    print(f"\n  Generated questions:")
    for q in questions:
        print(f"    Q{q['id']}: {q['question']}")
        print(f"         Purpose: {q['purpose']}")
        print(f"         Targets: {q['targets_reason_ids']}")
        print()

    if dry_run:
        print("\n  [DRY RUN] Skipping Phases 3 and 4. Questions saved.")
        return

    # ══════════════════════════════════════════
    # PHASE 3: ASK, LISTEN, RECORD
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  PHASE 3: CALL")
    print("=" * 60)

    speaker = EdgeTTSSpeaker(voice=TTS_VOICE)
    listener = WhisperListener()
    listener.connect()

    try:
        session = CallSession(speaker, listener, response_path)
        response_data = session.run(response_data)
    except KeyboardInterrupt:
        print("\n\n  [INTERRUPTED] Call stopped by user.")
    finally:
        listener.disconnect()

    # ══════════════════════════════════════════
    # PHASE 4: FINISH
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  PHASE 4: FINISH")
    print("=" * 60)

    # Speak a polite thank-you
    try:
        speaker.speak("Thank you very much for your time. Have a great day.")
    except Exception:
        pass  # person may have already hung up

    # Add completion timestamp
    response_data["meta"]["completed_at"] = get_time()
    write_file(response_path, json.dumps(response_data, indent=2, ensure_ascii=False))

    # Count answered questions
    answered_count = sum(1 for r in response_data["responses"] if r["status"] == "answered")
    total = len(response_data["responses"])

    print(f"\n  ✓ {response_filename} — {answered_count}/{total} questions answered.")
    speaker.shutdown()


if __name__ == "__main__":
    main()
