import json
import re
from typing import List, Dict, Any
from schemas.voice_schemas import VoiceVerificationSignals, ConversationMessage, VerificationState
from key_pool import key_pool

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_ollama import ChatOllama
    HAS_LANGCHAIN_OLLAMA = True
except ImportError:
    HAS_LANGCHAIN_OLLAMA = False


def extract_voice_signals(
    state: VerificationState,
    conversation_history: List[ConversationMessage]
) -> VoiceVerificationSignals:
    """
    Analyzes the full voice conversation transcript post-call and synthesizes
    structured verification signals for handoff to Objective 2.
    """
    transcript_text = "\n".join(
        [f"[{msg.speaker.upper()}]: {msg.text}" for msg in conversation_history]
    )

    if not transcript_text.strip():
        # Return state defaults if no transcript
        return VoiceVerificationSignals(
            company_reached=state.company_reached or False,
            role_actively_hiring=state.role_actively_hiring or False,
            recruiter_recognized=state.recruiter_recognized,
            company_denied_role=state.role_actively_hiring is False,
            suspicious_response=state.payment_requested is True,
            payment_requested=state.payment_requested,
            confidence=0.5
        )

    prompt_str = f"""Analyze this voice call transcript between an AI verification agent and a company contact.
Extract structured verification signals as JSON.

FULL TRANSCRIPT:
{transcript_text}

JSON schema requirement:
- "company_reached": bool (True if confirmed official company reached)
- "role_actively_hiring": bool (True if role is open and hiring)
- "recruiter_recognized": bool or null (True if recruiter is verified by company)
- "company_denied_role": bool (True if company explicitly denied the existence of this job role)
- "suspicious_response": bool (True if respondent requested money, used fake info, or gave evasive answers)
- "payment_requested": bool or null (True if any payment or fee was requested)
- "confidence": float between 0.0 and 1.0

Return valid JSON ONLY:"""

    raw_json = _call_llm(prompt_str)
    if raw_json:
        try:
            match = re.search(r"\{.*\}", raw_json, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return VoiceVerificationSignals(
                    company_reached=data.get("company_reached", state.company_reached or False),
                    role_actively_hiring=data.get("role_actively_hiring", state.role_actively_hiring or False),
                    recruiter_recognized=data.get("recruiter_recognized", state.recruiter_recognized),
                    company_denied_role=data.get("company_denied_role", state.role_actively_hiring is False),
                    suspicious_response=data.get("suspicious_response", state.payment_requested is True),
                    payment_requested=data.get("payment_requested", state.payment_requested),
                    confidence=float(data.get("confidence", 0.85))
                )
        except Exception as e:
            print(f"[Warning] Failed to parse FactExtractor JSON output ({e}).")

    # Fallback using state accumulation
    is_denied = state.role_actively_hiring is False
    is_suspicious = state.payment_requested is True or state.company_reached is False
    conf = 0.85 if state.company_reached else 0.50

    return VoiceVerificationSignals(
        company_reached=state.company_reached or False,
        role_actively_hiring=state.role_actively_hiring or False,
        recruiter_recognized=state.recruiter_recognized,
        company_denied_role=is_denied,
        suspicious_response=is_suspicious,
        payment_requested=state.payment_requested,
        confidence=conf
    )

def _call_llm(prompt_text: str) -> str:
    if HAS_LANGCHAIN_OLLAMA:
        try:
            llm = ChatOllama(model="qwen3-vl:latest", temperature=0.0)
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a verification signals extractor. Output valid JSON only."),
                ("human", "{prompt}")
            ])
            chain = prompt | llm
            res = chain.invoke({"prompt": prompt_text})
            return res.content.strip()
        except Exception:
            pass

    def api_call(client):
        try:
            response = client.chat.completions.create(
                model=key_pool.model,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.0,
                max_tokens=256,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return ""

    return key_pool.execute(api_call) if hasattr(key_pool, "execute") else ""
