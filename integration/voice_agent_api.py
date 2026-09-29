import uuid
from typing import Dict, Any, Optional
from schemas.voice_schemas import TranscriptTurn, AgentResponse, VoiceVerificationSignals, VerificationContext
from ai.state_manager import VerificationStateManager
from ai.conversation_agent import LLMConversationAgent
from ai.fact_extractor import extract_voice_signals
from engines.credibility_engine import Run_Objective_2_Filter

# Global in-memory session store
_SESSIONS: Dict[str, VerificationStateManager] = {}
_AGENT_ENGINE = LLMConversationAgent()


def init_verification_session(job_entities: Dict[str, Any]) -> Dict[str, Any]:
    """
    Person 1 or System initializes a voice verification call session
    using Objective 1 extracted entities.
    """
    session_id = str(uuid.uuid4())
    manager = VerificationStateManager.from_dict(job_entities)
    _SESSIONS[session_id] = manager
    
    # Generate initial greeting/question
    initial_obj = manager.get_active_objective()
    greeting_q = f"Hello, I'm calling to verify an open job position at {manager.context.company_name or 'your company'}. Is this the HR or recruitment office?"
    manager.add_message(speaker="agent", text=greeting_q)

    return {
        "session_id": session_id,
        "initial_question": greeting_q,
        "context": manager.context.model_dump()
    }


def process_transcript_turn(session_id: str, transcript_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Person 1 -> Person 2 API Endpoint.
    Input:
        transcript_data = {
            "type": "transcript",
            "text": "Yes, the software engineering internship is currently open.",
            "timestamp": 12.4
        }
    Output:
        AgentResponse dict (containing text, conversation_complete, next_objective, etc.)
    """
    manager = _SESSIONS.get(session_id)
    if not manager:
        raise ValueError(f"Session '{session_id}' not found.")

    turn = TranscriptTurn(
        type=transcript_data.get("type", "transcript"),
        text=transcript_data.get("text", ""),
        timestamp=float(transcript_data.get("timestamp", 0.0))
    )

    agent_response: AgentResponse = _AGENT_ENGINE.process_turn(manager, turn)
    return agent_response.model_dump()


def finalize_session(session_id: str) -> Dict[str, Any]:
    """
    Finalizes the voice call session and extracts structured verification signals.
    """
    manager = _SESSIONS.get(session_id)
    if not manager:
        raise ValueError(f"Session '{session_id}' not found.")

    signals: VoiceVerificationSignals = extract_voice_signals(
        state=manager.state,
        conversation_history=manager.state.conversation_history
    )
    
    return {
        "session_id": session_id,
        "verification_state": manager.state.model_dump(),
        "voice_signals": signals.model_dump()
    }


def hand_off_to_objective_2(
    job_entities: Dict[str, Any],
    retrieved_evidence: list,
    voice_signals: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Objective 2 Integration: Hands off extracted voice signals alongside
    Objective 1 entities and FAISS evidence into the Consensus Engine.
    """
    return Run_Objective_2_Filter(
        entities=job_entities,
        retrieved_evidence=retrieved_evidence,
        voice_signals=voice_signals
    )
