import json
import re
from typing import Dict, Any, Optional
from schemas.voice_schemas import AgentResponse, TranscriptTurn
from ai.state_manager import VerificationStateManager
from ai.question_bank import OBJECTIVES, get_objective_by_index, format_question_for_context
from key_pool import key_pool

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_ollama import ChatOllama
    HAS_LANGCHAIN_OLLAMA = True
except ImportError:
    HAS_LANGCHAIN_OLLAMA = False


class LLMConversationAgent:
    def __init__(self, model_name: str = "qwen3-vl:latest"):
        self.model_name = model_name

    def process_turn(
        self,
        manager: VerificationStateManager,
        transcript_turn: TranscriptTurn
    ) -> AgentResponse:
        # Record human caller's transcript
        manager.add_message(speaker="caller", text=transcript_turn.text, timestamp=transcript_turn.timestamp)

        active_obj = manager.get_active_objective()
        context_dict = manager.context.model_dump()
        state_dict = manager.state.model_dump()

        history_formatted = "\n".join(
            [f"{msg.speaker.upper()}: {msg.text}" for msg in manager.state.conversation_history[-6:]]
        )

        prompt_text = f"""You are an objective AI Verification Assistant calling a company HR to verify a job position.
CRITICAL RULE: DO NOT DECIDE OR JUDGE IF THE JOB IS FAKE OR FRAUDULENT. Your ONLY goal is to analyze the caller's response and extract objective facts.

CLAIM DETAILS:
Company: {context_dict.get('company_name')}
Role: {context_dict.get('job_role')}
Recruiter: {context_dict.get('recruiter_name')}

ACTIVE OBJECTIVE: {active_obj['index']} - {active_obj['title']}
Target fact to evaluate: {active_obj['target_fact']}

RECENT DIALOGUE:
{history_formatted}

LATEST CALLER TRANSCRIPT:
"{transcript_turn.text}"

TASK:
1. Determine if the latest transcript answers the target fact or provides info for any of these facts:
   - "company_reached" (boolean or null)
   - "role_actively_hiring" (boolean or null)
   - "recruiter_recognized" (boolean or null)
   - "application_channel_verified" (boolean or null)
   - "payment_requested" (boolean or null)
2. Decide if the current active objective is now answered. If yes, advance to next objective.

Return strictly a JSON object formatted as:
{{
  "updated_facts": {{
     "role_actively_hiring": true
  }},
  "objective_answered": true,
  "confidence_explanation": "Caller confirmed role is active."
}}
Output JSON ONLY:"""

        updated_facts = {}
        objective_answered = False

        raw_json_str = self._call_llm(prompt_text)
        if raw_json_str:
            try:
                json_match = re.search(r"\{.*\}", raw_json_str, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    updated_facts = parsed.get("updated_facts", {})
                    objective_answered = parsed.get("objective_answered", False)
            except Exception as e:
                print(f"[Warning] Failed to parse LLM response JSON ({e}). Falling back to heuristic parsing.")

        # Heuristic fallbacks if LLM outputs empty or fails
        if not updated_facts:
            lower_text = transcript_turn.text.lower()
            if active_obj["id"] == "company_awareness":
                if any(k in lower_text for k in ["yes", "speaking", "this is", "hr department", "correct"]):
                    updated_facts["company_reached"] = True
                    objective_answered = True
                elif any(k in lower_text for k in ["wrong number", "no", "not company"]):
                    updated_facts["company_reached"] = False
                    objective_answered = True

            elif active_obj["id"] == "role_active":
                if any(k in lower_text for k in ["hiring", "open", "accepting applications", "active"]):
                    updated_facts["role_actively_hiring"] = True
                    objective_answered = True
                elif any(k in lower_text for k in ["not hiring", "closed", "filled", "no such role"]):
                    updated_facts["role_actively_hiring"] = False
                    objective_answered = True

            elif active_obj["id"] == "recruiter_legitimacy":
                if any(k in lower_text for k in ["works here", "yes", "our recruiter", "legitimate"]):
                    updated_facts["recruiter_recognized"] = True
                    objective_answered = True
                elif any(k in lower_text for k in ["don't know", "never heard", "no one named", "imposter"]):
                    updated_facts["recruiter_recognized"] = False
                    objective_answered = True

            elif active_obj["id"] == "payment_request":
                if any(k in lower_text for k in ["fee", "pay", "deposit", "charge", "yes"]):
                    updated_facts["payment_requested"] = True
                    objective_answered = True
                elif any(k in lower_text for k in ["free", "never charge", "no payment", "no fee", "no"]):
                    updated_facts["payment_requested"] = False
                    objective_answered = True

            # If user spoke a clear sentence, mark objective as answered to move forward
            if len(transcript_turn.text.strip().split()) >= 3 and not objective_answered:
                objective_answered = True

        # Apply updated facts to state
        if updated_facts:
            manager.update_facts(updated_facts)

        if objective_answered:
            manager.advance_objective()

        conversation_complete = manager.is_complete()

        if conversation_complete:
            next_obj_id = "complete"
            next_q = "Thank you very much for your time and assistance. Have a great day!"
        else:
            current_obj = manager.get_active_objective()
            next_obj_id = current_obj["id"]
            next_q = format_question_for_context(
                current_obj["index"],
                {
                    "company_name": context_dict.get("company_name", ""),
                    "job_role": context_dict.get("job_role", ""),
                    "recruiter_name": context_dict.get("recruiter_name", "")
                }
            )

        # Record agent's turn in conversation history
        manager.add_message(speaker="agent", text=next_q)

        return AgentResponse(
            type="agent_response",
            text=next_q,
            updated_facts=updated_facts,
            next_objective=next_obj_id,
            next_question=next_q,
            conversation_complete=conversation_complete
        )

    def _call_llm(self, prompt_text: str) -> str:
        if HAS_LANGCHAIN_OLLAMA:
            try:
                llm = ChatOllama(model=self.model_name, temperature=0.0)
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are an AI data extraction agent. Output strictly valid JSON."),
                    ("human", "{prompt}")
                ])
                chain = prompt | llm
                res = chain.invoke({"prompt": prompt_text})
                return res.content.strip()
            except Exception as e:
                print(f"[Warning] ChatOllama failed ({e}), falling back to key_pool.")

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
