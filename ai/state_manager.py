from typing import Dict, Any, Optional
from schemas.voice_schemas import VerificationContext, VerificationState, ConversationMessage
from ai.question_bank import OBJECTIVES, get_objective_by_index

class VerificationStateManager:
    def __init__(self, context: VerificationContext):
        self.context = context
        self.state = VerificationState()

    @classmethod
    def from_dict(cls, context_dict: Dict[str, Any]) -> "VerificationStateManager":
        ctx = VerificationContext(
            company_name=context_dict.get("company_name") or context_dict.get("company", ""),
            job_role=context_dict.get("job_role") or context_dict.get("role", ""),
            location=context_dict.get("location"),
            recruiter_name=context_dict.get("recruiter_name") or context_dict.get("recruiter"),
            domain_or_website=context_dict.get("domain_or_website") or context_dict.get("domain")
        )
        return cls(ctx)

    def add_message(self, speaker: str, text: str, timestamp: float = 0.0):
        self.state.conversation_history.append(
            ConversationMessage(speaker=speaker, text=text, timestamp=timestamp)
        )

    def update_facts(self, updated_facts: Dict[str, Any]):
        for key, val in updated_facts.items():
            if hasattr(self.state, key) and val is not None:
                setattr(self.state, key, val)

    def get_active_objective(self) -> Dict[str, Any]:
        return get_objective_by_index(self.state.current_objective_index)

    def advance_objective(self):
        if self.state.current_objective_index < len(OBJECTIVES):
            self.state.current_objective_index += 1

    def is_complete(self) -> bool:
        return self.state.current_objective_index > len(OBJECTIVES)

    def get_missing_facts(self) -> Dict[str, bool]:
        state_dict = self.state.model_dump()
        return {
            "company_reached": state_dict.get("company_reached") is None,
            "role_actively_hiring": state_dict.get("role_actively_hiring") is None,
            "recruiter_recognized": state_dict.get("recruiter_recognized") is None if self.context.recruiter_name else False,
            "application_channel_verified": state_dict.get("application_channel_verified") is None,
            "payment_requested": state_dict.get("payment_requested") is None,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context": self.context.model_dump(),
            "state": self.state.model_dump()
        }
