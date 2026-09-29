from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class VerificationContext(BaseModel):
    company_name: str = Field(description="The claimed company name from Objective 1", default="")
    job_role: str = Field(description="The claimed job title or role", default="")
    location: Optional[str] = Field(description="Job location if specified", default=None)
    recruiter_name: Optional[str] = Field(description="Recruiter or contact person name", default=None)
    domain_or_website: Optional[str] = Field(description="Company website or domain", default=None)

class TranscriptTurn(BaseModel):
    type: str = Field(default="transcript", description="Message type")
    text: str = Field(description="Raw audio transcript text from caller or receiver")
    timestamp: float = Field(default=0.0, description="Timestamp in seconds")

class ConversationMessage(BaseModel):
    speaker: str = Field(description="'agent' or 'caller'")
    text: str = Field(description="Dialogue text")
    timestamp: float = Field(default=0.0)

class VerificationState(BaseModel):
    company_reached: Optional[bool] = Field(default=None, description="True if caller confirmed talking to official company")
    role_actively_hiring: Optional[bool] = Field(default=None, description="True if role is open/hiring")
    recruiter_recognized: Optional[bool] = Field(default=None, description="True if recruiter is verified by company")
    application_channel_verified: Optional[bool] = Field(default=None, description="True if official portal/email confirmed")
    payment_requested: Optional[bool] = Field(default=None, description="True if company/caller requests payment for job")
    current_objective_index: int = Field(default=1, description="Active objective being pursued (1-6)")
    conversation_history: List[ConversationMessage] = Field(default_factory=list)

class AgentResponse(BaseModel):
    type: str = Field(default="agent_response")
    text: str = Field(description="TTS text spoken by Person 2 voice agent")
    updated_facts: Dict[str, Any] = Field(default_factory=dict, description="Facts updated in this turn")
    next_objective: str = Field(default="", description="Identifier of current/next objective")
    next_question: str = Field(default="", description="Question being asked")
    conversation_complete: bool = Field(default=False, description="True when all objectives completed")

class VoiceVerificationSignals(BaseModel):
    company_reached: bool = Field(default=False)
    role_actively_hiring: bool = Field(default=False)
    recruiter_recognized: Optional[bool] = Field(default=None)
    company_denied_role: bool = Field(default=False)
    suspicious_response: bool = Field(default=False)
    payment_requested: Optional[bool] = Field(default=None)
    confidence: float = Field(default=0.0, description="Confidence score between 0.0 and 1.0")
