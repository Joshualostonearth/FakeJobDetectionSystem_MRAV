import os
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()

OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "ministral-3")

class JobOfferEntities(BaseModel):
    company_name: str = Field(description="The name of the company offering the job", default="")
    job_role: str = Field(description="The job title or role", default="")
    recruiter_name: Optional[str] = Field(description="Full name of the recruiter or person who signed the offer", default=None)
    recruiter_email: Optional[str] = Field(description="Contact email address of the recruiter", default=None)
    domain_or_website: Optional[str] = Field(description="The company's web domain or website URL", default=None)
    location: Optional[str] = Field(description="Job location", default=None)

    @property
    def email_or_phone(self) -> Optional[str]:
        return self.recruiter_email


def extract_entities_from_text(text: str) -> JobOfferEntities:
    if not text or not text.strip():
        print("[Warning] Empty text provided for extraction.")
        return JobOfferEntities()

    try:
        llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
        structured_llm = llm.with_structured_output(JobOfferEntities)
        
        # 🟢 UPGRADED STRICT PROMPT
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a strict entity extraction assistant.

Rules:
1. 'company_name': ALWAYS check the top header or letterhead (e.g., 'Richmond College'). Do not return empty if an organization header exists at the top.
2. 'job_role': Check the body or sign-off line (e.g., 'Administrative Officer').
3. 'recruiter_name': Extract the full name of the recruiter or the person who signed the letter. Leave empty if no person is named.
4. 'recruiter_email': Extract the contact email address.
5. 'domain_or_website': Extract domain from email or body text.
6. 'location': Extract city, state, zip code.
"""),
            ("human", "Extract job offer entities from this text:\n\n{text}")
        ])
        
        extraction_chain = prompt | structured_llm
        result = extraction_chain.invoke({"text": text})
        return result if result else JobOfferEntities()
    except Exception as e:
        print(f"[Error] LLM Extraction failed: {e}")
        return JobOfferEntities()