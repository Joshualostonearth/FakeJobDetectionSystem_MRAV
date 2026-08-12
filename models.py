from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

class JobOfferEntities(BaseModel):
    company_name: str = Field(description="The name of the company offering the job", default="")
    job_role: str = Field(description="The job title or role", default="")
    recruiter_email: Optional[str] = Field(description="Contact email address of the recruiter", default=None)
    domain_or_website: Optional[str] = Field(description="The company's web domain or website URL", default=None)
    location: Optional[str] = Field(description="Job location", default=None)

def extract_entities_from_text(text: str) -> JobOfferEntities:
    if not text or not text.strip():
        print("[Warning] Empty text provided for extraction.")
        return JobOfferEntities()

    try:
        llm = ChatOllama(model="llama3.2", temperature=0)
        structured_llm = llm.with_structured_output(JobOfferEntities)
        
        # 🟢 UPGRADED STRICT PROMPT FOR LLAMA 3.2
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a strict data extraction assistant. Parse the provided text and extract all job details into JSON format.

RULES:
1. 'company_name': Extract the exact organization or company name (e.g. 'Food52', 'Richmond College'). DO NOT leave this blank if a company name is mentioned.
2. 'job_role': Extract the job title (e.g. 'Marketing Intern').
3. 'recruiter_email': Extract any email address present in the text (e.g. 'recruiting-food52@gmail.com').
4. 'domain_or_website': Extract any website domain mentioned (e.g. 'food52.com').
5. 'location': Extract any city/state/country mentioned.

Extract every single field carefully from the text below."""),
            ("human", "{text}")
        ])
        
        extraction_chain = prompt | structured_llm
        result = extraction_chain.invoke({"text": text})
        return result if result else JobOfferEntities()
    except Exception as e:
        print(f"[Error] LLM Extraction failed: {e}")
        return JobOfferEntities()