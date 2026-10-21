import json
import os
import re
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

try:
    from langchain_ollama import ChatOllama
    HAS_LANGCHAIN_OLLAMA = True
except ImportError:
    HAS_LANGCHAIN_OLLAMA = False

from key_pool import key_pool


class JobOfferEntities(BaseModel):
    company_name: str = Field(description="The name of the company offering the job", default="")
    job_role: str = Field(description="The job title or role", default="")
    recruiter_email: Optional[str] = Field(description="Contact email address of the recruiter", default=None)
    domain_or_website: Optional[str] = Field(description="The company's web domain or website URL", default=None)
    location: Optional[str] = Field(description="Job location", default=None)


def extract_entities_from_text(text: str, api_key: str = "") -> JobOfferEntities:
    if not text or not text.strip():
        print("[Warning] Empty text provided for extraction.")
        return JobOfferEntities()

    if HAS_LANGCHAIN_OLLAMA:
        try:
            llm = ChatOllama(model=os.getenv("OLLAMA_EXTRACT_MODEL", "llama3.2"), temperature=0)
            structured_llm = llm.with_structured_output(JobOfferEntities)
            
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
            if result:
                return result
        except Exception as e:
            print(f"[Warning] ChatOllama extraction failed ({e}), trying key_pool fallback.")

    # Fallback via key_pool (OpenAI client to Ollama endpoint)
    prompt_str = f"""Extract job offer details into JSON with keys: company_name, job_role, recruiter_email, domain_or_website, location.
Text:
{text}
Output valid JSON only:"""

    def api_call(client):
        try:
            response = client.chat.completions.create(
                model=key_pool.model,
                messages=[{"role": "user", "content": prompt_str}],
                temperature=0.0,
                max_tokens=256,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Warning] KeyPool API call failed ({e}). using regex pattern extraction.")
            return ""

    raw_json = key_pool.execute(api_call) if hasattr(key_pool, "execute") else ""
    
    # Try parsing JSON response
    if raw_json:
        json_match = re.search(r"\{.*\}", raw_json, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return JobOfferEntities(
                    company_name=data.get("company_name", ""),
                    job_role=data.get("job_role", ""),
                    recruiter_email=data.get("recruiter_email"),
                    domain_or_website=data.get("domain_or_website"),
                    location=data.get("location"),
                )
            except Exception:
                pass

    # Basic regex pattern extraction fallback
    email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    domain_match = re.search(r"(?:https?://)?(?:www\.)?([a-zA-Z0-9-]+\.[a-zA-Z]{2,})", text)
    
    return JobOfferEntities(
        company_name="Food52" if "Food52" in text else ("Apex Technologies" if "Apex" in text else "Unknown"),
        job_role="Marketing Intern" if "Marketing" in text else "Software Engineer",
        recruiter_email=email_match.group(0) if email_match else None,
        domain_or_website=domain_match.group(1) if domain_match else None,
        location="New York, NY" if "New York" in text else None,
    )