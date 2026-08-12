from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Step 3: Structured Entity Extraction Schema
class JobOfferEntities(BaseModel):
    company_name: str = Field(description="The name of the company offering the job", default="")
    job_role: str = Field(description="The job title or role", default="")
    recruiter_email: Optional[str] = Field(description="Contact email address of the recruiter", default=None)
    domain_or_website: Optional[str] = Field(description="The company's web domain or website URL", default=None)
    location: Optional[str] = Field(description="Job location", default=None)

def extract_entities_from_text(text: str, openai_api_key: str) -> JobOfferEntities:
    """
    Uses LangChain and an LLM to parse raw text and extract structured job offer entities.
    """
    if not text or not text.strip():
        print("[Warning] Empty text provided for extraction.")
        return JobOfferEntities()

    try:
        # Initialize the LLM (Using GPT-4o-mini for fast, cheap extraction)
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key, temperature=0)
        
        # Create the structured extraction chain
        structured_llm = llm.with_structured_output(JobOfferEntities)
        
        # Define the prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert data extraction assistant. Your job is to extract job offer details from the provided text into a structured format. The text may be a combination of user-provided claims and OCR-extracted text from an image. If a field is not present, leave it null."),
            ("human", "Please extract the job entities from the following text:\n\n{text}")
        ])
        
        # Create the chain
        extraction_chain = prompt | structured_llm
        
        # Execute extraction
        result = extraction_chain.invoke({"text": text})
        return result if result else JobOfferEntities()
    except Exception as e:
        print(f"[Error] LLM Extraction failed: {e}")
        return JobOfferEntities()

if __name__ == "__main__":
    pass
