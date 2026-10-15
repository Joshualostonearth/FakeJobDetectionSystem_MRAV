import os
import pandas as pd
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def clean_html(raw_html: str) -> str:
    """Removes HTML tags and unescapes special characters from text."""
    if not isinstance(raw_html, str) or not raw_html.strip():
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(separator=" ").strip()

def build_faiss_from_emscad_csv(csv_path: str = "data/splits/train.csv", save_dir: str = "faiss_index"):
    print(f"1. Reading CSV from '{csv_path}'...")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ ERROR: Cannot find '{csv_path}'. Check your directory or file spelling.")

    df = pd.read_csv(csv_path)
    df.fillna("", inplace=True)
    
    documents = []
    print("2. Cleaning HTML tags and constructing LangChain Documents...")
    
    for idx, row in df.iterrows():
        # Handle boolean conversion ('t'/'f' or True/False or 1/0)
        is_fraud = str(row.get("fraudulent", "")).lower() in ["t", "true", "1"]
        has_logo = str(row.get("has_company_logo", "")).lower() in ["t", "true", "1"]
        
        # Clean rich text fields
        company_profile = clean_html(row.get("company_profile", ""))
        description = clean_html(row.get("description", ""))
        requirements = clean_html(row.get("requirements", ""))
        benefits = clean_html(row.get("benefits", ""))
        
        # Build unified page content for vector embeddings
        prefix = "[KNOWN FRAUDULENT SCAM] " if is_fraud else "[LEGITIMATE JOB RECORD] "
        page_content = (
            f"{prefix}"
            f"Title: {row.get('title', '')}. "
            f"Location: {row.get('location', '')}. "
            f"Department: {row.get('department', '')}. "
            f"Industry: {row.get('industry', '')}. "
            f"Employment Type: {row.get('employment_type', '')}. "
            f"Company Profile: {company_profile}. "
            f"Description: {description}. "
            f"Requirements: {requirements}. "
            f"Benefits: {benefits}."
        )

        # Store searchable tags inside metadata
        metadata = {
            "row_id": str(idx),
            "title": str(row.get("title", "")),
            "location": str(row.get("location", "")),
            "industry": str(row.get("industry", "")),
            "fraudulent": 1 if is_fraud else 0,
            "source_type": "scam_report" if is_fraud else "legitimate_job_record",
            "has_logo": 1 if has_logo else 0,
            "authority_tier": 4 if is_fraud else 3
        }

        documents.append(Document(page_content=page_content, metadata=metadata))

    print(f"3. Loaded {len(documents)} clean documents. Creating FAISS Index with HuggingFace Embeddings...")
    
    # Load embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Embed and create FAISS store
    vector_db = FAISS.from_documents(documents, embedding_model)
    vector_db.save_local(save_dir)
    print(f"✅ SUCCESS: Saved FAISS Index to '{save_dir}/' folder!")

if __name__ == "__main__":
    # Index only the training split so validation/test postings are never their own evidence
    build_faiss_from_emscad_csv(csv_path="data/splits/train.csv", save_dir="faiss_index")