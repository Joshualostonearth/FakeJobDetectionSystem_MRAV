import os
import json
from typing import List, Dict, Any
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from models import JobOfferEntities

class EvidenceRetriever:
    def __init__(self, index_folder: str = "faiss_index"):
        print("Initializing FAISS Retriever...")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_db = None
        
        # Exception handling for missing FAISS index
        if not os.path.exists(index_folder):
            print(f"[Warning] FAISS index folder '{index_folder}' not found. Searches will return empty results.")
        else:
            try:
                # Load local index from disk
                self.vector_db = FAISS.load_local(
                    index_folder, 
                    self.embedding_model,
                    allow_dangerous_deserialization=True # Required for local pickle loading
                )
            except Exception as e:
                print(f"[Error] Failed to load FAISS index from '{index_folder}': {e}")

    def retrieve_evidence_bundle(self, entities: JobOfferEntities, top_k: int = 5) -> Dict[str, Any]:
        """
        Executes targeted searches using OCR-extracted entities and returns 
        structured evidence for Objective 2.
        """
        if not self.vector_db:
            print("[Error] Vector database is not loaded. Returning empty bundle.")
            return {
                "status": "FAILED",
                "extracted_entities": entities.model_dump(),
                "retrieved_evidence": []
            }
            
        print(f"\n[FAISS Search] Querying database for: '{entities.company_name}'...")
        
        all_results = []
        
        try:
            # 1. Legitimate Job Record Query (Baseline)
            legit_query = f"{entities.company_name} {entities.job_role or ''} {entities.location or ''}".strip()
            if legit_query:
                legit_results = self.vector_db.similarity_search_with_score(
                    query=legit_query, 
                    k=top_k,
                    filter={"source_type": "legitimate_job_record"}
                )
                all_results.extend(legit_results)
            
            # 2. Scam Report Query
            scam_query = f"{entities.company_name} {entities.job_role or ''} {entities.email_or_phone or ''} scam fraud fake offer".strip()
            if scam_query:
                scam_results = self.vector_db.similarity_search_with_score(
                    query=scam_query, 
                    k=top_k,
                    filter={"source_type": "scam_report"}
                )
                all_results.extend(scam_results)
                
                
        except Exception as e:
            print(f"[Error] FAISS similarity search failed: {e}")
            return {
                "status": "FAILED",
                "extracted_entities": entities.model_dump(),
                "retrieved_evidence": []
            }
        
        # Deduplicate and format results
        bundle_docs = {
            "legitimate_job_records": [],
            "registry_records": [],
            "linkedin_data": [],
            "scam_reports": []
        }
        seen_contents = set()
        
        # We also need to map the output to the vector document format required by Obj 2
        import uuid
        
        for doc, score in all_results:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                vec_doc = {
                    "doc_id": str(uuid.uuid4()),
                    "content": doc.page_content,
                    "similarity_score": float(score),
                    "metadata": doc.metadata
                }
                
                source_type = doc.metadata.get("source_type", "")
                if source_type == "legitimate_job_record":
                    bundle_docs["legitimate_job_records"].append(vec_doc)
                elif source_type == "scam_report":
                    bundle_docs["scam_reports"].append(vec_doc)
                elif source_type == "linkedin_verified":
                    bundle_docs["linkedin_data"].append(vec_doc)
                else:
                    bundle_docs["registry_records"].append(vec_doc)
            
        return {
            "status": "SUCCESS",
            "extracted_entities": entities.model_dump(),
            "retrieved_evidence": bundle_docs
        }

if __name__ == "__main__":
    pass