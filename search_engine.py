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
            # 1. Registry Query
            registry_query = f"{entities.company_name} {entities.domain_or_website or ''}".strip()
            if registry_query:
                registry_results = self.vector_db.similarity_search_with_score(
                    query=registry_query, 
                    k=top_k,
                    filter={"source_type": "government_registry"}
                )
                all_results.extend(registry_results)
            
            # 2. LinkedIn Query
            linkedin_query = f"{entities.company_name} hiring {entities.job_role or ''}".strip()
            if linkedin_query:
                linkedin_results = self.vector_db.similarity_search_with_score(
                    query=linkedin_query, 
                    k=top_k,
                    filter={"source_type": "linkedin_verified"}
                )
                all_results.extend(linkedin_results)
            
            # 3. Scam Database Query
            scam_query = f"{entities.company_name} {entities.recruiter_email or ''} scam fraud fake offer".strip()
            if scam_query:
                scam_results = self.vector_db.similarity_search_with_score(
                    query=scam_query, 
                    k=top_k,
                    filter={"source_type": "scam_forum"}
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
        formatted_results = []
        seen_contents = set()
        for doc, score in all_results:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                formatted_results.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                    "vector_distance": float(score)
                })
            
        return {
            "status": "SUCCESS",
            "extracted_entities": entities.model_dump(),
            "retrieved_evidence": formatted_results
        }

if __name__ == "__main__":
    pass