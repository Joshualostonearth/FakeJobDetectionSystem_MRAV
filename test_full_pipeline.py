from models import extract_entities_from_text
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from engines.credibility_engine import Run_Objective_2_Filter

def run_test():
    print("--- 1. RUNNING OBJECTIVE 1a: Entity Extraction via Ollama ---")
    mock_raw_offer = """
    Food52 - Marketing Intern Offer
    Location: New York, NY
    Contact: recruiting-food52@gmail.com
    Website: food52.com
    Please reply to confirm your acceptance.
    """
    
    # Extract entities using local Ollama (Llama 3.2)
    entities_obj = extract_entities_from_text(mock_raw_offer)
    
    # 🟢 Form dictionary cleanly to guarantee all key names match Objective 2 requirements
    entities = {
        "company_name": entities_obj.company_name,
        "job_role": entities_obj.job_role,
        "recruiter_email": entities_obj.recruiter_email,
        "domain_or_website": entities_obj.domain_or_website,
        "location": entities_obj.location
    }
    print(f"Extracted Entities: {entities}\n")

    print("--- 2. RUNNING OBJECTIVE 1b: FAISS Database Search ---")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    
    query = f"{entities['company_name']} {entities['job_role']}".strip()
    results = vector_db.similarity_search_with_score(query, k=3)
    
    retrieved_evidence = []
    for doc, score in results:
        retrieved_evidence.append({
            "page_content": doc.page_content,
            "metadata": doc.metadata,
            "distance": float(score)
        })
    print(f"Retrieved {len(retrieved_evidence)} matching evidence documents from FAISS.\n")

    print("--- 3. RUNNING OBJECTIVE 2: Credibility Filtration Layer ---")
    results_obj2 = Run_Objective_2_Filter(entities, retrieved_evidence)
    
    print("\n================ FINAL REPORT ================")
    print(f"TRUST SCORE : {results_obj2['final_trust_score']}/100")
    print(f"RISK LEVEL  : {results_obj2['risk_level']}")
    print(f"FLAGS       : {results_obj2['flags_raised']}")
    print(f"\nM-RAV ANALYSIS:\n{results_obj2['mrav_explanation']}")
    print("==============================================")

if __name__ == "__main__":
    run_test()