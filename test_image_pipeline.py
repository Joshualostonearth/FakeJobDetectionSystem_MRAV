import os
from ocr_engine import init_ocr_reader, extract_text_from_image
from models import extract_entities_from_text
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from engines.credibility_engine import Run_Objective_2_Filter

def run_image_test(image_path: str):
    print("=" * 60)
    print("📸 STARTING END-TO-END OCR + OBJECTIVE 1 & 2 PIPELINE TEST")
    print("=" * 60)

    # 1. INITIALIZE EASYOCR & EXTRACT TEXT FROM IMAGE
    print("\n--- Step 1: Extracting Raw Text via EasyOCR ---")
    reader = init_ocr_reader(lang_list=['en'], gpu=False)
    
    raw_ocr_text = extract_text_from_image(reader, image_path)
    
    if not raw_ocr_text:
        print("❌ Error: Could not extract text from image.")
        return

    print(f"✅ Extracted {len(raw_ocr_text)} characters of raw text from image.")
    print("--- RAW OCR PREVIEW ---")
    print(raw_ocr_text[:300] + "...\n----------------------")

    # 2. OBJECTIVE 1a: ENTITY EXTRACTION VIA LOCAL OLLAMA (LLAMA 3.2)
    print("\n--- Step 2: Extracting Structured Entities with Ollama ---")
    entities_obj = extract_entities_from_text(raw_ocr_text)
    
    entities = {
        "company_name": entities_obj.company_name,
        "job_role": entities_obj.job_role,
        "recruiter_email": entities_obj.recruiter_email,
        "domain_or_website": entities_obj.domain_or_website,
        "location": entities_obj.location
    }
    
    print("Parsed Entities:")
    print(f"  • Company : {entities['company_name']}")
    print(f"  • Role    : {entities['job_role']}")
    print(f"  • Email   : {entities['recruiter_email']}")
    print(f"  • Domain  : {entities['domain_or_website']}")
    print(f"  • Location: {entities['location']}")

    # 3. OBJECTIVE 1b: FAISS DATABASE SEARCH
    print("\n--- Step 3: Querying FAISS Vector Database ---")
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
    print(f"Retrieved {len(retrieved_evidence)} evidence documents from FAISS.")

    # 4. OBJECTIVE 2: CREDIBILITY FILTRATION LAYER
    print("\n--- Step 4: Running Credibility Filtration Layer ---")
    results_obj2 = Run_Objective_2_Filter(entities, retrieved_evidence)
    
    print("\n================ FINAL REPORT ================")
    print(f"TRUST SCORE : {results_obj2['final_trust_score']}/100")
    print(f"RISK LEVEL  : {results_obj2['risk_level']}")
    print(f"FLAGS       : {results_obj2['flags_raised']}")
    print(f"\nM-RAV ANALYSIS:\n{results_obj2['mrav_explanation']}")
    print("==============================================")

if __name__ == "__main__":
    # Make sure richmond_offer.png exists in your FakeJobs folder
    IMAGE_FILE = "richmond_offer.png" 
    
    if os.path.exists(IMAGE_FILE):
        run_image_test(IMAGE_FILE)
    else:
        print(f"❌ File '{IMAGE_FILE}' not found. Please place the offer letter image in your folder!")