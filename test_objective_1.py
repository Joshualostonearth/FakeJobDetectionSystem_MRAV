import os
import json
from ocr_engine import init_ocr_reader, extract_text_from_image, merge_text_and_ocr
from models import extract_entities_from_text, JobOfferEntities
from search_engine import EvidenceRetriever

def run_pipeline(text: str, image_path: str, api_key: str):

    ocr_text = ""
    if image_path:
        reader = init_ocr_reader()
        ocr_text = extract_text_from_image(reader, image_path)
        
    merged_text = merge_text_and_ocr(text, ocr_text)
    
    if api_key:
        entities = extract_entities_from_text(merged_text, api_key)
    else:
        # Mock extraction if no API key is present so tests can run without failure
        print("[Mock] No OpenAI API Key found. Using mock entities.")
        entities = JobOfferEntities(
            company_name="Apex Tech Solutions" if "Apex" in merged_text else "Unknown",
            job_role="Software Engineer",
            recruiter_email="hr@apextech-careers.org" if "hr@" in merged_text else None,
            domain_or_website="apextech.com",
            location="San Francisco"
        )
        
    
    retriever = EvidenceRetriever()
    bundle = retriever.retrieve_evidence_bundle(entities, top_k=3)
    
    return bundle

def test_scenario_a():
    print("\n--- Running Scenario A: Text-only offer ---")
    text = "I received a job offer from Apex Tech Solutions for a Software Engineer role. They want me to email hr@apextech-careers.org."
    bundle = run_pipeline(text, "", os.getenv("OPENAI_API_KEY", ""))
    print(json.dumps(bundle, indent=2))
    assert bundle["status"] == "SUCCESS"
    assert "extracted_entities" in bundle
    assert "retrieved_evidence" in bundle
    assert isinstance(bundle["retrieved_evidence"], list)

def test_scenario_b():
    print("\n--- Running Scenario B: Image offer ---")
    # Using a fake image path to test OCR exception handling
    bundle = run_pipeline("", "fake_image_path.png", os.getenv("OPENAI_API_KEY", ""))
    print(json.dumps(bundle, indent=2))
    assert bundle["status"] == "SUCCESS"

def test_scenario_c():
    print("\n--- Running Scenario C: Edge Case (Ambiguous/Empty Text) ---")
    bundle = run_pipeline("Just a random text without any company info", "", os.getenv("OPENAI_API_KEY", ""))
    print(json.dumps(bundle, indent=2))
    assert bundle["status"] == "SUCCESS"

if __name__ == "__main__":
    test_scenario_a()
    test_scenario_b()
    test_scenario_c()
    print("\nAll tests passed!")
