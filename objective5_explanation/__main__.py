"""
Demo of the Objective 5 voice path:  python -m objective5_explanation
The confidence path runs inside full_pipeline_orchestrator.py (explain=True).
"""

from .explainer import explain_voice_result

demo_voice = {
    "company_reached": True,
    "role_actively_hiring": None,
    "recruiter_recognized": False,
    "company_denied_role": True,
    "suspicious_response": False,
    "source_confidence": 0.9,
    "extraction_confidence": 0.85,
    "number_source": "google_places",
    "response_summary": "HR said they have no Senior Software Engineer opening and do not know Sarah Johnson.",
}

report = explain_voice_result(
    demo_voice,
    entities={"company_name": "Apex Technologies Inc.", "job_role": "Senior Software Engineer"},
    offer_id="OFFER_DEMO_001",
)
print(report.to_text())
