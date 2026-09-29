import sys
import os

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from integration.voice_agent_api import (
    init_verification_session,
    process_transcript_turn,
    finalize_session,
    hand_off_to_objective_2
)

def test_person2_full_pipeline():
    print("=" * 70)
    print("[START] STARTING PERSON 2 (AI/VOICE CONVERSATION AGENT) PIPELINE TEST")
    print("=" * 70)

    # 1. Take Objective 1 Entity Extraction output
    obj1_data = {
        "company_name": "Microsoft",
        "job_role": "Software Engineering Intern",
        "location": "Bangalore",
        "recruiter_name": "John Doe",
        "recruiter_email": "recruiter-john@gmail.com",
        "domain_or_website": "microsoft.com"
    }

    print("\n--- 1. Initializing Voice Session from Objective 1 Context ---")
    session_info = init_verification_session(obj1_data)
    session_id = session_info["session_id"]
    print(f"[OK] Session Created ID: {session_id}")
    print(f"Initial Agent Greeting: '{session_info['initial_question']}'")

    # 2. Simulate Person 1 <-> Person 2 dialogue turns
    simulated_turns = [
        # Turn 1: Confirm company awareness
        {
            "type": "transcript",
            "text": "Yes, this is the official Microsoft Bangalore recruitment team.",
            "timestamp": 4.5
        },
        # Turn 2: Confirm role active
        {
            "type": "transcript",
            "text": "Yes, the Software Engineering Internship program is actively hiring candidates.",
            "timestamp": 12.0
        },
        # Turn 3: Recruiter verification
        {
            "type": "transcript",
            "text": "We do not have any recruiter named John Doe registered in our system. Please beware of unauthorized callers.",
            "timestamp": 20.1
        },
        # Turn 4: Application process
        {
            "type": "transcript",
            "text": "Applicants should apply only via careers.microsoft.com.",
            "timestamp": 27.8
        },
        # Turn 5: Payment request check
        {
            "type": "transcript",
            "text": "We never ask for any money or deposit fees during recruitment.",
            "timestamp": 35.2
        },
        # Turn 6: Official channel
        {
            "type": "transcript",
            "text": "You can verify all open roles at careers.microsoft.com.",
            "timestamp": 42.0
        }
    ]

    print("\n--- 2. Simulating Person 1 (STT) -> Person 2 (AI Agent) Interface Turns ---")
    for idx, turn_input in enumerate(simulated_turns, 1):
        print(f"\n[Turn {idx}] Person 1 Transcript: \"{turn_input['text']}\"")
        response = process_transcript_turn(session_id, turn_input)
        print(f"Person 2 Agent Response: \"{response['text']}\"")
        print(f"   - Updated Facts : {response['updated_facts']}")
        print(f"   - Next Objective: {response['next_objective']}")
        print(f"   - Call Complete : {response['conversation_complete']}")

    # 3. Finalize call session & extract structured signals
    print("\n--- 3. Finalizing Session & Extracting Voice Verification Signals ---")
    final_data = finalize_session(session_id)
    signals = final_data["voice_signals"]
    print("[OK] Extracted Voice Verification Signals:")
    print(f"   - Company Reached     : {signals['company_reached']}")
    print(f"   - Role Actively Hiring: {signals['role_actively_hiring']}")
    print(f"   - Recruiter Recognized: {signals['recruiter_recognized']}")
    print(f"   - Company Denied Role : {signals['company_denied_role']}")
    print(f"   - Suspicious Response : {signals['suspicious_response']}")
    print(f"   - Confidence Score    : {signals['confidence']}")

    # 4. Objective 2 Integration Handoff
    print("\n--- 4. Handoff to Objective 2 Consensus Engine (DST Fusion) ---")
    mock_evidence = [
        {
            "page_content": "Microsoft Bangalore Official Internship Post",
            "metadata": {"source_type": "legitimate_job_record", "fraudulent": 0, "doc_id": "MSFT_01"},
            "similarity_score": 0.90
        }
    ]
    
    obj2_result = hand_off_to_objective_2(obj1_data, mock_evidence, signals)

    print("\n================ FINAL VERIFICATION REPORT ================")
    print(f"TRUST SCORE : {obj2_result['final_trust_score']}/100")
    print(f"RISK LEVEL  : {obj2_result['risk_level']}")
    print("FLAGS RAISED:")
    for flag in obj2_result['flags_raised']:
        print(f"  - {flag}")
    print(f"\nM-RAV & DST ANALYSIS:\n{obj2_result['mrav_explanation']}")
    print("==========================================================")

if __name__ == "__main__":
    test_person2_full_pipeline()
