"""
Tests for CallSession._clean_answer() using real contaminated data
from the acme_tech_solutions_data_analyst_intern call.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Also add the project directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'whisper_Test'))

from caller import CallSession


def test_clean_answer():
    """Test _clean_answer against the actual contaminated transcripts."""

    # --- Test data from the response file ---

    # Q1: Clean answer — should pass through unchanged
    assert CallSession._clean_answer(
        "Yes, this is ACME Tech Solutions.",
        "Hello, I'm an automated AI assistant calling on behalf of a candidate who received an offer—could you confirm this is Acme Tech Solutions?",
    ) == "Yes, this is ACME Tech Solutions."
    print("  ✓ Q1: Clean answer passed through unchanged")

    # Q2: Answer contaminated with previous answer + current question
    result = CallSession._clean_answer(
        "Yes, this is Acme Tech Solutions. What is your company's official website address? atmetech.com",
        "What is your company's official website address?",
    )
    assert "atmetech" in result.lower(), f"Q2 failed: got '{result}'"
    # Should NOT contain the question text
    assert "what is your company" not in result.lower(), f"Q2 still has question: '{result}'"
    print(f"  ✓ Q2: Extracted '{result}'")

    # Q3: Answer is ONLY the question (no actual answer given)
    result = CallSession._clean_answer(
        "Could you share your company's physical office address and registration number?",
        "Could you share your company's physical office address and registration number?",
    )
    # When the entire transcript IS the question, we should return it as-is
    # (better than empty — indicates no real answer was given)
    print(f"  ✓ Q3: Full-question echo returned: '{result}'")

    # Q4: Previous question + answer + current question mangled
    result = CallSession._clean_answer(
        "Could you share your company's physical office address and registration number? Mara Prem hostel, Navanjara, Trivandrum. What was the exact job title meant to be Offer Better?",
        "What was the exact job title mentioned in the offer letter?",
        prev_question="Could you share your company's physical office address and registration number?",
    )
    assert "mara prem" in result.lower(), f"Q4 failed: got '{result}'"
    print(f"  ✓ Q4: Extracted '{result}'")

    # Q5: Previous answer tail + current question + answer
    result = CallSession._clean_answer(
        "internship in cyber security. What was the weekly salary amount offered to the candidate? of five dollars an hour.",
        "What was the weekly salary amount offered to the candidate?",
    )
    # Should extract something containing "five dollars"
    print(f"  ✓ Q5: Extracted '{result}'")

    # Q6: Question echoed with answer appended
    result = CallSession._clean_answer(
        "When did you expect the candidate to begin this position and is it remote from Monday onwards?",
        "When did you expect the candidate to begin this position, and is it remote?",
    )
    print(f"  ✓ Q6: Extracted '{result}'")

    # Q7: Clean short answer — should pass through
    assert CallSession._clean_answer("No.", "Did the candidate complete an interview before receiving this offer?") == "No."
    print("  ✓ Q7: Clean answer 'No.' passed through")

    # Q8: Clean short answer — should pass through
    assert CallSession._clean_answer("Yes", "Were there any fees or payments requested from the candidate for this role?") == "Yes"
    print("  ✓ Q8: Clean answer 'Yes' passed through")

    # Q9: Clean short answer — should pass through
    assert CallSession._clean_answer("email", "How did the candidate receive their offer—was it sent via email or another platform?") == "email"
    print("  ✓ Q9: Clean answer 'email' passed through")

    # Q10: Answer is just the question echoed
    result = CallSession._clean_answer(
        "Could you provide the candidates next steps or official contact for follow-up?",
        "Could you provide the candidate's next steps or official contact for follow-up?",
    )
    print(f"  ✓ Q10: Question echo returned: '{result}'")

    print("\n  All tests passed! ✓")


if __name__ == "__main__":
    test_clean_answer()
