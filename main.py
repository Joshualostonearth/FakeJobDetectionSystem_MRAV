"""
CLI Entrypoint for Objective 2: Credibility Filtration Layer.

Usage:
    # Run with a JSON input file
    python main.py --input payload.json

    # Run with the built-in demo payload (for quick testing)
    python main.py --demo

    # Read from stdin
    cat payload.json | python main.py --stdin
"""

import argparse
import json
import sys

from schemas.input_schema import Objective2InputPayload
from credibility_engine import evaluate_credibility


def get_demo_payload() -> dict:
    """
    Returns a hardcoded demo payload for quick testing.
    Simulates a suspicious job offer from "Apex Technologies"
    with mismatched domains and a free email provider.
    """
    return {
        "offer_id": "OFFER_DEMO_001",
        "extracted_entities": {
            "company_name": "Apex Technologies Inc.",
            "job_role": "Senior Software Engineer",
            "recruiter_name": "Sarah Johnson",
            "domain_or_website": "apextech-jobs.org",
            "email_or_phone": "sarah.johnson@gmail.com",
            "salary_or_stipend": "$150,000/year",
            "location": "New York, NY",
        },
        "retrieved_evidence": {
            "registry_records": [
                {
                    "doc_id": "REG_001",
                    "content": (
                        "Apex Technologies Inc. is a registered corporation in Delaware, USA. "
                        "Official website: apextech.com. Headquarters: 500 Park Avenue, New York, NY 10022. "
                        "Status: ACTIVE. Registration Date: 2015-03-12."
                    ),
                    "similarity_score": 0.92,
                    "metadata": {
                        "source_type": "registry",
                        "company_name": "Apex Technologies Inc.",
                        "domain": "apextech.com",
                        "credibility_tier": "official",
                        "last_updated": "2026-06-15",
                        "source_url": "https://registry.delaware.gov/apex-tech",
                    },
                }
            ],
            "linkedin_data": [
                {
                    "doc_id": "LI_001",
                    "content": (
                        "Apex Technologies - Enterprise Software Solutions. "
                        "Industry: Information Technology & Services. "
                        "Headquarters: New York, NY. Founded: 2015. "
                        "Employees: 500-1000. Website: apextech.com"
                    ),
                    "similarity_score": 0.85,
                    "metadata": {
                        "source_type": "linkedin",
                        "company_name": "Apex Technologies",
                        "domain": "apextech.com",
                        "credibility_tier": "high",
                        "last_updated": "2026-07-01",
                        "source_url": "https://linkedin.com/company/apex-technologies",
                    },
                }
            ],
            "scam_reports": [
                {
                    "doc_id": "SCAM_001",
                    "content": (
                        "WARNING: Job scam reported using domain 'apextech-jobs.org'. "
                        "Multiple users report receiving unsolicited job offers from "
                        "sarah.johnson@gmail.com claiming to represent Apex Technologies. "
                        "The real company uses apextech.com. Do not share personal information."
                    ),
                    "similarity_score": 0.78,
                    "metadata": {
                        "source_type": "scam_report",
                        "company_name": "Apex Technologies",
                        "domain": "apextech-jobs.org",
                        "credibility_tier": "community",
                        "last_updated": "2026-07-20",
                        "source_url": "https://scamreports.example.com/report/12345",
                    },
                }
            ],
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Objective 2: Credibility Filtration Layer"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input",
        type=str,
        help="Path to a JSON file containing an Objective2InputPayload",
    )
    group.add_argument(
        "--demo",
        action="store_true",
        help="Run with a built-in demo payload for quick testing",
    )
    group.add_argument(
        "--stdin",
        action="store_true",
        help="Read JSON payload from stdin",
    )

    args = parser.parse_args()

    # Load the payload
    if args.demo:
        print("[Main] Running with demo payload...\n")
        raw = get_demo_payload()
    elif args.stdin:
        raw = json.load(sys.stdin)
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            raw = json.load(f)

    # Validate input
    payload = Objective2InputPayload(**raw)

    # Run the credibility engine
    result = evaluate_credibility(payload)

    # Output as JSON
    output = result.model_dump(mode="json")
    print("\n" + "=" * 60)
    print("OBJECTIVE 2 OUTPUT PAYLOAD")
    print("=" * 60)
    print(json.dumps(output, indent=2))

    return result


if __name__ == "__main__":
    main()
