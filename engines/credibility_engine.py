import os

import tldextract
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()

OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "ministral-3")

# --- 1. RULE-BASED CHECKERS (Fast, Deterministic) ---

def check_domain_match(claimed_domain: str, official_domain: str) -> dict:
    """Checks if the offer email/website domain matches the retrieved official domain."""
    if not claimed_domain or not official_domain:
        return {"match": True, "flag": None}

    ext_claimed = tldextract.extract(claimed_domain)
    ext_official = tldextract.extract(official_domain)

    root_claimed = f"{ext_claimed.domain}.{ext_claimed.suffix}".lower()
    root_official = f"{ext_official.domain}.{ext_official.suffix}".lower()

    is_match = root_claimed == root_official
    flag = None if is_match else f"CRITICAL: Claimed domain '{root_claimed}' does not match registered domain '{root_official}'"
    return {"match": is_match, "flag": flag}

def check_free_email(email: str) -> bool:
    """Returns True if corporate recruiter uses a free public email host."""
    if not email:
        return False
    free_hosts = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com"]
    domain = email.split("@")[-1].lower() if "@" in email else ""
    return domain in free_hosts


# --- 2. M-RAV CONSISTENCY EVALUATOR (local Ollama) ---

def evaluate_mrav_consistency(entities: dict, retrieved_evidence: list) -> dict:
    """
    Uses the local Ollama model (OLLAMA_MODEL) to evaluate semantic consistency 
    between extracted job claims and FAISS evidence documents.
    """
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Credibility Filtration Agent for a job scam detection system.
Compare the user's JOB CLAIM against the RETRIEVED EVIDENCE from our database.

CRITICAL JUDGMENT RULES:
1. SEMANTIC TITLE EQUIVALENCE: Treat minor job title variations as CONSISTENT (e.g., 'Senior Software Engineer (Backend)' IS EQUIVALENT to 'Software Engineer Backend').
2. DOMAIN & LOCATION MISMATCH: Flag as SUSPICIOUS or FRAUDULENT if:
   - The company name is missing or blank.
   - The location listed (e.g., California) conflicts with the known location/entity of the domain (e.g., richmond.edu belongs to University of Richmond in Virginia, NOT Richmond, CA).
3. VERDICT DEFINITIONS:
   - LEGITIMATE: Company, role, location, and domain align cleanly with evidence.
   - SUSPICIOUS: Major structural mismatch, geographic/domain conflict, or missing company context.
   - FRAUDULENT: Outright scam patterns, check/wire transfer requests, or verified malicious database matches.

Output ONLY a 2-line summary in this exact format:
VERDICT: [LEGITIMATE / SUSPICIOUS / FRAUDULENT]
REASON: [1 concise sentence explaining your reasoning]"""),
        ("human", """JOB CLAIM:
Company: {company}
Role: {role}
Domain: {domain}
Location: {location}

RETRIEVED FAISS EVIDENCE:
{evidence}""")
    ])

    # Convert retrieved FAISS docs into readable text
    evidence_text = ""
    for idx, doc in enumerate(retrieved_evidence, 1):
        is_scam = "YES (Scam)" if doc.get("metadata", {}).get("fraudulent") == 1 else "NO (Legitimate)"
        evidence_text += f"\n--- Evidence #{idx} (Known Scam: {is_scam}) ---\n{doc.get('page_content', '')[:300]}\n"

    chain = prompt | llm
    try:
        response = chain.invoke({
            "company": entities.get("company_name", ""),
            "role": entities.get("job_role", ""),
            "domain": entities.get("domain_or_website", ""),
            "location": entities.get("location", ""),
            "evidence": evidence_text
        })
        return {"mrav_analysis": response.content.strip()}
    except Exception as e:
        return {"mrav_analysis": f"M-RAV evaluation error: {e}"}


# --- 3. SYNTHESIZED CREDIBILITY CALCULATOR ---

def Run_Objective_2_Filter(entities: dict, retrieved_evidence: list) -> dict:
    flags = []
    trust_score = 100.0  # Start at 100%

    # Check 0: Missing Company Entity Check
    if not entities.get("company_name", "").strip():
        flags.append("WARNING: Company/Organization name could not be reliably extracted from document")
        trust_score -= 15.0

    # Check 1: Free Email Provider
    if check_free_email(entities.get("recruiter_email", "")):
        flags.append("WARNING: Recruiter using free public email address (e.g. @gmail.com)")
        trust_score -= 25.0

    # Check 2: Top Match Known Scam Check (closest = lowest FAISS distance)
    top_match = min(retrieved_evidence, key=lambda d: d.get("distance", float("inf")), default=None)
    if top_match and top_match.get("metadata", {}).get("fraudulent") == 1:
        flags.append("CRITICAL: Top similarity match in FAISS database is a known fraudulent listing")
        trust_score -= 50.0

    # Check 3: M-RAV Ollama Consistency Analysis
    mrav_result = evaluate_mrav_consistency(entities, retrieved_evidence)

    final_score = max(0.0, trust_score)
    risk_level = "HIGH RISK" if final_score < 50 else ("MEDIUM RISK" if final_score < 75 else "LOW RISK")

    return {
        "final_trust_score": final_score,
        "risk_level": risk_level,
        "flags_raised": flags,
        "mrav_explanation": mrav_result["mrav_analysis"]
    }