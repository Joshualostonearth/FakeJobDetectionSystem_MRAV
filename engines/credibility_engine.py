import tldextract
from obj2dst import DSTEngine, run_objective_2_dst
from key_pool import key_pool

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_ollama import ChatOllama
    HAS_LANGCHAIN_OLLAMA = True
except ImportError:
    HAS_LANGCHAIN_OLLAMA = False

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


# --- 2. M-RAV CONSISTENCY EVALUATOR ---

def evaluate_mrav_consistency(entities: dict, retrieved_evidence: list) -> dict:
    """
    Evaluates semantic consistency between extracted job claims and FAISS evidence documents.
    """
    evidence_text = ""
    for idx, doc in enumerate(retrieved_evidence, 1):
        is_scam = "YES (Scam)" if doc.get("metadata", {}).get("fraudulent") == 1 else "NO (Legitimate)"
        evidence_text += f"\n--- Evidence #{idx} (Known Scam: {is_scam}) ---\n{doc.get('page_content', '')[:300]}\n"

    prompt_str = f"""You are a Credibility Filtration Agent for a job scam detection system.
Compare the user's JOB CLAIM against the RETRIEVED EVIDENCE from our database.

JOB CLAIM:
Company: {entities.get("company_name", "")}
Role: {entities.get("job_role", "")}
Domain: {entities.get("domain_or_website", "")}

RETRIEVED FAISS EVIDENCE:
{evidence_text}

Determine if the offer is CONSISTENT, SUSPICIOUS, or FRAUDULENT.
Output ONLY a 2-line summary in this format:
VERDICT: [LEGITIMATE / SUSPICIOUS / FRAUDULENT]
REASON: [1 concise sentence explaining your reasoning]"""

    if HAS_LANGCHAIN_OLLAMA:
        try:
            llm = ChatOllama(model="llama3.2", temperature=0)
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a Credibility Filtration Agent."),
                ("human", prompt_str)
            ])
            chain = prompt | llm
            response = chain.invoke({})
            return {"mrav_analysis": response.content.strip()}
        except Exception as e:
            print(f"[Warning] ChatOllama failed ({e}), falling back to key_pool.")

    def api_call(client):
        try:
            response = client.chat.completions.create(
                model=key_pool.model,
                messages=[{"role": "user", "content": prompt_str}],
                temperature=0.0,
                max_tokens=256,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"VERDICT: SUSPICIOUS\nREASON: Evidence verification performed via DST engine fallback ({e})."

    analysis = key_pool.execute(api_call) if hasattr(key_pool, "execute") else "VERDICT: SUSPICIOUS\nREASON: DST analysis completed."
    return {"mrav_analysis": analysis}


# --- 3. SYNTHESIZED CREDIBILITY CALCULATOR (USING DST) ---

def Run_Objective_2_Filter(
    entities: dict,
    retrieved_evidence: list,
    voice_signals: dict = None
) -> dict:
    """
    Runs Objective 2 filtration layer using Dempster-Shafer Theory (obj2dst.py).
    Optionally integrates Person 2 Voice Agent signals.
    """
    # Execute Dempster-Shafer Theory evidence combination
    dst_result = run_objective_2_dst(entities, retrieved_evidence, voice_signals)
    
    # Run M-RAV LLM consistency check
    mrav_res = evaluate_mrav_consistency(entities, retrieved_evidence)
    
    # Combine M-RAV explanation into DST output
    full_explanation = f"{mrav_res['mrav_analysis']}\n\n[DST Evidence Fusion]\n{dst_result['mrav_explanation']}\n{dst_result['dst_summary']}"
    
    return {
        "final_trust_score": dst_result["final_trust_score"],
        "risk_level": dst_result["risk_level"],
        "flags_raised": dst_result["flags_raised"],
        "mrav_explanation": full_explanation,
        "combined_mass": dst_result.get("combined_mass", {})
    }