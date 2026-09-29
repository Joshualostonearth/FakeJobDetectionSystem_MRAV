"""
DST (Dempster-Shafer Theory) Engine for combining evidence
Dempster's Rule for merging belief masses from multiple sources
"""

from typing import Dict, List


class DSTEngine:
    """
    Dempster-Shafer Theory implementation for combining evidence.
    Handles mass function assignment over {REAL, FAKE, UNCERTAIN} and
    Dempster's Rule combination to measure belief and conflict across multiple sources.
    """
    
    def __init__(self):
        self.epsilon = 1e-10  # Prevent division by zero
    
    def assign_mass_function(self, 
                            similarity_score: float,
                            source_type: str,
                            is_fraudulent: int,
                            metadata_quality: float = 0.7) -> Dict[str, float]:
        """
        Assign mass function m({REAL}), m({FAKE}), m({UNCERTAIN}) based on evidence.
        
        Args:
            similarity_score: Cosine/distance similarity (0-1, higher = more similar)
            source_type: Type of source (scam_db, linkedin, registry, etc.)
            is_fraudulent: Label from database (0 = legitimate, 1 = scam)
            metadata_quality: How complete/trustworthy is the metadata (0-1)
        
        Returns:
            Dictionary with m({REAL}), m({FAKE}), and m({UNCERTAIN}) masses
        """
        
        # Base belief from similarity score
        similarity_confidence = max(0.0, min(similarity_score, 1.0))
        
        # Source credibility weighting
        source_credibility = {
            "scam_db": 0.85,         # High trust - verified scam reports
            "scam_report": 0.85,     # High trust - scam report database
            "scam_report_forum": 0.70,
            "linkedin": 0.85,        # High trust - official company data
            "registry": 0.90,        # Very high - government verified
            "legitimate_job_record": 0.85,
            "user_reports": 0.60,     # Medium - user-submitted
            "news_sources": 0.75,    # Medium-high
            "voice_call": 0.95,      # HIGHEST - independent verification
            "unknown": 0.50          # Low - unknown source
        }
        
        credibility = source_credibility.get(source_type, 0.50)
        
        # Combine similarity + credibility + metadata quality
        evidence_strength = min(1.0, similarity_confidence * credibility * metadata_quality)
        
        # Assign masses over {REAL, FAKE, UNCERTAIN}
        if is_fraudulent == 1:
            # Evidence points to FAKE/Scam
            mass_fake = evidence_strength
            mass_real = 0.0
            mass_uncertain = 1.0 - mass_fake
        else:
            # Evidence points to REAL/Legitimate
            mass_real = evidence_strength
            mass_fake = 0.0
            mass_uncertain = 1.0 - mass_real
        
        return {
            "REAL": round(mass_real, 4),
            "FAKE": round(mass_fake, 4),
            "UNCERTAIN": round(mass_uncertain, 4),
            "source": source_type,
            "strength": round(evidence_strength, 4)
        }
    
    def dempster_combine(self, m1: Dict, m2: Dict) -> Dict[str, float]:
        """
        Combine two mass functions using Dempster's Rule over {REAL, FAKE, UNCERTAIN}.
        
        Formula: m₁₂(A) = (1/(1-K)) × Σ [m₁(B) × m₂(C)] where B ∩ C = A
        K = conflict factor = m1(REAL)*m2(FAKE) + m1(FAKE)*m2(REAL)
        
        Args:
            m1: First mass function
            m2: Second mass function
        
        Returns:
            Combined mass function dictionary
        """
        r1, f1 = m1.get("REAL", 0.0), m1.get("FAKE", 0.0)
        u1 = m1.get("UNCERTAIN", max(0.0, 1.0 - r1 - f1))
        
        r2, f2 = m2.get("REAL", 0.0), m2.get("FAKE", 0.0)
        u2 = m2.get("UNCERTAIN", max(0.0, 1.0 - r2 - f2))
        
        # Calculate intersections
        match_real = (r1 * r2) + (r1 * u2) + (u1 * r2)  # Supporting REAL
        match_fake = (f1 * f2) + (f1 * u2) + (u1 * f2)  # Supporting FAKE
        match_uncertain = u1 * u2                        # Pure uncertainty
        
        # Calculate conflict factor K
        conflict = (r1 * f2) + (f1 * r2)
        
        # Normalization factor (1 - K)
        k = 1.0 - conflict
        
        if k <= self.epsilon:
            # Complete conflict - return neutral mass distribution
            return {
                "REAL": 0.3333,
                "FAKE": 0.3333,
                "UNCERTAIN": 0.3334,
                "conflict": 1.0
            }
        
        # Apply Dempster's Rule: normalize by 1 - K
        combined = {
            "REAL": round(match_real / k, 4),
            "FAKE": round(match_fake / k, 4),
            "UNCERTAIN": round(match_uncertain / k, 4),
            "conflict": round(conflict, 4)
        }
        
        return combined
    
    def combine_all_evidence(self, masses_list: List[Dict]) -> Dict[str, float]:
        """
        Combine multiple mass functions (from multiple FAISS results).
        Iteratively applies Dempster's Rule.
        
        Args:
            masses_list: List of mass function dictionaries
        
        Returns:
            Combined final belief
        """
        
        if not masses_list:
            return {"REAL": 0.0, "FAKE": 0.0, "UNCERTAIN": 1.0, "conflict": 0.0, "error": "No evidence provided"}
        
        if len(masses_list) == 1:
            m = dict(masses_list[0])
            if "UNCERTAIN" not in m:
                r, f = m.get("REAL", 0.0), m.get("FAKE", 0.0)
                m["UNCERTAIN"] = max(0.0, round(1.0 - r - f, 4))
            if "conflict" not in m:
                m["conflict"] = 0.0
            return m
        
        # Start with first mass
        combined = dict(masses_list[0])
        if "UNCERTAIN" not in combined:
            r, f = combined.get("REAL", 0.0), combined.get("FAKE", 0.0)
            combined["UNCERTAIN"] = max(0.0, round(1.0 - r - f, 4))
        
        # Iteratively combine with each subsequent mass
        for i in range(1, len(masses_list)):
            combined = self.dempster_combine(combined, masses_list[i])
        
        return combined
    
    def get_belief_strength(self, mass_dict: Dict) -> str:
        """
        Get human-readable belief strength description.
        
        Args:
            mass_dict: Combined mass dictionary
        
        Returns:
            String describing belief strength
        """
        real_belief = mass_dict.get("REAL", 0.0)
        fake_belief = mass_dict.get("FAKE", 0.0)
        conflict = mass_dict.get("conflict", 0.0)
        
        if conflict > 0.45:
            return "UNCERTAIN - High Conflict Between Evidence Sources"
        
        if fake_belief >= 0.70:
            return "VERY STRONG - Fake / Fraudulent"
        elif fake_belief >= 0.45:
            return "STRONG - Fake / Fraudulent"
        elif fake_belief >= 0.25:
            return "MODERATE - Suspicious"
        elif real_belief >= 0.70:
            return "VERY STRONG - Legitimate"
        elif real_belief >= 0.45:
            return "STRONG - Legitimate"
        elif real_belief >= 0.25:
            return "MODERATE - Legitimate"
        else:
            return "UNCERTAIN - Low Confidence Evidence"
    
    def get_conflict_info(self, mass_dict: Dict) -> str:
        """
        Describe what the conflict factor means.
        """
        conflict = mass_dict.get("conflict", 0.0)
        
        if conflict > 0.5:
            return "HIGH CONFLICT - Strong disagreement between evidence sources"
        elif conflict > 0.2:
            return "MODERATE CONFLICT - Partial disagreement between evidence sources"
        else:
            return "LOW CONFLICT - Evidence sources largely agree"


def run_objective_2_dst(
    entities: Dict,
    retrieved_evidence: List[Dict],
    voice_signals: Dict = None
) -> Dict:
    """
    Top-level runner for Objective 2 using Dempster-Shafer Theory (DST).
    
    Args:
        entities: Dict of extracted offer details (company_name, recruiter_email, domain_or_website, etc.)
        retrieved_evidence: List of FAISS document dictionaries with page_content, metadata, and distance.
        voice_signals: Optional VoiceVerificationSignals dict from Person 2 Voice Agent.
        
    Returns:
        Structured output dictionary with final trust score, risk level, flags, and DST summary.
    """
    dst = DSTEngine()
    masses = []
    flags = []
    
    # Check 1: Free Email Provider
    recruiter_email = entities.get("recruiter_email") or entities.get("email_or_phone", "")
    free_email_domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com"]
    if recruiter_email and any(recruiter_email.lower().endswith("@" + d) for d in free_email_domains):
        flags.append("WARNING: Recruiter uses free public email provider (e.g. @gmail.com)")
    
    # Check 2: Convert retrieved evidence to mass functions
    for doc in retrieved_evidence:
        meta = doc.get("metadata", {})
        # Support similarity score or distance (FAISS lower distance = higher similarity)
        if "similarity_score" in doc:
            sim = float(doc["similarity_score"])
        elif "distance" in doc:
            dist = float(doc["distance"])
            sim = max(0.0, 1.0 - (dist / 2.0))  # Convert L2 / cosine distance to similarity
        else:
            sim = 0.7
            
        source_type = meta.get("source_type", "unknown")
        is_fraudulent = int(meta.get("fraudulent", 1 if source_type == "scam_report" else 0))
        
        mass = dst.assign_mass_function(
            similarity_score=sim,
            source_type=source_type,
            is_fraudulent=is_fraudulent,
            metadata_quality=0.8
        )
        masses.append(mass)
        
        if is_fraudulent == 1:
            flags.append(f"CRITICAL: Scam report pattern detected from database record (doc: {meta.get('doc_id', 'N/A')})")

    # Check 3: Process Person 2 Voice Signals Evidence
    if voice_signals:
        voice_conf = float(voice_signals.get("confidence", 0.90))
        is_fraud = 1 if (
            voice_signals.get("company_denied_role") or
            voice_signals.get("payment_requested") or
            voice_signals.get("suspicious_response")
        ) else (
            0 if (voice_signals.get("company_reached") and voice_signals.get("role_actively_hiring")) else None
        )
        
        if is_fraud is not None:
            voice_mass = dst.assign_mass_function(
                similarity_score=voice_conf,
                source_type="voice_call",
                is_fraudulent=is_fraud,
                metadata_quality=0.95
            )
            masses.append(voice_mass)
            
            if is_fraud == 1:
                flags.append("CRITICAL VOICE SIGNAL: Phone verification failed or confirmed fake offer/payment request")
            else:
                flags.append("VERIFIED VOICE SIGNAL: Phone call to company confirmed legitimate hiring position")

    # Combine all masses using Dempster's Rule
    combined_mass = dst.combine_all_evidence(masses)
    
    # Convert DST belief into 0-100 Trust Score
    real_belief = combined_mass.get("REAL", 0.0)
    fake_belief = combined_mass.get("FAKE", 0.0)
    
    trust_score = round(max(0.0, (real_belief - fake_belief + 1.0) / 2.0 * 100.0), 2)
    if any("WARNING" in f or "CRITICAL" in f for f in flags):
        trust_score = min(trust_score, 60.0 if not any("CRITICAL" in f for f in flags) else 35.0)
    
    risk_level = "HIGH RISK" if trust_score < 50 else ("MEDIUM RISK" if trust_score < 75 else "LOW RISK")
    belief_summary = dst.get_belief_strength(combined_mass)
    conflict_summary = dst.get_conflict_info(combined_mass)
    
    return {
        "final_trust_score": trust_score,
        "risk_level": risk_level,
        "flags_raised": list(dict.fromkeys(flags)),
        "dst_summary": f"Belief: {belief_summary} | Conflict: {conflict_summary}",
        "mrav_explanation": f"DST Evidence Fusion complete. REAL belief: {real_belief:.4f}, FAKE belief: {fake_belief:.4f}, UNCERTAIN: {combined_mass.get('UNCERTAIN', 0.0):.4f}, Conflict: {combined_mass.get('conflict', 0.0):.4f}",
        "combined_mass": combined_mass
    }


if __name__ == "__main__":
    print("--- Dempster-Shafer Theory (DST) Objective 2 Demo ---")
    mock_entities = {
        "company_name": "Food52",
        "job_role": "Marketing Intern",
        "recruiter_email": "recruiting-food52@gmail.com",
        "domain_or_website": "food52.com"
    }
    mock_evidence = [
        {
            "page_content": "Food52 official job record",
            "metadata": {"source_type": "legitimate_job_record", "fraudulent": 0, "doc_id": "LEGIT_01"},
            "similarity_score": 0.88
        },
        {
            "page_content": "Scam alert: gmail recruitment from Food52 impersonator",
            "metadata": {"source_type": "scam_report", "fraudulent": 1, "doc_id": "SCAM_01"},
            "similarity_score": 0.75
        }
    ]
    res = run_objective_2_dst(mock_entities, mock_evidence)
    print(f"Trust Score : {res['final_trust_score']}/100")
    print(f"Risk Level  : {res['risk_level']}")
    print(f"Flags       : {res['flags_raised']}")
    print(f"DST Summary : {res['dst_summary']}")
    print(f"Masses      : {res['combined_mass']}")


