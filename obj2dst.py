"""
DST (Dempster-Shafer Theory) Engine for combining evidence
Dempster's Rule for merging belief masses from multiple sources

Frame of discernment: {REAL, FAKE}. Every mass function assigns belief to
three focal elements:
    REAL       — m({REAL})
    FAKE       — m({FAKE})
    UNCERTAIN  — m({REAL, FAKE}), i.e. ignorance / "evidence doesn't say"

Weak evidence leaves most of its mass on UNCERTAIN instead of pushing it
onto the opposite hypothesis.
"""

from typing import Dict, List, Optional


class DSTEngine:
    """
    Dempster-Shafer Theory implementation for combining evidence.
    Handles 3-focal-element mass functions m({REAL}), m({FAKE}), m({UNCERTAIN})
    and calculates combined belief with conflict tracking (K).
    """

    def __init__(self):
        self.epsilon = 1e-10

    def assign_mass_function(
        self,
        similarity_score: float,
        source_type: str,
        is_fraudulent: int,
        metadata_quality: float = 0.7,
    ) -> Dict[str, float]:
        """
        Assign mass function m({REAL}), m({FAKE}), m({REAL, FAKE}) based on evidence
        
        Args:
            similarity_score: Similarity / credibility (0-1, higher = stronger evidence)
            source_type: Type of source (scam_report, linkedin, registry, etc.)
            is_fraudulent: Label from database (0 = legitimate, 1 = scam)
            metadata_quality: How complete/trustworthy is the metadata (0-1)
        
        Returns:
            Dictionary with REAL, FAKE and UNCERTAIN masses
        """
        
        # Base belief from similarity score
        similarity_confidence = max(0.0, min(similarity_score, 1.0))
        
        # Source credibility weighting
        source_credibility = {
            "scam_db": 0.8,          # High trust - verified scam reports
            "linkedin": 0.85,        # High trust - official company data
            "registry": 0.90,        # Very high - government verified
            "user_reports": 0.6,     # Medium - user-submitted
            "news_sources": 0.75,    # Medium-high
            "voice_call": 0.95,      # HIGHEST - independent verification
            "unknown": 0.5,          # Low - unknown source

            # Source types emitted by Objective 1 / Objective 2
            "government_registry": 0.90,
            "linkedin_verified": 0.85,
            "scam_report": 0.8,
            "scam_report_forum": 0.6,
            "legitimate_job_record": 0.7,   # EMSCAD labelled listing
            "third_party_aggregator": 0.6,
            "glassdoor": 0.6,
            "crunchbase": 0.6,
            "job_board": 0.6,
        }
        
        credibility = source_credibility.get(source_type, 0.5)
        
        # Combine similarity + credibility + metadata quality
        evidence_strength = similarity_confidence * credibility * max(0.0, min(metadata_quality, 1.0))
        evidence_strength = min(evidence_strength, 1.0)
        
        # Evidence supports its own label; the remainder is ignorance,
        # not support for the opposite hypothesis.
        if is_fraudulent == 1:
            mass_real, mass_fake = 0.0, evidence_strength
        else:
            mass_real, mass_fake = evidence_strength, 0.0
        
        return {
            "REAL": mass_real,
            "FAKE": mass_fake,
            "UNCERTAIN": 1.0 - evidence_strength,
            "source": source_type,
            "strength": round(evidence_strength, 4),
        }
    
    def _uncertain_mass(self, m: Dict) -> float:
        """m({REAL, FAKE}); derived from REAL/FAKE for masses without an UNCERTAIN key."""
        if "UNCERTAIN" in m:
            return m["UNCERTAIN"]
        return max(0.0, 1.0 - m["REAL"] - m["FAKE"])
    
    def dempster_combine(self, m1: Dict, m2: Dict) -> Dict[str, float]:
        """
        Combine two 3-focal mass functions m1 and m2 using Dempster's Rule of Combination.

        Formula:
            K (conflict) = m1(REAL)*m2(FAKE) + m1(FAKE)*m2(REAL)
            1 - K = Normalisation factor

            m12(REAL)      = [m1(REAL)*m2(REAL) + m1(REAL)*m2(UNCERTAIN) + m1(UNCERTAIN)*m2(REAL)] / (1 - K)
            m12(FAKE)      = [m1(FAKE)*m2(FAKE) + m1(FAKE)*m2(UNCERTAIN) + m1(UNCERTAIN)*m2(FAKE)] / (1 - K)
            m12(UNCERTAIN) = [m1(UNCERTAIN)*m2(UNCERTAIN)] / (1 - K)
        """
        u1 = self._uncertain_mass(m1)
        u2 = self._uncertain_mass(m2)
        
        # Intersections that land on each focal element
        # ({REAL} ∩ {REAL, FAKE} = {REAL}, etc.)
        match_real = m1["REAL"] * m2["REAL"] + m1["REAL"] * u2 + u1 * m2["REAL"]
        match_fake = m1["FAKE"] * m2["FAKE"] + m1["FAKE"] * u2 + u1 * m2["FAKE"]
        match_uncertain = u1 * u2
        
        # Calculate conflict (when they disagree)
        conflict = (m1["REAL"] * m2["FAKE"]) + (m1["FAKE"] * m2["REAL"])
        
        # Normalization factor
        k = 1.0 - conflict
        
        if k <= self.epsilon:
            # Complete conflict - nothing can be concluded
            return {
                "REAL": 0.0,
                "FAKE": 0.0,
                "UNCERTAIN": 1.0,
                "conflict": 1.0
            }
        
        # Apply Dempster's Rule: normalize by 1-K
        combined = {
            "REAL": match_real / k,
            "FAKE": match_fake / k,
            "UNCERTAIN": match_uncertain / k,
            "conflict": conflict
        }
        
        return combined
    
    def combine_all_evidence(self, masses_list: List[Dict]) -> Dict[str, float]:
        """
        Combine multiple mass functions (from multiple FAISS results)
        Iteratively applies Dempster's Rule
        
        Args:
            masses_list: List of mass function dictionaries
        
        Returns:
            Combined final belief. "conflict" is the total conflict
            accumulated across all pairwise combinations.
        """
        
        if not masses_list:
            return {"REAL": 0.0, "FAKE": 0.0, "UNCERTAIN": 1.0, "error": "No evidence provided"}
        
        # Start with first mass
        combined = {
            "REAL": masses_list[0]["REAL"],
            "FAKE": masses_list[0]["FAKE"],
            "UNCERTAIN": self._uncertain_mass(masses_list[0]),
            "conflict": 0.0
        }
        
        # Iteratively combine with each subsequent mass
        agreement = 1.0
        for i in range(1, len(masses_list)):
            combined = self.dempster_combine(combined, masses_list[i])
            agreement *= 1.0 - combined["conflict"]
        combined["conflict"] = 1.0 - agreement
        
        return combined

    def get_belief_strength(self, mass_dict: Dict) -> str:
        """
        Get human-readable belief strength description
        
        Uses the pignistic probability BetP(REAL) = m(REAL) + m(UNCERTAIN)/2,
        so ignorance is split evenly instead of counting against REAL.
        
        Args:
            mass_dict: Combined mass dictionary
        
        Returns:
            String describing belief strength
        """
        uncertain = self._uncertain_mass(mass_dict) if "REAL" in mass_dict and "FAKE" in mass_dict else 1.0
        if uncertain >= 0.6:
            return "UNCERTAIN - Insufficient evidence"
        
        real_belief = mass_dict.get("REAL", 0.0) + uncertain / 2.0
        
        if real_belief >= 0.85:
            return "VERY STRONG - Real"
        elif real_belief >= 0.70:
            return "STRONG - Real"
        elif real_belief >= 0.60:
            return "MODERATE - Real"
        elif real_belief <= 0.15:
            return "VERY STRONG - Fake"
        elif real_belief <= 0.30:
            return "STRONG - Fake"
        elif real_belief <= 0.40:
            return "MODERATE - Fake"
        else:
            return "UNCERTAIN - Conflicting"

    def get_conflict_info(self, mass_dict: Dict) -> str:
        """Describe what the conflict factor means."""
        conflict = mass_dict.get("conflict", 0.0)
        if conflict > 0.5:
            return "HIGH CONFLICT - Strong disagreement between sources"
        elif conflict > 0.2:
            return "MODERATE CONFLICT - Some disagreement"
        else:
            return "LOW CONFLICT - Sources mostly agree"


# ──────────────────────────────────────────────────────────
# Person 3 Standard Standalone Deliverable Functions
# ──────────────────────────────────────────────────────────

_default_engine = DSTEngine()


def combine_two(m1: Dict[str, float], m2: Dict[str, float]) -> Dict[str, float]:
    """Combines two mass functions and exposes conflict value K."""
    return _default_engine.dempster_combine(m1, m2)


def combine_all(masses_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Combines a list of mass functions (registry, linkedin, scam-reports, voice)."""
    return _default_engine.combine_all_evidence(masses_list)
