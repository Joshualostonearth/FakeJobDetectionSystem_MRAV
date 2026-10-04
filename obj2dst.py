"""
DST (Dempster-Shafer Theory) Engine for combining evidence.
Dempster's Rule for merging belief masses from multiple sources:
Registry, LinkedIn, Scam Reports, and Voice Call Mass.

Focal elements supported: {REAL}, {FAKE}, {UNCERTAIN}.
Exposes top-level combine_two() and combine_all() functions and conflict factor K.
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
        Assign a 3-focal mass function m({REAL}), m({FAKE}), m({UNCERTAIN}) based on evidence.

        Args:
            similarity_score: Similarity / relevance score (0-1)
            source_type: Type of source (scam_db, linkedin, registry, voice_call, etc.)
            is_fraudulent: Label from record (0 = legitimate, 1 = scam)
            metadata_quality: Source completeness / trustworthiness (0-1)

        Returns:
            Dict with REAL, FAKE, UNCERTAIN masses and metadata.
        """
        similarity_confidence = min(max(float(similarity_score), 0.0), 1.0)

        source_credibility = {
            "registry": 0.95,
            "voice_call": 0.95,
            "linkedin": 0.85,
            "scam_db": 0.85,
            "scam_report": 0.85,
            "news_sources": 0.75,
            "user_reports": 0.60,
            "unknown": 0.50,
        }

        credibility = source_credibility.get(source_type, 0.50)
        evidence_strength = similarity_confidence * credibility * metadata_quality
        evidence_strength = min(max(evidence_strength, 0.0), 0.95)

        if is_fraudulent == 1:
            mass_fake = round(evidence_strength, 6)
            mass_real = 0.0
            mass_uncertain = round(1.0 - mass_fake, 6)
        else:
            mass_real = round(evidence_strength, 6)
            mass_fake = 0.0
            mass_uncertain = round(1.0 - mass_real, 6)

        return {
            "REAL": mass_real,
            "FAKE": mass_fake,
            "UNCERTAIN": mass_uncertain,
            "source": source_type,
            "strength": round(evidence_strength, 4),
        }

    def dempster_combine(self, m1: Dict[str, float], m2: Dict[str, float]) -> Dict[str, float]:
        """
        Combine two 3-focal mass functions m1 and m2 using Dempster's Rule of Combination.

        Formula:
            K (conflict) = m1(REAL)*m2(FAKE) + m1(FAKE)*m2(REAL)
            1 - K = Normalisation factor

            m12(REAL)      = [m1(REAL)*m2(REAL) + m1(REAL)*m2(UNCERTAIN) + m1(UNCERTAIN)*m2(REAL)] / (1 - K)
            m12(FAKE)      = [m1(FAKE)*m2(FAKE) + m1(FAKE)*m2(UNCERTAIN) + m1(UNCERTAIN)*m2(FAKE)] / (1 - K)
            m12(UNCERTAIN) = [m1(UNCERTAIN)*m2(UNCERTAIN)] / (1 - K)
        """
        m1_real = m1.get("REAL", 0.0)
        m1_fake = m1.get("FAKE", 0.0)
        m1_unc = m1.get("UNCERTAIN", 1.0 - m1_real - m1_fake)

        m2_real = m2.get("REAL", 0.0)
        m2_fake = m2.get("FAKE", 0.0)
        m2_unc = m2.get("UNCERTAIN", 1.0 - m2_real - m2_fake)

        # Conflict factor K (empty set intersections)
        conflict = (m1_real * m2_fake) + (m1_fake * m2_real)
        norm = 1.0 - conflict

        if norm <= self.epsilon:
            # Complete conflict — default to max uncertainty
            return {
                "REAL": 0.0,
                "FAKE": 0.0,
                "UNCERTAIN": 1.0,
                "conflict": 1.0,
            }

        # Intersections that yield {REAL}
        raw_real = (m1_real * m2_real) + (m1_real * m2_unc) + (m1_unc * m2_real)

        # Intersections that yield {FAKE}
        raw_fake = (m1_fake * m2_fake) + (m1_fake * m2_unc) + (m1_unc * m2_fake)

        # Intersections that yield {UNCERTAIN}
        raw_unc = m1_unc * m2_unc

        combined_real = round(raw_real / norm, 6)
        combined_fake = round(raw_fake / norm, 6)
        combined_unc = round(raw_unc / norm, 6)

        return {
            "REAL": combined_real,
            "FAKE": combined_fake,
            "UNCERTAIN": combined_unc,
            "conflict": round(conflict, 6),
        }

    def combine_all_evidence(self, masses_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Iteratively combine a list of mass functions using Dempster's Rule."""
        if not masses_list:
            return {"REAL": 0.0, "FAKE": 0.0, "UNCERTAIN": 1.0, "conflict": 0.0}

        if len(masses_list) == 1:
            res = dict(masses_list[0])
            if "UNCERTAIN" not in res:
                res["UNCERTAIN"] = round(1.0 - res.get("REAL", 0.0) - res.get("FAKE", 0.0), 6)
            if "conflict" not in res:
                res["conflict"] = 0.0
            return res

        combined = dict(masses_list[0])
        max_conflict = 0.0

        for i in range(1, len(masses_list)):
            combined = self.dempster_combine(combined, masses_list[i])
            if combined.get("conflict", 0.0) > max_conflict:
                max_conflict = combined["conflict"]

        combined["conflict"] = round(max_conflict, 6)
        return combined

    def get_belief_strength(self, mass_dict: Dict) -> str:
        """Get human-readable belief strength description."""
        real_belief = mass_dict.get("REAL", 0.0)
        fake_belief = mass_dict.get("FAKE", 0.0)

        if real_belief >= 0.80:
            return "VERY STRONG - Real"
        elif real_belief >= 0.60:
            return "STRONG - Real"
        elif fake_belief >= 0.80:
            return "VERY STRONG - Fake"
        elif fake_belief >= 0.60:
            return "STRONG - Fake"
        else:
            return "UNCERTAIN - Mixed Evidence"

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
