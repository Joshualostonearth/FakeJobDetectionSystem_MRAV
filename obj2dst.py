"""
DST (Dempster-Shafer Theory) Engine for combining evidence
Dempster's Rule for merging belief masses from multiple sources
"""

from typing import Dict, List


class DSTEngine:
    """
    Dempster-Shafer Theory implementation for combining evidence
    Handles mass function assignment and Dempster's Rule combination
    """
    
    def __init__(self):
        self.epsilon = 1e-10  # Prevent division by zero
    
    def assign_mass_function(self, 
                            similarity_score: float,
                            source_type: str,
                            is_fraudulent: int,
                            metadata_quality: float = 0.7) -> Dict[str, float]:
        """
        Assign mass function m({REAL}), m({FAKE}) based on evidence
        
        Args:
            similarity_score: FAISS cosine similarity (0-1, higher = more similar)
            source_type: Type of source (scam_db, linkedin, registry, etc.)
            is_fraudulent: Label from database (0 = legitimate, 1 = scam)
            metadata_quality: How complete/trustworthy is the metadata (0-1)
        
        Returns:
            Dictionary with m({REAL}) and m({FAKE}) masses
        """
        
        # Base belief from similarity score
        similarity_confidence = min(similarity_score, 1.0)
        
        # Source credibility weighting
        source_credibility = {
            "scam_db": 0.8,          # High trust - verified scam reports
            "linkedin": 0.85,        # High trust - official company data
            "registry": 0.90,        # Very high - government verified
            "user_reports": 0.6,     # Medium - user-submitted
            "news_sources": 0.75,    # Medium-high
            "voice_call": 0.95,      # HIGHEST - independent verification
            "unknown": 0.5           # Low - unknown source
        }
        
        credibility = source_credibility.get(source_type, 0.5)
        
        # Combine similarity + credibility + metadata quality
        evidence_strength = similarity_confidence * credibility * metadata_quality
        
        # Assign masses based on label
        if is_fraudulent == 1:
            # Evidence points to FAKE/Scam
            mass_fake = min(evidence_strength, 1.0)
            mass_real = 1.0 - mass_fake
        else:
            # Evidence points to REAL/Legitimate
            mass_real = min(evidence_strength, 1.0)
            mass_fake = 1.0 - mass_real
        
        return {
            "REAL": mass_real,
            "FAKE": mass_fake,
            "source": source_type,
            "strength": evidence_strength
        }
    
    def dempster_combine(self, m1: Dict, m2: Dict) -> Dict[str, float]:
        """
        Combine two mass functions using Dempster's Rule
        
        Formula: m₁₂(A) = (1/(1-K)) × Σ [m₁(B) × m₂(C)] where B ∩ C = A
        K = conflict factor (when evidence disagrees)
        
        Args:
            m1: First mass function
            m2: Second mass function
        
        Returns:
            Combined mass function
        """
        
        # Calculate intersections (when both agree)
        match_real = m1["REAL"] * m2["REAL"]  # Both say REAL
        match_fake = m1["FAKE"] * m2["FAKE"]  # Both say FAKE
        
        # Calculate conflict (when they disagree)
        conflict = (m1["REAL"] * m2["FAKE"]) + (m1["FAKE"] * m2["REAL"])
        
        # Normalization factor
        k = 1.0 - conflict
        
        if k <= self.epsilon:
            # Complete conflict - return neutral mass
            return {
                "REAL": 0.5,
                "FAKE": 0.5,
                "conflict": 1.0
            }
        
        # Apply Dempster's Rule: normalize by 1-K
        combined = {
            "REAL": match_real / k,
            "FAKE": match_fake / k,
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
            Combined final belief
        """
        
        if not masses_list:
            return {"REAL": 0.5, "FAKE": 0.5, "error": "No evidence provided"}
        
        if len(masses_list) == 1:
            return masses_list[0]
        
        # Start with first mass
        combined = {
            "REAL": masses_list[0]["REAL"],
            "FAKE": masses_list[0]["FAKE"]
        }
        
        # Iteratively combine with each subsequent mass
        for i in range(1, len(masses_list)):
            combined = self.dempster_combine(combined, masses_list[i])
        
        return combined
    
    def get_belief_strength(self, mass_dict: Dict) -> str:
        """
        Get human-readable belief strength description
        
        Args:
            mass_dict: Combined mass dictionary
        
        Returns:
            String describing belief strength
        """
        real_belief = mass_dict.get("REAL", 0.5)
        
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
        """
        Describe what the conflict factor means
        """
        conflict = mass_dict.get("conflict", 0)
        
        if conflict > 0.5:
            return "HIGH CONFLICT - Strong disagreement between sources"
        elif conflict > 0.2:
            return "MODERATE CONFLICT - Some disagreement"
        else:
            return "LOW CONFLICT - Sources mostly agree"
