"""
evidence_builder.py — Builds the grounded EvidenceSheet for Objective 5.

The verdict reaches Objective 5 through one of two paths:

    CONFIDENCE_PATH  Objective 4 output (full_pipeline_orchestrator.run_full_pipeline):
                     source credibility scores, DST masses, conflict K, flags,
                     calibrated P(Real), safety overrides.
    VOICE_PATH       Objective 3 voice verification yes/no decision plus the
                     extracted call signals (consensus_handoff.push_voice_signals).

Both are converted into the same list of numbered, citable facts. Nothing in
this module calls an LLM: the sheet is a deterministic restatement of the
pipeline's own outputs, which is what the explanation is later checked against.

Public API:
    build_from_confidence_result(pipeline_result, entities=None) -> EvidenceSheet
    build_from_voice_result(voice_analysis, voice_decision=None, entities=None, offer_id="UNKNOWN") -> EvidenceSheet
"""

from typing import Any, Dict, List, Optional, Union

from .schemas import Contradiction, EvidenceFact, EvidenceSheet

# Objective 4 verdict labels -> Objective 5 three-way verdict
OBJ4_VERDICT_MAP: Dict[str, str] = {
    "VERIFIED_REAL": "Real",
    "HIGH_RISK_SCAM": "Fake",
    "SUSPICIOUS_FAKE": "Fake",
    "UNCERTAIN_NEEDS_VERIFICATION": "Unable to Verify",
}

# Same threshold as obj2dst.HIGH_CONFLICT_THRESHOLD
HIGH_CONFLICT_THRESHOLD = 0.40
MODERATE_CONFLICT_THRESHOLD = 0.20

SCAM_SOURCE_TYPES = {"scam_report", "scam_report_forum", "scam_db"}

# signal -> (text when True, text when False, stance when True, stance when False)
VOICE_SIGNALS = {
    "company_reached": (
        "Someone at the company answered the verification call",
        "Nobody at the company could be reached on the independently sourced number",
        "neutral", "neutral",  # reaching the company enables verification; it is not evidence either way
    ),
    "role_actively_hiring": (
        "The company confirmed that the offered role is open",
        "The company said the offered role is not open",
        "supports_real", "supports_fake",
    ),
    "recruiter_recognized": (
        "The company recognised the recruiter named in the offer",
        "The company did not recognise the recruiter named in the offer",
        "supports_real", "supports_fake",
    ),
    "company_denied_role": (
        "The company explicitly denied that this role exists",
        None,
        "supports_fake", "neutral",
    ),
    "suspicious_response": (
        "The response on the call was evasive, hostile or scam-like",
        None,
        "supports_fake", "neutral",
    ),
    "payment_requested": (
        "A payment was requested from the candidate during the call",
        "No payment was requested during the call",
        "supports_fake", "neutral",
    ),
}

VOICE_SIGNAL_TOPICS = {
    "company_reached": "whether the company was reached",
    "role_actively_hiring": "whether the role is open",
    "recruiter_recognized": "whether the recruiter is known to the company",
}


class _FactList:
    """Assigns sequential, prefix-based fact IDs (S1, S2, F1, ...)."""

    def __init__(self):
        self.facts: List[EvidenceFact] = []
        self._counters: Dict[str, int] = {}

    def add(self, prefix: str, category: str, text: str, stance: str = "neutral") -> str:
        self._counters[prefix] = self._counters.get(prefix, 0) + 1
        fact_id = f"{prefix}{self._counters[prefix]}"
        self.facts.append(EvidenceFact(fact_id=fact_id, category=category, text=text, stance=stance))
        return fact_id


# ──────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────

def _add_claims(facts: _FactList, entities: Optional[Dict[str, Any]]):
    if not entities:
        return
    labels = [
        ("company_name", "company"),
        ("job_role", "role"),
        ("recruiter_name", "recruiter"),
        ("email_or_phone", "contact"),
        ("recruiter_email", "contact"),
        ("domain_or_website", "website"),
        ("location", "location"),
    ]
    parts, seen = [], set()
    for key, label in labels:
        value = entities.get(key)
        if value and label not in seen:
            parts.append(f"{label} '{value}'")
            seen.add(label)
    if parts:
        facts.add("C", "claim", "The offer claims: " + ", ".join(parts) + ".")


def _flag_parts(flag: Any) -> tuple:
    if isinstance(flag, dict):
        return flag.get("severity", "INFO"), flag.get("message", "")
    if hasattr(flag, "severity"):
        return flag.severity, flag.message
    text = str(flag)
    severity = "CRITICAL" if "CRITICAL" in text.upper() else ("WARNING" if "WARNING" in text.upper() else "INFO")
    return severity, text


def _conflict_level(k: float) -> str:
    if k > HIGH_CONFLICT_THRESHOLD:
        return "high: the evidence sources strongly disagree"
    if k > MODERATE_CONFLICT_THRESHOLD:
        return "moderate: the evidence sources partly disagree"
    return "low: the evidence sources largely agree"


def _find_contradictions(facts: List[EvidenceFact], conflict_fact_id: Optional[str] = None) -> List[Contradiction]:
    """Group facts pointing to Real against facts pointing to Fake."""
    evidence_categories = {"source", "voice", "flag"}
    real_ids = [f.fact_id for f in facts if f.category in evidence_categories and f.stance == "supports_real"]
    fake_ids = [f.fact_id for f in facts if f.category in evidence_categories and f.stance == "supports_fake"]

    if not real_ids or not fake_ids:
        return []

    ids = real_ids + fake_ids + ([conflict_fact_id] if conflict_fact_id else [])
    text = (
        f"Evidence indicating a legitimate offer ({', '.join(real_ids)}) disagrees with "
        f"evidence indicating fraud ({', '.join(fake_ids)})."
    )
    return [Contradiction(fact_ids=ids, text=text)]


# ──────────────────────────────────────────────────────────
# Path 1: Objective 4 calibrated-confidence result
# ──────────────────────────────────────────────────────────

def _source_fact(doc: Dict[str, Any]) -> tuple:
    source_type = doc.get("source_type", "unknown")
    triple = doc.get("mrav_triple") or {}
    hypothesis = triple.get("hypothesis", "Neutral")

    text = (
        f"{source_type} record ({doc.get('source', 'unknown source')}): credibility index "
        f"{doc.get('credibility_index', 0.0):.3f} [{doc.get('credibility_tier', 'unknown')}] "
        f"(authority {doc.get('authority', 0.0):.2f}, freshness {doc.get('freshness', 0.0):.2f}, "
        f"consensus {doc.get('consensus', 0.0):.2f}); M-RAV alignment: {hypothesis} "
        f"(consistency {triple.get('consistency_score', 0.0):.2f})"
    )
    if triple.get("explanation"):
        text += f" - {triple['explanation']}"
    if doc.get("domain_mismatch"):
        text += "; the record's domain does not match the offer's domain"

    if source_type in SCAM_SOURCE_TYPES:
        stance = "supports_fake"
    elif doc.get("domain_mismatch") or hypothesis == "Contradicts":
        stance = "supports_fake"
    elif hypothesis == "Confirms":
        stance = "supports_real"
    else:
        stance = "neutral"
    return text, stance


def build_from_confidence_result(
    pipeline_result: Dict[str, Any],
    entities: Optional[Dict[str, Any]] = None,
) -> EvidenceSheet:
    """Build the evidence sheet from a run_full_pipeline() result (Objective 4 path)."""
    final = pipeline_result.get("final_verification") or {}
    masses = pipeline_result.get("dst_combined_masses") or final.get("dst_masses") or {}
    obj2 = pipeline_result.get("obj2_credibility") or {}
    obj2_dst = pipeline_result.get("obj2_dst") or {}
    voice = pipeline_result.get("voice_verification") or {}
    entities = entities or (pipeline_result.get("objective1") or {}).get("extracted_entities")

    facts = _FactList()
    _add_claims(facts, entities)

    # Source credibility (Objective 2)
    for doc in obj2.get("evaluated_evidence", []):
        text, stance = _source_fact(doc)
        facts.add("S", "source", text, stance)

    discarded = obj2.get("discarded_evidence", [])
    if discarded:
        names = ", ".join(f"{d.get('source_type', 'unknown')} ({d.get('credibility_index', 0.0):.3f})" for d in discarded)
        facts.add(
            "S", "source",
            f"{len(discarded)} retrieved record(s) were discarded for low credibility "
            f"(credibility index below 0.3) and were not used as evidence: {names}.",
        )

    for flag in (obj2.get("credibility_evaluation") or {}).get("flags_triggered", []):
        severity, message = _flag_parts(flag)
        if severity in ("CRITICAL", "WARNING") and message:
            facts.add("F", "flag", f"{severity} flag: {message}", "supports_fake")

    # DST fusion
    real = float(masses.get("REAL", 0.0))
    fake = float(masses.get("FAKE", 0.0))
    uncertain = float(masses.get("UNCERTAIN", max(0.0, 1.0 - real - fake)))
    facts.add(
        "D", "dst",
        f"Dempster-Shafer fusion of all evidence gives m(Real) = {real:.3f}, "
        f"m(Fake) = {fake:.3f}, m(Uncertain) = {uncertain:.3f}.",
        "supports_real" if real > fake else ("supports_fake" if fake > real else "neutral"),
    )
    facts.add(
        "D", "dst",
        f"Belief-plausibility intervals: Real [{real:.3f}, {real + uncertain:.3f}], "
        f"Fake [{fake:.3f}, {fake + uncertain:.3f}].",
    )
    if obj2_dst.get("dst_status"):
        facts.add(
            "D", "dst",
            f"Evidence-only sufficiency check (before any voice call): {obj2_dst['dst_status']} "
            f"({obj2_dst.get('reason', 'no reason given')}).",
        )

    conflict = float(masses.get("conflict", final.get("conflict_factor_k", 0.0)) or 0.0)
    steps = len(masses.get("conflict_history", []) or [])
    conflict_text = f"Conflict factor K = {conflict:.3f} ({_conflict_level(conflict)})"
    if steps:
        conflict_text += f", measured over {steps} combination step(s)"
    conflict_id = facts.add("K", "conflict", conflict_text + ".")

    # Voice verification (Objective 3) as an input to DST
    status = voice.get("status")
    voice_mass = voice.get("voice_mass")
    if status == "COMPLETED" and voice_mass:
        v_real, v_fake = float(voice_mass.get("REAL", 0.0)), float(voice_mass.get("FAKE", 0.0))
        facts.add(
            "V", "voice",
            f"Voice verification call completed; its mass function is m(Real) = {v_real:.3f}, "
            f"m(Fake) = {v_fake:.3f}, m(Uncertain) = {float(voice_mass.get('UNCERTAIN', 0.0)):.3f}.",
            "supports_real" if v_real > v_fake else ("supports_fake" if v_fake > v_real else "neutral"),
        )
        for signal, contribution in (voice_mass.get("details") or {}).items():
            if contribution:
                stance = "supports_real" if contribution > 0 else "supports_fake"
                facts.add(
                    "V", "voice",
                    f"Call signal '{signal}' contributed {contribution:+.3f} to the voice score.",
                    "neutral" if signal == "company_reached" else stance,
                )
    elif status == "REQUIRED_NOT_RUN":
        facts.add(
            "V", "voice",
            "Voice verification was required because static evidence was insufficient, "
            "but no call result was available, so the verdict is provisional.",
        )
    elif status == "SKIPPED":
        facts.add("V", "voice", "Voice verification was skipped because the static evidence was sufficient.")

    # Objective 4 decision
    raw_verdict = final.get("final_verdict", "UNCERTAIN_NEEDS_VERIFICATION")
    confidence = final.get("calibrated_confidence")
    if confidence is not None:
        facts.add(
            "Q", "decision",
            f"Objective 4 calibrated confidence P(Real) = {float(confidence):.3f}, "
            f"giving the label {raw_verdict} (Real at 0.75 or above, Fake at 0.45 or below, "
            f"otherwise Unable to Verify).",
        )
    if final.get("override_applied") and final.get("override_reason"):
        facts.add("Q", "decision", f"Safety override applied: {final['override_reason']}", "supports_fake")

    return EvidenceSheet(
        offer_id=str(pipeline_result.get("offer_id", "UNKNOWN")),
        decision_path="CONFIDENCE_PATH",
        verdict=OBJ4_VERDICT_MAP.get(raw_verdict, "Unable to Verify"),
        raw_verdict=raw_verdict,
        confidence=float(confidence) if confidence is not None else None,
        provisional=bool(final.get("provisional", False)),
        override_reason=final.get("override_reason"),
        facts=facts.facts,
        contradictions=_find_contradictions(facts.facts, conflict_id if conflict > MODERATE_CONFLICT_THRESHOLD else None),
    )


# ──────────────────────────────────────────────────────────
# Path 2: Objective 3 voice-verification yes/no
# ──────────────────────────────────────────────────────────

def normalize_voice_decision(decision: Union[bool, str, None]) -> Optional[bool]:
    """Map yes/no style inputs to True / False / None (inconclusive)."""
    if decision is None or isinstance(decision, bool):
        return decision
    value = str(decision).strip().lower()
    if value in {"yes", "y", "true", "real", "verified", "legitimate"}:
        return True
    if value in {"no", "n", "false", "fake", "scam", "fraud"}:
        return False
    return None


def derive_voice_decision(voice_analysis: Dict[str, Any]) -> tuple:
    """Rule used when the voice agent did not report an explicit yes/no."""
    if voice_analysis.get("company_denied_role") is True:
        return False, "the company denied the role"
    if voice_analysis.get("suspicious_response") is True:
        return False, "the call response was suspicious"
    if voice_analysis.get("payment_requested") is True:
        return False, "a payment was requested"
    if voice_analysis.get("company_reached") is True and voice_analysis.get("role_actively_hiring") is True:
        return True, "the company was reached and confirmed the role"
    return None, "the call did not establish whether the role is genuine"


def build_from_voice_result(
    voice_analysis: Dict[str, Any],
    voice_decision: Union[bool, str, None] = None,
    entities: Optional[Dict[str, Any]] = None,
    offer_id: str = "UNKNOWN",
) -> EvidenceSheet:
    """Build the evidence sheet from an Objective 3 voice yes/no result."""
    voice_analysis = voice_analysis or {}
    facts = _FactList()
    _add_claims(facts, entities)

    number_source = voice_analysis.get("number_source")
    if number_source:
        text = f"The company phone number was sourced independently from {number_source}, not from the offer"
        if voice_analysis.get("source_confidence") is not None:
            text += f" (source confidence {float(voice_analysis['source_confidence']):.2f})"
        facts.add("V", "voice", text + ".")

    for signal, (text_true, text_false, stance_true, stance_false) in VOICE_SIGNALS.items():
        value = voice_analysis.get(signal)
        if value is True:
            facts.add("V", "voice", text_true + ".", stance_true)
        elif value is False and text_false:
            facts.add("V", "voice", text_false + ".", stance_false)
        elif value is None and signal in VOICE_SIGNAL_TOPICS:
            facts.add("V", "voice", f"Not established during the call: {VOICE_SIGNAL_TOPICS[signal]}.")

    if voice_analysis.get("extraction_confidence") is not None:
        facts.add(
            "V", "voice",
            f"Confidence of the transcript signal extraction: {float(voice_analysis['extraction_confidence']):.2f}.",
        )
    if voice_analysis.get("response_summary"):
        facts.add("V", "voice", f"Call summary from the transcript: \"{voice_analysis['response_summary']}\"")

    decision = normalize_voice_decision(voice_decision)
    if voice_decision is None:
        decision, rule = derive_voice_decision(voice_analysis)
        origin = f"derived from the call signals because {rule}"
    else:
        origin = "as reported by the voice agent"

    raw_verdict = {True: "VOICE_YES", False: "VOICE_NO", None: "VOICE_INCONCLUSIVE"}[decision]
    facts.add(
        "Q", "decision",
        f"Objective 3 voice decision: {'YES' if decision else ('NO' if decision is False else 'INCONCLUSIVE')} ({origin}).",
    )

    return EvidenceSheet(
        offer_id=str(offer_id),
        decision_path="VOICE_PATH",
        verdict={True: "Real", False: "Fake", None: "Unable to Verify"}[decision],
        raw_verdict=raw_verdict,
        facts=facts.facts,
        contradictions=_find_contradictions(facts.facts),
    )
