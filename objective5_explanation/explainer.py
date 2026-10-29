"""
explainer.py — Objective 5: LLM-based Explanation Module.

Converts the verdict from either the calibrated-confidence path (Objective 4)
or the voice-verification path (Objective 3) into a readable report that
surfaces the key evidence, contradictions and findings behind the
Real / Fake / Unable to Verify decision.

Grounding (explainable AI, not free generation):
    1. evidence_builder turns the pipeline outputs into numbered facts
       (source credibility scores, DST masses, conflict K, flags, voice findings).
    2. The LLM (Groq, or OpenAI GPT as fallback) only rephrases those facts and
       must cite fact IDs for every statement. It never chooses the verdict.
    3. Every LLM statement is checked: cited IDs must exist and every number
       must appear in the evidence sheet. Ungrounded statements are dropped and
       logged in grounding_issues.
    4. Without an API key, or if the LLM output fails the checks, a deterministic
       template report is produced from the same facts.

Public API:
    explain_confidence_result(pipeline_result, entities=None, client=None) -> ExplanationReport
    explain_voice_result(voice_analysis, voice_decision=None, entities=None, offer_id="UNKNOWN", client=None) -> ExplanationReport
    explain_evidence_sheet(sheet, client=None, use_llm=True) -> ExplanationReport
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from . import config
from .evidence_builder import build_from_confidence_result, build_from_voice_result
from .schemas import EvidenceSheet, ExplanationReport, ReportPoint

RECOMMENDATIONS = {
    "Real": (
        "The available evidence supports this offer. Still apply normal caution: never pay any fee, "
        "and confirm joining details only through the company's official website or HR contact."
    ),
    "Fake": (
        "Treat this offer as fraudulent. Do not pay any fee or share personal, banking or identity "
        "documents, stop contact through the channels given in the offer, and report it to the "
        "job platform or cybercrime portal."
    ),
    "Unable to Verify": (
        "The evidence is insufficient or conflicting, so no decision was forced. Contact the company "
        "directly through its official website or phone number, and do not pay or share documents "
        "until the offer is independently confirmed."
    ),
}

SYSTEM_PROMPT = """You are the explanation module of a fake job offer detection system.
You receive an EVIDENCE SHEET produced by the system's own pipeline. Write a clear explanation of the
verdict for a non-technical job seeker.

STRICT RULES:
1. Use ONLY the facts in the evidence sheet. Do not add outside knowledge, guesses or new evidence.
2. The verdict is already decided. Never change it, question it or suggest a different one.
3. Every statement must cite the fact IDs it is based on (e.g. ["S1", "K1"]).
4. Only use numbers that appear in the evidence sheet, written the same way.
5. Explain technical terms briefly (e.g. conflict factor K = how much the sources disagree).
6. Keep each statement to one or two sentences.

Respond with JSON only, in exactly this shape:
{
  "summary": "2-3 sentences explaining why the verdict was reached",
  "summary_cites": ["fact IDs"],
  "key_evidence": [{"text": "...", "cites": ["fact IDs"]}],
  "contradictions": [{"text": "...", "cites": ["fact IDs"]}]
}
List at most 5 key_evidence items, most decisive first. Use an empty contradictions list if the sheet has none."""

_NUMBER_RE = re.compile(r"(?<![A-Za-z])-?\d+(?:\.\d+)?%?")
_FACT_ID_RE = re.compile(r"\b[A-Z]\d+\b")
_TOLERANCE = 0.0051


# ──────────────────────────────────────────────────────────
# LLM client
# ──────────────────────────────────────────────────────────

def get_llm_client() -> Optional[Tuple[Any, str, str]]:
    """Return (client, model, provider) for Groq, else OpenAI, else None when no key is set."""
    groq_key = config.GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
    openai_key = config.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
    if not groq_key and not openai_key:
        return None

    from openai import OpenAI

    if groq_key:
        client = OpenAI(base_url=config.GROQ_BASE_URL, api_key=groq_key, timeout=config.REQUEST_TIMEOUT_SECONDS)
        return client, os.getenv("GROQ_MODEL", config.GROQ_MODEL), "groq"
    client = OpenAI(api_key=openai_key, timeout=config.REQUEST_TIMEOUT_SECONDS)
    return client, os.getenv("OPENAI_MODEL", config.OPENAI_MODEL), "openai"


def _call_llm(sheet: EvidenceSheet, client: Any, model: str) -> Dict[str, Any]:
    sheet_json = sheet.model_dump(
        include={"verdict", "decision_path", "confidence", "provisional", "facts", "contradictions"}
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "EVIDENCE SHEET:\n" + json.dumps(sheet_json, indent=2)},
        ],
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or ""
    match = re.search(r"\{.*\}", content, re.DOTALL)
    return json.loads(match.group(0) if match else content)


# ──────────────────────────────────────────────────────────
# Grounding checks
# ──────────────────────────────────────────────────────────

def _allowed_numbers(sheet: EvidenceSheet) -> Tuple[List[float], List[float]]:
    """Return (raw values, percent values) that the explanation may mention."""
    texts = [f.text for f in sheet.facts] + [c.text for c in sheet.contradictions]
    values = []
    for text in texts:
        for token in _NUMBER_RE.findall(_FACT_ID_RE.sub(" ", text)):
            values.append(float(token.rstrip("%")))
    if sheet.confidence is not None:
        values.append(sheet.confidence)
    # Fractions may also be written as percentages (0.123 -> 12.3%)
    return values, [v * 100 for v in values if abs(v) <= 1.0]


def _ungrounded_numbers(text: str, allowed: Tuple[List[float], List[float]]) -> List[str]:
    raw_values, percent_values = allowed
    bad = []
    for token in _NUMBER_RE.findall(_FACT_ID_RE.sub(" ", text)):
        value = float(token.rstrip("%"))
        if token.endswith("%"):
            grounded = any(abs(value - a) <= 0.51 for a in percent_values)
        else:
            grounded = any(abs(value - a) <= _TOLERANCE for a in raw_values)
        if not grounded:
            bad.append(token)
    return bad


def _check_point(text: str, cites: Any, sheet: EvidenceSheet, allowed: Tuple[List[float], List[float]]) -> Tuple[Optional[ReportPoint], Optional[str]]:
    text = str(text or "").strip()
    cites = [str(c).strip("[] ") for c in (cites or []) if str(c).strip()]
    if not text:
        return None, None

    valid_ids = sheet.fact_ids()
    unknown = [c for c in cites if c not in valid_ids]
    if not cites:
        return None, f"No citation: {text!r}"
    if unknown:
        return None, f"Unknown fact ID(s) {unknown}: {text!r}"
    bad_numbers = _ungrounded_numbers(text, allowed)
    if bad_numbers:
        return None, f"Number(s) {bad_numbers} not in the evidence sheet: {text!r}"
    return ReportPoint(text=text, cites=cites), None


def _ground_llm_output(raw: Dict[str, Any], sheet: EvidenceSheet) -> Tuple[Optional[str], List[ReportPoint], List[ReportPoint], List[str]]:
    allowed = _allowed_numbers(sheet)
    issues: List[str] = []

    summary_point, issue = _check_point(raw.get("summary"), raw.get("summary_cites"), sheet, allowed)
    if issue:
        issues.append("summary: " + issue)

    def _points(key: str) -> List[ReportPoint]:
        points = []
        for item in raw.get(key) or []:
            if not isinstance(item, dict):
                continue
            point, item_issue = _check_point(item.get("text"), item.get("cites"), sheet, allowed)
            if point:
                points.append(point)
            elif item_issue:
                issues.append(f"{key}: {item_issue}")
        return points

    key_evidence = _points("key_evidence")[:5]
    contradictions = _points("contradictions")
    return (summary_point.text if summary_point else None), key_evidence, contradictions, issues


# ──────────────────────────────────────────────────────────
# Deterministic template (no LLM / fallback)
# ──────────────────────────────────────────────────────────

def _template_summary(sheet: EvidenceSheet) -> str:
    if sheet.decision_path == "VOICE_PATH":
        decision = sheet.facts[-1].text if sheet.facts else ""
        return f"The offer is classified as {sheet.verdict} based on the voice verification call. {decision}"

    text = f"The offer is classified as {sheet.verdict}"
    if sheet.confidence is not None:
        text += f" with a calibrated confidence P(Real) of {sheet.confidence:.3f}"
    text += ", after fusing the credibility-scored evidence with Dempster-Shafer theory."
    if sheet.override_reason:
        text += f" A safety override was applied: {sheet.override_reason}"
    if sheet.provisional:
        text += " The verdict is provisional because the required voice verification was not run."
    return text


def _template_report_points(sheet: EvidenceSheet) -> Tuple[List[ReportPoint], List[ReportPoint]]:
    wanted = {
        "Real": {"supports_real"},
        "Fake": {"supports_fake"},
        "Unable to Verify": {"supports_real", "supports_fake"},
    }[sheet.verdict]

    decisive = [f for f in sheet.facts if f.stance in wanted and f.category in {"source", "flag", "voice", "decision"}]
    context = [f for f in sheet.facts if f.category in {"dst", "conflict"}][:2]
    key_evidence = [ReportPoint(text=f.text, cites=[f.fact_id]) for f in (decisive[:4] + context)][:6]

    contradictions = [ReportPoint(text=c.text, cites=c.fact_ids) for c in sheet.contradictions]
    return key_evidence, contradictions


# ──────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────

def explain_evidence_sheet(
    sheet: EvidenceSheet,
    client: Optional[Tuple[Any, str, str]] = None,
    use_llm: bool = True,
) -> ExplanationReport:
    """
    Generate the explanation report for an evidence sheet.

    Args:
        sheet:   EvidenceSheet from evidence_builder
        client:  Optional (client, model, provider) tuple; defaults to get_llm_client()
        use_llm: False forces the deterministic template report
    """
    issues: List[str] = []
    llm = (client or get_llm_client()) if use_llm else None
    template_evidence, template_contradictions = _template_report_points(sheet)

    common = dict(
        offer_id=sheet.offer_id,
        decision_path=sheet.decision_path,
        verdict=sheet.verdict,
        confidence=sheet.confidence,
        provisional=sheet.provisional,
        recommendation=RECOMMENDATIONS[sheet.verdict],
        evidence_sheet=sheet,
    )

    if llm:
        llm_client, model, provider = llm
        try:
            raw = _call_llm(sheet, llm_client, model)
            summary, key_evidence, contradictions, issues = _ground_llm_output(raw, sheet)
            if summary and key_evidence:
                return ExplanationReport(
                    summary=summary,
                    key_evidence=key_evidence,
                    # Keep the pipeline-detected contradictions if the LLM omitted them
                    contradictions=contradictions or template_contradictions,
                    generated_by="llm",
                    model=f"{provider}:{model}",
                    grounding_issues=issues,
                    **common,
                )
            issues.append("LLM output had no grounded summary or key evidence; used template instead.")
        except Exception as e:
            issues.append(f"LLM call failed ({type(e).__name__}: {e}); used template instead.")

    return ExplanationReport(
        summary=_template_summary(sheet),
        key_evidence=template_evidence,
        contradictions=template_contradictions,
        generated_by="template",
        grounding_issues=issues,
        **common,
    )


def explain_confidence_result(
    pipeline_result: Dict[str, Any],
    entities: Optional[Dict[str, Any]] = None,
    client: Optional[Tuple[Any, str, str]] = None,
    use_llm: bool = True,
) -> ExplanationReport:
    """Explain a verdict from the calibrated-confidence path (Objective 4 / run_full_pipeline output)."""
    sheet = build_from_confidence_result(pipeline_result, entities)
    return explain_evidence_sheet(sheet, client=client, use_llm=use_llm)


def explain_voice_result(
    voice_analysis: Dict[str, Any],
    voice_decision: Union[bool, str, None] = None,
    entities: Optional[Dict[str, Any]] = None,
    offer_id: str = "UNKNOWN",
    client: Optional[Tuple[Any, str, str]] = None,
    use_llm: bool = True,
) -> ExplanationReport:
    """Explain a verdict from the voice-verification path (Objective 3 yes/no)."""
    sheet = build_from_voice_result(voice_analysis, voice_decision, entities, offer_id)
    return explain_evidence_sheet(sheet, client=client, use_llm=use_llm)

