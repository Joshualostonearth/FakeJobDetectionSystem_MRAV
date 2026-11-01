"""
entity_extractor.py - entities the voice agent needs (company, role, recruiter contact),
extracted with regular expressions. No LLM, no network.

The M4 verdict never depends on this module's LLM path. The optional Ollama
extractor (models.extract_entities_from_text) is used only by
`enrich_with_llm`, which the bridge calls when a phone call is requested and
Ollama is reachable; it only fills fields the regexes left empty.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

ENTITY_KEYS = ("company_name", "job_role", "recruiter_name", "recruiter_email",
               "recruiter_phone", "domain_or_website", "location")

FREE_MAIL = {"gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "live.com", "aol.com",
             "icloud.com", "protonmail.com", "proton.me", "mail.com", "gmx.com", "yandex.com",
             "rediffmail.com", "ymail.com", "msn.com", "zoho.com"}

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")
URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>\"')]+", re.IGNORECASE)
BARE_DOMAIN_RE = re.compile(
    r"\b(?:[a-z0-9-]+\.)+(?:com|org|net|edu|gov|io|co|in|uk|us|ca|au|biz|info|ai|tech|jobs)\b",
    re.IGNORECASE)
# 8-15 digits with optional +, spaces, dots, dashes, parentheses
PHONE_RE = re.compile(r"(?<![\w])(\+?\(?\d[\d\s().-]{6,18}\d)(?![\w])")

# "Label: value" lines
_LABELS = {
    "company_name": r"company(?:\s+name)?|organi[sz]ation|employer|firm|institution|hiring\s+company",
    "job_role": r"job\s+title|position(?:\s+title)?|role|designation|job\s+role|vacancy|post|title",
    "recruiter_name": r"recruiter(?:\s+name)?|contact(?:\s+person)?|hiring\s+manager|hr\s+contact|signed\s+by",
    "location": r"location|job\s+location|work\s+location|city|address|place\s+of\s+work",
}
_LABEL_RES = {k: re.compile(rf"^\s*(?:{v})\s*[:\-–]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
              for k, v in _LABELS.items()}

ORG_SUFFIX = (r"(?:Inc\.?|Incorporated|Ltd\.?|Limited|LLC|LLP|PLC|Corp\.?|Corporation|Company|Co\.|"
              r"Pvt\.?(?:\s+Ltd\.?)?|GmbH|S\.A\.|Technologies|Technology|Solutions|Systems|Services|"
              r"Group|Holdings|Consulting|Labs|Software|College|University|Institute|School|Academy|"
              r"Hospital|Bank|Agency|Enterprises|International|Global|Partners|Foundation)")
ORG_RE = re.compile(rf"\b((?:[A-Z][\w&'.-]*\s+){{1,4}}(?i:{ORG_SUFFIX}))(?![\w])")
ORG_CONTEXT_RE = re.compile(
    r"\b(?:join|joining|welcome\s+to|on\s+behalf\s+of|at|with|from)\s+((?:[A-Z][\w&'.-]*\s?){1,5})",
)

ROLE_WORDS = (r"(?:Engineer|Developer|Programmer|Manager|Intern|Internship|Trainee|Analyst|Assistant|"
              r"Executive|Officer|Representative|Agent|Designer|Consultant|Specialist|Coordinator|Clerk|"
              r"Driver|Teacher|Tutor|Nurse|Technician|Administrator|Associate|Accountant|Operator|Writer|"
              r"Editor|Scientist|Architect|Lead|Director|Supervisor|Advisor|Recruiter|Receptionist|"
              r"Secretary|Cashier|Salesperson|Marketer|Researcher|Lecturer|Professor|Instructor|Auditor|"
              r"Planner|Tester|Support|Head|Partner|Worker|Staff|Handler|Packer|Chef|Cook|Guard)")
ROLE_PHRASE_RE = re.compile(rf"\b((?:[A-Z][A-Za-z/&+-]*\s+){{0,3}}{ROLE_WORDS}s?)\b")
ROLE_CONTEXT_RE = re.compile(
    rf"\b(?:position\s+of|role\s+of|post\s+of|as\s+(?:an?|the)|hiring\s+(?:an?|for)?|looking\s+for\s+(?:an?)?)\s+"
    rf"((?:[A-Za-z/&+-]+\s+){{0,3}}?{ROLE_WORDS}s?)\b",
    re.IGNORECASE)

# Vocabulary that marks text as being about a job; used for the title-line fallback
JOB_CONTEXT_RE = re.compile(
    r"\b(?:hiring|hire|vacanc\w*|position|apply|applicants?|applications?|salary|wages?|stipend|"
    r"responsibilit\w*|requirements?|qualifications?|jobs?|careers?|recruit\w*|candidates?|"
    r"employment|internships?|full[- ]time|part[- ]time|work\s+from\s+home|benefits)\b",
    re.IGNORECASE)
MIN_JOB_CONTEXT = 2

SIGNOFF_RE = re.compile(
    r"(?:regards|sincerely|best|thanks|thank\s+you|yours\s+(?:truly|faithfully))\s*,?\s*\n+\s*"
    r"([A-Z][a-z]+(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-z]+){0,2})\s*$",
    re.IGNORECASE | re.MULTILINE)

SOURCE_TAG_RE = re.compile(r"\[SOURCE: [A-Z_]+\]")


def strip_source_tags(text: str) -> str:
    """Remove the [SOURCE: ...] markers merge_text_and_ocr adds."""
    return SOURCE_TAG_RE.sub("", text or "").strip()


def _clean_value(v: Optional[str], max_words: int = 8) -> Optional[str]:
    if not v:
        return None
    v = re.split(r"[|;\n]", v)[0]
    v = re.sub(r"\s+", " ", v).strip(" .,:-–")
    if not v or len(v.split()) > max_words:
        return None
    return v


def _first(regex: re.Pattern, text: str, max_words: int = 8) -> Optional[str]:
    for m in regex.finditer(text):
        v = _clean_value(m.group(1), max_words)
        if v:
            return v
    return None


def _phone(text: str) -> Optional[str]:
    for m in PHONE_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group(1))
        if 8 <= len(digits) <= 15 and not re.fullmatch(r"(19|20)\d{6}", digits):  # skip dates like 20240101
            return re.sub(r"\s+", " ", m.group(1)).strip()
    return None


def _domain(text: str, email: Optional[str]) -> Optional[str]:
    m = URL_RE.search(text)
    if m:
        return m.group(0).rstrip(".,")
    for m in BARE_DOMAIN_RE.finditer(text):
        dom = m.group(0).lower()
        if "@" + dom not in text.lower() and dom not in FREE_MAIL:
            return dom
    if email:
        dom = email.split("@", 1)[1].lower()
        if dom not in FREE_MAIL:
            return dom
    return None


def extract_entities(text: str, fields: Optional[Dict[str, Any]] = None) -> Dict[str, Optional[str]]:
    """
    Regex entity extraction. `fields` (title, location, company_name, ...) from the UI win over
    anything found in the text.
    """
    text = strip_source_tags(text)
    fields = fields or {}
    ent: Dict[str, Optional[str]] = {k: None for k in ENTITY_KEYS}

    for key, rx in _LABEL_RES.items():
        ent[key] = _first(rx, text)

    emails = EMAIL_RE.findall(text)
    ent["recruiter_email"] = emails[0] if emails else None
    ent["recruiter_phone"] = _phone(text)
    ent["domain_or_website"] = _domain(text, ent["recruiter_email"])

    if not ent["company_name"]:
        ent["company_name"] = _first(ORG_RE, text, 6)
    if not ent["company_name"]:
        m = ORG_CONTEXT_RE.search(text)
        cand = _clean_value(m.group(1), 5) if m else None
        # ignore generic capitalised words ("at The", "with Us")
        if cand and cand.split()[0].lower() not in {"the", "us", "our", "we", "you", "your", "a", "an"}:
            ent["company_name"] = cand
    if not ent["job_role"]:
        ent["job_role"] = _first(ROLE_CONTEXT_RE, text, 6) or _first(ROLE_PHRASE_RE, text, 5)
    if not ent["job_role"] and len(JOB_CONTEXT_RE.findall(text)) >= MIN_JOB_CONTEXT:
        # Posters usually lead with the job title (often all caps or an unlisted role word):
        # take a short first line that is not a sentence or a "Label: value" line
        first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        if ":" not in first and not re.search(r"[.!?]$", first):
            ent["job_role"] = _clean_value(first, 8)
    if not ent["recruiter_name"]:
        ent["recruiter_name"] = _first(SIGNOFF_RE, text, 4)

    overrides = {"company_name": fields.get("company_name") or fields.get("company"),
                 "job_role": fields.get("title") or fields.get("job_role"),
                 "location": fields.get("location"),
                 "recruiter_name": fields.get("recruiter_name"),
                 "recruiter_email": fields.get("recruiter_email"),
                 "recruiter_phone": fields.get("recruiter_phone"),
                 "domain_or_website": fields.get("domain_or_website")}
    for k, v in overrides.items():
        if isinstance(v, str) and v.strip():
            ent[k] = v.strip()
    return ent


def has_job_offer(entities: Dict[str, Optional[str]]) -> bool:
    """Ported idea from integrated2 (a3c42c3): without a company or a role there is nothing to verify."""
    return bool((entities.get("company_name") or "").strip() or (entities.get("job_role") or "").strip())


def enrich_with_llm(entities: Dict[str, Optional[str]], text: str) -> Dict[str, Any]:
    """
    Fill empty fields with the existing Ollama extractor. Only call this when a phone call is
    requested and Ollama is reachable; regex values are never overwritten.

    Returns {"entities": ..., "filled": [keys filled by the LLM], "error": str|None}.
    """
    filled: List[str] = []
    try:
        from models import extract_entities_from_text  # imports langchain_ollama; voice-only dependency
        llm = extract_entities_from_text(strip_source_tags(text)).model_dump()
    except Exception as e:  # missing langchain, Ollama down, bad JSON ...
        return {"entities": entities, "filled": filled, "error": str(e)}
    out = dict(entities)
    for k, v in llm.items():
        if k in out and not out.get(k) and isinstance(v, str) and v.strip():
            out[k] = v.strip()
            filled.append(k)
    return {"entities": out, "filled": filled, "error": None}
