"""
red_flags.py - rule-based scam red flags shown next to the M4 verdict.

The rules never change M4's verdict or P(fraud). A hit raises the risk level one step
(LOW -> MEDIUM: call recommended; MEDIUM -> HIGH: call starts automatically), see
pipeline_bridge.RED_FLAG_RAISE.
M4's validated test metrics therefore still describe the verdict exactly.

Each rule looks for something that legitimate employers essentially never do (ask the
candidate for money, ask them to deposit a cheque or buy equipment, move the chat to
WhatsApp/Telegram, ask for bank or ID details up front). The hit rates on EMSCAD are in
results/red_flags_report.md.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from entity_extractor import FREE_MAIL

# "no registration fee", "we never ask for payment" must not count
_NEGATION = re.compile(r"\b(?:no|never|not|without|free\s+of|don'?t|do\s+not|won'?t|will\s+not)\b[^.\n]{0,25}$",
                       re.IGNORECASE)

_FEE_RES = [
    re.compile(r"\b(?:registration|training|processing|enrol+ment|security|onboarding|internship|joining|"
               r"kit|refundable|verification|activation)\s+(?:fees?|charges?|deposit)\b", re.IGNORECASE),
    # an instruction to pay, not a salary line ("pay is $15/hr", "Pay: $8")
    re.compile(r"\b(?:to\s+pay|must\s+pay|please\s+pay|pay\s+(?:a|an|the|us|rs\.?|inr|₹|\d)|make\s+(?:a\s+)?payment|"
               r"payment\s+of|deposit|transfer|remit)\b[^.\n]{0,40}\b(?:fees?|charges?)\b", re.IGNORECASE),
]
_CHEQUE_RES = [
    re.compile(r"\bmobile\s+deposit\b", re.IGNORECASE),
    re.compile(r"\b(?:deposit|cash)\s+(?:the|this|a|your)?\s*(?:check|cheque)\b", re.IGNORECASE),
    re.compile(r"\b(?:check|cheque)\b[^.\n]{0,60}\b(?:deposit|mobile\s+app|bank\s+app)\b", re.IGNORECASE),
    re.compile(r"\b(?:buy|purchase|get|send)\b[^.\n]{0,40}\bgift\s*cards?\b", re.IGNORECASE),  # not "gift cards" as perks
    re.compile(r"\b(?:buy|gather\s+pricing|get\s+(?:a\s+)?quotes?)\b[^.\n]{0,60}"
               r"\b(?:office\s+supplies|equipment|webcam|projector|printer|laptop)\b", re.IGNORECASE),
]
# contact instructions only; a company profile may simply name WhatsApp or Hangouts
_CHAT_APP = r"(?:whats\s?app|telegram|wickr|google\s+hangouts|hangouts)"
_OFF_PLATFORM_RE = re.compile(
    rf"\b(?:contact|message|text|reach|chat|add|interview(?:ed)?|connect|join|ping)\b[^.\n]{{0,40}}"
    rf"\b(?:on|via|through|using|in)\s+{_CHAT_APP}\b"
    rf"|\b{_CHAT_APP}\s*(?:number|no\.?|id|:)\s*\+?\d", re.IGNORECASE)
_PERSONAL_DATA_RE = re.compile(
    r"\b(?:send|provide|submit|share|forward)\b[^.\n]{0,40}\byour\s+"
    r"(?:bank\s+(?:account|details)|account\s+(?:number|details)|social\s+security\s+number|ssn|"
    r"(?:scanned\s+)?(?:copy\s+of\s+(?:your\s+)?)?(?:passport|driver'?s\s+licen[cs]e|national\s+id|aadhaa?r|"
    r"pan\s+card|id\b))",
    re.IGNORECASE)
_INSTITUTION_RE = re.compile(
    r"\b(?:university|college|career\s+(?:services|center|centre)|department\s+of|"
    r"(?:inc|ltd|llc|corp|pvt)\b\.?)", re.IGNORECASE)


def _first_hit(regexes, text: str) -> Optional[str]:
    for rx in regexes:
        for m in rx.finditer(text):
            if _NEGATION.search(text[max(0, m.start() - 40):m.start()]):
                continue
            return re.sub(r"\s+", " ", m.group(0)).strip()
    return None


def find_red_flags(text: str, entities: Optional[Dict[str, Optional[str]]] = None) -> List[Dict[str, str]]:
    """
    Return [{"id", "label", "evidence"}] for each rule that fires on the offer text.
    `entities` (from entity_extractor) supplies the recruiter email for the free-mail rule.
    """
    text = text or ""
    entities = entities or {}
    flags: List[Dict[str, str]] = []

    def add(rule_id: str, label: str, evidence: Optional[str]):
        if evidence:
            flags.append({"id": rule_id, "label": label, "evidence": evidence})

    add("upfront_fee", "Asks the candidate to pay money (fee, deposit or charge)", _first_hit(_FEE_RES, text))
    add("cheque_or_purchase", "Mentions depositing a cheque, gift cards or buying equipment for the job",
        _first_hit(_CHEQUE_RES, text))
    m = _OFF_PLATFORM_RE.search(text)
    add("off_platform_chat", "Moves the conversation to WhatsApp, Telegram or a similar chat app",
        m.group(0) if m else None)
    add("personal_data", "Asks for bank or identity details up front", _first_hit([_PERSONAL_DATA_RE], text))

    email = (entities.get("recruiter_email") or "").lower()
    if "@" in email and email.split("@", 1)[1] in FREE_MAIL:
        inst = _INSTITUTION_RE.search(text)
        if inst:
            add("free_mail_institution",
                "A university or company recruiting from a free email address",
                f"{email} (text mentions '{inst.group(0).strip()}')")
    return flags
