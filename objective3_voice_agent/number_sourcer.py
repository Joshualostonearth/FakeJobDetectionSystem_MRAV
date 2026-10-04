"""
number_sourcer.py — Step 1: Independent phone number sourcing.

HARD RULE: The phone number from the job posting entity dict is NEVER used here.
This module independently sources official company contact numbers from external APIs.

Cascade (tried in order, stops at first result with confidence >= threshold):
  1. Google Places API  — confidence 0.9 (OPERATIONAL) / 0.5
  2. OpenCorporates API — confidence 0.85
  3. SerpAPI            — confidence 0.5 (regex-extracted from search results)
"""

import logging
import os
import re
from typing import Optional

import requests
from dotenv import load_dotenv

from schemas import SourcedNumber

load_dotenv(override=False)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# API keys (read from env — never hardcoded)
# ──────────────────────────────────────────────────────────
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")
OPENCORPORATES_API_KEY = os.getenv("OPENCORPORATES_API_KEY", "")
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")

PHONE_REGEX = re.compile(r"\+?[\d\s\-\(\)]{10,15}")


# ──────────────────────────────────────────────────────────
# Tier 1: Google Places
# ──────────────────────────────────────────────────────────

def _source_google_places(company_name: str, location: str) -> Optional[SourcedNumber]:
    """Query Google Places API for the company's official phone number."""
    if not GOOGLE_PLACES_API_KEY:
        logger.debug("[NumberSourcer] Google Places skipped — no API key")
        return None

    try:
        query = f"{company_name} {location} headquarters"
        # Step 1: Find Place ID
        find_url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
        find_resp = requests.get(
            find_url,
            params={
                "input": query,
                "inputtype": "textquery",
                "fields": "place_id,name",
                "key": GOOGLE_PLACES_API_KEY,
            },
            timeout=10,
        )
        find_resp.raise_for_status()
        find_data = find_resp.json()

        candidates = find_data.get("candidates", [])
        if not candidates:
            logger.debug("[NumberSourcer] Google Places: no candidates found")
            return None

        place_id = candidates[0]["place_id"]
        matched_name = candidates[0].get("name", company_name)

        # Step 2: Get Place Details including phone
        details_url = "https://maps.googleapis.com/maps/api/place/details/json"
        details_resp = requests.get(
            details_url,
            params={
                "place_id": place_id,
                "fields": "formatted_phone_number,business_status,name",
                "key": GOOGLE_PLACES_API_KEY,
            },
            timeout=10,
        )
        details_resp.raise_for_status()
        details = details_resp.json().get("result", {})

        phone = details.get("formatted_phone_number")
        if not phone:
            logger.debug("[NumberSourcer] Google Places: no phone in result")
            return None

        # Normalise to E.164-ish (remove spaces and dashes)
        phone_clean = re.sub(r"[\s\-\(\)]", "", phone)
        if not phone_clean.startswith("+"):
            phone_clean = "+" + phone_clean

        business_status = details.get("business_status", "")
        confidence = 0.9 if business_status == "OPERATIONAL" else 0.5

        logger.info(
            f"[NumberSourcer] Google Places: {phone_clean} "
            f"(confidence={confidence}, status={business_status})"
        )
        return SourcedNumber(
            phone_number=phone_clean,
            source="google_places",
            company_name_matched=matched_name,
            confidence=confidence,
        )

    except Exception as e:
        logger.warning(f"[NumberSourcer] Google Places error: {e}")
        return None


# ──────────────────────────────────────────────────────────
# Tier 2: OpenCorporates
# ──────────────────────────────────────────────────────────

def _source_opencorporates(company_name: str, location: str) -> Optional[SourcedNumber]:
    """Query OpenCorporates for the company's registered contact number."""
    if not OPENCORPORATES_API_KEY:
        logger.debug("[NumberSourcer] OpenCorporates skipped — no API key")
        return None

    try:
        # Infer jurisdiction from location
        loc_lower = location.lower()
        if any(x in loc_lower for x in ["india", "bangalore", "mumbai", "delhi", "hyderabad"]):
            jurisdiction = "in"
        elif any(x in loc_lower for x in ["uk", "london", "england"]):
            jurisdiction = "gb"
        elif any(x in loc_lower for x in ["us", "usa", "new york", "california"]):
            jurisdiction = "us_de"  # Delaware (most common US jurisdiction)
        else:
            jurisdiction = None

        params: dict = {
            "q": company_name,
            "api_token": OPENCORPORATES_API_KEY,
        }
        if jurisdiction:
            params["jurisdiction_code"] = jurisdiction

        resp = requests.get(
            "https://api.opencorporates.com/v0.4/companies/search",
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        companies = data.get("results", {}).get("companies", [])
        if not companies:
            logger.debug("[NumberSourcer] OpenCorporates: no results")
            return None

        # Take the first result — look for a phone in agent_address or registered_address
        company = companies[0]["company"]
        matched_name = company.get("name", company_name)

        # OpenCorporates doesn't always return phone — look in officers/agents
        # Return None if no phone found (do not guess)
        phone = company.get("telephone_number") or company.get("phone")
        if not phone:
            logger.debug("[NumberSourcer] OpenCorporates: company found but no phone field")
            return None

        phone_clean = re.sub(r"[\s\-\(\)]", "", str(phone))
        if not phone_clean.startswith("+"):
            phone_clean = "+" + phone_clean

        logger.info(f"[NumberSourcer] OpenCorporates: {phone_clean} (confidence=0.85)")
        return SourcedNumber(
            phone_number=phone_clean,
            source="opencorporates",
            company_name_matched=matched_name,
            confidence=0.85,
        )

    except Exception as e:
        logger.warning(f"[NumberSourcer] OpenCorporates error: {e}")
        return None


# ──────────────────────────────────────────────────────────
# Tier 3: SerpAPI
# ──────────────────────────────────────────────────────────

def _source_serpapi(company_name: str) -> Optional[SourcedNumber]:
    """Query SerpAPI and extract phone number via regex from search results."""
    if not SERPAPI_KEY:
        logger.debug("[NumberSourcer] SerpAPI skipped — no API key")
        return None

    try:
        query = f'"{company_name}" official HR phone site:linkedin.com'
        resp = requests.get(
            "https://serpapi.com/search",
            params={
                "q": query,
                "api_key": SERPAPI_KEY,
                "engine": "google",
                "num": 5,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        # Scan organic results for any phone pattern
        organic = data.get("organic_results", [])
        for result in organic:
            text = result.get("snippet", "") + " " + result.get("title", "")
            matches = PHONE_REGEX.findall(text)
            for match in matches:
                clean = re.sub(r"[\s\-\(\)]", "", match)
                if len(clean.lstrip("+")) >= 10:
                    if not clean.startswith("+"):
                        clean = "+" + clean
                    logger.info(f"[NumberSourcer] SerpAPI: {clean} (confidence=0.5)")
                    return SourcedNumber(
                        phone_number=clean,
                        source="serpapi",
                        company_name_matched=company_name,
                        confidence=0.5,
                    )

        logger.debug("[NumberSourcer] SerpAPI: no phone found in results")
        return None

    except Exception as e:
        logger.warning(f"[NumberSourcer] SerpAPI error: {e}")
        return None


# ──────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────

def source_official_number(
    company_name: str,
    location: str,
    config: dict,
) -> Optional[SourcedNumber]:
    """
    Source an official phone number for a company using the cascade:
      Google Places → OpenCorporates → SerpAPI

    Returns the first result whose confidence >= config["number_source_min_confidence"].
    Returns None if all sources fail or confidence is below the threshold.

    NEVER reads the phone number from the job posting entity dict.
    """
    threshold: float = config.get("number_source_min_confidence", 0.6)

    logger.info(f"[NumberSourcer] Sourcing number for '{company_name}' in '{location}'")

    for sourcer_fn, args in [
        (_source_google_places, (company_name, location)),
        (_source_opencorporates, (company_name, location)),
        (_source_serpapi, (company_name,)),
    ]:
        result = sourcer_fn(*args)
        if result is not None and result.confidence >= threshold:
            logger.info(
                f"[NumberSourcer] ✅ Using {result.source}: "
                f"{result.phone_number} (confidence={result.confidence})"
            )
            return result
        elif result is not None:
            logger.debug(
                f"[NumberSourcer] {result.source} returned {result.phone_number} "
                f"but confidence {result.confidence} < threshold {threshold}"
            )

    logger.warning(f"[NumberSourcer] All sources exhausted — no number found for '{company_name}'")
    return None
