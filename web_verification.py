import os
import re
import json
from urllib.parse import urlparse, unquote
from firecrawl import Firecrawl


# ============================================================
# FIRECRAWL SETUP
# ============================================================
api_key = os.getenv("FIRECRAWL_API_KEY")

if not api_key:
    raise ValueError("FIRECRAWL_API_KEY is not set.")

app = Firecrawl(api_key=api_key)


# ============================================================
# URL HELPERS
# ============================================================
def normalize_url(url):
    if not url:
        return ""
    return url.strip().rstrip("/")


def get_domain(url):
    try:
        domain = urlparse(url).netloc.lower()

        if domain.startswith("www."):
            domain = domain[4:]

        return domain

    except Exception:
        return ""


def get_root_domain(url):
    domain = get_domain(url)
    parts = domain.split(".")

    if len(parts) >= 2:
        return ".".join(parts[-2:])

    return domain


# ============================================================
# BAD DOMAINS
# ============================================================
BAD_DOMAINS = [
    "wikipedia.org",
    "linkedin.com",
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "x.com",
    "youtube.com",
    "glassdoor.com",
    "indeed.com",
    "ambitionbox.com",
    "naukri.com",
    "wellfound.com",
    "reddit.com",
]


def is_bad_domain(url):
    domain = get_domain(url)

    return any(
        domain == bad
        or domain.endswith("." + bad)
        for bad in BAD_DOMAINS
    )


# ============================================================
# BAD URL PATHS
# ============================================================
BAD_PATH_PATTERNS = [
    "/maps/",
    "/maps?",
    "/search",
    "/search?",
    "/local/",
    "/reviews/",
]


def is_bad_path(url):
    lower = url.lower()

    return any(
        pattern in lower
        for pattern in BAD_PATH_PATTERNS
    )


# ============================================================
# COMPANY NAME NORMALIZATION
# ============================================================
def normalize_company_name(company):

    company = company.lower()

    removable = [
        "incorporated",
        "inc.",
        "inc",
        "limited",
        "ltd.",
        "ltd",
        "llc",
        "plc",
        "corp.",
        "corp",
        "corporation",
        "company",
        "co.",
        "co"
    ]

    for word in removable:
        company = company.replace(
            word,
            " "
        )

    company = re.sub(
        r"[^a-z0-9\s]",
        " ",
        company
    )

    company = re.sub(
        r"\s+",
        " ",
        company
    ).strip()

    return company


def company_tokens(company):

    normalized = normalize_company_name(company)

    return [
        token
        for token in normalized.split()
        if len(token) >= 3
    ]


def company_matches_text(text, company):

    if not text:
        return False

    text_lower = text.lower()

    normalized = normalize_company_name(company)

    if normalized and normalized in text_lower:
        return True

    tokens = company_tokens(company)

    if not tokens:
        return False

    matches = 0

    for token in tokens:

        if re.search(
            r"\b"
            + re.escape(token)
            + r"\b",
            text_lower
        ):
            matches += 1

    if len(tokens) >= 2:
        return matches >= 2

    return matches >= 1


# ============================================================
# OFFICIAL WEBSITE SEARCH SCORING
# ============================================================
def score_search_result(
    url,
    title,
    description,
    company
):

    if not url:
        return -100

    if is_bad_domain(url):
        return -100

    if is_bad_path(url):
        return -50

    lower_url = url.lower()

    lower_title = (
        title or ""
    ).lower()

    lower_description = (
        description or ""
    ).lower()

    company_normalized = normalize_company_name(
        company
    )

    company_clean = re.sub(
        r"[^a-z0-9]",
        "",
        company_normalized
    )

    domain = get_domain(url)

    domain_clean = re.sub(
        r"[^a-z0-9]",
        "",
        domain
    )

    score = 0

    if domain.startswith("about."):
        score += 15

    if any(
        keyword in lower_url
        for keyword in [
            "/company",
            "/company-info",
            "/corporate",
            "/about"
        ]
    ):
        score += 10

    if company_clean and company_clean in domain_clean:
        score += 15

    title_clean = re.sub(
        r"[^a-z0-9]",
        "",
        lower_title
    )

    if company_clean and company_clean in title_clean:
        score += 10

    if (
        company_normalized
        and company_normalized in lower_description
    ):
        score += 5

    if "/job/results" in lower_url:
        score -= 10

    return score


# ============================================================
# FIND OFFICIAL WEBSITE
# ============================================================
def find_official_website(company):

    print(
        f"\nSearching for official company information: {company}"
    )

    result = app.search(
        f"{company} official company website company information",
        limit=10
    )

    candidates = []

    for item in result.web:

        if not item.url:
            continue

        url = normalize_url(item.url)

        score = score_search_result(
            url,
            item.title,
            item.description,
            company
        )

        candidates.append({
            "url": url,
            "title": item.title,
            "description": item.description,
            "score": score
        })

    if not candidates:
        raise RuntimeError(
            f"No search results found for {company}"
        )

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    print("\nSearch candidates:")

    for candidate in candidates:

        print(
            f"{candidate['score']:>4} | "
            f"{candidate['url']}"
        )

    best = candidates[0]

    if best["score"] < 0:

        raise RuntimeError(
            f"Could not identify an official website "
            f"for {company}"
        )

    print(
        "\nSelected company website:"
    )

    print(best["url"])

    return best["url"]


# ============================================================
# SEARCH FOR SPECIFIC COMPANY PAGE
# ============================================================
def search_company_page(
    company,
    page_type,
    official_domain
):

    queries = {

        "contact": [
            f"{company} official corporate contact",
            f"{company} official contact us",
            f"{company} corporate office phone number",
        ],

        "locations": [
            f"{company} official office locations",
            f"{company} corporate offices",
        ],

        "careers": [
            f"{company} official careers",
            f"{company} official jobs",
        ],

        "about": [
            f"{company} official about company",
        ]

    }

    official_root = get_root_domain(
        "https://" + official_domain
    )

    candidates = []

    # --------------------------------------------------------
    # Run multiple targeted searches
    # --------------------------------------------------------
    for query in queries[page_type]:

        print(
            f"\nSearching for {page_type}:"
        )

        print(query)

        result = app.search(
            query,
            limit=10
        )

        for item in result.web:

            if not item.url:
                continue

            url = normalize_url(
                item.url
            )

            domain = get_domain(url)

            if is_bad_domain(url):
                continue

            if is_bad_path(url):
                continue

            lower = url.lower()

            score = 0

            candidate_root = get_root_domain(
                url
            )

            # ------------------------------------------------
            # ONLY ACCEPT OFFICIAL COMPANY DOMAIN
            # ------------------------------------------------
            if candidate_root != official_root:
                continue

            # ------------------------------------------------
            # Exact official domain
            # ------------------------------------------------
            if domain == official_domain:
                score += 50

            # ------------------------------------------------
            # Same root domain
            # ------------------------------------------------
            if candidate_root == official_root:
                score += 30

            # ------------------------------------------------
            # Contact relevance
            # ------------------------------------------------
            if page_type == "contact":

                if "contact" in lower:
                    score += 35

                if "contact-us" in lower:
                    score += 10

                if "corporate" in lower:
                    score += 15

                if "support" in lower:
                    score -= 15

                if "/docs/" in lower:
                    score -= 30

                if "documentation" in lower:
                    score -= 30

            # ------------------------------------------------
            # Location relevance
            # ------------------------------------------------
            elif page_type == "locations":

                if "location" in lower:
                    score += 35

                if "office" in lower:
                    score += 35

                if "offices" in lower:
                    score += 20

            # ------------------------------------------------
            # Careers relevance
            # ------------------------------------------------
            elif page_type == "careers":

                if "career" in lower:
                    score += 35

                if "jobs" in lower:
                    score += 30

                if "work-with-us" in lower:
                    score += 20

            # ------------------------------------------------
            # About relevance
            # ------------------------------------------------
            elif page_type == "about":

                if "about" in lower:
                    score += 35

                if "company" in lower:
                    score += 10

            # ------------------------------------------------
            # Search result title / description
            # ------------------------------------------------
            title = (
                item.title or ""
            ).lower()

            description = (
                item.description or ""
            ).lower()

            normalized_company = normalize_company_name(
                company
            )

            if normalized_company in title:
                score += 20

            if normalized_company in description:
                score += 10

            candidates.append({
                "url": url,
                "title": item.title,
                "description": item.description,
                "score": score
            })

    # --------------------------------------------------------
    # Deduplicate
    # --------------------------------------------------------
    unique = {}

    for candidate in candidates:

        url = candidate["url"]

        if (
            url not in unique
            or candidate["score"]
            > unique[url]["score"]
        ):
            unique[url] = candidate

    candidates = list(
        unique.values()
    )

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return candidates


# ============================================================
# SEARCH LINKEDIN
# ============================================================
def search_linkedin_company(company):

    print(
        "\nSearching for official LinkedIn company page..."
    )

    result = app.search(
        f"{company} official LinkedIn company",
        limit=10
    )

    candidates = []

    normalized_company = normalize_company_name(
        company
    )

    for item in result.web:

        if not item.url:
            continue

        url = normalize_url(
            item.url
        )

        lower = url.lower()

        if "linkedin.com/company/" not in lower:
            continue

        score = 40

        title = (
            item.title or ""
        ).lower()

        description = (
            item.description or ""
        ).lower()

        if normalized_company in title:
            score += 30

        if normalized_company in description:
            score += 20

        if company_matches_text(
            title + " " + description,
            company
        ):
            score += 15

        candidates.append({
            "url": url,
            "score": score
        })

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    if candidates[0]["score"] < 50:
        return None

    return candidates[0]["url"]


# ============================================================
# SCRAPE PAGE
# ============================================================
def scrape_page(url):

    try:

        # IMPORTANT:
        # Get BOTH markdown and HTML.
        #
        # Markdown is useful for visible text.
        # HTML is useful for tel: links and mailto: links.

        result = app.scrape(
            url,
            formats=[
                "markdown",
                "html"
            ]
        )

        markdown = getattr(
            result,
            "markdown",
            ""
        ) or ""

        html = getattr(
            result,
            "html",
            ""
        ) or ""

        return {
            "markdown": markdown,
            "html": html
        }

    except Exception as e:

        print(
            f"Could not scrape {url}"
        )

        print(
            f"Reason: {e}"
        )

        return {
            "markdown": "",
            "html": ""
        }


# ============================================================
# PAGE CONTENT VALIDATION
# ============================================================
def page_is_relevant(
    scraped,
    company,
    page_type
):

    if not scraped:
        return False

    text = (
        scraped.get("markdown", "")
        + "\n"
        + scraped.get("html", "")
    )

    if not text:
        return False

    if company_matches_text(
        text,
        company
    ):
        return True

    return False


# ============================================================
# EXTRACT EMAILS
# ============================================================
EMAIL_PATTERN = (
    r"\b[A-Za-z0-9._%+-]+"
    r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)


BAD_EMAIL_DOMAINS = {
    "mhtml.blink",
    "localhost",
    "example.com",
    "example.org",
    "example.net",
}


def is_valid_email(email):

    if not email:
        return False

    email = email.strip().lower()

    if "@" not in email:
        return False

    local, domain = email.rsplit(
        "@",
        1
    )

    if not local or not domain:
        return False

    # --------------------------------------------------------
    # Reject known technical/browser artifacts
    # --------------------------------------------------------
    if domain in BAD_EMAIL_DOMAINS:
        return False

    if domain.endswith(".blink"):
        return False

    if local.startswith("frame-"):
        return False

    # --------------------------------------------------------
    # Basic domain validation
    # --------------------------------------------------------
    if "." not in domain:
        return False

    if ".." in domain:
        return False

    if ".." in local:
        return False

    return True


def extract_emails(
    markdown,
    html
):

    emails = []

    # ========================================================
    # 1. Extract emails from visible Markdown
    # ========================================================
    if markdown:

        matches = re.findall(
            EMAIL_PATTERN,
            markdown
        )

        for email in matches:

            email = email.strip(
                ".,;:()[]{}<>\"'"
            ).lower()

            if not is_valid_email(
                email
            ):
                continue

            if email not in emails:
                emails.append(email)

    # ========================================================
    # 2. Extract ONLY explicit mailto: links from HTML
    # ========================================================
    if html:

        mailto_pattern = (
            r'href\s*=\s*["\']'
            r'mailto:([^"\'>?#\s]+)'
        )

        matches = re.findall(
            mailto_pattern,
            html,
            flags=re.IGNORECASE
        )

        for email in matches:

            email = unquote(
                email
            ).strip(
                ".,;:()[]{}<>\"'"
            ).lower()

            if not is_valid_email(
                email
            ):
                continue

            if email not in emails:
                emails.append(email)

    return emails


# ============================================================
# PHONE NORMALIZATION
# ============================================================
def normalize_phone(phone):

    if not phone:
        return ""

    phone = unquote(phone)

    phone = phone.replace(
        "\u00a0",
        " "
    )

    phone = re.sub(
        r"\s+",
        " ",
        phone
    ).strip()

    phone = phone.strip(
        ".,;:()[]{}<>\"'"
    )

    return phone


# ============================================================
# PHONE VALIDATION
# ============================================================
def is_valid_phone(phone):

    if not phone:
        return False

    digits = re.sub(
        r"\D",
        "",
        phone
    )

    if len(digits) < 7:
        return False

    if len(digits) > 15:
        return False

    # --------------------------------------------------------
    # Reject invalid international numbers
    # --------------------------------------------------------
    if phone.startswith("+"):

        if digits.startswith("0"):
            return False

        if len(digits) < 10:
            return False

    # --------------------------------------------------------
    # Reject obvious dates
    # --------------------------------------------------------
    if re.fullmatch(
        r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}",
        phone
    ):
        return False

    # --------------------------------------------------------
    # Reject repeated digits
    # --------------------------------------------------------
    if len(set(digits)) == 1:
        return False

    return True


# ============================================================
# EXTRACT PHONE NUMBERS
# ============================================================
def extract_phone_numbers(
    markdown,
    html
):

    numbers = []

    # ========================================================
    # 1. Extract TEL links from HTML
    # ========================================================
    if html:

        tel_patterns = [

            r'href\s*=\s*["\']tel:([^"\']+)["\']',

            r'href\s*=\s*["\']tel%3A([^"\']+)["\']',

        ]

        for pattern in tel_patterns:

            matches = re.findall(
                pattern,
                html,
                flags=re.IGNORECASE
            )

            for number in matches:

                number = normalize_phone(
                    number
                )

                if is_valid_phone(
                    number
                ):

                    if number not in numbers:
                        numbers.append(number)

    # ========================================================
    # 2. Extract phone-like visible text
    # ========================================================
    text = markdown or ""

    patterns = [

        # International
        r"""
        \+\d{1,3}
        (?:[\s.\-()]?\d){7,14}
        """,

        # India mobile
        r"\b[6-9]\d{4}[\s\-]\d{5}\b",

        # India landline
        r"\b0\d{2,4}[\s\-]\d{3,4}[\s\-]\d{4}\b",

        # US
        r"\(\d{3}\)\s*\d{3}[\s\-]\d{4}",

        # US / generic
        r"\b\d{3}[\s\-]\d{3}[\s\-]\d{4}\b",

        # Generic with dots
        r"\b\d{3}\.\d{3}\.\d{4}\b",

        # Generic 3-4-4
        r"\b\d{3}[\s\-]\d{4}[\s\-]\d{4}\b",

    ]

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text,
            flags=re.VERBOSE
        )

        for number in matches:

            if isinstance(
                number,
                tuple
            ):
                number = " ".join(
                    number
                )

            number = normalize_phone(
                number
            )

            if not is_valid_phone(
                number
            ):
                continue

            if number not in numbers:
                numbers.append(number)

    return numbers


# ============================================================
# EXTRACT URLS
# ============================================================
def extract_urls(text):

    if not text:
        return []

    urls = re.findall(
        r"https?://[^\s)\]>]+",
        text
    )

    cleaned = []

    for url in urls:

        url = url.rstrip(
            ".,;:"
        )

        if url not in cleaned:
            cleaned.append(url)

    return cleaned


# ============================================================
# MAIN FUNCTION
# ============================================================
def get_company_information(company):

    # ========================================================
    # STEP 1
    # Find official website
    # ========================================================
    official_website = find_official_website(
        company
    )

    official_domain = get_domain(
        official_website
    )

    print(
        f"\nOfficial domain: {official_domain}"
    )

    # ========================================================
    # STEP 2
    # Find pages
    # ========================================================
    selected_pages = {}

    scraped_pages = []

    sources = []

    for page_type in [
        "contact",
        "locations",
        "careers",
        "about"
    ]:

        candidates = search_company_page(
            company,
            page_type,
            official_domain
        )

        if not candidates:

            print(
                f"No {page_type} candidates found."
            )

            continue

        print(
            f"\n{page_type.upper()} candidates:"
        )

        for candidate in candidates[:7]:

            print(
                f"{candidate['score']:>4} | "
                f"{candidate['url']}"
            )

        accepted = False

        # ====================================================
        # Try candidates
        # ====================================================
        for candidate in candidates:

            url = candidate["url"]

            if is_bad_path(url):
                continue

            print(
                f"\nChecking {page_type}:"
            )

            print(url)

            scraped = scrape_page(
                url
            )

            if (
                not scraped["markdown"]
                and not scraped["html"]
            ):
                continue

            if page_is_relevant(
                scraped,
                company,
                page_type
            ):

                print(
                    f"Accepted {page_type} page."
                )

                selected_pages[
                    page_type
                ] = url

                scraped_pages.append({

                    "page_type": page_type,

                    "url": url,

                    "markdown": scraped["markdown"],

                    "html": scraped["html"]

                })

                sources.append({

                    "page_type": page_type,

                    "source_url": url

                })

                accepted = True

                break

            print(
                f"Rejected {page_type} "
                f"(content not relevant)."
            )

        if not accepted:

            print(
                f"No valid {page_type} page accepted."
            )

    # ========================================================
    # STEP 3
    # LinkedIn
    # ========================================================
    linkedin_url = search_linkedin_company(
        company
    )

    # ========================================================
    # STEP 4
    # Extract evidence from EVERY accepted page
    # ========================================================
    phone_evidence = []

    email_evidence = []

    for page in scraped_pages:

        source_url = page["url"]

        markdown = page["markdown"]

        html = page["html"]

        # ----------------------------------------------------
        # Emails
        # ----------------------------------------------------
        emails = extract_emails(
            markdown,
            html
        )

        for email in emails:

            exists = any(
                item["value"] == email
                and item["source_url"] == source_url
                for item in email_evidence
            )

            if not exists:

                email_evidence.append({

                    "value": email,

                    "source_url": source_url,

                    "source_type": "official_company"

                })

        # ----------------------------------------------------
        # Phone numbers
        # ----------------------------------------------------
        phones = extract_phone_numbers(
            markdown,
            html
        )

        for phone in phones:

            exists = any(
                item["value"] == phone
                and item["source_url"] == source_url
                for item in phone_evidence
            )

            if not exists:

                phone_evidence.append({

                    "value": phone,

                    "source_url": source_url,

                    "source_type": "official_company"

                })

    # ========================================================
    # STEP 5
    # Final result
    # ========================================================
    result = {

        "company":
            company,

        "official_website":
            official_website,

        "contact_page":
            selected_pages.get(
                "contact"
            ),

        "locations_page":
            selected_pages.get(
                "locations"
            ),

        "careers_page":
            selected_pages.get(
                "careers"
            ),

        "linkedin_company":
            linkedin_url,

        "phone_numbers":
            phone_evidence,

        "email_addresses":
            email_evidence,

        "sources":
            sources

    }

    return result


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================
if __name__ == "__main__":

    company = input(
        "Enter company name: "
    ).strip()

    if not company:

        print(
            "Company name cannot be empty."
        )

        exit()

    result = get_company_information(
        company
    )

    print("\n")

    print(
        "=" * 70
    )

    print(
        "FINAL COMPANY INFORMATION"
    )

    print(
        "=" * 70
    )

    print(
        json.dumps(
            result,
            indent=4
        )
    )