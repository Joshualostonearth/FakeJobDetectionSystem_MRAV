import sys
import json
from scrapegraphai.graphs import SmartScraperGraph
#Local llm
# Local pipeline configuration pointing strictly to Ollama
graph_config = {
    "llm": {
        "model": "ollama/llama3.2",
        "temperature": 0,
        "format": "json",
        "base_url": "http://localhost:11434",
    },
    "embeddings": {
        "model": "ollama/nomic-embed-text",
        "base_url": "http://localhost:11434",
    },
    "verbose": True,
    "headless": True,
}

DEFAULT_PROMPT = (
    "Extract the official company name, primary contact phone number, "
    "corporate office address, recruiter name, and job role."
)


def fallback_scrape(url: str) -> Dict[str, Any]:
    """
    Fallback HTTP scraper used when ScrapeGraphAI or Ollama is unavailable.
    Fetches raw HTML and extracts company name, domain, emails, phones, and address heuristics.
    """
    domain = urlparse(url).netloc.replace("www.", "")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0'}

    page_title = domain.split('.')[0].capitalize() if domain else "Unknown Company"
    emails = []
    phones = []
    text_content = ""

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as response:
            html = response.read().decode('utf-8', errors='ignore')
            text_content = re.sub(r'<[^>]+>', ' ', html)

            # Title extraction
            title_match = re.search(r'<title>(.*?)</title>', html, re.IGNORECASE)
            if title_match:
                title_clean = title_match.group(1).strip()
                if title_clean:
                    page_title = title_clean.split('|')[0].split('-')[0].strip()

            # Email extraction
            emails = list(set(re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text_content)))
            # Phone extraction
            phones = list(set(re.findall(r'\+?\d{1,3}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}', text_content)))
    except Exception as e:
        print(f"[Fallback Scraper Notice] Could not fetch live URL directly ({e}). Using domain-derived heuristics.")

    contact = emails[0] if emails else (phones[0] if phones else f"contact@{domain}" if domain else "N/A")

    return {
        "company_name": page_title if page_title else "Target Company",
        "job_role": "Open Position",
        "recruiter_name": "Talent Acquisition Team",
        "email_or_phone": contact,
        "office_address": f"Corporate Office, {domain}" if domain else "Headquarters",
        "domain_or_website": domain or url,
        "raw_extracted_text": text_content[:300] if text_content else "No text extracted",
        "scraper_engine": "Fallback Web Extractor (HTTP Heuristic)"
    }


def scrape_job_page(source_url: str, prompt: Optional[str] = None) -> Dict[str, Any]:
    """
    Executes web scraping using SmartScraperGraph if available,
    falling back to standard Web Extractor if Ollama/ScrapeGraphAI is offline.
    """
    if not prompt:
        prompt = DEFAULT_PROMPT

    # Try SmartScraperGraph (ScrapeGraphAI)
    try:
        from scrapegraphai.graphs import SmartScraperGraph
        print(f"[Scraper] Initializing SmartScraperGraph for: {source_url}")
        smart_scraper_graph = SmartScraperGraph(
            prompt=prompt,
            source=source_url,
            config=GRAPH_CONFIG
        )
        result = smart_scraper_graph.run()

        if isinstance(result, dict) and result:
            # Map scraped output keys to standard voice verification entities
            domain = urlparse(source_url).netloc.replace("www.", "")
            normalized = {
                "company_name": result.get("company_name") or result.get("official_company_name") or result.get("company") or domain,
                "job_role": result.get("job_role") or result.get("role") or result.get("position") or "Open Position",
                "recruiter_name": result.get("recruiter_name") or result.get("contact_person") or "HR Representative",
                "email_or_phone": result.get("primary_contact_phone_number") or result.get("phone") or result.get("contact") or f"contact@{domain}",
                "office_address": result.get("corporate_office_address") or result.get("address") or result.get("location") or "Headquarters",
                "domain_or_website": domain or source_url,
                "raw_extracted": result,
                "scraper_engine": "SmartScraperGraph (Ollama)"
            }
            return normalized

    except Exception as e:
        print(f"[Scraper Warning] SmartScraperGraph execution encountered: {e}. Switching to fallback scraper.")

    # Fallback execution
    return fallback_scrape(source_url)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TestScraper: Webscraping for Voice Verification Agent")
    parser.add_argument("--url", type=str, default="https://www.google.com/about/careers/applications/", help="Target URL to scrape")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Extraction prompt")
    parser.add_argument("--json-only", action="store_true", help="Output strictly raw JSON")

    args = parser.parse_args()

    scraped_data = scrape_job_page(args.url, args.prompt)

    if args.json_only:
        print(json.dumps(scraped_data))
    else:
        print("\n--- EXTRACTED JSON OUTPUT ---")
        print(json.dumps(scraped_data, indent=2))
