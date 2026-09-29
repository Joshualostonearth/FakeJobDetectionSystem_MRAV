# Web Verification Module

## 1. Objective

The Web Verification Module collects publicly available company information from the web to support the Fake Job and Internship Detection System.

Given a company name, the module attempts to identify:

* Official company website
* Contact, locations, careers and about pages
* LinkedIn company profile
* Company phone numbers
* Company email addresses
* Source URLs for the collected information

The output is returned as structured JSON.

---

## 2. Working Process

The module follows these steps:

```text
Company Name
     ↓
Firecrawl Web Search
     ↓
Candidate URLs
     ↓
URL / Domain / Relevance Filtering
     ↓
Official Website
     ↓
Search for Contact / Locations / Careers / About
     ↓
Scrape Selected Pages
     ↓
Extract Emails & Phone Numbers
     ↓
Validate Extracted Data
     ↓
Store Source URLs
     ↓
Structured JSON Output
```

---

## 3. Web Search and Website Identification

**Firecrawl** is used as the web retrieval layer. It searches for the company and returns candidate webpages.

The Python code then evaluates the results using factors such as:

* Company name in the domain
* Company name in the title or description
* Relevant URL paths such as `/about`, `/company` and `/corporate`
* Known third-party domains
* Unwanted paths such as search, maps and reviews

The highest-scoring valid candidate is selected as the official website.

For company-specific pages, the module checks that the webpage belongs to the same root domain as the selected official website. This prevents websites such as job portals or recruitment platforms from being incorrectly classified as official company pages.

---

## 4. Page Extraction

The selected webpages are retrieved using Firecrawl in both:

* **Markdown** – for visible webpage content
* **HTML** – for structured links such as `tel:` and `mailto:`

The module searches separately for:

* Contact pages
* Office/location pages
* Careers pages
* About pages

The scraped content is also checked for company relevance before being accepted.

---

## 5. Email and Phone Extraction

Emails and phone numbers are extracted using Python regular expressions rather than relying on generated values.

### Email validation

The system removes malformed addresses and technical scraping artifacts such as:

```text
frame-...@mhtml.blink
```

### Phone validation

Phone numbers are checked for:

* Valid digit length
* Invalid international formats
* Date-like values
* Repeated digits
* Other obvious false positives

Every accepted phone number or email address is stored with its source URL.

Example:

```json
{
    "value": "recrops@ca.ibm.com",
    "source_url": "https://www.ibm.com/careers",
    "source_type": "official_company"
}
```

---

## 6. Problems Found and Fixes

### Third-party careers pages

Search results sometimes returned external job websites as careers pages.

**Fix:** Company pages are now required to belong to the identified official root domain.

### False phone numbers

A Google page produced the phone-like value:

```text
+04543-011
```

**Fix:** Added validation for international numbers and minimum digit requirements.

### MHTML email artifacts

IBM pages contained technical browser-generated email-like values.

**Fix:** Added validation to reject `.blink`, `localhost`, `example.*` and `frame-*` artifacts.

### Third-party company databases

Testing with a smaller company showed that websites such as Tracxn can contain company information without being the company's official website.

**Current status:** This remains a limitation for companies with weak or poorly indexed official web presence and requires stronger website-ownership verification.

---

## 7. Testing

The module has been tested with companies including **Google** and **IBM**.

For IBM, the system successfully identified the official website, contact, locations and careers pages, LinkedIn profile, phone numbers and company email while retaining the source URL for each extracted value.

Google testing also confirmed that invalid phone values and unrelated third-party pages could be filtered out.

---

## 8. Design Approach

The module separates web retrieval from evidence extraction:

```text
Firecrawl → Search and webpage retrieval
Python    → Filtering and validation
Regex     → Exact email/phone extraction
Sources   → Evidence traceability
```

This approach reduces false positives and avoids relying on generated values for exact contact information.
