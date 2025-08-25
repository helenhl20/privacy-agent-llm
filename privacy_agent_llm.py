#!/usr/bin/env python3
"""
Privacy Agent:
Find web pages that mention a target email, then use OpenAI to
infer what information the site has about that email and what API
actions could be taken. Detects obfuscated emails like name [at] example [dot] com.

Outputs
  <out>/discoveries.csv   # verified pages
  <out>/report.md         # human-readable LLM responses per page
  <out>/report.pdf        # PDF version of the report
  <out>/domains.json      # grouping + contacts
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
import tldextract

# PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Search (no key required)
# DuckDuckGo has a lightweight, unofficial but widely used Python library that lets you pull search results programmatically without requiring an API key. 
# Google Search doesn’t allow scraping directly — they block automated queries aggressively. To use Google legally, you’d need their Custom Search API. 
try:
    from ddgs import DDGS  # updated import
    HAS_DDG = True
except Exception:
    HAS_DDG = False

# OpenAI client
try:
    from openai import OpenAI  # type: ignore
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

USER_AGENT = "PrivacyAgentLLM/1.0 (+contact: user)"
REQUEST_TIMEOUT = 20
CRAWL_DELAY_SEC = 1.0
MAX_CONTENT_BYTES = 2_000_000  # 2 MB cap

EMAIL_OBFUSCATION_PATTERNS = [
    re.compile(r"([\w.+-]+)\s*\[at\]\s*([\w.-]+)\s*\[dot\]\s*([a-z]{2,})", re.I),
    re.compile(r"([\w.+-]+)\s*\(at\)\s*([\w.-]+)\s*\(dot\)\s*([a-z]{2,})", re.I),
]

@dataclass
class Finding:
    url: str
    domain: str
    title: str
    contains_email: bool
    snippet: str
    status_code: int

@dataclass
class DomainInfo:
    domain: str
    pages: List[Finding] = field(default_factory=list)
    contacts: Dict[str, List[str]] = field(default_factory=dict)

@dataclass
class PageAnalysis:
    url: str
    title: str
    domain: str
    context: str
    llm_response: str = ""
    error: Optional[str] = None

# ============================ Helpers ============================ #

def robots_allows(url: str) -> bool:
    from urllib import robotparser
    try:
        parts = requests.utils.urlparse(url)
        robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
        rp = robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(USER_AGENT, url)
    except Exception:
        return True

def domain_of(url: str) -> str:
    parts = tldextract.extract(url)
    return ".".join(p for p in [parts.domain, parts.suffix] if p)

def clean_text(s: str, n: int = 280) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s[:n]

@retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(3))
def http_get(url: str) -> requests.Response:
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp

def detect_obfuscated_email(text: str, target_user: str) -> bool:
    for pat in EMAIL_OBFUSCATION_PATTERNS:
        for match in pat.finditer(text):
            user, domain, tld = match.groups()
            if target_user.lower() in user.lower():
                return True
    return False

# ============================ Search ============================ #

def search_queries_for_email(email_addr: str) -> List[str]:
    return [f'"{email_addr}"']


def search_duckduckgo(query: str, max_results: int = 50) -> List[str]:
    if not HAS_DDG:
        return []
    urls: List[str] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            u = r.get("href") or r.get("url")
            if u:
                urls.append(u)
    return urls


def search_web(email_addr: str, max_results: int = 100) -> List[str]:
    urls: List[str] = []
    for q in search_queries_for_email(email_addr):
        urls.extend(search_duckduckgo(q, max_results=max_results))
    # dedupe
    seen: Set[str] = set()
    uniq: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq

# ============================ Crawl & Verify ============================ #

def fetch_and_verify(url: str, needle_email: str) -> Optional[Finding]:
    if not robots_allows(url):
        return None
    try:
        resp = http_get(url)
        content = resp.content[:MAX_CONTENT_BYTES]
        soup = BeautifulSoup(content, "lxml")
        text = soup.get_text(" ")
        user_part = needle_email.split("@")[0].lower()
        has_email = (
            needle_email.lower() in text.lower()
            or user_part in text.lower()
            or detect_obfuscated_email(text, user_part)
        )
        title = soup.title.get_text().strip() if soup.title else ""
        snippet = ""
        if has_email:
            idx = text.lower().find(user_part)
            start = max(0, idx-160); end = idx + len(needle_email) + 160
            snippet = clean_text(text[start:end], n=420)
        return Finding(url=url, domain=domain_of(url), title=title, contains_email=has_email, snippet=snippet, status_code=resp.status_code)
    except Exception:
        return None

# ============================ LLM ============================ #

LLM_SYSTEM_PROMPT = (
    "You are a privacy and API analyst. Given a web page excerpt that includes a target email, "
    "analyze and answer in Markdown with the following sections:\n"
    "### Platform\nIdentify the platform or service if possible.\n\n"
    "### Public Data\nList what kinds of public or semi-public data could be tied to this email.\n\n"
    "### Possible API Actions\nSuggest what API endpoints or platform features might expose or return this data.\n\n"
    "### Recommended Action\nGive the user a clear recommendation on what to do."
)

def llm_analyze(email_addr: str, finding: Finding, model: str = "gpt-4o-mini", max_tokens: int = 800) -> PageAnalysis:
    pa = PageAnalysis(url=finding.url, title=finding.title, domain=finding.domain, context=finding.snippet)
    if not HAS_OPENAI or not os.getenv("OPENAI_API_KEY"):
        pa.error = "OPENAI_API_KEY not set or openai package missing; skipped LLM analysis."
        return pa
    client = OpenAI()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": f"Target email: {email_addr}\nURL: {finding.url}\nTitle: {finding.title}\nContext: {finding.snippet}"},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        pa.llm_response = resp.choices[0].message.content.strip()
        return pa
    except Exception as e:
        pa.error = f"LLM error: {e}"
        return pa

# ============================ I/O ============================ #

def write_discoveries_csv(path: str, findings: List[Finding]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["url", "domain", "title", "contains_email", "status_code", "snippet"])
        for r in findings:
            if r and r.contains_email:
                w.writerow([r.url, r.domain, r.title, r.contains_email, r.status_code, r.snippet])

def write_domains_json(path: str, domain_map: Dict[str, DomainInfo]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {}
    for d, info in domain_map.items():
        data[d] = {
            "pages": [dataclasses.asdict(p) for p in info.pages],
            "contacts": info.contacts,
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def write_report_md(path: str, target_email: str, analyses: List[PageAnalysis]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [f"# Privacy Agent Report (LLM Edition)\n\n", f"**Target email:** `{target_email}`\n\n"]
    for a in analyses:
        lines.append(f"## {a.domain}\n")
        lines.append(f"URL: {a.url}\n\n")
        if a.llm_response:
            lines.append(a.llm_response + "\n\n")
        if a.error:
            lines.append(f"*(Error: {a.error})*\n\n")
    text = "".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text

def write_report_pdf(path: str, md_text: str) -> None:
    doc = SimpleDocTemplate(path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    for line in md_text.splitlines():
        if line.startswith("# "):
            story.append(Paragraph(f"<b><font size=16>{line[2:].strip()}</font></b>", styles["Heading1"]))
        elif line.startswith("## "):
            story.append(Paragraph(f"<b><font size=14>{line[3:].strip()}</font></b>", styles["Heading2"]))
        elif line.startswith("### "):
            story.append(Paragraph(f"<b><font size=12>{line[4:].strip()}</font></b>", styles["Heading3"]))
        else:
            story.append(Paragraph(line, styles["Normal"]))
        story.append(Spacer(1, 8))
    doc.build(story)

# ============================ Main ============================ #

def run(email_addr: str, out_dir: str, max_results: int, model: str, max_tokens: int) -> None:
    load_dotenv()
    if not HAS_DDG:
        print("ddgs not installed; try: pip install ddgs", file=sys.stderr)
        sys.exit(1)

    print("\n[1/3] Searching the web…")
    urls = search_web(email_addr, max_results=max_results)
    if not urls:
        print("No URLs returned from search.")
        return

    print(f"Found {len(urls)} candidate URLs. Verifying…")
    findings: List[Finding] = []
    for u in tqdm(urls, desc="Verify", unit="page"):
        f = fetch_and_verify(u, email_addr)
        time.sleep(CRAWL_DELAY_SEC)
        if f and f.contains_email:
            findings.append(f)

    if not findings:
        print("No verified pages contained the email.")
        return

    domain_map = {}
    for f in findings:
        domain_map.setdefault(f.domain, DomainInfo(domain=f.domain)).pages.append(f)

    print("\n[2/3] Writing discoveries…")
    out_discover = os.path.join(out_dir, "discoveries.csv")
    os.makedirs(out_dir, exist_ok=True)
    write_discoveries_csv(out_discover, findings)
    out_domains = os.path.join(out_dir, "domains.json")
    write_domains_json(out_domains, domain_map)

    print("[3/3] Analyzing pages with LLM…")
    analyses: List[PageAnalysis] = []
    for f in tqdm(findings, desc="Analyze", unit="page"):
        analyses.append(llm_analyze(email_addr, f, model=model, max_tokens=max_tokens))
        time.sleep(0.3)

    out_report_md = os.path.join(out_dir, "report.md")
    md_text = write_report_md(out_report_md, email_addr, analyses)
    out_report_pdf = os.path.join(out_dir, "report.pdf")
    write_report_pdf(out_report_pdf, md_text)

    print(f"\n✅ Done. Outputs in: {os.path.abspath(out_dir)}")
    print(f" - {out_discover}")
    print(f" - {out_domains}")
    print(f" - {out_report_md}")
    print(f" - {out_report_pdf}")

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Find pages with your email; LLM infers platform info and API actions.")
    ap.add_argument("--email", required=True, help="Target email to search for")
    ap.add_argument("--out", default="outputs", help="Output directory")
    ap.add_argument("--max", dest="max_results", type=int, default=80, help="Max search results to fetch")
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=800, help="Max tokens for LLM completion")
    return ap.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    run(
        email_addr=args.email,
        out_dir=args.out,
        max_results=args.max_results,
        model=args.model,
        max_tokens=args.max_tokens,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
