import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

OUTPUT_DIR = "data/documents"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Gambian websites to scrape
TARGETS = [
    # Government
    {"url": "https://www.statehouse.gm", "name": "statehouse"},
    {"url": "https://www.moj.gov.gm", "name": "ministry_justice"},
    {"url": "https://www.moh.gov.gm", "name": "ministry_health"},
    {"url": "https://www.moe.gov.gm", "name": "ministry_education"},
    {"url": "https://www.mofa.gov.gm", "name": "ministry_foreign_affairs"},
    {"url": "https://www.grts.gm", "name": "grts_tv"},
    {"url": "https://whatson-gambia.com", "name": "whatson_gambia"},
    {"url": "https://www.kerrfatou.com", "name": "kerr_fatou"},
    {"url": "https://fatunetwork.net", "name": "fatu_network"},
    {"url": "https://www.moj.gov.gm", "name": "ministry_justice"},
    {"url": "https://www.judiciary.gm", "name": "judiciary_gambia"},
    {"url": "https://www.gra.gm", "name": "gambia_revenue"},
    {"url": "https://www.nhrc.gm", "name": "human_rights_commission"},
    {"url": "https://www.giepa.gm", "name": "investment_promotion"},

    # News
    {"url": "https://thepoint.gm", "name": "thepoint"},
    {"url": "https://foroyaa.net", "name": "foroyaa"},
    {"url": "https://gainako.com", "name": "gainako"},
    {"url": "https://www.thestandard.gm", "name": "standard_newspaper"},
    {"url": "https://smbcnewsgambia.com", "name": "smbc_news"},

    # Tourism & Business
    {"url": "https://www.visitthegambia.gm", "name": "tourism"},
    {"url": "https://www.gcci.gm", "name": "chamber_commerce"},
    {"url": "https://www.gnpc.gm", "name": "petroleum"},

    # Education
    {"url": "https://www.utm.edu.gm", "name": "university_gambia"},
    {"url": "https://www.utm.edu.gm", "name": "university_gambia"},
    {"url": "https://www.gambiacollege.edu.gm", "name": "gambia_college"},
    {"url": "https://www.gtti.edu.gm", "name": "gtti"},

    # Health
    {"url": "https://www.mrc.gm", "name": "medical_research"},
]

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; GambiaGPT-bot/1.0)"}
MAX_PAGES = 50  # per site

def get_links(base_url, soup):
    links = set()
    for tag in soup.find_all("a", href=True):
        href = urljoin(base_url, tag["href"])
        if urlparse(href).netloc == urlparse(base_url).netloc:
            links.add(href.split("#")[0])
    return links

def scrape_site(url, name):
    visited = set()
    to_visit = {url}
    all_text = []
    pages = 0

    print(f"\nScraping: {name} ({url})")

    while to_visit and pages < MAX_PAGES:
        current = to_visit.pop()
        if current in visited:
            continue
        try:
            r = requests.get(current, headers=HEADERS, timeout=10)
            if "text/html" not in r.headers.get("Content-Type", ""):
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            if len(text) > 200:
                all_text.append(f"--- Page: {current} ---\n{text}\n")
                pages += 1
                print(f"  [{pages}/{MAX_PAGES}] {current}")
            to_visit |= get_links(url, soup) - visited
            visited.add(current)
            time.sleep(1)
        except Exception as e:
            print(f"  Error: {e}")

    if all_text:
        out_path = os.path.join(OUTPUT_DIR, f"{name}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(all_text))
        print(f"  Saved {pages} pages to {out_path}")

if __name__ == "__main__":
    for target in TARGETS:
        scrape_site(target["url"], target["name"])
    print("\nAll done! Now run: python ingest.py")