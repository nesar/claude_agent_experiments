#!/usr/bin/env python3
"""
Scrape OpenCosmo documentation and save it in JSONL format for RAG systems.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import time
import re

BASE_URL = "https://opencosmo.readthedocs.io/en/latest/"
OUTPUT_FILE = "opencosmo_docs.jsonl"

visited_urls = set()

def is_valid_url(url):
    parsed = urlparse(url)
    return (
        parsed.netloc == "opencosmo.readthedocs.io" and
        parsed.path.startswith("/en/latest/") and
        "genindex" not in url and
        "search" not in url and
        "#" not in url
    )

def extract_text_from_main(soup):
    main = soup.find('div', {'role': 'main'})
    if not main:
        return ""
    for tag in main(['script', 'style', 'nav', 'footer']):
        tag.decompose()

    # Preserve headings (for better chunking later)
    for h in main.find_all(re.compile('^h[1-6]$')):
        h.insert_before('\n' + '#' * int(h.name[1]) + ' ' + h.get_text(strip=True) + '\n')

    text = main.get_text(separator="\n", strip=True)
    return re.sub(r'\n{2,}', '\n\n', text.strip())

def scrape_page(url, output_file, depth=0):
    if depth > 5 or url in visited_urls:
        return
    visited_urls.add(url)
    print(f"Scraping: {url}")

    try:
        time.sleep(1)
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed: {url} ({response.status_code})")
            return

        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.title.string.strip() if soup.title else url
        content = extract_text_from_main(soup)

        data = {
            "url": url,
            "title": title,
            "content": content
        }

        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

        links = soup.find_all('a', href=True)
        for link in links:
            full_url = urljoin(url, link['href'])
            if is_valid_url(full_url) and full_url not in visited_urls:
                scrape_page(full_url, output_file, depth + 1)

    except Exception as e:
        print(f"Error processing {url}: {e}")

def main():
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        pass  # Just clear file
    scrape_page(BASE_URL, OUTPUT_FILE)
    print(f"\n✅ Scraping complete. JSONL saved to {OUTPUT_FILE}")
    print(f"📄 Pages scraped: {len(visited_urls)}")

if __name__ == "__main__":
    main()