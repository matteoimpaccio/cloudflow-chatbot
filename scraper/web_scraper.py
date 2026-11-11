
from pathlib import Path

import requests
from bs4 import BeautifulSoup


BASE_URL = "http://localhost:8000"

PAGES = [
    "index.html",
    "products.html",
    "pricing.html",
    "about.html",
    "faq.html",
    "contact.html",
]

OUTPUT_DIR = Path(__file__).parent / "scraped_data"


def scrape_page(page):
    url = f"{BASE_URL}/{page}"

    print(f"Scraping: {url}")

    response = requests.get(url, timeout=15)
    response.raise_for_status()

    # Force UTF-8 decoding to preserve special characters.
    response.encoding = "utf-8"

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove elements that do not contain useful page content.
    for element in soup.select(
        "script, style, nav, footer, header, button"
    ):
        element.decompose()

    title = soup.title.get_text(strip=True) if soup.title else page

    main_content = soup.find("main") or soup.body

    if main_content is None:
        print(f"No content found for {page}")
        return

    text = main_content.get_text(separator="\n", strip=True)

    # Remove excessive blank lines.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    clean_text = "\n".join(lines)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / f"{Path(page).stem}.txt"

    output_file.write_text(
        f"TITLE: {title}\nURL: {url}\n\n{clean_text}",
        encoding="utf-8",
    )

    print(f"Saved: {output_file}\n")


def main():
    for page in PAGES:
        try:
            scrape_page(page)
        except requests.RequestException as error:
            print(f"Error scraping {page}: {error}")


if __name__ == "__main__":
    main()
