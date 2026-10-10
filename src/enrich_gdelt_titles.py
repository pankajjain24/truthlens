from pathlib import Path
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

from src.data_store import save_cache


ROOT = Path(__file__).resolve().parents[1]
CACHE_FILE = ROOT / "data" / "raw" / "gdelt" / "articles_cache.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/140.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}


def clean_title(title):
    if not title:
        return ""

    return " ".join(str(title).split()).strip(" -|")


def get_title(url):

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=(5, 10),
            allow_redirects=True,
        )

        if response.status_code != 200:
            return "", f"HTTP {response.status_code}"

        soup = BeautifulSoup(
            response.content,
            "html.parser",
        )

        # Standard HTML title
        if soup.title:
            title = clean_title(soup.title.get_text())

            if title:
                return title, "html_title"

        # OpenGraph title
        for attrs in [
            {"property": "og:title"},
            {"name": "og:title"},
        ]:
            tag = soup.find("meta", attrs=attrs)

            if tag and tag.get("content"):
                title = clean_title(tag["content"])

                if title:
                    return title, "og:title"

        # Twitter/X title
        tag = soup.find(
            "meta",
            attrs={"name": "twitter:title"},
        )

        if tag and tag.get("content"):
            title = clean_title(tag["content"])

            if title:
                return title, "twitter:title"

        # Article headline
        for selector in [
            "h1",
            '[itemprop="headline"]',
            ".headline",
            ".article-title",
        ]:
            tag = soup.select_one(selector)

            if tag:
                title = clean_title(tag.get_text())

                if title:
                    return title, selector

        return "", "no_title_found"

    except requests.exceptions.Timeout:
        return "", "timeout"

    except requests.exceptions.RequestException as exc:
        return "", f"request_error:{type(exc).__name__}"

    except Exception as exc:
        return "", f"parse_error:{type(exc).__name__}"


def save_progress(df):

    save_cache(
        df,
        query="GDELT GKG snapshot 20260915 - article title enrichment",
        source="GDELT GKG + real article webpage metadata",
    )


def main():

    if not CACHE_FILE.exists():
        raise FileNotFoundError(
            f"Cache not found: {CACHE_FILE}"
        )

    df = pd.read_csv(CACHE_FILE)

    if "title" not in df.columns:
        df["title"] = ""

    df["title"] = df["title"].fillna("").astype(str)

    total = len(df)

    already_done = (
        df["title"].str.strip().ne("").sum()
    )

    print(f"Loaded {total:,} GDELT records.")
    print(f"Existing titles: {already_done:,}")
    print(
        f"Remaining: {total - already_done:,}"
    )

    success = already_done

    for idx, url in enumerate(
        df["url"].fillna(""),
        start=1,
    ):

        # Resume support:
        # skip records that already have a title.
        if df.loc[idx - 1, "title"].strip():
            continue

        if not url:
            continue

        title, status = get_title(url)

        if title:
            df.loc[idx - 1, "title"] = title
            success += 1

        print(
            f"[{idx:03d}/{total}] "
            f"titles={success} "
            f"status={status}"
        )

        # Save every 10 processed records.
        if idx % 10 == 0:
            save_progress(df)
            print("  -> Progress saved.")

        # Be polite to external websites.
        time.sleep(0.5)

    # Final save
    save_progress(df)

    empty = (
        df["title"]
        .str.strip()
        .eq("")
        .sum()
    )

    print()
    print("=" * 60)
    print("TITLE ENRICHMENT COMPLETE")
    print("=" * 60)
    print(f"Total records:      {total:,}")
    print(f"Titles found:       {success:,}")
    print(f"Titles still empty: {empty:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()