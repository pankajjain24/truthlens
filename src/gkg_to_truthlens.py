from pathlib import Path
import pandas as pd
import re

from src.data_store import save_cache


ROOT = Path(__file__).resolve().parents[1]

GKG_FILE = ROOT / "data" / "raw" / "gdelt" / "20260915.gkg.csv"


COLUMNS = [
    "DATE",
    "NUMARTS",
    "COUNTS",
    "THEMES",
    "LOCATIONS",
    "PERSONS",
    "ORGANIZATIONS",
    "TONE",
    "CAMEOEVENTIDS",
    "SOURCES",
    "SOURCEURLS",
]


KEYWORDS = [
    "misinformation",
    "disinformation",
    "deepfake",
    "fake news",
    "fact check",
    "false claim",
    "false claims",
    "fake",
    "hoax",
]


def split_urls(value):
    """Extract individual URLs from the GKG SOURCEURLS field."""
    if pd.isna(value):
        return []

    text = str(value)

    # GKG normally separates multiple URLs with semicolons.
    parts = re.split(r"[;,\s]+(?=https?://)", text)

    urls = []
    for part in parts:
        part = part.strip()

        if part.startswith("http://") or part.startswith("https://"):
            urls.append(part)

    return list(dict.fromkeys(urls))


def extract_domain(url):
    match = re.search(r"https?://(?:www\.)?([^/]+)", url)

    if match:
        return match.group(1).lower()

    return "unknown"


def relevant_record(row):
    text = " ".join(
        str(row.get(col, ""))
        for col in [
            "THEMES",
            "LOCATIONS",
            "PERSONS",
            "ORGANIZATIONS",
        ]
    ).lower()

    return any(keyword in text for keyword in KEYWORDS)


def main():

    if not GKG_FILE.exists():
        raise FileNotFoundError(
            f"GDELT file not found: {GKG_FILE}"
        )

    print("Reading real GDELT GKG data...")
    
    df = pd.read_csv(
        GKG_FILE,
        sep="\t",
        header=None,
        names=COLUMNS,
        low_memory=False,
    )

    print(f"GKG rows loaded: {len(df):,}")

    # Keep records related to the TruthLens research topic.
    relevant = df[df.apply(relevant_record, axis=1)].copy()

    print(f"Topic-relevant GKG rows: {len(relevant):,}")

    records = []

    for _, row in relevant.iterrows():

        urls = split_urls(row["SOURCEURLS"])

        for url in urls:

            records.append(
                {
                    "title": "",
                    "url": url,
                    "domain": extract_domain(url),
                    "language": "Unknown",
                    "sourcecountry": "Unknown",
                    "seendate": str(row["DATE"]),
                    "themes": str(row["THEMES"]),
                    "tone": str(row["TONE"]),
                    "persons": str(row["PERSONS"]),
                    "organizations": str(row["ORGANIZATIONS"]),
                }
            )

    result = pd.DataFrame(records)

    if result.empty:
        print("No relevant URLs were found.")
        return

    result = result.drop_duplicates(
        subset=["url"],
        keep="first",
    )

    # Keep a manageable portfolio/demo dataset.
    result = result.head(500)

    save_cache(
        result,
        query="GDELT GKG snapshot 20260915 - misinformation/deepfake/fact-check related",
        source="GDELT GKG daily snapshot",
    )

    print()
    print(f"Created {len(result):,} real-world GDELT records.")
    print(
        "Saved cache:",
        ROOT / "data" / "raw" / "gdelt" / "articles_cache.csv",
    )

    print()
    print("Sample records:")
    print(
        result[
            [
                "url",
                "domain",
                "seendate",
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()