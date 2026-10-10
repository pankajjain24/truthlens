import argparse
from pathlib import Path
import requests
import pandas as pd

from src.data_store import save_cache

API = "https://api.gdeltproject.org/api/v2/doc/doc"


def fetch(query, timespan="30d", maxrecords=250):
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": min(int(maxrecords), 50),
        "timespan": timespan,
        "sort": "datedesc",
    }
    headers = {"User-Agent": "TruthLens/2.1 research-dashboard"}
    r = requests.get(API, params=params, headers=headers, timeout=60)
    r.raise_for_status()
    payload = r.json()
    return pd.DataFrame(payload.get("articles", []))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default='"misinformation" OR "deepfake"')
    ap.add_argument("--timespan", default="30d")
    ap.add_argument("--maxrecords", type=int, default=50)
    ap.add_argument("--out", default="data/raw/gdelt/articles.csv")
    args = ap.parse_args()

    df = fetch(args.query, args.timespan, args.maxrecords)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    save_cache(df, query=args.query, source="GDELT DOC 2.0")
    print(f"Fetched {len(df):,} real-world GDELT records")
    print(f"Local cache: data/raw/gdelt/articles_cache.csv")
