from pathlib import Path
import pandas as pd

src = Path("data/raw/gdelt/articles.csv")
out = Path("data/processed/powerbi")
out.mkdir(parents=True, exist_ok=True)

if not src.exists():
    raise SystemExit("Run src/fetch_gdelt.py first.")

df = pd.read_csv(src)

if "seendate" in df.columns:
    df["seendate"] = pd.to_datetime(df["seendate"], errors="coerce")

# Keep metadata/URLs rather than copyrighted article bodies.
keep=[c for c in [
    "url","url_mobile","title","seendate","socialimage","domain",
    "language","sourcecountry"
] if c in df.columns]

df[keep].to_csv(out/"gdelt_articles.csv", index=False)

if "seendate" in df.columns:
    daily=(df.dropna(subset=["seendate"])
        .assign(date=lambda x:x.seendate.dt.date)
        .groupby("date")
        .size().reset_index(name="article_count"))
    daily.to_csv(out/"daily_trend.csv", index=False)

if "domain" in df.columns:
    domains=(df.groupby("domain").size()
        .reset_index(name="article_count")
        .sort_values("article_count",ascending=False))
    domains.to_csv(out/"source_distribution.csv", index=False)

print("Power BI exports written to", out)
