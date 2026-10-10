from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "raw" / "gdelt"
CACHE_CSV = CACHE_DIR / "articles_cache.csv"
META_JSON = CACHE_DIR / "cache_meta.json"


def save_cache(df: pd.DataFrame, query: str = "", source: str = "GDELT DOC 2.0") -> Path:
    """Persist real-world records locally so the dashboard can run without repeated API calls."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_CSV, index=False, encoding="utf-8-sig")
    meta = {
        "source": source,
        "query": query,
        "rows": int(len(df)),
        "saved_at_utc": pd.Timestamp.utcnow().isoformat(),
    }
    META_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return CACHE_CSV


def load_cache() -> tuple[pd.DataFrame, dict]:
    if not CACHE_CSV.exists():
        return pd.DataFrame(), {}

    try:
        df = pd.read_csv(CACHE_CSV)
    except Exception:
        return pd.DataFrame(), {}

    meta = {}
    if META_JSON.exists():
        try:
            meta = json.loads(META_JSON.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    return df, meta


def cache_age_minutes(meta: dict) -> Optional[float]:
    stamp = meta.get("saved_at_utc")
    if not stamp:
        return None
    try:
        saved = pd.Timestamp(stamp)
        now = pd.Timestamp.now(tz="UTC")
        return max(0.0, (now - saved).total_seconds() / 60.0)
    except Exception:
        return None
