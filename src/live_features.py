import re
import pandas as pd
import numpy as np

EMOTION = {
    "urgent": ["urgent","breaking","shocking","warning","alert","must share","exposed"],
    "fear": ["danger","scam","attack","threat","warning","panic","deadly"],
    "certainty": ["100%","always","never","guaranteed","proof","definitely"]
}

def add_text_features(df):
    x=df.copy()
    text=x.get("title", pd.Series("", index=x.index)).fillna("").astype(str).str.lower()

    x["text_length"]=text.str.len()
    for name, words in EMOTION.items():
        pat="|".join(re.escape(w) for w in words)
        x[name+"_hits"]=text.str.count(pat)

    x["emotion_signal"]=(x["urgent_hits"]+x["fear_hits"]+x["certainty_hits"]).clip(0,8)
    x["emotion_signal"]=100*x["emotion_signal"]/8
    return x

def host_column(df):
    if "domain" in df.columns:
        return df["domain"]
    if "url" in df.columns:
        return df["url"].str.extract(r"https?://([^/]+)", expand=False)
    return pd.Series("unknown", index=df.index)
