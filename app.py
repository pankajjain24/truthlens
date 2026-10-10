import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from src.live_features import add_text_features, host_column
from src.factcheck_client import search_fact_checks
from src.data_store import load_cache, save_cache, cache_age_minutes


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="TruthLens V2 — Stage 6",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .hero { padding: 1.2rem 1.4rem; border: 1px solid rgba(128,128,128,.25);
            border-radius: 18px; margin-bottom: 1rem; }
    .kpi { border: 1px solid rgba(128,128,128,.25); border-radius: 16px;
           padding: 16px; min-height: 115px; }
    .kpi-label { font-size: 13px; opacity: .7; }
    .kpi-value { font-size: 28px; font-weight: 700; margin-top: 6px; }
    .risk-high { padding: 12px 16px; border-radius: 12px; background: rgba(220,38,38,.12);
                 border: 1px solid rgba(220,38,38,.35); }
    .risk-medium { padding: 12px 16px; border-radius: 12px; background: rgba(234,179,8,.12);
                   border: 1px solid rgba(234,179,8,.35); }
    .risk-low { padding: 12px 16px; border-radius: 12px; background: rgba(34,197,94,.12);
                border: 1px solid rgba(34,197,94,.35); }
    .signal-box { border: 1px solid rgba(128,128,128,.25); border-radius: 14px; padding: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# CONSTANTS
# ============================================================
GDELT_API = "https://api.gdeltproject.org/api/v2/doc/doc"
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "data" / "processed" / "model" / "liar_tfidf_logreg.joblib"
METRICS_PATH = ROOT / "data" / "processed" / "model" / "metrics.json"

# ============================================================
# SESSION STATE
# ============================================================
DEFAULT_QUERY = '"misinformation" OR "deepfake" OR "fake news"'

for key, value in {
    "gdelt": pd.DataFrame(),
    "query": "",
    "last_fetch": None,
    "data_source": "none",
    "cache_meta": {},
    "gdelt_rate_limit_until": 0.0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPERS
# ============================================================
def fetch_gdelt(query: str, timespan: str, maxrecords: int):
    """Fetch real-world GDELT article metadata with conservative retries."""
    maxrecords = min(int(maxrecords), 50)
    params = {
        "query": query.strip(),
        "mode": "artlist",
        "format": "json",
        "maxrecords": maxrecords,
        "timespan": timespan,
        "sort": "datedesc",
    }
    headers = {"User-Agent": "TruthLens/2.0 research-dashboard"}
    last_error = None

    for attempt in range(4):
        try:
            response = requests.get(GDELT_API, params=params, headers=headers, timeout=60)

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                try:
                    wait = max(30, int(retry_after))
                except (TypeError, ValueError):
                    wait = 300
                # Do not repeatedly hammer a rate-limited endpoint. A single
                # request is enough to detect the condition; the UI then enters
                # a cooldown and encourages local-cache/import workflows.
                return None, (
                    f"GDELT is temporarily rate-limiting this connection (HTTP 429). "
                    f"Wait about {wait} seconds, then try again. No synthetic fallback is used."
                )

            if response.status_code in [500, 502, 503, 504]:
                wait = 5 * (attempt + 1)
                if attempt < 3:
                    st.warning(f"GDELT returned HTTP {response.status_code}. Retrying in {wait} seconds...")
                    time.sleep(wait)
                    continue

            response.raise_for_status()
            payload = response.json()
            return pd.DataFrame(payload.get("articles", [])), None

        except requests.exceptions.Timeout:
            last_error = "The GDELT request timed out."
        except requests.exceptions.RequestException as exc:
            last_error = str(exc)
        except ValueError:
            last_error = "GDELT returned a response that was not valid JSON."

        if attempt < 3:
            time.sleep(3 * (attempt + 1))

    return None, last_error or "Unknown GDELT error."


def prepare_live_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich GDELT article metadata."""
    df = raw_df.copy()
    if df.empty:
        return df

    for col in ["title", "url", "domain", "language", "sourcecountry", "seendate"]:
        if col not in df.columns:
            df[col] = ""

    df["title"] = df["title"].fillna("").astype(str)
    df["url"] = df["url"].fillna("").astype(str)
    df["time"] = pd.to_datetime(df["seendate"], errors="coerce", utc=True)
    df["domain"] = host_column(df).fillna("unknown").astype(str)
    df = add_text_features(df)
    return df.drop_duplicates(subset=["url"], keep="first")


def risk_band(score):
    if pd.isna(score):
        return "Not available"
    if score >= 70:
        return "HIGH"
    if score >= 40:
        return "MEDIUM"
    return "LOW"


def source_signal_table(df):
    if df.empty:
        return pd.DataFrame()
    out = (
        df.groupby("domain", dropna=False)
        .agg(articles=("title", "count"), avg_emotion=("emotion_signal", "mean"))
        .reset_index()
    )
    out["attention_signal"] = (
        0.65 * out["avg_emotion"] + 0.35 * np.minimum(out["articles"] * 5, 100)
    ).clip(0, 100)
    return out.sort_values("attention_signal", ascending=False)


def load_model_and_metrics():
    model = None
    metrics = {}
    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
        except Exception as exc:
            st.warning(f"Model file exists but could not be loaded: {exc}")
    if METRICS_PATH.exists():
        try:
            metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            metrics = {}
    return model, metrics


def model_feature_explanation(model, text: str, top_n: int = 8):
    """Explain TF-IDF x Logistic Regression contributions using meaningful terms."""
    try:
        vectorizer = model.named_steps["tfidf"]
        classifier = model.named_steps["clf"]
        matrix = vectorizer.transform([text])
        feature_names = np.asarray(vectorizer.get_feature_names_out())
        nonzero = matrix.nonzero()[1]
        if len(nonzero) == 0:
            return pd.DataFrame(columns=["term", "contribution", "direction"])

        values = matrix.toarray()[0]
        coefs = classifier.coef_[0]
        contributions = values[nonzero] * coefs[nonzero]
        terms = feature_names[nonzero]

        explanation = pd.DataFrame({"term": terms, "contribution": contributions})
        # Filter very short/stop-word-like tokens. Keep n-grams with useful lexical content.
        stop_like = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
            "was", "were", "with", "will", "would", "you", "your",
        }
        explanation = explanation[~explanation["term"].str.lower().isin(stop_like)]
        explanation = explanation[explanation["term"].str.len() >= 3]
        if explanation.empty:
            return pd.DataFrame(columns=["term", "contribution", "direction"])

        explanation["direction"] = np.where(
            explanation["contribution"] >= 0,
            "pushes toward benchmark-positive class",
            "pushes toward benchmark-negative class",
        )
        explanation["contribution"] = explanation["contribution"].round(4)

        positive = explanation[explanation["contribution"] > 0].sort_values("contribution", ascending=False).head(top_n)
        negative = explanation[explanation["contribution"] < 0].sort_values("contribution").head(top_n)
        return pd.concat([positive, negative]).drop_duplicates("term")
    except Exception:
        return pd.DataFrame(columns=["term", "contribution", "direction"])


def model_metrics_table(metrics):
    rows = []
    for split in ["valid", "test"]:
        item = metrics.get(split, {})
        report = item.get("classification_report", {})
        rows.append({
            "split": split,
            "roc_auc": item.get("roc_auc"),
            "accuracy": report.get("accuracy"),
            "macro_f1": report.get("macro avg", {}).get("f1-score"),
            "weighted_f1": report.get("weighted avg", {}).get("f1-score"),
        })
    return pd.DataFrame(rows)


# ============================================================
# LOAD LOCAL CACHE FIRST
# ============================================================
# The cache is intentionally loaded before any network action. This means
# opening/rerunning the dashboard does not call GDELT. The cache directory is
# created by save_cache() after the first successful fetch/import.
if st.session_state["gdelt"].empty:
    cached_df, cached_meta = load_cache()
    if not cached_df.empty:
        st.session_state["gdelt"] = prepare_live_data(cached_df)
        st.session_state["cache_meta"] = cached_meta
        st.session_state["query"] = cached_meta.get("query", "")
        st.session_state["data_source"] = cached_meta.get("source", "local cache")


# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div class="hero">
        <h1>🛡️ TruthLens V2</h1>
        <p>Real-world information intelligence • evidence-first • human-verifiable</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "TruthLens does not declare an article true or false. It combines measurable signals "
    "to prioritize items for verification. Live article metadata comes from GDELT. "
    "The classifier is trained on the real LIAR benchmark and produces a benchmark-derived "
    "model signal, not a real-world truth verdict."
)

cache_meta = st.session_state.get("cache_meta", {})
source_label = st.session_state.get("data_source", "none")
query_label = cache_meta.get("query", st.session_state.get("query", ""))
st.caption(
    f"Data source: **{source_label}** • Query: **{query_label or 'not set'}** • "
    "No synthetic news records are generated."
)


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("🌐 Real-world data")
query = st.sidebar.text_input(
    "GDELT query",
    value=st.session_state["query"] or DEFAULT_QUERY,
    help="Try misinformation, deepfake, AI scam, fake news, or another GDELT search term.",
)

timespan = st.sidebar.selectbox("Time window", ["1d", "3d", "7d", "30d"], index=2)
maxrecords = st.sidebar.slider("Maximum articles", 10, 50, 50, 10)

cache_age = cache_age_minutes(st.session_state.get("cache_meta", {}))
if cache_age is not None:
    st.sidebar.caption(f"Local cache: {len(st.session_state['gdelt']):,} rows • {cache_age:.0f} min old")
else:
    st.sidebar.caption("No local cache yet. A successful fetch/import will create one at data/raw/gdelt/.")

rate_limit_until = float(st.session_state.get("gdelt_rate_limit_until", 0.0))
rate_limit_active = time.time() < rate_limit_until
if rate_limit_active:
    remaining = max(1, int(rate_limit_until - time.time()))
    st.sidebar.warning(f"GDELT cooldown active: {remaining}s remaining. Use local cache/import while waiting.")
fetch_button = st.sidebar.button(
    "🔄 Refresh from GDELT",
    type="primary",
    width="stretch",
    disabled=rate_limit_active,
)

with st.sidebar.expander("📥 Import real-world CSV", expanded=False):
    uploaded = st.file_uploader(
        "Upload a GDELT ArticleList CSV",
        type=["csv"],
        help="Import only real-world records obtained from sources such as GDELT.",
    )
    if uploaded is not None and st.button("Use uploaded data", width="stretch"):
        try:
            imported = pd.read_csv(uploaded)
            cleaned = prepare_live_data(imported)
            if cleaned.empty:
                st.error("The CSV contained no usable rows.")
            else:
                save_cache(imported, query=query, source="User-imported real-world CSV")
                st.session_state["gdelt"] = cleaned
                st.session_state["cache_meta"] = {
                    "source": "User-imported real-world CSV",
                    "query": query,
                    "rows": int(len(imported)),
                    "saved_at_utc": pd.Timestamp.utcnow().isoformat(),
                }
                st.session_state["data_source"] = "uploaded CSV"
                st.success(f"Loaded {len(cleaned):,} real-world rows.")
                st.rerun()
        except Exception as exc:
            st.error(f"Could not read the CSV: {exc}")


# ============================================================
# FETCH DATA
# ============================================================
if fetch_button:
    if not query.strip():
        st.sidebar.error("Please enter a search query.")
    else:
        with st.spinner("Fetching real-world GDELT data..."):
            data, error = fetch_gdelt(query, timespan, maxrecords)

        if error:
            if "HTTP 429" in error:
                st.session_state["gdelt_rate_limit_until"] = time.time() + 300
            st.error(f"❌ {error}")
            if not st.session_state["gdelt"].empty:
                st.info("Existing local real-world data was retained. No synthetic fallback is used.")
            else:
                st.info("No local dataset is available. Wait for the rate limit to clear or import a real-world CSV.")
        elif data is not None and not data.empty:
            cleaned = prepare_live_data(data)
            save_cache(data, query=query, source="GDELT DOC 2.0")
            st.session_state["gdelt"] = cleaned
            st.session_state["query"] = query
            st.session_state["last_fetch"] = pd.Timestamp.now()
            st.session_state["cache_meta"] = {
                "source": "GDELT DOC 2.0",
                "query": query,
                "rows": int(len(data)),
                "saved_at_utc": pd.Timestamp.utcnow().isoformat(),
            }
            st.session_state["data_source"] = "GDELT live + local cache"
            st.success(f"✅ Loaded {len(cleaned):,} real-world articles and saved them locally.")
        else:
            st.warning("GDELT returned no matching articles. Try a simpler query.")


# ============================================================
# DATA
# ============================================================
df = st.session_state["gdelt"].copy()

if df.empty:
    model, metrics = load_model_and_metrics()
    st.warning("👈 Load real-world data using **Refresh from GDELT** or **Import real-world CSV**.")
    st.info(
        "No local GDELT cache is currently available at `data/raw/gdelt/articles_cache.csv`. "
        "The dashboard will not invent or synthesize news records. Once a real-world CSV is "
        "successfully fetched/imported, it will be persisted locally for future runs."
    )
    if model is not None:
        st.success("🤖 LIAR benchmark model: READY")
        test_item = metrics.get("test", {})
        test_report = test_item.get("classification_report", {})
        a,b,c = st.columns(3)
        a.metric("Test ROC-AUC", f"{test_item.get('roc_auc', 0):.3f}")
        b.metric("Test accuracy", f"{test_report.get('accuracy', 0):.1%}")
        c.metric("Test macro F1", f"{test_report.get('macro avg', {}).get('f1-score', 0):.3f}")
    else:
        st.warning("🤖 LIAR benchmark model: NOT FOUND. Run `python src\prepare_liar.py` then `python src\train_model.py`.")
    st.markdown(
        """
        ### Try these queries
        **Deepfakes** — `deepfake`

        **Misinformation** — `misinformation`

        **AI scams** — `"AI scam"`

        **Fake news** — `"fake news"`

        **Digital fraud** — `"digital fraud"`
        """
    )
    st.stop()


# ============================================================
# MODEL INTEGRATION — STAGE 5
# ============================================================
model, metrics = load_model_and_metrics()
model_available = model is not None

if model_available:
    text = df["title"].fillna("").astype(str)
    probabilities = model.predict_proba(text)[:, 1]
    df["benchmark_probability"] = probabilities
    df["model_risk_signal"] = (100 * (1 - probabilities)).round(1)
    df["risk_band"] = df["model_risk_signal"].apply(risk_band)
else:
    df["benchmark_probability"] = np.nan
    df["model_risk_signal"] = np.nan
    df["risk_band"] = "Not available"


# ============================================================
# KPI CARDS
# ============================================================
high_emotion = int((df["emotion_signal"] >= 50).sum())
high_risk = int((df["model_risk_signal"] >= 70).sum()) if model_available else 0
unique_domains = int(df["domain"].nunique())

c1, c2, c3, c4, c5 = st.columns(5)
for col, (label, value) in zip(
    [c1, c2, c3, c4, c5],
    [
        ("Real articles", f"{len(df):,}"),
        ("Unique domains", f"{unique_domains:,}"),
        ("High-emotion", f"{high_emotion:,}"),
        ("Model", "READY" if model_available else "NOT TRAINED"),
        ("High model-risk signals", f"{high_risk:,}" if model_available else "—"),
    ],
):
    col.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>',
        unsafe_allow_html=True,
    )

st.divider()


# ============================================================
# MODEL PERFORMANCE — NEW STAGE 5 SECTION
# ============================================================
st.subheader("🤖 ML model status & benchmark performance")

if model_available:
    m1, m2, m3, m4 = st.columns(4)
    test_item = metrics.get("test", {})
    test_report = test_item.get("classification_report", {})
    m1.metric("Test ROC-AUC", f"{test_item.get('roc_auc', 0):.3f}")
    m2.metric("Test accuracy", f"{test_report.get('accuracy', 0):.1%}")
    m3.metric("Test macro F1", f"{test_report.get('macro avg', {}).get('f1-score', 0):.3f}")
    m4.metric("Validation ROC-AUC", f"{metrics.get('valid', {}).get('roc_auc', 0):.3f}")

    metric_df = model_metrics_table(metrics)
    if not metric_df.empty:
        st.dataframe(metric_df, width="stretch", hide_index=True)

    st.caption(
        "These metrics describe performance on the LIAR benchmark splits. The live GDELT articles are a different data distribution, so the model signal should not be interpreted as a calibrated probability of misinformation."
    )
else:
    st.info("Train the LIAR benchmark model to enable article-level ML signals.")


# ============================================================
# AUTOMATIC DATA INSIGHT
# ============================================================
st.subheader("💡 Automatic data insight")
top_domain = df.groupby("domain").size().sort_values(ascending=False).index[0]
top_topic_signal = "emotional-language intensity" if df["emotion_signal"].mean() >= 35 else "relatively neutral language"
st.write(
    f"The retrieved sample contains **{len(df):,} articles** across **{unique_domains:,} domains**. "
    f"The most represented domain is **{top_domain}**, while the overall language signal is characterized by **{top_topic_signal}**."
)
st.caption("This describes the retrieved GDELT sample only; it is not a prevalence estimate for misinformation in society.")


# ============================================================
# TRENDS + DOMAINS
# ============================================================
left, right = st.columns(2)
with left:
    st.subheader("📈 Coverage over time")
    trend_df = df.dropna(subset=["time"]).set_index("time").resample("D").size().reset_index(name="articles")
    if not trend_df.empty:
        fig = px.line(trend_df, x="time", y="articles", markers=True, labels={"time": "Date", "articles": "Articles"})
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No usable date information returned by GDELT.")

with right:
    st.subheader("🌐 Top domains in retrieved sample")
    domain_df = df.groupby("domain").size().reset_index(name="articles").sort_values("articles", ascending=False).head(15)
    fig = px.bar(domain_df, x="articles", y="domain", orientation="h", labels={"articles": "Article count", "domain": "Domain"})
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, width="stretch")


# ============================================================
# EMOTION + MODEL RISK
# ============================================================
left, right = st.columns(2)
with left:
    st.subheader("🧠 Emotional / urgency signal")
    fig = px.histogram(df, x="emotion_signal", nbins=20, labels={"emotion_signal": "Language signal"})
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, width="stretch")

with right:
    st.subheader("⚠️ Benchmark-derived risk signal")
    if model_available:
        fig = px.histogram(df, x="model_risk_signal", nbins=20, labels={"model_risk_signal": "Risk prioritization signal"})
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("The score is derived from the LIAR-trained classifier. It is not a probability that the live article is false.")
    else:
        st.info("Train the real LIAR benchmark model to enable this analysis.")


# ============================================================
# RISK VS EMOTION
# ============================================================
if model_available:
    st.divider()
    st.subheader("🔬 Model risk signal vs. emotional language")
    corr = df[["emotion_signal", "model_risk_signal"]].corr().iloc[0, 1]
    fig = px.scatter(
        df,
        x="emotion_signal",
        y="model_risk_signal",
        hover_data=["title", "domain"],
        labels={"emotion_signal": "Emotional / urgency signal", "model_risk_signal": "Model-derived risk signal"},
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, width="stretch")
    if pd.notna(corr):
        st.metric("Sample correlation", f"{corr:.2f}")
    st.caption("Correlation is descriptive for this retrieved sample and does not establish causation.")


# ============================================================
# SOURCE ANALYTICS
# ============================================================
st.divider()
st.subheader("🌐 Source-level attention signals")
source_df = source_signal_table(df)
if not source_df.empty:
    st.dataframe(source_df.head(20), width="stretch", hide_index=True)
    st.caption("Attention signal combines article representation and language signals. It is not a publisher credibility score.")


# ============================================================
# ARTICLE INVESTIGATION
# ============================================================
st.divider()
st.subheader("🔎 Article investigation")
article_options = df.index.tolist()
selected_idx = st.selectbox(
    "Select an article",
    article_options,
    format_func=lambda i: f"{df.loc[i, 'title'][:150]} — {df.loc[i, 'domain']}",
)
row = df.loc[selected_idx]

st.markdown(f"### {row.get('title', 'Untitled')}")
st.write(
    f"**Domain:** {row.get('domain', 'Unknown')}  \n"
    f"**Language:** {row.get('language', 'Unknown')}  \n"
    f"**Country:** {row.get('sourcecountry', 'Unknown')}  \n"
    f"**Published/seen:** {row.get('seendate', 'Unknown')}"
)

if row.get("url"):
    st.markdown(f"**Source article:** [{row.get('url')}]({row.get('url')})")

if model_available and pd.notna(row.get("model_risk_signal")):
    score = float(row["model_risk_signal"])
    band = risk_band(score)
    x1, x2, x3, x4 = st.columns(4)
    x1.metric("Model-derived risk", f"{score:.0f}/100")
    x2.metric("Benchmark-positive probability", f"{row['benchmark_probability']:.1%}")
    x3.metric("Language signal", f"{row['emotion_signal']:.0f}/100")
    x4.metric("Review priority", band)

    if band == "HIGH":
        st.markdown('<div class="risk-high">⚠️ <b>HIGH PRIORITY FOR REVIEW</b><br>This item has a high benchmark-derived risk signal. Verify it using primary sources and independent evidence.</div>', unsafe_allow_html=True)
    elif band == "MEDIUM":
        st.markdown('<div class="risk-medium">🟡 <b>MEDIUM PRIORITY FOR REVIEW</b><br>Additional evidence should be checked before drawing a conclusion.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="risk-low">🟢 <b>LOW MODEL-RISK SIGNAL</b><br>This does not establish that the article is true. Normal source verification still applies.</div>', unsafe_allow_html=True)

    # Explain the selected prediction.
    with st.expander("🧩 Explain this model signal", expanded=True):
        explanation = model_feature_explanation(model, str(row.get("title", "")))
        if explanation.empty:
            st.info("No usable TF-IDF features were found for this title.")
        else:
            st.dataframe(explanation, width="stretch", hide_index=True)
            st.caption("A positive contribution pushes the model toward its benchmark-positive class; a negative contribution pushes it toward its benchmark-negative class. These are model features, not evidence that the underlying claim is true or false.")

else:
    st.info("The ML model is not available yet. Complete the LIAR training step to enable model-derived signals.")


# ============================================================
# FACT-CHECK EVIDENCE — STAGE 6
# ============================================================
st.markdown("#### 🔍 External fact-check evidence")

# Show configuration status without exposing the secret.
try:
    from src.factcheck_client import get_api_key
    factcheck_configured = bool(get_api_key())
except Exception:
    factcheck_configured = False

if factcheck_configured:
    st.success("Google Fact Check Tools API: configured")
else:
    st.info("Google Fact Check Tools API: not configured. Add GOOGLE_FACTCHECK_API_KEY to enable evidence retrieval.")

fact_query = st.text_area(
    "Claim to search",
    value=str(row.get("title", ""))[:500],
    height=90,
    help="Search for an existing fact-check of this claim. A match is evidence to inspect, not an automatic verdict on the selected article.",
)

if st.button("🔎 Search fact-check evidence", disabled=not fact_query.strip()):
    with st.spinner("Searching available fact-check records..."):
        result = search_fact_checks(fact_query, page_size=10)

    if "error" in result:
        st.error(result["error"])
        if result.get("code") == "missing_key":
            st.markdown("Configure the key as described in the Stage 6 setup instructions.")
        elif result.get("details"):
            with st.expander("Technical response"):
                st.code(result["details"])
    else:
        claims = result.get("claims", [])
        if not claims:
            st.warning("No matching fact-check records were returned for this query.")
            st.caption("No match does not mean the claim is true; it only means the API returned no matching indexed fact-check record.")
        else:
            st.success(f"Found {len(claims)} matching fact-check record(s).")
            for claim in claims:
                st.markdown(f"**Claim indexed by fact-check source:** {claim.get('text', 'Unknown')}")
                reviews = claim.get("claimReview", [])
                if not reviews:
                    st.caption("The API returned the claim but no ClaimReview details in this record.")
                for review in reviews:
                    publisher = review.get("publisher", {}).get("name", "Unknown")
                    site = review.get("publisher", {}).get("site", "")
                    rating = review.get("textualRating", "Unknown")
                    review_date = review.get("reviewDate", "Unknown")
                    title = review.get("title", "")
                    url = review.get("url", "")

                    a, b = st.columns([2, 1])
                    with a:
                        st.markdown(f"**Reviewer:** {publisher}")
                        if site:
                            st.caption(f"Publisher site: {site}")
                        if title:
                            st.write(f"**Review title:** {title}")
                    with b:
                        st.metric("Reported rating", rating)
                        st.caption(f"Review date: {review_date}")
                    if url:
                        st.markdown(f"🔗 [Open the original fact-check]({url})")
                st.divider()

st.caption("Fact-check results are external evidence retrieved from indexed fact-check sources. TruthLens does not convert a rating into a definitive truth/falsehood verdict.")


# ============================================================
# REVIEW QUEUE
# ============================================================
st.divider()
st.subheader("🚨 Review queue")
queue_columns = [
    c for c in ["title", "domain", "language", "sourcecountry", "emotion_signal", "model_risk_signal", "risk_band", "url"]
    if c in df.columns
]
queue = df.sort_values(["model_risk_signal", "emotion_signal"], ascending=False, na_position="last")[queue_columns].head(50)
st.dataframe(queue, width="stretch", hide_index=True)

csv_data = queue.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download review queue", data=csv_data, file_name="truthlens_review_queue.csv", mime="text/csv")


# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("TruthLens V2 • Real-world GDELT data • LIAR benchmark model • Stage 6 ML + external fact-check evidence • Evidence-first workflow • Human verification required")
