# 🛡️ TruthLens V2 — Real-World Misinformation Intelligence

TruthLens V2 is an end-to-end **Data Analyst + ML + product analytics project** built around real-world public data.

## Important: no synthetic dataset

V2 deliberately does **not** ship with fabricated records or fabricated findings.

It uses:

1. **GDELT DOC 2.0** for live/current news coverage and trends.
2. **LIAR** for supervised model training/evaluation. LIAR contains 12.8K human-labeled real-world statements collected from PolitiFact.
3. **FakeNewsNet** as an optional second benchmark/source. Its public repository provides minimal Politifact/GossipCop CSVs and documents its data/usage limitations.
4. **Google Fact Check Tools API** as an optional evidence layer for searching fact-checked claims.

Sources and methodology are documented in `docs/data_sources.md`.

## Product idea

TruthLens does not pretend an ML model can determine objective truth from text alone.

Instead, it produces an **Information Risk Assessment** based on:

- model probability,
- language/emotion signals,
- evidence availability,
- historical fact-check matches,
- source/claim context,
- and unusual coverage/engagement signals where available.

The product clearly separates:

**Model signal → Evidence → Human-verifiable fact-check**

## Architecture

```text
                 REAL-WORLD DATA
                       │
        ┌──────────────┼─────────────────┐
        ↓              ↓                 ↓
     GDELT           LIAR          Google Fact Check
   live news       benchmark            API
        │              │                 │
        ↓              ↓                 ↓
       ingestion     training        evidence lookup
        │              │                 │
        └──────────────┼─────────────────┘
                       ↓
                 PostgreSQL / CSV
                       ↓
             Feature engineering
                       ↓
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
       NLP          Risk engine     Trend engine
        │              │              │
        └──────────────┼──────────────┘
                       ↓
               Streamlit product
                       +
                 Power BI-ready
                    datasets
```

## V2 features

### Dashboard
- live GDELT search
- risk distribution
- topic trends
- source/host distribution
- emotion/tone analysis
- high-risk queue
- claim investigation
- fact-check evidence lookup
- model confidence
- uncertainty indicator
- downloadable analysis dataset

### ML
- TF-IDF + Logistic Regression baseline
- optional calibrated probability
- optional transformer upgrade
- temporal holdout evaluation
- precision / recall / F1 / confusion matrix
- model card and limitations

### Analytics
- topic-level risk
- source-level patterns
- tone vs coverage
- daily trend/anomaly flags
- evidence coverage
- false-positive review workflow

## Quick start

### 1. Install

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Get real LIAR data

The repository intentionally does not redistribute the LIAR data.

Download `train.tsv`, `valid.tsv`, and `test.tsv` from the public LIAR repository listed in `docs/data_sources.md`, then place them here:

```text
data/raw/liar/
    train.tsv
    valid.tsv
    test.tsv
```

Run:

```bash
python src/prepare_liar.py
python src/train_model.py
```

### 3. Run the application with resilient local caching

```bash
streamlit run app.py
```

TruthLens now persists every successful GDELT retrieval under:

```text
data/raw/gdelt/articles_cache.csv
data/raw/gdelt/cache_meta.json
```

The dashboard loads that real-world cache automatically on restart, so a GDELT 429 rate-limit response does not erase your previous analysis. If GDELT is temporarily unavailable, you can also import a real-world GDELT ArticleList CSV from the sidebar. No synthetic fallback data is used.

### 4. Run the live application

```bash
streamlit run app.py
```

The application can query GDELT directly, persist successful results locally, and continue analysing the local cache when the API is rate-limited.

### 4. Optional Google Fact Check API

Set:

```bash
# Windows PowerShell
$env:GOOGLE_FACTCHECK_API_KEY="YOUR_KEY"

# macOS/Linux
export GOOGLE_FACTCHECK_API_KEY="YOUR_KEY"
```

Then the claim investigation page can search fact-checked claims.

## GDELT

TruthLens uses GDELT DOC 2.0 because it supports article search, timelines, multilingual coverage and tone-related query operators.

Example:

```bash
python src/fetch_gdelt.py --query '"deepfake" OR "misinformation"' --timespan 30d
```

The script writes only the retrieved real-world records to `data/raw/gdelt/`.

## LIAR

LIAR is a benchmark built from 12,836 human-labeled statements from PolitiFact.

The project uses it for model development, not as proof that a current article is true/false.

Labels are preserved from the original dataset and mapped only for the binary benchmark experiment.

## FakeNewsNet

FakeNewsNet can be added as a second benchmark. The original repository explicitly notes that the complete dataset cannot be distributed because of Twitter privacy policies and publisher copyrights. Use its current repository and respect its terms.

## SQL

`sql/schema.sql` creates tables for:

- articles
- model_predictions
- fact_checks
- daily_trends

`sql/analytics_queries.sql` contains portfolio-grade SQL questions.

## Power BI

Run the export script:

```bash
python src/export_powerbi.py
```

It creates Power BI-ready CSVs from real fetched data.

## Suggested portfolio questions

1. Which topics have the highest model-risk concentration?
2. How does news tone vary by topic?
3. Which claims have matching external fact-check evidence?
4. Does high emotional intensity coincide with higher risk predictions?
5. Where is the model uncertain?
6. Which domains generate the most high-risk model flags?
7. How does coverage change over time?
8. Which errors are most important for human reviewers?

## What NOT to claim

Do not write:

> "TruthLens detects fake news with 95% accuracy in the real world."

Instead write:

> "TruthLens uses a benchmark-trained classifier and evidence retrieval to prioritize information for human verification."

Benchmark performance is reported separately from live-data behavior.

## Portfolio deliverables

- Streamlit app
- ML experiment
- SQL analytics
- Power BI-ready exports
- data-source documentation
- model card
- architecture diagram
- case study
- GitHub README
