# V2 execution checklist

## A. Real benchmark model
1. Download LIAR `train.tsv`, `valid.tsv`, `test.tsv`.
2. Put them in `data/raw/liar/`.
3. Run:
   `python src/prepare_liar.py`
4. Run:
   `python src/train_model.py`

## B. Real current news
Run:
`python src/fetch_gdelt.py --query '"misinformation" OR "deepfake"' --timespan 30d --maxrecords 250`

## C. Power BI
Run:
`python src/export_powerbi.py`

Load:
- `data/processed/powerbi/gdelt_articles.csv`
- `data/processed/powerbi/daily_trend.csv`
- `data/processed/powerbi/source_distribution.csv`

## D. App
`streamlit run app.py`

## E. Optional fact-check evidence
Configure `GOOGLE_FACTCHECK_API_KEY`, then use the evidence search inside the app.
