# TruthLens V2 — Stage 6 Setup

Stage 6 adds optional external fact-check evidence retrieval through Google's Fact Check Tools API.

## 1. Create/choose a Google Cloud project

Use the Google Cloud Console and create or select a project. Google documents API-key creation and recommends restricting API keys. Do not paste the key into `app.py` or commit it to Git.

Official documentation:
- https://developers.google.com/fact-check/tools/api/reference/rest
- https://docs.cloud.google.com/docs/authentication/api-keys

## 2. Enable the Fact Check Tools API

In Google Cloud Console, open **APIs & Services → Library**, search for **Fact Check Tools API**, and enable it for the project.

## 3. Create an API key

Open **APIs & Services → Credentials → Create credentials → API key**.

For a portfolio project, add API restrictions so the key can only call the API you need. Keep the key private.

## 4. Configure Windows

In the same CMD window used for TruthLens, with the virtual environment active, run:

```cmd
set GOOGLE_FACTCHECK_API_KEY=PASTE_YOUR_KEY_HERE
```

This sets the variable for the current CMD session only.

Then start Streamlit:

```cmd
streamlit run app.py
```

Do not send your API key to ChatGPT or include it in screenshots.

## 5. Optional: Streamlit secrets

For Streamlit deployments, you can use `.streamlit/secrets.toml`:

```toml
GOOGLE_FACTCHECK_API_KEY = "YOUR_KEY_HERE"
```

Keep this file out of Git.

## 6. Test

Open an article in **Article investigation**. Under **External fact-check evidence**, enter a specific claim and click **Search fact-check evidence**.

A match means an indexed fact-check source exists for the query. No match does not establish that a claim is true.
