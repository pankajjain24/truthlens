import os
from typing import Any

import requests

API = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


def get_api_key() -> str | None:
    """Read the Fact Check Tools API key without hard-coding it in source."""
    key = os.getenv("GOOGLE_FACTCHECK_API_KEY", "").strip()
    if key:
        return key

    # Optional Streamlit secrets support for local/deployed apps.
    try:
        import streamlit as st
        key = str(st.secrets.get("GOOGLE_FACTCHECK_API_KEY", "")).strip()
        return key or None
    except Exception:
        return None


def search_fact_checks(query: str, page_size: int = 10) -> dict[str, Any]:
    """Search Google Fact Check Tools for previously fact-checked claims."""
    key = get_api_key()
    if not key:
        return {
            "error": "GOOGLE_FACTCHECK_API_KEY is not configured.",
            "code": "missing_key",
        }

    query = (query or "").strip()
    if not query:
        return {"error": "Enter a claim to search.", "code": "empty_query"}

    params = {
        "query": query[:500],
        "pageSize": max(1, min(int(page_size), 10)),
        "key": key,
    }

    try:
        response = requests.get(API, params=params, timeout=30)
    except requests.RequestException as exc:
        return {"error": f"Fact Check API request failed: {exc}", "code": "network"}

    if response.status_code == 429:
        return {
            "error": "Fact Check API rate limit reached. Wait briefly and try again.",
            "code": "rate_limited",
        }

    if response.status_code in (401, 403):
        return {
            "error": "Fact Check API rejected the API key. Check that the API is enabled and the key is valid/restricted correctly.",
            "code": "auth",
            "details": response.text[:500],
        }

    if not response.ok:
        return {
            "error": f"Fact Check API HTTP {response.status_code}",
            "code": "http_error",
            "details": response.text[:500],
        }

    try:
        return response.json()
    except ValueError:
        return {
            "error": "Fact Check API returned an unexpected response.",
            "code": "invalid_json",
            "details": response.text[:500],
        }
