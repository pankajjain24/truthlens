# Real-world data sources

## 1. GDELT DOC 2.0

Official:
https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/

Use:
- live/recent news discovery
- multilingual coverage
- timeline analysis
- tone/emotionality signals

Important:
GDELT is a news-coverage source, not a truth oracle. Coverage volume is not the same as prevalence or truth.

## 2. LIAR

Repository:
https://github.com/tfs4/liar_dataset

Paper:
https://arxiv.org/abs/1705.00648

The repository describes:
- train.tsv
- valid.tsv
- test.tsv
- 12.8K statements
- labels derived from PolitiFact
- speaker/context/history metadata

Usage:
research/benchmark model development. Respect the source's stated terms.

## 3. FakeNewsNet

Repository:
https://github.com/KaiDMML/FakeNewsNet

The repository states that its complete dataset cannot be distributed because of Twitter privacy policies and publisher copyrights. Its minimal public dataset includes Politifact/GossipCop fake/real CSVs.

Use:
optional second benchmark and social-context research.

## 4. Google Fact Check Tools API

Official:
https://developers.google.com/fact-check/tools/api/reference/rest

Use:
search fact-checked claims by text or image.

Requires:
Google API configuration/key.

The API returns claim, publisher/reviewer, review URL, review date and textual rating information when available.

## Data governance

The application stores URLs/metadata rather than republishing full copyrighted article bodies.

Do not scrape or redistribute protected article text.

Do not treat source frequency, model score or sentiment as proof of falsity.
