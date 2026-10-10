CREATE TABLE articles (
    article_id BIGSERIAL PRIMARY KEY,
    url TEXT UNIQUE,
    title TEXT,
    domain TEXT,
    published_at TIMESTAMP,
    language TEXT,
    source_country TEXT,
    query TEXT
);

CREATE TABLE model_predictions (
    article_id BIGINT REFERENCES articles(article_id),
    model_name TEXT,
    positive_probability NUMERIC(7,5),
    risk_band TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE fact_checks (
    article_id BIGINT REFERENCES articles(article_id),
    claim_text TEXT,
    reviewer TEXT,
    review_url TEXT,
    rating TEXT,
    review_date DATE
);

CREATE INDEX idx_articles_published ON articles(published_at);
CREATE INDEX idx_articles_domain ON articles(domain);
CREATE INDEX idx_predictions_risk ON model_predictions(risk_band);
