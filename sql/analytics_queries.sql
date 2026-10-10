-- 1. Daily coverage trend
SELECT DATE(published_at) AS day, COUNT(*) AS articles
FROM articles
GROUP BY 1
ORDER BY 1;

-- 2. Top domains in the retrieved real-world sample
SELECT domain, COUNT(*) AS articles
FROM articles
GROUP BY domain
ORDER BY articles DESC
LIMIT 25;

-- 3. High-risk queue with evidence
SELECT a.title, a.domain, p.positive_probability, f.reviewer, f.rating
FROM articles a
JOIN model_predictions p ON p.article_id=a.article_id
LEFT JOIN fact_checks f ON f.article_id=a.article_id
WHERE p.risk_band='HIGH'
ORDER BY p.positive_probability DESC;

-- 4. Evidence coverage
SELECT
  COUNT(*) FILTER (WHERE f.article_id IS NOT NULL)::float / COUNT(*) AS evidence_match_rate
FROM articles a
LEFT JOIN fact_checks f ON f.article_id=a.article_id;
