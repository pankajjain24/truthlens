from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

df = pd.read_csv("data/processed/liar_prepared.csv")
train_df = df[df.split=="train"].copy()
valid_df = df[df.split=="valid"].copy()
test_df = df[df.split=="test"].copy()

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1,2),
        min_df=2,
        max_features=120000,
        sublinear_tf=True
    )),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

model.fit(train_df.statement, train_df.binary_label)

metrics = {}
for name, part in [("valid", valid_df), ("test", test_df)]:
    pred = model.predict(part.statement)
    proba = model.predict_proba(part.statement)[:,1]
    metrics[name] = {
        "classification_report": classification_report(
            part.binary_label, pred, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(part.binary_label, pred).tolist(),
        "roc_auc": float(roc_auc_score(part.binary_label, proba))
    }

Path("data/processed/model").mkdir(parents=True, exist_ok=True)
joblib.dump(model, "data/processed/model/liar_tfidf_logreg.joblib")

with open("data/processed/model/metrics.json","w") as f:
    json.dump(metrics, f, indent=2)

print(json.dumps(
    {k: {"roc_auc":v["roc_auc"]} for k,v in metrics.items()},
    indent=2
))
