# TruthLens V2 Model Card

## Model
TF-IDF word/bi-gram features + Logistic Regression.

## Training data
LIAR benchmark from PolitiFact, using the original train/validation/test splits.

## Target
A binary benchmark mapping:
- pants-fire / false / barely-true → 0
- half-true / mostly-true / true → 1

This is a project-specific binary simplification and should not be interpreted as an objective truth scale.

## Evaluation
The training script produces:
- precision
- recall
- F1
- confusion matrix
- ROC AUC

These metrics are calculated on the original held-out test split.

## Known limitations
- LIAR consists of short statements and is not equivalent to arbitrary web articles.
- A model trained on historical PolitiFact statements may not generalize to current news, other countries, languages or domains.
- The live GDELT feed is a distribution shift.
- Headlines can omit context.
- Fact-check coverage is incomplete.
- High model risk does not mean an item is false.
- Low model risk does not mean an item is true.

## Safe product behavior
TruthLens should route high-risk or uncertain items to human verification and surface evidence instead of presenting an automated "truth" verdict.
