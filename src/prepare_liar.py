from pathlib import Path
import pandas as pd

COLS = [
    "id","label","statement","subjects","speaker","job_title","state",
    "party","barely_true","false","half_true","mostly_true","pants_fire","context"
]

LABELS = ["pants-fire","false","barely-true","half-true","mostly-true","true"]

def load(path, split):
    df = pd.read_csv(path, sep="\t", header=None, names=COLS, quoting=3)
    df["split"] = split
    return df

base = Path("data/raw/liar")
frames = []
for split in ["train","valid","test"]:
    p = base / f"{split}.tsv"
    if p.exists():
        frames.append(load(p, split))

if not frames:
    raise SystemExit("Put real LIAR train.tsv/valid.tsv/test.tsv files in data/raw/liar/ first.")

df = pd.concat(frames, ignore_index=True)
df = df[df.label.isin(LABELS)].copy()

# Conservative binary benchmark:
# lower three labels -> potentially misleading
# upper three labels -> comparatively more truthful
df["binary_label"] = df["label"].map({
    "pants-fire": 0,
    "false": 0,
    "barely-true": 0,
    "half-true": 1,
    "mostly-true": 1,
    "true": 1
})

df["statement"] = df["statement"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
df.to_csv("data/processed/liar_prepared.csv", index=False)
print("Prepared real LIAR records:", len(df))
