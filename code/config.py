## config
## Settings and paths shared by all notebooks and modules. Change them only here.
## Use: from config import SEED, THRESHOLD, TABLES, load_cv

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
SEED = 42         # one random seed for the whole project (split, tie-breaks, models)
THRESHOLD = 450   # legal threshold in MJ/m² per year (3-year average)

# ---------------------------------------------------------------------------
# Paths (relative to this file, so notebooks work from any working directory)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"              # frozen raw files, never edited
PROCESSED = ROOT / "data" / "processed"  # building_clean.csv, cv.csv, test.csv
TABLES = ROOT / "output" / "tables"
FIGURES = ROOT / "output" / "figures"

for folder in (PROCESSED, TABLES, FIGURES):
    folder.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Loading the split data (written by 03_split_pipeline.ipynb)
# ---------------------------------------------------------------------------
def load_cv():
    """The buildings for cross-validation (one row per building, column `fold` = 0-4).
    Every notebook up to the model choice uses only this data; the test set is not in it."""
    data = pd.read_csv(PROCESSED / "cv.csv")
    data["fold"] = data["fold"].astype(int)
    assert (data.groupby("group_id")["fold"].nunique() == 1).all(), "a group appears in more than one fold"
    return data


def load_test():
    """The test set (task 16). Use it ONLY in 05_evaluation.ipynb, section 'Test set',
    exactly once, after the final model is chosen and written into the README (task 26)."""
    return pd.read_csv(PROCESSED / "test.csv")
