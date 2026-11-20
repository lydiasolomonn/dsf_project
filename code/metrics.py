##metrics 
##All models are scored with these functions, so every result is comparable.



import numpy as np
import pandas as pd

THRESHOLD = 450  # legal threshold in MJ/m² per year (3-year average)


def _top_positions(score, top):
#Positions of the top share of buildings, ranked by score (highest first). Equal scores are ordered by a shuffle that is the same every run, not by row order. This matters for the training-mean baseline, where all scores are equal.
    
    score = np.asarray(score, dtype=float)
    n_top = max(1, int(round(top * len(score))))       # for example 20% of 250 = 50 buildings
    tie_breaker = np.random.default_rng(0).random(len(score))
    order = np.lexsort((tie_breaker, -score))          # sort by score, highest first
    return order[:n_top]


def share_above_in_top(y_true, score, top, threshold=THRESHOLD):
    #Selection measure: of the buildings in the top share, how many are above the threshold
    y_true = np.asarray(y_true, dtype=float)
    top_pos = _top_positions(score, top)
    return np.mean(y_true[top_pos] > threshold)


def share_captured(y_true, score, top, threshold=THRESHOLD):
    #Of all buildings above the threshold, how many land in the top share
    y_true = np.asarray(y_true, dtype=float)
    above = y_true > threshold
    if above.sum() == 0:
        return np.nan                                  # nothing to capture
    top_pos = _top_positions(score, top)
    return above[top_pos].sum() / above.sum()


def mae(y_true, y_pred):
    #Mean absolute error in MJ/m² per year. Only for models that predict the index
    return np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)))


def evaluate(y_true,score, is_house, tops=(0.10, 0.20), predicts_index=True):
    """Helps us compare models on equal terms by scoring each one in the same table:
        share above 450, share captured and MAE, for houses and all buildings. score: higher = more likely above 450. predicts_index: False for classification (no MAE).
"""
    y_true = np.asarray(y_true, dtype=float)
    score = np.asarray(score, dtype=float)
    is_house = np.asarray(is_house, dtype=bool)

    rows = []
    for group, mask in [("houses", is_house), ("all buildings", np.ones(len(y_true), bool))]:
        row = {"group": group, "n": int(mask.sum())}
        for top in tops:
            row[f"share_above_top{int(top * 100)}"] = share_above_in_top(y_true[mask], score[mask], top)
            row[f"captured_top{int(top * 100)}"] = share_captured(y_true[mask], score[mask], top)
        row["mae"] = mae(y_true[mask], score[mask]) if predicts_index else np.nan
        rows.append(row)
    return pd.DataFrame(rows)