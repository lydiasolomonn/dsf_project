## pipeline
## Turns the cleaned table (data/processed/building_clean.csv) into numbers a model can use.
## Everything that learns from data (medians, categories, scaling, rule averages) is fitted
## inside the pipeline, so it only ever sees the training part of a fold (rule 2 in TASKS.md).

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from config import THRESHOLD


# ---------------------------------------------------------------------------
# A. Columns and settings
# ---------------------------------------------------------------------------
NUMERIC = ["floor_area", "floors", "dwellings"]
CATEGORICAL = ["construction_period", "building_class"]
HEATING = ["energy_group", "heating_system"]
MUNICIPALITY = ["municipality"]
RULE_COLUMNS = ["construction_period", "energy_group"]  # the expert rule: construction period x energy source

FEATURES = NUMERIC + CATEGORICAL + HEATING + MUNICIPALITY

MIN_FREQUENCY = 20  # categories seen fewer than 20 times in training are merged into "infrequent"
SMOOTHING = 20      # weight m of the overall mean in the expert rule (see ExpertRuleEncoder)


# ---------------------------------------------------------------------------
# B. Split the table into features, target, house flag and groups
# ---------------------------------------------------------------------------
def split_xy(df):
    # Same split of the table for everyone. Not used as features: EGID, group_id, house, y, ref_year,
    # the *_code columns (we use the readable labels) and energy_source (energy_group is used instead).
    X = df[FEATURES].copy()
    y = df["y"].to_numpy(dtype=float)
    is_house = df["house"].to_numpy(dtype=bool)
    groups = df["group_id"].to_numpy()
    return X, y, is_house, groups


# ---------------------------------------------------------------------------
# C. The expert rule as one feature (target encoding)
# ---------------------------------------------------------------------------
class ExpertRuleEncoder(BaseEstimator, TransformerMixin):
    """Replaces construction period x energy group by the average target of that combination.

    fit:       learns, from the training part only, the mean of y for every combination.
               Small combinations are noisy, so they are pulled towards the overall mean:
                   value = (n * mean_of_combination + m * overall_mean) / (n + m)
               With n much larger than m the value is close to the combination's own mean;
               with n small it is close to the overall mean.
    transform: looks the value up for every building. A combination not seen in training
               gets the overall mean.

    Input: two columns, construction period and energy group (in this order).
    """

    def __init__(self, smoothing=SMOOTHING):
        self.smoothing = smoothing

    def _keys(self, X):
        X = pd.DataFrame(X)
        return X.iloc[:, 0].astype(str) + "|" + X.iloc[:, 1].astype(str)  # e.g. "1961-1970|oil"

    def fit(self, X, y):
        y = pd.Series(np.asarray(y, dtype=float))
        keys = self._keys(X).reset_index(drop=True)
        self.global_mean_ = y.mean()

        stats = y.groupby(keys).agg(["count", "mean"])  # n and mean per combination
        m = self.smoothing
        smoothed = (stats["count"] * stats["mean"] + m * self.global_mean_) / (stats["count"] + m)
        self.rule_values_ = smoothed.to_dict()
        self.counts_ = stats["count"].to_dict()  # kept for checks, not used in transform
        return self

    def transform(self, X):
        values = self._keys(X).map(self.rule_values_).fillna(self.global_mean_)
        return values.to_numpy(dtype=float).reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array(["expert_rule"])


# ---------------------------------------------------------------------------
# D. The four branches
# ---------------------------------------------------------------------------
def _numeric_branch():
    # median from training only -> log against the very skewed areas -> same scale for ridge and lasso
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])


def _one_hot():
    # one 0/1 column per category. Rare categories, and categories that appear only in the
    # validation or test part, all go into one "infrequent" column. No category is dropped:
    # ridge and lasso do not need it, and predictions of linear regression do not change.
    return OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=MIN_FREQUENCY,
                         sparse_output=False)


def _expert_rule_branch(smoothing):
    return Pipeline([
        ("rule", ExpertRuleEncoder(smoothing=smoothing)),
        ("scale", StandardScaler()),
    ])


# ---------------------------------------------------------------------------
# E. Put the branches together
# ---------------------------------------------------------------------------
def build_preprocessor(expert_rule=True, heating=True, municipality=True, smoothing=SMOOTHING):
    """Switches for the feature sets (task 22):
    expert_rule=False  without the expert-rule column
    heating=False      without energy group and heating system; the expert rule is then also left out,
                       because it needs the energy group
    municipality=False without municipality
    """
    categorical = CATEGORICAL + (HEATING if heating else [])

    branches = [
        ("numeric", _numeric_branch(), NUMERIC),
        ("categorical", _one_hot(), categorical),
    ]
    if municipality:
        branches.append(("municipality", _one_hot(), MUNICIPALITY))
    if expert_rule and heating:
        branches.append(("expert_rule", _expert_rule_branch(smoothing), RULE_COLUMNS))

    return ColumnTransformer(branches, remainder="drop", verbose_feature_names_out=False)


# ---------------------------------------------------------------------------
# F. Preprocessing + model
# ---------------------------------------------------------------------------
def build_pipeline(model, expert_rule=True, heating=True, municipality=True, smoothing=SMOOTHING):
    # Returns an unfitted pipeline. Fit it on the training part of a fold only.
    # model: any sklearn model, e.g. LinearRegression(), Ridge(), Lasso(), LogisticRegression()
    return Pipeline([
        ("prep", build_preprocessor(expert_rule, heating, municipality, smoothing)),
        ("model", model),
    ])


# ---------------------------------------------------------------------------
# G. Models that need a small adapter
# ---------------------------------------------------------------------------
class AboveThreshold(ClassifierMixin, BaseEstimator):
    """Classification variant (task 21): learns "above 450 or not" from the heating index.

    The pipeline passes the index y to every step, so the preprocessing (and the expert rule in it)
    still learns from the index, exactly as for the regressions. Only this last step turns y into
    above / not above the threshold. Use it with predicts_index=False in cv.cross_validate:
    the score is the predicted probability of being above the threshold.

    Example: build_pipeline(AboveThreshold(LogisticRegression(C=10, max_iter=5000)))
    """

    def __init__(self, model=None, threshold=THRESHOLD):
        self.model = model
        self.threshold = threshold

    def fit(self, X, y):
        self.model_ = clone(self.model).fit(X, np.asarray(y, dtype=float) > self.threshold)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X)


class ExpertRuleBaseline(RegressorMixin, BaseEstimator):
    """Baseline expert rule (task 18, README decision 12): every building gets the mean index of its
    construction period x energy group in the training part, without smoothing. A combination not seen
    in training gets the overall mean. Takes the raw table (no preprocessing)."""

    def fit(self, X, y):
        self.rule_ = ExpertRuleEncoder(smoothing=0).fit(X[RULE_COLUMNS], y)
        return self

    def predict(self, X):
        return self.rule_.transform(X[RULE_COLUMNS]).ravel()
