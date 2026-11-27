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
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, SplineTransformer, StandardScaler

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
SPLINE_KNOTS = 5    # knots of the floor-area spline (switch area_spline, README decision 25)

# columns that get a copy "x house" with the switch house_interactions (README decision 27), by the start
# of their name after preprocessing: floor area, floors, construction period, energy group, expert rule,
# and the compactness columns ("inv_...") if they are switched on
INTERACT_PREFIXES = ("floor_area", "floors", "construction_period_", "energy_group_", "expert_rule", "inv_")


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
    # the house flag is only used by the switch house_interactions (README decision 27); all other branches
    # pick their columns by name and drop the rest, so every other model ignores it
    X["house"] = is_house
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
# D. The branches
# ---------------------------------------------------------------------------
def _numeric_branch():
    # median from training only -> log against the very skewed areas -> same scale for ridge and lasso
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])


def _area_spline_branch():
    # floor area as a curve instead of a straight line (README decision 25): after the log, a spline
    # splits the area range at 5 knots (placed at quantiles, so each piece has about as many buildings)
    # and fits a smooth cubic piece between them. Beyond the training range it continues as a straight line.
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("spline", SplineTransformer(n_knots=SPLINE_KNOTS, degree=3, knots="quantile", extrapolation="linear")),
        ("scale", StandardScaler()),
    ])


def _compactness(X):
    # X: floor area (footprint) and floors, missing values already filled (array or table)
    X = np.asarray(X, dtype=float)
    area = np.maximum(X[:, 0], 1)
    floors = np.maximum(X[:, 1], 1)
    return np.column_stack([1 / np.sqrt(area), 1 / floors])


def _compactness_branch():
    # outer surface per m² of heated area (README decision 26): roof and floor slab give about 2 / floors,
    # the walls a term in 1 / sqrt(footprint). Small, low houses have the most surface per m².
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("compactness", FunctionTransformer(_compactness,
                                            feature_names_out=lambda self, names: np.array(["inv_sqrt_floor_area", "inv_floors"]))),
        ("scale", StandardScaler()),
    ])


class HouseInteractions(BaseEstimator, TransformerMixin):
    """House interactions (README decision 27). Gets the named columns after preprocessing (a DataFrame with
    a "house" column) and adds a copy "<column> x house" of every column whose name starts with one of the
    prefixes: equal to the column for houses, 0 for all other buildings. The model then learns one effect
    from all buildings plus a deviation for houses. Learns nothing itself (fit only stores the column names)."""

    def __init__(self, prefixes=INTERACT_PREFIXES):
        self.prefixes = prefixes

    def fit(self, X, y=None):
        self.columns_ = [c for c in X.columns if c.startswith(tuple(self.prefixes))]
        self.feature_names_out_ = np.array(list(X.columns) + [f"{c} x house" for c in self.columns_])
        return self

    def transform(self, X):
        house = X["house"].astype(float)
        copies = X[self.columns_].mul(house, axis=0)
        copies.columns = [f"{c} x house" for c in self.columns_]
        return pd.concat([X.astype(float), copies], axis=1)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


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
def build_preprocessor(expert_rule=True, heating=True, municipality=True, smoothing=SMOOTHING,
                       area_spline=False, compactness=False, house_interactions=False):
    """Switches for the feature sets (task 22):
    expert_rule=False  without the expert-rule column
    heating=False      without energy group and heating system; the expert rule is then also left out,
                       because it needs the energy group
    municipality=False without municipality
    area_spline=True   floor area as a spline (a curve) instead of one straight line (README decision 25)
    compactness=True   two extra columns, 1 / sqrt(floor area) and 1 / floors (README decision 26)
    house_interactions=True  a house dummy plus a copy "x house" of the columns in INTERACT_PREFIXES (decision 27)
    """
    categorical = CATEGORICAL + (HEATING if heating else [])
    numeric = [c for c in NUMERIC if not (area_spline and c == "floor_area")]

    branches = [
        ("numeric", _numeric_branch(), numeric),
        ("categorical", _one_hot(), categorical),
    ]
    if area_spline:
        branches.append(("area", _area_spline_branch(), ["floor_area"]))
    if municipality:
        branches.append(("municipality", _one_hot(), MUNICIPALITY))
    if expert_rule and heating:
        branches.append(("expert_rule", _expert_rule_branch(smoothing), RULE_COLUMNS))
    if compactness:
        branches.append(("compactness", _compactness_branch(), ["floor_area", "floors"]))

    if not house_interactions:
        return ColumnTransformer(branches, remainder="drop", verbose_feature_names_out=False)

    branches.append(("house", "passthrough", ["house"]))
    columns = ColumnTransformer(branches, remainder="drop", verbose_feature_names_out=False)
    columns.set_output(transform="pandas")   # named columns, so HouseInteractions can find them
    return Pipeline([("columns", columns), ("interactions", HouseInteractions())])


# ---------------------------------------------------------------------------
# F. Preprocessing + model
# ---------------------------------------------------------------------------
def build_pipeline(model, expert_rule=True, heating=True, municipality=True, smoothing=SMOOTHING,
                   area_spline=False, compactness=False, house_interactions=False):
    # Returns an unfitted pipeline. Fit it on the training part of a fold only.
    # model: any sklearn model, e.g. LinearRegression(), Ridge(), Lasso(), LogisticRegression()
    return Pipeline([
        ("prep", build_preprocessor(expert_rule, heating, municipality, smoothing, area_spline,
                                    compactness, house_interactions)),
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
    """Baseline expert rule (task 18, README decision 16): every building gets the mean index of its
    construction period x energy group in the training part, without smoothing. A combination not seen
    in training gets the overall mean. Takes the raw table (no preprocessing)."""

    def fit(self, X, y):
        self.rule_ = ExpertRuleEncoder(smoothing=0).fit(X[RULE_COLUMNS], y)
        return self

    def predict(self, X):
        return self.rule_.transform(X[RULE_COLUMNS]).ravel()
