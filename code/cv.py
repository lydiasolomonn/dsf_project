## cv
## Cross-validation on the 5 folds from 03_split_pipeline (task 16), scored with the shared metrics (task 17).
## Every model is fitted on all buildings of the training folds and judged on the houses of the
## validation fold (feedback point 1). Used for all model variants in 04_models.ipynb (tasks 18, 20-23).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import evaluate
from pipeline import build_pipeline

SELECTION = "share_above_top20"  # selection measure (README decision 1), scored on houses


# ---------------------------------------------------------------------------
# 1. One model through all folds
# ---------------------------------------------------------------------------
def cross_validate(make_model, X, y, is_house, folds, predicts_index=True, train_rows=None):
    """make_model: function without arguments that returns a NEW unfitted pipeline,
                   e.g. lambda: build_pipeline(Ridge(alpha=10)). A new one per fold, so nothing carries over.
    folds:         fold number (0-4) of every row; test buildings must already be removed.
    predicts_index: True for regression (score = predicted index);
                    False for classification (score = predicted probability of being above 450, no MAE).
    train_rows:    None = learn from all buildings of the training folds (default);
                   a True/False array = learn only from these rows, e.g. is_house for "trained on houses only"
                   (task 23). Validation always uses all buildings of the validation fold.
    Returns: per_fold (one row per fold and group: houses / all buildings) and the out-of-fold scores.
    """
    y = np.asarray(y, dtype=float)
    is_house = np.asarray(is_house, dtype=bool)
    folds = np.asarray(folds)
    train_rows = np.ones(len(y), bool) if train_rows is None else np.asarray(train_rows, dtype=bool)
    oof_score = np.full(len(y), np.nan)  # out-of-fold: every building is scored by the model that did not see it

    rows = []
    for k in np.unique(folds):
        train, val = (folds != k) & train_rows, folds == k
        model = make_model()
        model.fit(X[train], y[train])
        if predicts_index:
            score = model.predict(X[val])
        else:
            score = model.predict_proba(X[val])[:, 1]
        oof_score[val] = score

        result = evaluate(y[val], score, is_house[val], predicts_index=predicts_index)
        result.insert(0, "fold", k)
        rows.append(result)

    per_fold = pd.concat(rows, ignore_index=True)
    return per_fold, oof_score


# ---------------------------------------------------------------------------
# 2. Average over the folds
# ---------------------------------------------------------------------------
def summarize(per_fold):
    """Mean, standard deviation and standard error (std / sqrt(number of folds)) of every measure,
    for houses and for all buildings. Returns one row per group with columns like
    share_above_top20_mean, share_above_top20_se, ..."""
    measures = [c for c in per_fold.columns if c not in ("fold", "group", "n")]
    n_folds = per_fold["fold"].nunique()
    grouped = per_fold.groupby("group", sort=False)[measures]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std().add_suffix("_std")
    se = (grouped.std() / np.sqrt(n_folds)).add_suffix("_se")
    return pd.concat([mean, std, se], axis=1).reset_index()


# ---------------------------------------------------------------------------
# 3. Try many values of alpha (ridge, lasso) or C (logistic regression)
# ---------------------------------------------------------------------------
def tune_alpha(model_class, alphas, X, y, is_house, folds, model_kwargs=None, param="alpha",
               predicts_index=True, train_rows=None, **switches):
    """Runs cross_validate for every value of the setting. switches go to build_pipeline
    (expert_rule, heating, municipality).
    model_class: a model class, e.g. Ridge, or a function that takes the setting as keyword,
                 e.g. lambda C: AboveThreshold(LogisticRegression(C=C, max_iter=5000)).
    param:       name of the setting: "alpha" (ridge, lasso) or "C" (logistic regression).
    Returns one row per value and group with the summary measures; the value is in the column `param`."""
    model_kwargs = model_kwargs or {}
    tables = []
    for alpha in alphas:
        make_model = lambda: build_pipeline(model_class(**{param: alpha}, **model_kwargs), **switches)
        per_fold, _ = cross_validate(make_model, X, y, is_house, folds,
                                     predicts_index=predicts_index, train_rows=train_rows)
        summary = summarize(per_fold)
        summary.insert(0, param, alpha)
        tables.append(summary)
    return pd.concat(tables, ignore_index=True)


# ---------------------------------------------------------------------------
# 4. Choose alpha (or C)
# ---------------------------------------------------------------------------
def best_alpha(table, metric=SELECTION, group="houses", param="alpha"):
    """The value with the highest mean of the selection measure on houses over the folds.
    If several values have the same mean, the one with the smaller MAE on houses wins;
    for classification (no MAE), the one that comes first in the list of values."""
    t = table[table["group"] == group]
    t = t.sort_values([f"{metric}_mean", "mae_mean"], ascending=[False, True], kind="stable")
    return t[param].iloc[0]


# ---------------------------------------------------------------------------
# 5. Plot: which alpha?
# ---------------------------------------------------------------------------
def plot_validation_curve(tables, title, reference=None, path=None, metric=SELECTION, param="alpha"):
    """tables:    dict label -> table from tune_alpha, e.g. {"with expert rule": t1, "without": t2}
    reference: dict label -> value, horizontal lines (e.g. random share, linear regression, baselines)
    Top: selection measure on houses (mean over the folds, band = +- one standard error) with the best alpha.
    Bottom: MAE on houses."""
    reference = reference or {}
    fig, (ax_top, ax_mae) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                         gridspec_kw={"height_ratios": [2, 1]})

    for i, (label, table) in enumerate(tables.items()):
        color = f"C{i}"
        t = table[table["group"] == "houses"].sort_values(param)
        mean, se = t[f"{metric}_mean"] * 100, t[f"{metric}_se"] * 100
        ax_top.plot(t[param], mean, color=color, marker="o", markersize=3, label=label)
        ax_top.fill_between(t[param], mean - se, mean + se, color=color, alpha=0.15)

        best = best_alpha(table, metric, param=param)
        best_value = t.loc[t[param] == best, f"{metric}_mean"].iloc[0] * 100
        ax_top.scatter([best], [best_value], color=color, marker="*", s=150, zorder=3,
                       label=f"{label}: best {param} = {best:.3g}")

        ax_mae.plot(t[param], t["mae_mean"], color=color, marker="o", markersize=3)
        ax_mae.fill_between(t[param], t["mae_mean"] - t["mae_se"], t["mae_mean"] + t["mae_se"],
                            color=color, alpha=0.15)

    # reference lines: own colour and style each, none of them C0/C1 (used by the curves)
    styles = ["--", ":", "-.", (0, (5, 1)), (0, (1, 2))]
    colors = ["grey", "black", "C3", "C2", "C4"]
    for j, (label, value) in enumerate(reference.items()):
        ax_top.axhline(value * 100, color=colors[j % len(colors)], linestyle=styles[j % len(styles)],
                       linewidth=1.2, label=label)

    ax_top.set_xscale("log")
    ax_top.set_ylabel("Houses above 450 in top 20% (%)")
    ax_top.set_title(title)
    ax_top.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))  # next to the plot, covers nothing
    ax_mae.set_xlabel(f"{param} (penalty strength, log scale)")
    ax_mae.set_ylabel("MAE on houses")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# 6. Plot: which features survive in lasso?
# ---------------------------------------------------------------------------
def plot_lasso_path(X, y, alphas, n_show=8, path=None, **switches):
    """Fits lasso on all given buildings (training part only, no test set) for every alpha.
    Only descriptive: shows how many coefficients stay non-zero and the paths of the strongest ones.
    The expert-rule column is drawn thicker."""
    from sklearn.linear_model import Lasso

    coefs = []
    names = None
    for alpha in alphas:
        pipe = build_pipeline(Lasso(alpha=alpha, max_iter=10000), **switches).fit(X, y)
        coefs.append(pipe.named_steps["model"].coef_)
        names = pipe.named_steps["prep"].get_feature_names_out()
    coefs = pd.DataFrame(coefs, index=alphas, columns=names)

    fig, (ax_n, ax_path) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                        gridspec_kw={"height_ratios": [1, 2]})
    ax_n.plot(coefs.index, (coefs != 0).sum(axis=1), marker="o", markersize=3)
    ax_n.set_ylabel("Non-zero coefficients")
    ax_n.set_title("Lasso: which features survive a stronger penalty?")

    strongest = coefs.abs().max().sort_values(ascending=False).index[:n_show]
    for name in strongest:
        width = 3 if name == "expert_rule" else 1.2
        ax_path.plot(coefs.index, coefs[name], linewidth=width, label=name)
    ax_path.axhline(0, color="grey", linewidth=0.8)
    ax_path.set_xscale("log")
    ax_path.set_xlabel("alpha (penalty strength, log scale)")
    ax_path.set_ylabel("Coefficient")
    ax_path.legend(fontsize=8, loc="best")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig, coefs


# ---------------------------------------------------------------------------
# 7. Show the main numbers and compare two variants fold by fold
# ---------------------------------------------------------------------------
def show(summary, group="houses"):
    """The main numbers of a summary table (from summarize) for one group, in percent, rounded."""
    s = summary[summary["group"] == group].iloc[0]
    return {
        "top 20%: above 450 (%)": round(s["share_above_top20_mean"] * 100, 1),
        "± SE": round(s["share_above_top20_se"] * 100, 1),
        "top 10%: above 450 (%)": round(s["share_above_top10_mean"] * 100, 1),
        "top 20%: captured (%)": round(s["captured_top20_mean"] * 100, 1),
        "MAE": round(s["mae_mean"], 1),
    }


def paired_difference(per_fold_a, per_fold_b, metric=SELECTION, group="houses"):
    """Compares variant A with variant B fold by fold, on the same buildings.
    Returns the difference A - B in percentage points per fold and one summary row:
    mean difference, its standard error and in how many folds A is better / worse."""
    a = per_fold_a[per_fold_a["group"] == group].set_index("fold")[metric] * 100
    b = per_fold_b[per_fold_b["group"] == group].set_index("fold")[metric] * 100
    diff = a - b
    summary = {
        "difference (points)": round(diff.mean(), 1) + 0.0,   # + 0.0 turns -0.0 into 0.0
        "± SE": round(diff.std() / np.sqrt(len(diff)), 1),
        "better in": f"{(diff > 0).sum()} of {len(diff)} folds",
        "worse in": f"{(diff < 0).sum()} of {len(diff)} folds",
    }
    return diff, summary
