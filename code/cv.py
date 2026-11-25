## cv
## Cross-validation on the 5 folds from 03_split_pipeline (task 16), scored with the shared metrics (task 17).
## Every model is fitted on all buildings of the training folds and judged on the houses of the
## validation fold (feedback point 1). Used for all model variants in 04_models.ipynb (tasks 18, 20-23).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import Lasso

from config import THRESHOLD
from metrics import evaluate, _top_positions   # _top_positions: the same ranking and tie-break as the selection measure
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


# ---------------------------------------------------------------------------
# 8. Error analysis by feature (README decision 20)
# ---------------------------------------------------------------------------
def error_by_group(error, groups, min_n=20, compare=None):
    """Mean error per category of a feature, e.g. per construction period.
    error:   prediction - true value for every house (positive = predicted too high), out-of-fold
    groups:  the category of every house (one array, or a DataFrame with several columns for combinations)
    compare: optional errors of a second model for the same houses (e.g. gradient boosting), shown next to it
    Returns one row per category with at least min_n houses:
      n, mean error, ± 95% (1.96 * standard deviation / sqrt(n)), clear (0 lies outside the interval),
      and the mean error of the second model."""
    data = pd.DataFrame(groups).reset_index(drop=True)
    keys = list(data.columns)
    data["error"] = np.asarray(error, dtype=float)
    if compare is not None:
        data["compare"] = np.asarray(compare, dtype=float)
    grouped = data.groupby(keys, observed=True)

    table = grouped["error"].agg(n="size", mean_error="mean", sd="std")
    table["ci"] = 1.96 * table["sd"] / np.sqrt(table["n"])
    table["clear"] = table["mean_error"].abs() > table["ci"]   # the model is off for this group, more than chance
    if compare is not None:
        table["compare_mean_error"] = grouped["compare"].mean()
    table = table[table["n"] >= min_n].drop(columns="sd")
    return table.rename(columns={"mean_error": "mean error", "ci": "± 95%",
                                 "compare_mean_error": "second model: mean error"}).round(1)


def contact_list(score, y, folds, top=0.20, threshold=THRESHOLD):
    """Rebuilds the office's list in every validation fold: the top 20% of the houses by score are contacted,
    exactly as in the selection measure. Give it the houses only.
    Returns one outcome per house: hit, false alarm, missed or correctly left out."""
    score, y, folds = np.asarray(score, dtype=float), np.asarray(y, dtype=float), np.asarray(folds)
    contacted = np.zeros(len(y), dtype=bool)
    for k in np.unique(folds):
        rows = np.flatnonzero(folds == k)
        contacted[rows[_top_positions(score[rows], top)]] = True
    above = y > threshold
    return np.select([contacted & above, contacted & ~above, ~contacted & above],
                     ["hit", "false alarm", "missed"], default="correctly left out")


def plot_error_by_feature(tables, labels=("model", "second model"), ncols=3, path=None):
    """Forest plot, one panel per feature. tables: dict title -> table from error_by_group.
    Dot = mean error of the category, bar = 95% interval, vertical line = no error.
    Filled dot: the model is clearly off for this group (0 outside the interval); open dot: within chance.
    The second model (if given) is the small grey diamond."""
    nrows = int(np.ceil(len(tables) / ncols))
    heights = [max(len(t) for t in list(tables.values())[r * ncols:(r + 1) * ncols]) for r in range(nrows)]
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 0.32 * sum(heights) + 1.2 * nrows),
                             sharex=True, squeeze=False, gridspec_kw={"height_ratios": heights})
    for ax, (title, t) in zip(axes.flat, tables.items()):
        pos = np.arange(len(t))[::-1]   # first category at the top
        ax.axvline(0, color="grey", linewidth=1)
        ax.errorbar(t["mean error"], pos, xerr=t["± 95%"], fmt="none", ecolor="C0", elinewidth=1.2, capsize=2)
        ax.scatter(t["mean error"], pos, s=36, zorder=3, edgecolor="C0", linewidth=1.2,
                   facecolor=np.where(t["clear"], "C0", "white"))
        if "second model: mean error" in t:
            ax.scatter(t["second model: mean error"], pos + 0.25, marker="D", s=14, color="grey", zorder=2)
        names = [" × ".join(map(str, i)) if isinstance(i, tuple) else str(i) for i in t.index]
        ax.set_yticks(pos)
        ax.set_yticklabels([f"{name} ({n})" for name, n in zip(names, t["n"])], fontsize=8)
        ax.set_title(title, fontsize=10)

    # x-axis labels under the lowest panel of every column (also when the panel below is empty)
    for i, ax in enumerate(axes.flat):
        if i >= len(tables):
            ax.axis("off")
        elif i + ncols >= len(tables):
            ax.tick_params(labelbottom=True)
            ax.set_xlabel("mean error (prediction − true value)")

    handles = [Line2D([], [], marker="o", linestyle="", color="C0", label=f"{labels[0]}: clearly off (0 outside the 95% interval)"),
               Line2D([], [], marker="o", linestyle="", markerfacecolor="white", color="C0", label=f"{labels[0]}: within chance")]
    if any("second model: mean error" in t for t in tables.values()):
        handles.append(Line2D([], [], marker="D", linestyle="", color="grey", markersize=4, label=labels[1]))
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig
