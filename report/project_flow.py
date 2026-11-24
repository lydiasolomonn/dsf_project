"""Draws the project flow as a 16:9 figure for the slides and the report.
Usage (from the project folder): python report/project_flow.py
Writes report/project_flow.png (for the slides) and report/project_flow_with_files.png (with the notebooks)."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from pathlib import Path

HERE = Path(__file__).resolve().parent

STAGES = [
    ("1", "Data", "#2F6690", "01_preprocess",
     ["5 raw files, linked by\nbuilding number (EGID)",
      "Clean the heating index,\nbuild the target",
      "Register features,\nhouse flag",
      "Check heating changes",
      "Groups for the split"],
     "14,827 buildings\n1,366 houses"),
    ("2", "Explore", "#3A7D7C", "02_describe",
     ["Target group:\nhouses without a value",
      "Declared vs.\nundeclared houses",
      "Distribution of the target",
      "By building class and\nconstruction period"],
     "What the data can\nand cannot tell us"),
    ("3", "Split & pipeline", "#5B7B3A", "03_split_pipeline",
     ["Test set locked away\n(2,117 buildings)",
      "5 folds, boiler groups\nkept together",
      "Measure: houses above\n450 in the top 20%",
      "Preprocessing fitted on\ntraining folds only"],
     "12,710 buildings for\ncross-validation"),
    ("4", "Models", "#B0722E", "04_models",
     ["Baselines: training mean,\nexpert rule",
      "Linear regression,\nridge, lasso",
      "Classification\n(above 450 or not)",
      "Feature sets: heating,\nmunicipality",
      "Trained on houses only\nvs. all buildings"],
     "One comparison table,\nscored on houses"),
    ("5", "Evaluation", "#A14A4A", "05_evaluation",
     ["Choose the final model",
      "Error analysis",
      "Test set, used once",
      "Hit-rate curve for\nseveral list lengths"],
     "Ranking of houses\nfor the energy office"),
]

INK, MUTED, CARD, LINE = "#1F2933", "#5F6B76", "#F6F7F8", "#D5DADF"


def draw(SHOW_FILES, path):
    fig = plt.figure(figsize=(16, 9), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(0.6, 8.35, "Predicting the heating energy index of Geneva buildings", fontsize=24, fontweight="bold",
            color=INK, va="center")
    ax.text(0.6, 7.82, "From five public files to a ranking of houses the energy office should contact first",
            fontsize=14, color=MUTED, va="center")

    n = len(STAGES)
    left, right, gap = 0.6, 15.4, 0.38
    width = (right - left - gap * (n - 1)) / n
    top, header_h, body_bottom, out_h = 7.25, 0.85, 2.55, 0.95


    def box(x, y, w, h, face, edge=None, radius=0.08, lw=1.0, z=1):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                    facecolor=face, edgecolor=edge or face, linewidth=lw, zorder=z))


    for i, (number, title, color, files, steps, output) in enumerate(STAGES):
        x = left + i * (width + gap)

        # card with the steps
        box(x, body_bottom, width, top - body_bottom, CARD, LINE)
        # header
        box(x, top - header_h, width, header_h, color)
        ax.text(x + 0.22, top - header_h / 2, number, fontsize=22, fontweight="bold", color="white",
                va="center", alpha=0.75)
        ax.text(x + 0.58, top - header_h / 2, title, fontsize=15.5, fontweight="bold", color="white", va="center")

        # steps, evenly spaced
        area_top, area_bottom = top - header_h - 0.25, body_bottom + 0.2
        slot = (area_top - area_bottom) / 5
        for j, step in enumerate(steps):
            y = area_top - slot * (j + 0.5)
            ax.plot([x + 0.24], [y], marker="o", markersize=5, color=color, zorder=3)
            ax.text(x + 0.45, y, step, fontsize=11.5, color=INK, va="center", linespacing=1.25)

        # result of the stage
        out_y = body_bottom - 0.3 - out_h
        box(x, out_y, width, out_h, "white", color, lw=1.8)
        ax.text(x + width / 2, out_y + out_h / 2, output, fontsize=12, fontweight="bold", color=color,
                ha="center", va="center", linespacing=1.3)
        ax.add_patch(FancyArrowPatch((x + width / 2, body_bottom - 0.02), (x + width / 2, out_y + out_h + 0.02),
                                     arrowstyle="-|>", mutation_scale=13, color=color, linewidth=1.5, zorder=2))

        if SHOW_FILES:
            ax.text(x + width / 2, out_y - 0.22, files, fontsize=10, color=MUTED, ha="center", va="center",
                    family="monospace")

        # arrow to the next stage
        if i < n - 1:
            y = top - header_h / 2
            ax.add_patch(FancyArrowPatch((x + width + 0.04, y), (x + width + gap - 0.04, y), arrowstyle="-|>",
                                         mutation_scale=18, color=MUTED, linewidth=2, zorder=4))

    # rules that hold in every stage
    rule_y = 0.35 if not SHOW_FILES else 0.22
    box(left, rule_y, right - left, 0.55, "white", LINE)
    ax.text(left + 0.25, rule_y + 0.275, ("Code:" if SHOW_FILES else "Throughout:"), fontsize=11.5, fontweight="bold", color=INK, va="center")
    rules = "one seed   ·   one metric function for all models   ·   models are judged on houses   ·   the test set stays closed until the end"
    if SHOW_FILES:
        rules = "config.py, pipeline.py, metrics.py, cv.py are shared by all notebooks   ·   00_main.ipynb runs steps 1 to 5 in order"
    ax.text(left + 1.75, rule_y + 0.275, rules, fontsize=11.5, color=MUTED, va="center")

    fig.savefig(path, facecolor="white")
    print("saved", path)


draw(False, HERE / "project_flow.png")
draw(True, HERE / "project_flow_with_files.png")
