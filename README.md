# Predicting the Heating Energy Index (IDC) of Geneva Buildings
## Data
All raw files are stored unchanged in `data/raw/`. They were downloaded on 13 November 2026.
Do not edit them; all cleaning happens in the notebooks in `code/`.

| File | Dataset | Publisher | Used for | Source |
|---|---|---|---|---|
| SCANE_INDICE_MOYENNES_3_ANS.csv | IDC des bâtiments | Canton of Geneva (SITG) | Target (INDICE_MOY3) | https://ckan.opendata.swiss/perma/SCANE_INDICE_MOYENNES_3_ANS@canton-geneve |
| gebaeude_batiment_edificio.csv | Federal Register of Buildings and Dwellings (GWR), canton GE | Federal Statistical Office | Features | https://www.housing-stat.ch/__publicdata |
| kodes_codes_codici.csv | GWR code list | Federal Statistical Office | Readable labels for the codes | https://www.housing-stat.ch/__publicdata |
| SCANE_RELATION_BATI_CHAUDIERE.csv | Bâtiments et chaudières centrales | Canton of Geneva (SITG) | Groups for the split | https://sitg.ge.ch/donnees/scane-relation-bati-chaudiere |
| OCEN_ETAT_IDC_PUBLIC.csv | Declaration status per building | Canton of Geneva (SITG) | Buildings without a value | https://ckan.opendata.swiss/en/perma/OCEN_ETAT_IDC_PUBLIC@canton-geneve |

All files are linked by the federal building number (EGID).

## How to run
**Requirements:** Python 3 with `pandas`, `numpy`, `matplotlib`, `scikit-learn` and `nbformat`
(`nbformat` is needed to run notebooks with `%run`; the first cell of `04_main.ipynb` installs it).

**Run:** open `code/04_main.ipynb`, select the kernel and choose **Run All**. It runs the other notebooks in order.
Run it from the `code/` folder, because the notebooks use relative paths (`../data`, `../output`).
All notebooks and modules take the seed from `config.py` (decision 15).

| File | What it does | Output |
|---|---|---|
| `04_main.ipynb` | runs the notebooks below in order | – |
| `01_preprocess.ipynb` | target, implausible values, register features, house flag, year effect, heating checks, split groups, merge loss | `data/processed/building_clean.csv`, `output/merge_loss.csv`, `output/year_effect.png` |
| `01b_split.ipynb` | test set and cross-validation folds (task 16) | `data/processed/split.csv` |
| `01c_baselines.ipynb` | baselines in cross-validation: training mean and expert rule (task 18) | `output/tables/task18_baselines.csv`, `task18_oof_predictions.csv` |
| `02_estimate.ipynb` | linear regression, ridge and lasso in cross-validation (task 20) | `output/tables/task20_*.csv`, `output/figures/task20_*.png` |
| `03_evaluate.ipynb` | checks of the metric functions | – |
| `config.py` | project seed and threshold | – |
| `pipeline.py` | preprocessing pipeline used by all models | – |
| `metrics.py` | shared metric functions (task 17) | – |
| `cv.py` | cross-validation, choice of alpha and plots (task 20) | – |

## Repository structure
- `code/` – notebooks and Python modules (see above)
- `data/raw/` – frozen raw files, never edited
- `data/processed/` – `building_clean.csv`, the modelling table (one row per building)
- `output/` – tables and figures (`figures/`, `tables/`, merge loss, year effect)
- `old/` – earlier project idea (Zurich injury accidents), not used
- `TASKS.md` – task list with owners and status
- `Predicting the Heating Energy Index of Geneva Buildings (3).md` – project description with the lecturer's feedback
- `Project_report.tex` – report


## Decisions:

1. Our selection measure: We choose models by the share of houses above 450 among the top 20% of the ranking. It fits the outreach decision, and it applies to every model we compare: regression models rank by predicted index, the classification model by predicted probability, and the baselines by their group averages.
2. Implausible target values: We remove buildings whose 3-year average is below 50 or above 1,500 MJ/m² per year(25 buildings, 1 house). Even very efficient buildings lie above 50, so lower values point to a wrong floor area; values above 1,500, about three times the threshold,
point to data errors. The rule was fixed before the test set was used.
3. List length: The right length is the office’s capacity. Since we don’t know it, we show the whole curve and report 10% and 20% as examples of a small and a medium campaign, both below the 37.7% share of problem houses. Up to that share, a perfect model could fill the whole list with problem houses only.

### Preprocessing pipeline (fixed 22 Nov 2026)
All steps are in `code/pipeline.py` and are fitted on the training part of each fold only.

4. Expert rule as one feature (target encoding): Instead of a dummy for each of the roughly 100 combinations of construction period and energy source, the model gets the expert rule's value, the mean heating index of the building's combination, as one extra column. We agreed on this with the lecturer. Small combinations are noisy, so they are pulled towards the overall mean: (n · mean of the combination + m · overall mean) / (n + m), with m = 20. We use our own class `ExpertRuleEncoder` instead of sklearn's `TargetEncoder`, because its internal cross-fitting ignores the boiler groups. Models are compared with and without this column.
5. Energy groups: We use `energy_group` (8 groups) instead of the 18 register codes, the same grouping as in the heating check. The group "other" mainly holds "Unbestimmt" (617 buildings) and "Andere" (196).
6. Rare categories: Categories seen fewer than 20 times in training, and categories not seen in training at all, go into one "infrequent" column.
7. Numbers: Missing floors are filled with the training median. Floor area, floors and dwellings are transformed with log(1 + x) against their strong skew and then standardised, which ridge and lasso need.
8. No dropped dummy: Every category keeps its own column. Ridge and lasso do not need a reference category and the predictions of linear regression do not change, but single linear-regression coefficients cannot be interpreted on their own.

### Split (task 16, fixed 23 Nov 2026)
The split is made once in `code/01b_split.ipynb` and saved in `data/processed/split.csv` (EGID, test, fold).

9. Test set and folds: `StratifiedGroupKFold` splits the buildings into 7 parts; one part is the test set (2,117 buildings, 195 houses, 14.3%), used only once at the end (task 26). It has no option for the test size, and 1 of 7 parts is closest to the usual 15%. The other 12,710 buildings are split the same way into 5 folds for cross-validation.
10. Groups and houses: buildings in the same group (shared boiler or twins) always stay in the same part, so a model cannot predict a building by copying its twin. Houses have the same share (9.2%) in every part. The share of houses above 450 is not balanced on purpose and varies between 36% and 40%.

### Baselines (task 18, fixed 23 Nov 2026)
Computed in `code/01c_baselines.ipynb`, after the split and before the models, on the same 5 folds and with the same measures as the models.

11. Training mean: every building gets the average index of the training folds. All scores are equal, so the ranking is random; this is a sanity check and should give about the share of houses above 450.
12. Expert rule: every building gets the average index of its construction period × energy group in the training folds, without smoothing, as an expert would use it. A combination not seen in training gets the overall mean. The models must beat this rule to be useful (lecturer's feedback point 4).

### Models (task 20, fixed 23 Nov 2026)
All models run in `code/02_estimate.ipynb` with the cross-validation in `code/cv.py` and the folds from `01b_split.ipynb`.

13. Cross-validation: In every fold the model is trained on all buildings of the other four folds and scored on the houses of the validation fold; results for all buildings are reported next to them. Linear regression, ridge and lasso each run with and without the expert-rule column.
14. Choosing alpha: For ridge (0.01 to 100,000) and lasso (0.001 to 100, 26 to 29 values on a log scale) we take the alpha with the highest mean selection measure on houses over the 5 folds; if several alphas have the same mean, the smaller MAE on houses decides. The selection measure only depends on the order of the houses, so it changes little with alpha; differences between alphas mostly lie within one standard error (about 47 houses in the top 20% of a fold).
15. One seed: `code/config.py` holds the project seed (42) and the threshold (450); all notebooks and modules import them from there.