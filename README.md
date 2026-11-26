# Predicting the Heating Energy Index (IDC) of Geneva Buildings
## Data
All raw files are stored unchanged in `data/raw/`. They were downloaded on 13 November 2026.
Do not edit them; all cleaning happens in the notebooks in `code/`.

| File | Dataset | Publisher | Used for | Source |
|---|---|---|---|---|
| SCANE_INDICE_MOYENNES_3_ANS.csv.gz (gzip, read directly by pandas) | IDC des bâtiments | Canton of Geneva (SITG) | Target (INDICE_MOY3) | https://ckan.opendata.swiss/perma/SCANE_INDICE_MOYENNES_3_ANS@canton-geneve |
| gebaeude_batiment_edificio.csv | Federal Register of Buildings and Dwellings (GWR), canton GE | Federal Statistical Office | Features | https://www.housing-stat.ch/__publicdata |
| kodes_codes_codici.csv | GWR code list | Federal Statistical Office | Readable labels for the codes | https://www.housing-stat.ch/__publicdata |
| SCANE_RELATION_BATI_CHAUDIERE.csv | Bâtiments et chaudières centrales | Canton of Geneva (SITG) | Groups for the split | https://sitg.ge.ch/donnees/scane-relation-bati-chaudiere |
| OCEN_ETAT_IDC_PUBLIC.csv | Declaration status per building | Canton of Geneva (SITG) | Buildings without a value | https://ckan.opendata.swiss/en/perma/OCEN_ETAT_IDC_PUBLIC@canton-geneve |

All files are linked by the federal building number (EGID).

## How to run
**Requirements:** Python 3 with `pandas`, `numpy`, `matplotlib`, `scikit-learn` and `nbformat`
(`nbformat` is needed to run notebooks with `%run`; the first cell of `00_main.ipynb` installs it).

**Run:** open `code/00_main.ipynb`, select the kernel and choose **Run All**. It runs the notebooks of the five steps
in order (about 11 minutes, most of it `04_models.ipynb` and the model race in `05_evaluation.ipynb`). Every notebook can also be run on its own once the
notebooks before it have written their files. Paths come from `config.py`, so the working directory does not matter.
All notebooks and modules take the seed from `config.py` (decision 15).

| Step | File | What it does | Output |
|---|---|---|---|
| – | `00_main.ipynb` | runs the notebooks below in order | – |
| 1 Data | `01_preprocess.ipynb` | target, implausible values, register features, house flag, year effect, heating checks, split groups, merge loss (tasks 6–12) | `data/processed/building_clean.csv`, `output/tables/preprocess_merge_loss.csv`, `output/figures/preprocess_year_effect.png` |
| 2 Split & pipeline | `02_split_pipeline.ipynb` | test set and folds (task 16); checks of the metric functions, the expert rule and the pipeline (tasks 17, 19) | `data/processed/cv.csv`, `data/processed/test.csv` |
| 3 Explore | `03_describe.ipynb` | target group, declared vs. undeclared houses, descriptive analysis (tasks 13–15); reads `cv.csv`, so it runs after the split | `output/*/describe_*` |
| 4 Models | `04_models.ipynb` | all model variants in cross-validation, trained once and compared: baselines (18), linear regression, ridge, lasso (20), classification (21), feature sets (22), houses only (23) | `output/tables/models_*.csv`, `output/figures/models_*.png` |
| 5 Evaluation | `05_evaluation.ipynb` | choice of the final model in a race of four candidates on five splits (decisions 25–28), error analysis of the final model (24), test set (26), hit-rate curve (27) | `output/tables/choice_*.csv`, `output/tables/evaluation_*.csv`, `output/figures/choice_race.png`, `output/figures/evaluation_*.png` |
| shared | `config.py` | seed, threshold, paths, `load_cv()` and `load_test()` | – |
| shared | `pipeline.py` | preprocessing pipeline, classification wrapper and expert-rule baseline used by all models | – |
| shared | `metrics.py` | shared metric functions (task 17) | – |
| shared | `cv.py` | cross-validation, choice of alpha / C, comparisons and plots | – |

**The test set is locked away.** `02_split_pipeline.ipynb` writes the cross-validation buildings and the test set into
two separate files. All notebooks up to the model choice read only `cv.csv` (`load_cv()`); only `05_evaluation.ipynb`
opens `test.csv` (`load_test()`), once, after the final model is chosen (task 26).

**Adding a model variant:** one line in the list of variants in `04_models.ipynb`, section 2 (and the model in `MODELS`
if it is a new kind of model, e.g. a tree).

## Repository structure
- `code/` – notebooks and Python modules (see above)
- `data/raw/` – frozen raw files, never edited
- `data/processed/` – `building_clean.csv` (modelling table, one row per building), `cv.csv` and `test.csv` (the split)
- `output/tables/`, `output/figures/` – results of the notebooks, named after their step (`preprocess_`, `models_`, `evaluation_`)
- `report/` – report (`Project_report.tex`) and material for the presentation (`project_flow*.png`, made by `project_flow.py`)
- `TASKS.md` – task list with owners and status
- `Predicting the Heating Energy Index of Geneva Buildings (3).md` – project description with the lecturer's feedback


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
The split is made once in `code/02_split_pipeline.ipynb` and saved in `data/processed/cv.csv` (with the fold) and `data/processed/test.csv`.

9. Test set and folds: `StratifiedGroupKFold` splits the buildings into 7 parts; one part is the test set (2,117 buildings, 195 houses, 14.3%), used only once at the end (task 26). It has no option for the test size, and 1 of 7 parts is closest to the usual 15%. The other 12,710 buildings are split the same way into 5 folds for cross-validation.
10. Groups and houses: buildings in the same group (shared boiler or twins) always stay in the same part, so a model cannot predict a building by copying its twin. Houses have the same share (9.2%) in every part. The share of houses above 450 is not balanced on purpose and varies between 36% and 40%.

### Baselines (task 18, fixed 23 Nov 2026)
Computed in `code/04_models.ipynb` (section 4), on the same 5 folds and with the same measures as the models.

11. Training mean: every building gets the average index of the training folds. All scores are equal, so the ranking is random; this is a sanity check and should give about the share of houses above 450.
12. Expert rule: every building gets the average index of its construction period × energy group in the training folds, without smoothing, as an expert would use it. A combination not seen in training gets the overall mean. The models must beat this rule to be useful (lecturer's feedback point 4).

### Models (task 20, fixed 23 Nov 2026)
All models run in `code/04_models.ipynb` with the cross-validation in `code/cv.py` and the folds from `02_split_pipeline.ipynb`.

13. Cross-validation: In every fold the model is trained on all buildings of the other four folds and scored on the houses of the validation fold; results for all buildings are reported next to them. Linear regression, ridge and lasso each run with and without the expert-rule column.
14. Choosing alpha: For ridge (0.01 to 100,000) and lasso (0.001 to 100, 26 to 29 values on a log scale) we take the alpha with the highest mean selection measure on houses over the 5 folds; if several alphas have the same mean, the smaller MAE on houses decides. The selection measure only depends on the order of the houses, so it changes little with alpha; differences between alphas mostly lie within one standard error (about 47 houses in the top 20% of a fold).
15. One seed: `code/config.py` holds the project seed (42) and the threshold (450); all notebooks and modules import them from there.

### Feature sets (task 22, fixed 24 Nov 2026)
Computed in `code/04_models.ipynb` (section 7) with the switches of `build_pipeline` (`heating`, `municipality`, `expert_rule`).

16. Feature sets: We compare five sets: full (as in task 20), full without the expert rule, without municipality, without heating information (energy group and heating system; the expert rule then drops out too, because it needs the energy group), and without both. Everything else is as in task 20: same folds, trained on all buildings, scored on houses, linear regression, ridge and lasso with the same alpha values. Alpha is chosen again for every feature set with the rule of decision 14, so that no set is held back by a penalty tuned for another one. Each set is compared with the full set fold by fold, on the same houses. Reason: the register shows today's heating, which we cannot check for undeclared houses (lecturer's feedback point 2), so we want to know how much the ranking depends on it.
### Extra model: gradient boosting (fixed 25 Nov 2026)
17. Gradient boosting (extra check): a tree model that finds interactions (e.g. construction period × energy
source × size) by itself. We test whether it finds more in the register than our linear models. Same pipeline,
folds and selection measure as ridge; we tune the number of leaves per tree (7, 15, 31), with learning rate 0.05,
300 trees and early stopping off (it would make the result depend on a random hold-out).
18. Result: it ranks houses well (61.7% trained on all buildings, 59.1% on houses only; expert rule 45.1%,
random 38%), but not better than ridge trained on houses only: +1.3 ± 2.9 and −1.3 ± 3.1 points fold by fold,
each winning only 3 of 5 folds. The limit is the information in the register, not the flexibility of the model.

### Final model choice (task 26, fixed 25 Nov 2026)
19. *(Replaced by decision 28.)* Final model: ridge trained on houses only (alpha = 31.6). Its mean is only slightly lower than that of gradient
boosting trained on all buildings (60.4% vs. 61.7%), and the difference is within one standard error (+1.3 ± 2.9
points). We choose ridge because it is simpler and easier to understand: each prediction is a sum of effects of the
register characteristics, while gradient boosting combines hundreds of trees. 

### Error analysis and non-linear floor area (fixed 25 Nov 2026)
Computed in `code/04_models.ipynb` (sections 9 and 10), on the cross-validation predictions only; the test set is not used.

20. Error analysis by feature: we take the out-of-fold predictions of the best candidate so far (ridge trained on houses only)
and compute, for every category of a feature, the mean error (prediction − true value; positive = predicted too high) with a
95% interval (± 1.96 · standard deviation / √n), for groups with at least 20 houses. A mean error outside the interval means the
model misses something about this group; a dummy for it would shift its predictions by about that amount. Because we test many
groups, we compare the number of such groups with the number expected by chance (5%). We also count how many houses of each
group land on the contact list (top 20%) as hits or false alarms, or are missed, because an error only matters for us if it
changes the list. Gradient boosting is shown next to ridge. The reference year (`ref_year`) is only shown, not used as a
feature: undeclared houses have no reference year.
21. Non-linear floor area: the only feature the error analysis flags with an effect on the list is the floor area (small houses
predicted too high, medium ones too low). We therefore add exactly one variant, fixed before running it: ridge trained on houses
only, with the floor area as a spline (log floor area, 5 knots at quantiles, degree 3, linear beyond the training range;
not tuned, alpha chosen as in decision 14). It is compared fold by fold with ridge trained on houses only.
Result: 60.9% vs. 60.4% (+0.4 ± 2.6 points, better in 2 of 5 folds). The chosen alpha (56,234) lies near the edge of the
grid, where the predictions are pulled towards the average (MAE 116.3). With the same alpha as ridge on houses only (31.6),
the spline removes the wave by floor area, but does not rank houses better (57.9%). The spline is kept as an experiment,
not pursued and not a candidate for the final model.

### Houses: lasso, house interactions and compactness (fixed 25 Nov 2026)
Computed in `code/04_models.ipynb` (section 11). Houses behave differently from large buildings (task 23), so we add four
variants that focus on houses. They are fixed before running them and are the only additions before the finalists are chosen;
the finalists and the rule for choosing among them follow. Same folds, selection measure on houses, alpha as in decision 14;
each is compared fold by fold with ridge trained on houses only.

22. Lasso trained on houses only, with the same alpha values as lasso (0.001 to 100); it was missing in the houses-only family.
23. House interactions: ridge trained on all buildings, with a house dummy and a copy "× house" of these columns: floor area,
floors, construction period, energy group and the expert rule. The copy equals the column for houses and is 0 for all other
buildings, so the model learns one effect from all buildings plus a deviation for houses. The penalty pulls every deviation
towards 0, so houses keep their own effect only where the data show it; otherwise they share the effect of all buildings.
This uses all buildings and still lets houses differ (between "all the same" and "houses only"). Municipality is not
interacted (too few houses per municipality), nor dwellings and building class (for houses they only repeat "house").
24. Compactness: two extra columns, 1 / floors and 1 / √floor area. `floor_area` is the building's footprint (GAREA), so the
heated area is about footprint × floors. The heat a house loses per m² of heated area grows with its outer surface per m²:
about 2 / floors (roof and floor slab) plus a term in 1 / √footprint (walls). Small, low houses have the most surface per m².
The columns are added to the log features, both for ridge on houses only and for ridge with house interactions.

### Model choice: repeated cross-validation (fixed 25 Nov 2026)
Computed in `code/05_evaluation.ipynb` (section 1), on the cross-validation data only; the test set is not used. Several variants lie within
the noise of one split (standard error about 3 points, about 47 houses in the top 20% of a fold), and the more variants we compare,
the more optimistic the best value is. So a small list of candidates, fixed here before the run, competes on five different splits.
This replaces decision 19 once the race and the coaching are done.

25. Candidates: (1) ridge trained on houses only, (2) the same with compactness, (3) ridge with house interactions and compactness,
(4) gradient boosting trained on all buildings, the better of the two tree variants in `04_models.ipynb` (61.7% vs. 59.1%), with its
settings unchanged (learning rate 0.05, 300 trees, early stopping off, number of leaves 7, 15 or 31). Not in the race: lasso on houses
only (better than ridge in 0 of 5 folds), linear regression on houses only (ridge with alpha close to 0, so already covered), the
spline (decision 21), the linear models trained on all buildings (about 6 points behind), and logistic regression on houses only
(57.4%, −3.0 ± 1.1 points against ridge on houses only, better in 0 of 5 folds; in `04_models.ipynb`). The two baselines run along as
references, not as candidates.
26. Five runs: run 1 uses the folds of `cv.csv`; runs 2–5 split the same 12,710 buildings again into 5 folds with the seeds 43–46
(project seed + 1 to + 4), with the same rules as in decision 9 (`StratifiedGroupKFold`, groups together, same share of houses).
In every run each candidate is tuned again (alpha or number of leaves, decision 14), so no setting is learned on another run's folds.
27. Rule: the leader is the candidate with the highest mean selection measure over all 25 folds. Every other candidate is compared
with the leader fold by fold on the same houses. Because the five runs reuse the same houses, the 25 differences are not independent,
and their usual standard error would be too small; we use the corrected standard error of Nadeau and Bengio:
standard deviation × √(1/25 + 1/4). If a simpler candidate is at most one corrected standard error behind the leader, the simplest such
candidate wins; from simple to complex: (1), (2), (3), (4). The final model gets the setting with the highest mean over all 25 folds
(decision 14 applied to the tuning curve averaged over the five runs).

### Final model (coaching 25 Nov 2026)
28. Final model: ridge trained on houses only, with compactness (candidate 2), alpha = 3.16 (the best mean over all
25 folds, as in decision 27). The rule of decision 27 picks the simpler ridge without compactness, because compactness is only
1.6 ± 1.7 points ahead. In the coaching we chose compactness anyway, following the lecturer: preferring the simpler model pays
off when it saves something, for example a data source. Compactness needs no extra data (only floor area and floors, which the
model uses anyway), has no drawback, has a physical reason, and is ahead in 4 of 5 runs. A meta-model that combines the
candidates (stacking) was suggested in the coaching; we checked it and dropped it: the errors of the candidates are very
similar, and a simple average of their predictions does not rank houses better (section 1 of `05_evaluation.ipynb`).
For transparency: the earlier final model (decision 19, ridge on houses only) had already been evaluated once on the test set
before the coaching. The choice in the coaching is based on the cross-validation race only, not on that test result.
