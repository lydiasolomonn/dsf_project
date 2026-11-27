# Predicting the Heating Energy Index (IDC) of Geneva Buildings

**Question:** Owners of small residential buildings in Geneva do not have to declare their heating energy index (IDC).
Can the federal building register rank these houses so that the energy office contacts mostly owners whose house is above
the legal threshold of 450 MJ/m² per year?

**Answer in short:** Our final model, ridge regression trained on houses only with two compactness features, puts on
average 60.1% problem houses into the top 20% of its ranking in cross-validation (five splits × five folds), against 44.7% for
the expert rule (mean index of construction period × energy source) and 38% for a random choice. On the test set, evaluated
once, it reaches 61.5% (24 of 39 houses; random 35.9%). With only 39 test houses on the list the test set cannot separate the
model from the expert rule (59.0%); the cross-validation over 1,171 houses can (+15.4 ± 4.6 points, better in 24 of 25 folds).
The limit is the information in the register: more flexible models, interactions and combinations of models do not rank
houses better.

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
All notebooks and modules take the seed from `config.py` (decision 19).

| Step | File | What it does | Output |
|---|---|---|---|
| – | `00_main.ipynb` | runs the notebooks below in order | – |
| 1 Data | `01_preprocess.ipynb` | target, implausible values, register features, house flag, year effect, heating checks, split groups, merge loss (tasks 6–12) | `data/processed/building_clean.csv`, `output/tables/preprocess_merge_loss.csv`, `output/figures/preprocess_year_effect.png` |
| 2 Split & pipeline | `02_split_pipeline.ipynb` | test set and folds (task 16); checks of the metric functions, the expert rule and the pipeline (tasks 17, 19) | `data/processed/cv.csv`, `data/processed/test.csv` |
| 3 Explore | `03_describe.ipynb` | target group, declared vs. undeclared houses, descriptive analysis (tasks 13–15); reads `cv.csv`, so it runs after the split | `output/*/describe_*` |
| 4 Models | `04_models.ipynb` | all model variants in cross-validation, trained once and compared: baselines (18), linear regression, ridge, lasso (20), classification (21), feature sets (22), houses only (23) | `output/tables/models_*.csv`, `output/figures/models_*.png` |
| 5 Evaluation | `05_evaluation.ipynb` | choice of the final model in a race of four candidates on five splits (decisions 29–34), error analysis of the final model (24), test set (26), hit-rate curve (27) | `output/tables/choice_*.csv`, `output/tables/evaluation_*.csv`, `output/figures/choice_race.png`, `output/figures/evaluation_*.png` |
| shared | `config.py` | seed, threshold, paths, `load_cv()` and `load_test()` | – |
| shared | `pipeline.py` | preprocessing pipeline, classification wrapper and expert-rule baseline used by all models | – |
| shared | `metrics.py` | shared metric functions (task 17) | – |
| shared | `cv.py` | cross-validation, choice of alpha / C, comparisons and plots | – |

**The test set is locked away.** `02_split_pipeline.ipynb` writes the cross-validation buildings and the test set into
two separate files. All notebooks up to the model choice read only `cv.csv` (`load_cv()`); only `05_evaluation.ipynb`
opens `test.csv` (`load_test()`), once, after the final model is chosen.

## Repository structure
- `code/` – notebooks and Python modules (see above)
- `data/raw/` – frozen raw files, never edited
- `data/processed/` – `building_clean.csv` (modelling table, one row per building), `cv.csv` and `test.csv` (the split)
- `output/tables/`, `output/figures/` – results of the notebooks, named after their step (`preprocess_`, `describe_`, `models_`, `choice_`, `evaluation_`)
- `report/` – report (`Project_report.tex`) and material for the presentation (`project_flow*.png`, made by `project_flow.py`)
- `TASKS.md` – task list with owners and status
- `Predicting the Heating Energy Index of Geneva Buildings (3).md` – project description with the lecturer's feedback

## Decisions

### Question and measure
1. Selection measure: we choose models by the share of houses above 450 among the top 20% of the ranking. It fits the outreach
decision and applies to every model: regression models rank by predicted index, the classification model by predicted
probability, the baselines by their group averages. MAE and the share of problem houses captured are reported beside it.
2. List length: the right length is the office's capacity. Since we do not know it, we show the whole hit-rate curve and report
10% and 20% as examples of a small and a medium campaign, both below the 37.7% share of problem houses. Up to that share, a
perfect model could fill the whole list with problem houses only.
3. Houses: buildings with one or two dwellings in the register (GKLAS 1110 and 1121). They approximate the target group
(residential buildings with fewer than 5 heat consumers, which do not have to declare): the definition covers 95% of its
23,123 buildings, and the building class is known for every building in the register, declared or not. Models are chosen
and evaluated on houses (lecturer's feedback point 1).

### Data (`01_preprocess.ipynb`)
4. Target: one value per building, its latest 3-year average ending in 2023 or later. Yearly values of the same building are
so similar that a model could copy the answer across training and test data. Older averages describe buildings before
renovations; using only periods ending in 2025 or later would cost about 450 of our 1,531 houses. Values have fallen over
time (about −43 MJ/m² for the same buildings between periods ending in 2023 and 2026); we accept this year effect.
5. Implausible values: we remove 3-year averages below 50 or above 1,500 MJ/m² per year (25 buildings, 1 house). Even very
efficient buildings lie above 50, so lower values point to a wrong floor area; values above 1,500, about three times the
threshold, point to data errors.
6. Heating that does not match the target: the register shows today's heating, the target an earlier period. We drop buildings
whose energy source changed within the 3 target years, and buildings whose energy source in the heating file differs from
the register's. Their features would not describe the building that produced the value.
7. Groups: buildings that share a central boiler, and "twins" (same reference year, value and heated floor area), get the same
value. They form one group and always stay on the same side of every split, so a model cannot predict a building by
copying its twin. Variables known only after a declaration (heated floor area, reference year) are never features.

### Split (`02_split_pipeline.ipynb`)
8. Test set and folds: `StratifiedGroupKFold` splits the buildings into 7 parts; one part is the test set (2,117 buildings,
195 houses, 14.3%), the closest to the usual 15% this method allows. The other 12,710 buildings are split the same way into
5 folds for cross-validation. Both are saved in separate files (`cv.csv`, `test.csv`), so the test set stays locked away.
9. Groups and houses: groups always stay in one part, and houses have the same share (9.2%) in every part. The share of houses
above 450 is not balanced on purpose and varies between 36% and 40%.

### Preprocessing pipeline (`pipeline.py`)
Every step is fitted on the training part of each fold only.

10. Expert rule as one feature (target encoding): instead of a dummy for each of the roughly 100 combinations of construction
period and energy source, the model gets the mean index of the building's combination as one column. Small combinations are
pulled towards the overall mean: (n · mean of the combination + m · overall mean) / (n + m), with m = 20. We use our own class
`ExpertRuleEncoder`, because sklearn's `TargetEncoder` cross-fits without the boiler groups. Agreed with the lecturer.
11. Energy groups: `energy_group` (8 groups) instead of the 18 register codes; "other" mainly holds "Unbestimmt" and "Andere".
12. Rare categories: categories seen fewer than 20 times in training, or not at all, go into one "infrequent" column.
13. Numbers: missing floors are filled with the training median; floor area, floors and dwellings are log(1 + x)-transformed
against their strong skew and standardised, which ridge and lasso need.
14. No dropped dummy: every category keeps its own column. Ridge and lasso do not need a reference category, and the
predictions of linear regression do not change.

### Baselines (`04_models.ipynb`, section 4)
15. Training mean: every building gets the mean index of the training folds. All scores are equal, so the ranking is random;
a sanity check.
16. Expert rule: every building gets the mean index of its construction period × energy group in the training folds, without
smoothing, as an expert would use it. The models must beat this rule to be useful (lecturer's feedback point 4).

### Models in cross-validation (`04_models.ipynb`)
17. Cross-validation: in every fold the model is trained on the other four folds and scored on the houses of the validation
fold; results for all buildings are reported beside them. Linear regression, ridge and lasso run with and without the
expert-rule column.
18. Choosing alpha (and C, number of leaves): the value with the highest mean selection measure on houses over the folds; on a
tie, the smaller MAE on houses. Ridge 0.01 to 100,000 and lasso 0.001 to 100, 26 to 29 values on a log scale. The measure
depends only on the order of the houses, so it changes little with alpha.
19. One seed: `config.py` holds the project seed (42) and the threshold (450); all notebooks and modules import them.
20. Classification: logistic regression on "above 450 or not" ranks houses as well as regression, not better. The best
logistic regression (54.5%) is within 0.0 to 0.9 points of the six regression models (53.6% to 54.5%), always less than one
standard error; trained on houses only it is behind ridge on houses only (−3.0 ± 1.1 points). We keep regression: the
distance to the threshold carries information that a yes/no label throws away.
21. Feature sets: without heating information, without municipality, or without both, the ranking of houses is not
clearly worse (all differences within one standard error). We keep the full set: heating and municipality help the level a
little, and the expert rule needs the energy source. The register shows today's heating, which we cannot check for
undeclared houses; the model does not depend on it for the ranking.
22. Trained on houses only: houses behave differently from large buildings. Ridge trained on the houses of the
training folds ranks houses better than ridge trained on all buildings (60.4% vs. 54.5%, +6.0 ± 2.4 points, better in 4 of
5 folds). From here on the candidates focus on houses.
23. Gradient boosting (extra check): a tree model that finds interactions by itself (learning rate 0.05, 300 trees, early
stopping off, 7, 15 or 31 leaves). It is not clearly better than ridge on houses only (+1.3 ± 2.9 points trained on all
buildings): the limit is the information in the register, not the flexibility of the model.

### Error analysis and new features (`04_models.ipynb`, sections 9–11)
24. Error analysis by feature: for every category of a feature, the mean out-of-fold error (prediction − true value) of ridge on
houses only, with a 95% interval (± 1.96 · standard deviation / √n), for groups of at least 20 houses. We compare the number of
clearly-off groups with the number expected by chance (5%), and count what each group does to the contact list, because an
error only matters if it changes the list. Result: 9 of 74 groups are clearly off (3.7 by chance); construction period ×
energy source is captured by the expert-rule column (1 of 20 combinations, as by chance). The floor area is the only feature
with a systematic error across many houses on the list (small houses too high, medium ones too low); the other clear groups
are small or put only a few houses on the list.
25. Floor area as a spline (experiment): log floor area as a curve with 5 knots. It removes the wave by floor area but does not
rank houses better; not pursued.
26. Compactness: two extra columns, 1 / floors and 1 / √floor area. `floor_area` is the footprint, so the heated area is about
footprint × floors; the heat a house loses per m² grows with its outer surface per m², about 2 / floors (roof and floor slab)
plus a term in 1 / √footprint (walls). It removes the wave by floor area with 2 instead of 7 columns.
27. House interactions: ridge trained on all buildings with a house dummy and a copy "× house" of floor area, floors,
construction period, energy group, the expert rule and compactness (0 for other buildings). The model learns one effect from
all buildings plus a deviation for houses, which the penalty pulls towards 0 unless the data show it. Municipality,
dwellings and building class are not interacted.
28. Lasso trained on houses only completes the houses-only family; it is never ahead of ridge on houses only (worse in 3 of 5
folds, equal in 2).

### Model choice (`05_evaluation.ipynb`, section 1)
Several variants lie within the noise of one split (standard error about 3 points, about 47 houses in the top 20% of a fold),
and the more variants we compare, the more optimistic the best value is. So a small list of candidates, fixed before the run,
competes on five splits.

29. Candidates: (1) ridge on houses only, (2) ridge on houses only with compactness, (3) ridge with house interactions and
compactness, (4) gradient boosting on all buildings. Not in the race: lasso and linear regression on houses only (covered by
ridge), the spline, the models trained on all buildings without house interactions (about 6 points behind) and logistic
regression. The baselines run along as references.
30. Five runs: run 1 uses the folds of `cv.csv`; runs 2–5 split the same 12,710 buildings again with the seeds 43–46 and the
rules of decisions 8 and 9. Every candidate is tuned again in every run.
31. Uncertainty: the five runs reuse the same houses, so the 25 fold values are not independent and their usual standard error
would be too small. We use the corrected standard error of Nadeau and Bengio: standard deviation × √(1/25 + 1/4). The repeats
average out the luck of one split, but not the limited number of houses.
32. Rule: the leader has the highest mean over all 25 folds; a simpler candidate at most one corrected standard error behind it
wins; order from simple to complex (1), (2), (3), (4). The setting of the final model is the one with the highest mean over
all 25 folds.
33. Final model (chosen in the coaching): ridge on houses only with compactness, alpha = 3.16. It leads the race (60.1%, ahead in
4 of 5 runs); the rule would pick ridge without compactness, 1.6 ± 1.7 points behind. Following the lecturer, preferring the
simpler model pays off when it saves something, for example a data source; compactness needs no extra data, has no drawback
and has a physical reason.
34. No meta-model (stacking): the errors of the candidates correlate 0.94 to 0.99, and a simple average of their predictions
does not rank houses better than the final model (−0.7 ± 2.2 and −1.2 ± 1.8 points).

### Test set and hit-rate curve (`05_evaluation.ipynb`, sections 3 and 4)
35. Test set: the final model is trained once on the houses of all cross-validation buildings and evaluated once on the test
houses, with the expert rule as reference. Only houses are reported: the model is trained on houses and meant for houses.
Nothing is changed after the result.
36. Hit-rate curve: share of contacted test houses above 450 for every list length (decision 2), with random choice, the expert
rule and a perfect model as references. A second axis translates the list length into owners of the 21,377 undeclared houses
in the target group (`03_describe.ipynb`): the top 20% means about 4,275 owners.
