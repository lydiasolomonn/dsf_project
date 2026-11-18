# Predicting the Heating Energy Index of Geneva Buildings
Group 7 - Jenny Braunstorfer, Anastasija Stirbanovic, Lydia Kristina Solomon, Kangyue Wang, Elisa Stepanski

In the canton of Geneva, building owners must declare each year how much energy their building uses for heating. The canton converts this into a heating energy index (IDC, in MJ per m² per year), and owners whose buildings exceed 450 on a 3-year average must take improvement measures. So far, this duty applies to larger buildings; for small houses the declaration is still voluntary and is planned to become mandatory later. As a result, the cantonal energy office has no value for roughly 24,000 buildings, most of them small houses. We predict the 3-year average IDC of a building from characteristics that are registered for every building: construction period, heating system, energy source, building type, size and municipality. The model is trained on buildings that have already been declared. Its result is a ranking that tells the office in advance which houses are most likely above the threshold, so that it can contact those owners first with information and advice.
## 1. Problem Statement
**Who would use the result, and for which decision?** The cantonal energy office of Geneva (OCEN) uses the heating energy index (IDC) to decide which owners must act: above 450 MJ/m² per year (3-year average) an audit and improvement measures are required, above 800 a renovation ([ge.ch](https://www.ge.ch/connaitre-consommation-energie-batiment-idc/que-faire-resultat-idc-votre-immeuble)). For about 24,000 buildings the office has no value. Most are small houses, for which the declaration is still voluntary ([ge.ch](https://www.ge.ch/connaitre-consommation-energie-batiment-idc/proprietaires-villas-petits-immeubles-habitation)). The office wants to know in advance which houses are likely to use too much heating energy once the obligation starts, so it can reach out to those owners first with information and advice. Its capacity is limited, so it must decide where to start.

**What should improve, and how would we measure it?** Information and advice should first reach the owners whose houses are most likely above the threshold. Among the top 10% and 20% of buildings ranked by the model, we measure the share that truly exceeds 450 and the share of all such buildings captured, and compare with the baselines.

**What will the model predict, for which unit, at what moment, with what information?** Target: the 3-year average IDC of a building in MJ/m² per year, the value the law uses. Unit: one building (federal building number EGID). Cutoff: when the office plans its outreach, before the obligation starts and before the building has declared. Information: only characteristics from the federal building register, which exist for every building. We do not use a building's earlier IDC values, because they do not exist for the buildings concerned.

**What can the result not tell the decision-maker?**

- The exact consumption of one building. Occupant behaviour is in no register and values vary from year to year. The result is a ranking, not a replacement for the declaration.

- Whether a renovation or a new heating system would lower consumption. That is a causal question and out of scope.

- How undeclared buildings really behave. They are mostly single-family houses, the declared ones mostly apartment blocks, and the small houses in our data declared voluntarily: a selected sample.

- Register entries can be outdated, and thresholds and heating systems change over time.
## 2. Data
**Sources.** Four datasets, all linked by the federal building number (EGID).

| Dataset | Publisher | Role |
| --- | --- | --- |
| [IDC des bâtiments](https://ckan.opendata.swiss/perma/SCANE_INDICE_MOYENNES_3_ANS@canton-geneve) | Canton of Geneva | Target |
| [Federal Register of Buildings and Dwellings](https://www.housing-stat.ch/__publicdata), canton GE | Federal Statistical Office | Features |
| [Buildings and central boilers](https://sitg.ge.ch/donnees/scane-relation-bati-chaudiere) | Canton of Geneva | Grouping for the split |
| [Declaration status per building](https://ckan.opendata.swiss/en/perma/OCEN_ETAT_IDC_PUBLIC@canton-geneve) | Canton of Geneva | Identifies buildings without a value |
**Observations, time span and type.** 249,398 building-year records for 21,206 buildings, mainly 2011–2026. We keep one value per building, the latest 3-year average from 2023 onwards, because several values of one building are near-copies. After merging: 16,376 buildings. Cross-section.

**Target variable.** INDICE_MOY3, the average IDC of the last three years (regression). It is the value the law applies the threshold to, and it varies less than a single year. Buildings without three years of data are left out.

**Minimum requirements.** About 1,000 observations required: we have 16,376. 28.4% lie above 450, well above 10%, so no techniques for imbalanced data are needed. 1,527 are houses with one or two dwellings, enough to report them separately.

**Features (8).** Construction period, heating energy source, heating system, building class, ground area, number of floors, number of dwellings, municipality. Excluded: energy quantities and the yearly IDC values (the target is calculated from them) and the heated area (known only with the declaration).

**Existing analyses.** A [University of Geneva paper](https://archive-ouverte.unige.ch/unige:189755) compares climate-correction methods on the IDC database; an [EPFL project](https://ipese-web.epfl.ch/loustau/qbuildings-validation.html) uses it to validate a physical building model. Neither predicts the IDC of undeclared buildings to prioritise them.

**Kaggle.** We checked kaggle.com and found none of our datasets there, and no similar project.
## 3. First Plan
**Split:** by group. Buildings on the same central boiler often have identical IDC values, so they stay together (13,768 groups). About 70% training, 15% validation, 15% test; the test set is used once at the end. We do not split by time, because the model estimates buildings at the same moment, not a future year.

**Baselines:** the training mean; and a simple rule, the mean IDC by construction period and energy source, which an expert could look up.

**Models:** linear regression, ridge and lasso, compared with the baselines and across feature sets.

**Metrics:** mean absolute error in MJ/m² per year; share above 450 in the top 10% and 20%; both also separately for small houses.

**Leakage and sanity checks:** preprocessing fitted on training data only; implausible target values and duplicates removed; one row per building after merging; predictions checked for plausibility and errors by building class; declared and undeclared buildings compared.

**Feedback November 14 (Johannes Binswanger)**

Thank you, this is a clear and well-designed proposal. **You are good to go.** Please work the following points into your plan, and answer them in this document.

**1. Judge your model on the buildings it is meant for.** The office will use your model to rank houses whose owners have not declared yet. But most buildings in your data are declared apartment blocks; only 1,527 are houses with one or two dwellings. If you choose and report your model by its error on all buildings, the apartment blocks dominate the numbers. A model that works well for blocks but poorly for houses would then look good. Here are some ideas:

- **Train on all building** **types**: the blocks can help the model learn. But compute the validation error that you use to **choose between models** on the **houses** in your validation data. Report the **test results** accordingly. Results for all buildings can go next to them.

- **Make sure houses appear in the training, validation and test data in about their overall share**, for example by splitting the boiler groups separately within each building class.

- **With a 15% test set, you have only about 230 houses for testing, and similarly few for validation.** Results on houses will therefore vary a lot by chance. Instead of a single validation set, consider cross-validation on the training data, with folds that keep each boiler group together (deck 08, P1). The test set stays untouched until the end.

- **Even then, your houses declared voluntarily and may differ from the undeclared ones.** Compare the register characteristics of declared and undeclared houses, and say what this means for your result.

**2. Check the dates of your features.** The register shows today’s heating system and energy source, while your target covers earlier years. A building above 450 may already have replaced its heating *because of* that value. Check when the register entries were last updated, and handle buildings whose heating changed after their target period.

**3. Choose the error measure for model selection with the decision in mind.** Linear regression, ridge and lasso are trained by minimizing squared errors, and you plan to report the MAE in MJ/m² per year. Both are fine. But the measure you use to *choose between models* on validation data need not be either of them. Training with one loss and choosing or reporting with another is normal (deck 11; notebook 108b trained with squared errors and reported the MAE).

For your application, we suggest not choosing by raw MAE. The office’s decision is whether a house is above 450:

- An error of 100 for a house at 200 changes nothing. The same error for a house at 420 can move it across the threshold. The MAE counts both the same.

- The MAE is also driven by a few buildings with very high values.

A measure closer to the decision would be, for example, the share of houses above 450 among the model’s top 10% or 20% on the validation data. Another option is to treat the problem as classification (above 450 or not), with logistic regression and the measures of deck 12. Choose one measure before you compare models, say why it fits the outreach decision, and keep reporting the MAE as an easy-to-understand number.

**4. Can your models beat your own expert rule?** Your expert baseline uses the *combination* of construction period and energy source. How will your models represent that combination?

**5. Smaller points:**

- Which buildings are lost in the merge (21,206 → 16,376), and how many of them are houses?

- What is the rule for removing implausible values? Fix it before you look at the test set.

- Why the top 10% and 20%? How many owners can the office contact?

Does this make sense to you?
