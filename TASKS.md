# Project Tasks: Heating Energy Index Geneva

Task list for our group project. It is a guide, not a contract: adjust it as the project develops.

## How to use this file

- Change the **Status** of your own tasks: `Open` → `In progress` → `Done`.
- Edit only your own rows, so we do not get merge conflicts.
- Pull the latest version before you edit (`git pull`), or edit directly in the GitHub browser.
- `FB` refers to the lecturer's feedback of 14 November (points 1 to 5).

## Roles

| Person | Name | Role | Responsible for |
| --- | --- | --- | --- |
| Person 1 | | Data: target | Cleaning the heating index and building the target, merge loss, year effect, classification variant, houses-only experiment, figures |
| Person 2 | | Data: features | Building register, heating-change check, split groups, merging, error analysis, replication test |
| Person 3 | | Evaluation | Fixed decisions (selection measure, cleaning rule, list lengths), split and folds, metrics, baselines, test evaluation, hit-rate curve, final check |
| Person 4 | | Models | Pipeline with interaction terms, linear models, feature sets, slides |
| Person 5 | | Coordination and report | Repo, answers to the feedback, target group, declared vs. undeclared houses, descriptive analysis, report, AI-use statement, submission |

## Phase 1: Start

| No | Task | Owner | Reviewer | Due | Status | FB |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Set up the GitHub repo: folder structure, README skeleton, fixed random seeds | All | Person 3 | 17 Nov | Done | |
| 2 | Freeze the raw data: store all 4 files with their download date, add source links to the README | Person 1 | Person 2 | 18 Nov | Done | |
| 3 | Answer the lecturer's feedback (14 Nov) point by point in the project description document | Person 5 | All | 19 Nov | Open | 1-5 |
| 4 | Coaching on 18 Nov, 18:15: bring questions on the selection measure, the split by building class and the interaction terms | All | | 18 Nov | Done | 1, 3, 4 |
| 5 | Fix three decisions in the README before any modelling: (a) the selection measure; (b) the rule for implausible target values; (c) the list lengths and why | Person 3 | All | 19 Nov | Done, needs to be reviewed | 3, 5 |

## Phase 2: Data

| No | Task | Owner | Reviewer | Due | Status | FB |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | Clean the heating index file with the rule from task 5: duplicates, implausible values, year 0. Build the target: latest 3-year average from 2023 onwards | Person 1 | Person 3 | 19 Nov | Done, ready for review | 5 |
| 7 | Document the merge loss from 21,206 to 16,376: which buildings drop out at which step, and how many of them are houses | Person 1 | Person 2 | 20 Nov | Done | 5 |
| 8 | Prepare the building register: select the features, turn codes into readable labels, handle missing values | Person 2 | Person 1 | 19 Nov | Done | |
| 9 | Check heating changes: compare the register's energy source with `AGENT_ENERGETIQUE_1` in the target years, and look at the update date of the heating entry. Decide how to handle buildings whose heating changed after the target period | Person 2 | Person 4 | 20 Nov | Done, for review | 2 |
| 10 | Check the year effect: median heating index per year. Decide whether a fixed reference period is needed | Person 1 | Person 4 | 20 Nov | Done, for review | |
| 11 | Build the groups for the split: boiler file plus twins with the same year, value and floor area | Person 2 | Person 3 | 20 Nov | Done, for review | 1 |
| 12 | Merge target and features, check the row count, add a flag for houses (one or two dwellings) as the evaluation group | Person 2 | Person 1 | 21 Nov | Done | 1 |
| 13 | Define the target group: houses without a value and with status "fewer than 5 heat consumers". Exclude densification zones and new buildings | Person 5 | Person 2 | 20 Nov | Open | |
| 14 | Compare declared and undeclared houses on the register characteristics, and write what the differences mean for the result | Person 5 | Person 1 | 21 Nov | Open | 1 |
| 15 | Descriptive analysis: distribution of the target, by building class and construction period | Person 5 | Person 1 | 21 Nov | Open | |

## Phase 3: Model

| No | Task | Owner | Reviewer | Due | Status | FB |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | Split: hold out 15% as test set, by group and stratified by building class. Lock it away. Set up 5-fold cross-validation on the rest, groups kept together, also by building class | Person 3 | Person 4 | 21 Nov | Done | 1 |
| 17 | Write the metric functions: the selection measure from task 5 on houses; MAE and share captured; each for houses and for all buildings | Person 3 | Person 5 | 21 Nov | done, needs to be reviewed | 1, 3 |
| 18 | Compute the baselines in cross-validation: training mean, and the expert rule construction period by energy source; scored on houses | Person 3 | Person 4 | 22 Nov | Done | 1, 4 |
| 19 | Preprocessing as a pipeline, fitted on the training part of each fold only: encoding, scaling, and the interaction of construction period and energy source | Person 4 | Person 3 | 22 Nov | Done, needs review  | 4 |
| 20 | Train linear regression, ridge and lasso, with and without the interaction; choose settings by cross-validation on houses with the selection measure | Person 4 | Person 3 | 23 Nov | Done, needs review | 3, 4 |
| 21 | Classification variant (above 450 or not) with logistic regression, compared on the same selection measure | Person 5 | Person 4 | 24 Nov | DONE, NEEDS REVIEW | 3 |
| 22 | Compare feature sets: with and without heating information, with and without municipality | Person 4 | Person 2 | 24 Nov | Open | 2 |
| 23 | Experiment: trained on houses only versus trained on all, measured on the same houses in cross-validation | Person 1 | Person 4 | 24 Nov | Open | 1 |
| 24 | Error analysis: errors by building class and construction period, check predictions for plausibility | Person 2 | Person 5 | 24 Nov | Open | |
| 25 | Coaching on 25 Nov, 18:15: show the cross-validation results and bring one result you do not trust | All | | 25 Nov | Open | |
| 26 | Choose the model by the cross-validation on houses, evaluate exactly once on the test set. Report houses first, all buildings beside them | Person 3 | Person 4 | 25 Nov | Open | 1 |
| 27 | Hit-rate curve on test houses for several list lengths, with the number of owners each length means | Person 3 | Person 5 | 26 Nov | Open | 5 |

## Phase 4: Submission

| No | Task | Owner | Reviewer | Due | Status | FB |
| --- | --- | --- | --- | --- | --- | --- |
| 28 | Set up the report skeleton (3 to 6 pages) from the project description, assign sections | Person 5 | All | 24 Nov | Open | |
| 29 | Figures for report and slides: distribution, model comparison, hit-rate curve | Person 1 | Person 5 | 26 Nov | Open | |
| 30 | Write the report: everyone writes their own part | All | | 27 Nov | Open | |
| 31 | Write the AI-use statement: which tools, for which parts | Person 5 | All | 27 Nov | Open | |
| 32 | Build the slides (8 at most) plus backup slides with details | All | Person 5 | 28 Nov | Open | |
| 33 | Test replication: clone the repo fresh and run everything from scratch | Person 2 | Person 3 | 28 Nov | Open | |
| 34 | Proofread report and slides: do all numbers match, and is every feedback point answered? | Person 3 | All | 29 Nov | Open | 1-5 |
| 35 | Submit by 23:59: folder GroupNN-topic with report, slides, data and code | All | Person 1 | 30 Nov | Open | |

## Phase 5: Presentation

| No | Task | Owner | Reviewer | Due | Status | FB |
| --- | --- | --- | --- | --- | --- | --- |
| 36 | Rehearsal: 8-minute talk, then ask each other interview questions, including on the feedback points | All | | 1 Dec | Open | |
| 37 | Presentation evenings on 2 Dec and 9 Dec, 18:15 (attendance mandatory at both) | All | | 9 Dec | Open | |

## Rules for all code

1. The test set stays closed until task 26.
2. Everything that learns from data (encoding, scaling, rule averages) is fitted inside the training part of each fold.
3. Buildings of the same group always stay in the same fold and on the same side of the split.
4. Fixed seeds, the frozen raw data and the shared metric function from task 17.
5. Every decision goes into the README with date and reason before the code is written.
