# Predicting the Heating Energy Index (IDC) of Geneva Buildings
## Data
All raw files are stored unchanged in `data/raw/`. They were downloaded on 13 November 2026.
Do not edit them; all cleaning happens in the scripts.

| File | Dataset | Publisher | Used for | Source |
|---|---|---|---|---|
| SCANE_INDICE_MOYENNES_3_ANS.csv | IDC des bâtiments | Canton of Geneva (SITG) | Target (INDICE_MOY3) | https://ckan.opendata.swiss/perma/SCANE_INDICE_MOYENNES_3_ANS@canton-geneve |
| gebaeude_batiment_edificio.csv | Federal Register of Buildings and Dwellings (GWR), canton GE | Federal Statistical Office | Features | https://www.housing-stat.ch/__publicdata |
| kodes_codes_codici.csv | GWR code list | Federal Statistical Office | Readable labels for the codes | https://www.housing-stat.ch/__publicdata |
| SCANE_RELATION_BATI_CHAUDIERE.csv | Bâtiments et chaudières centrales | Canton of Geneva (SITG) | Groups for the split | https://sitg.ge.ch/donnees/scane-relation-bati-chaudiere |
| OCEN_ETAT_IDC_PUBLIC.csv | Declaration status per building | Canton of Geneva (SITG) | Buildings without a value | https://ckan.opendata.swiss/en/perma/OCEN_ETAT_IDC_PUBLIC@canton-geneve |

All files are linked by the federal building number (EGID).

## How to run
Run the scripts in order:
1. `01_target.py` – cleans the IDC file, builds the target
2. `02_features.py` – prepares the register features
3. `03_merge.py` – merges everything, adds house flag and group ID → `data/processed/model_table.csv`


##Decisions:

1. Our selection measure: We choose models by the share of houses above 450 among the top 20% of the ranking. It fits the outreach decision, and it applies to every model we compare: regression models rank by predicted index, the classification model by predicted probability, and the baselines by their group averages.
2. Implausible target values: We remove buildings whose 3-year average is below 50 or above 1,500 MJ/m² per year(25 buildings, 1 house). Even very efficient buildings lie above 50, so lower values point to a wrong floor area; values above 1,500, about three times the threshold,
point to data errors. The rule was fixed before the test set was used.
3. List length: The right length is the office’s capacity. Since we don’t know it, we show the whole curve and report 10% and 20% as examples of a small and a medium campaign, both below the 37.5% share of problem houses. Up to that share, a perfect model could fill the whole list with problem houses only.