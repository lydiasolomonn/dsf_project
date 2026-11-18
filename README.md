# Predicting the Heating Energy Index (IDC) of Geneva Buildings
## Data
All raw files are stored unchanged in `data/raw/`. They were downloaded on 13 November 2026.
Do not edit them; all cleaning happens in the scripts.

| File | Dataset | Publisher | Used for | Source |
|---|---|---|---|---|
| SCANE_INDICE_MOYENNES_3_ANS.csv | IDC des bâtiments | Canton of Geneva (SITG) | Target (INDICE_MOY3) | https://ckan.opendata.swiss/perma/SCANE_INDICE_MOYENNES_3_ANS@canton-geneve |
| gebaeude_batiment_edificio.csv | Federal Register of Buildings and Dwellings (GWR), canton GE | Federal Statistical Office | Features | https://www.housing-stat.ch/__publicdata |
| SCANE_RELATION_BATI_CHAUDIERE.csv | Bâtiments et chaudières centrales | Canton of Geneva (SITG) | Groups for the split | https://sitg.ge.ch/donnees/scane-relation-bati-chaudiere |
| OCEN_ETAT_IDC_PUBLIC.csv | Declaration status per building | Canton of Geneva (SITG) | Buildings without a value | https://ckan.opendata.swiss/en/perma/OCEN_ETAT_IDC_PUBLIC@canton-geneve |

All files are linked by the federal building number (EGID).

## How to run
Run the scripts in order:
1. `01_target.py` – cleans the IDC file, builds the target
2. `02_features.py` – prepares the register features
3. `03_merge.py` – merges everything, adds house flag and group ID → `data/processed/model_table.csv`