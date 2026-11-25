# Replication of Toccafondi, Di Paolo and Di Guida (2025)

Problem Set 3, Empirical Methods, University of Zurich, HS 2026.
Group: A. S. Cojocaru, M. V. Petric, A. Roceanu, F. Rohde.

Paper: Toccafondi, N., Di Paolo, R. and Di Guida, S. (2025). Virtual currencies in online gaming increase the willingness to pay for loot boxes: an experimental analysis. Experimental Economics, 28(6), 1262-1280. https://doi.org/10.1017/eec.2026.10049

Original replication package (source of the data): https://osf.io/cuvre

## How to run

```
pip install -r requirements.txt
python code/00_master.py
```

The master script runs all scripts in order and writes the results to `output/`.

## Folder structure

```
data/raw/      raw data from the authors' replication package (not modified)
data/paper/    Table 3 as published in the paper, used to check our Table 1
data/clean/    created by 01_clean.py
code/          scripts, in the order they are run:
  00_master.py            runs all scripts below
  01_clean.py             cleans the raw data
  02_table2.py            descriptive statistics (Table 2 of the paper)
  03_figure3.py           average WTP by treatment and lottery (Figure 3 of the paper)
  04_table3.py            main regressions (Table 3 of the paper)
  05_check_vs_authors.py  compares our cleaned data with the authors' cleaned data (optional, see Data)
  06_extension.py         extension and robustness checks
  07_report_tables.py     LaTeX tables for the report, checks Table 1 against the published Table 3
  helpers.py              functions used by 06 and 07
output/        tables and figures
```

## Tables and figures in the report

- Table 1: `output/report_tables/table1_replication.tex`
- Figure 1: `output/figures/figure3.png`
- Table 2: `output/report_tables/table_balance.tex`
- Table 3: `output/report_tables/table_wave.tex`
- Other numbers in Sections 4 and 5: `output/tables/ext_*.csv`

## Data

The raw data (`RawData_submission_Anonymized.xlsx`, `Characteristics.xlsx`) come from the authors' replication package. The authors' cleaned data are not included. To run `05_check_vs_authors.py`, download `LongData.dta` from the OSF package (folder `DTADataSubmission`) and save it as `data/raw/authors_LongData.dta`. Without this file the check is skipped.

## Software

Python 3.9.6 with the packages in `requirements.txt`.
