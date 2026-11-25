"""
00_master.py — Runs the whole replication from the raw data.

Usage (from the project root or anywhere):  python code/00_master.py
"""
import runpy
from pathlib import Path

CODE = Path(__file__).resolve().parent
STEPS = [
    "01_clean.py",            # raw -> data/clean/wide.csv, long.csv, entries.csv
    "02_table2.py",           # Table 2 (paper): descriptive statistics
    "03_figure3.py",          # Figure 3 (paper): mean WTP by lottery and treatment
    "04_table3.py",           # Table 3 (paper): main regression result
    "05_check_vs_authors.py", # check: our clean data vs authors' LongData.dta (stops on mismatch)
    "06_extension.py",        # Extension: wave FE, attrition, balance, T93 vs T107, OVB for T0.1
    "07_report_tables.py",    # LaTeX tables used in the report (Table 1, wave table, balance table)
]

for step in STEPS:
    print(f"\n===== {step} =====")
    try:
        runpy.run_path(str(CODE / step), run_name="__main__")
    except SystemExit as e:
        if e.code not in (0, None):
            raise
print("\nDone. Results are in output/.")
