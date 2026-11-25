"""
02_table2.py — Replicate Table 2 (descriptive statistics by treatment).

Reproduces the authors' notebook coding: in Table 2, Female and Student
are 1 only for "Female"/"Yes" and 0 otherwise (missing and "Prefer not to
say" count as 0), and duration is in minutes.seconds format.
Output: output/tables/table2.csv, output/tables/table2.tex
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

w = pd.read_csv(ROOT / "data" / "clean" / "wide.csv")
w["Female_t2"] = (w["Female"] == 1).astype(int)   # authors' Table 2 coding
w["Student_t2"] = (w["Student"] == 1).astype(int)

VARS = {"Age": "Age", "Female_t2": "Female", "Risk_Taking_Falk_1": "Risk Preference",
        "Student_t2": "Student", "Knowledge_Loot_box": "Loot Box Knowledge",
        "duration_mss": "Duration in minutes"}
ORDER = ["TC", "T1", "T0.1", "T100", "T93", "T107"]

rows = []
for v, lab in VARS.items():
    m = w.groupby("treat")[v].mean().reindex(ORDER)
    s = w.groupby("treat")[v].sem().reindex(ORDER)
    m["Total"], s["Total"] = w[v].mean(), w[v].sem()
    rows.append(pd.Series(m, name=lab))
    rows.append(pd.Series(s, name=f"  (s.e.) {lab}"))
n = w["treat"].value_counts().reindex(ORDER)
n["Total"] = len(w)
rows.append(pd.Series(n, name="Observations"))

t2 = pd.DataFrame(rows)
t2.to_csv(OUT / "table2.csv", float_format="%.3f")
t2.to_latex(OUT / "table2.tex", float_format="%.2f")
print(t2.round(2).to_string())
