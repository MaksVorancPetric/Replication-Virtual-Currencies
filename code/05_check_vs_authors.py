"""
05_check_vs_authors.py — Verify our cleaned data against the authors'
own cleaned file (LongData.dta from the OSF package).
The file is not included in this repository; if it is missing, the check is skipped.

This is a check of our pipeline, not part of the analysis: the analysis
scripts only use data/clean/*, which we build from the raw files.
The script stops with an error if our data do not match the authors'.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUTHORS_FILE = ROOT / "data" / "raw" / "authors_LongData.dta"
if not AUTHORS_FILE.exists():
    # The authors' cleaned data are not part of this repository (course rules: no original
    # authors' output). Download LongData.dta from https://osf.io/cuvre (folder
    # DTADataSubmission), save it as data/raw/authors_LongData.dta and rerun to do this check.
    print("[check] skipped: data/raw/authors_LongData.dta not found "
          "(download LongData.dta from https://osf.io/cuvre to run this check)")
    raise SystemExit(0)
ours = pd.read_csv(ROOT / "data" / "clean" / "long.csv")
theirs = pd.read_stata(AUTHORS_FILE, convert_categoricals=False)

theirs = theirs.rename(columns={"Lottery": "lottery_id", "C_": "wtp_authors"})
m = ours.merge(theirs[["Q_ID", "lottery_id", "wtp_authors", "treatment", "attention", "Age",
                       "Female", "Student", "Risk_Taking_Falk_1", "Knowledge_Loot_box"]],
               on=["Q_ID", "lottery_id"], how="outer", suffixes=("", "_a"), indicator=True)

n_both = int((m["_merge"] == "both").sum())
max_diff = float(np.nanmax(np.abs(m["wtp"] - m["wtp_authors"])))
lines = [f"Rows: ours {len(ours)}, authors {len(theirs)}, matched {n_both}",
         f"Max |WTP difference|: {max_diff:.2e} (authors store floats in single precision)"]

# every variable must be identical (missing values must coincide)
share = {}
for ours_v, their_v in [("treatment", "treatment_a"), ("attention", "attention_a"), ("Age", "Age_a"),
                        ("Female_authors", "Female_a"), ("Student", "Student_a"),
                        ("Risk_Taking_Falk_1", "Risk_Taking_Falk_1_a"),
                        ("Knowledge_Loot_box", "Knowledge_Loot_box_a")]:
    a, b = m[ours_v], m[their_v]
    share[ours_v] = ((a == b) | (a.isna() & b.isna())).mean()
    lines.append(f"{ours_v:20s} vs authors' {their_v:22s}: {share[ours_v]:.4%} identical")

out = "\n".join(lines)
print(out)
(ROOT / "output").mkdir(exist_ok=True)
(ROOT / "output" / "check_vs_authors.txt").write_text(out + "\n")

# ---- hard checks: the pipeline must reproduce the authors' data
assert len(ours) == len(theirs) == n_both == 12048, "row count / merge mismatch"
assert max_diff < 1e-6, f"WTP differs from authors' data by {max_diff:.2e}"
assert all(s == 1.0 for s in share.values()), f"variables differ from authors' data: {share}"
# Female_paper differs from the authors' variable only in the code of "Prefer not to say" (2 -> 3)
fp, fa = ours["Female_paper"], ours["Female_authors"]
assert ((fp == fa) | (fa == 2) & (fp == 3) | (fa.isna() & fp.isna())).all(), "Female_paper inconsistent"
print("[check] OK: cleaned data are identical to the authors' LongData.dta")
