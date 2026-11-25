"""
06_extension.py — Extension and robustness checks for the report.

Needs data/clean/{long,wide,entries}.csv (written by 01_clean.py).
Only numpy / pandas / scipy are used (cluster-robust OLS in helpers.py with the
same small-sample correction as 04_table3.py).

Outputs (output/tables/):
  ext_E1_wave_fe.csv             Effects within data-collection waves (report Table 2)
  ext_E1b_wave_difference_T1.csv T1 effect, June minus January 2023
  ext_E2_attrition.csv           Entries, completion and screening by treatment arm
  ext_E2b_attrition_tests.csv    Fisher tests and wave-level non-completion
  ext_E2c_reentries.csv          Repeated Prolific IDs in the raw export
  ext_E3_balance.csv             Balance: 2025 wave vs 2023 waves; across 2023 arms
  ext_E4_t93_t107.csv            T93 and T107 estimated separately
  ext_E5_ovb_t01.csv             Back-of-the-envelope omitted-variable bias for T0.1
  ext_E6_attention.csv           Share passing both attention checks, by treatment
  ext_E7_female_coding.csv       Sensitivity of columns (2) and (4) to the coding of Female
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from helpers import cluster_ols, lincom, stars

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "output" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

long = pd.read_csv(CLEAN / "long.csv")
wide = pd.read_csv(CLEAN / "wide.csv")
entries = pd.read_csv(CLEAN / "entries.csv")

assert set(long["wave"].unique()) == {"2023-01", "2023-06", "2025-07"}, "unexpected waves"
assert set(wide.loc[wide["treat"] == "T0.1", "wave"]) == {"2025-07"}, "T0.1 must be 2025 only"
assert len(wide) == 753 and len(long) == 12048

# ------------------------------------------------------------------ helpers
ORDER = ["TC", "T1", "T0.1", "T100", "T93", "T107"]
CONTROLS = ["attention", "Age", "Female_reg", "Student", "Risk_Taking_Falk_1", "Knowledge_Loot_box"]
long["Female_reg"] = long["Female_paper"]                 # coding that reproduces the published Table 3
long["T1"] = (long["treat5"] == "T1").astype(int)
long["T100"] = (long["treat5"] == "T100").astype(int)
long["T93_T107"] = (long["treat5"] == "T93&T107").astype(int)
long["T0_1"] = (long["treat5"] == "T0.1").astype(int)
long["T93"] = (long["treat"] == "T93").astype(int)
long["T107"] = (long["treat"] == "T107").astype(int)
long["w_jun"] = (long["wave"] == "2023-06").astype(int)


def fit(sub, tv, risky):
    d = sub[sub["risky"] == risky].dropna(subset=tv + CONTROLS + ["wtp"])
    return cluster_ols(d, tv + CONTROLS)


def cell(b, se, p):
    return f"{b:.3f}{stars(p)} ({se:.3f})"


# ------------------------------------------------ E1: wave fixed effects
specs = {
    "(1) Baseline": (long, ["T1", "T100", "T93_T107", "T0_1"]),
    "(2) 2023 + wave FE": (long[long["wave"] != "2025-07"], ["T1", "T100", "T93_T107", "w_jun"]),
    "(3) June 2023": (long[long["wave"] == "2023-06"], ["T1", "T100", "T93_T107"]),
    "(4) January 2023": (long[long["wave"] == "2023-01"], ["T1", "T100"]),
}
rows, res = {}, {}
for panel, risky in [("A: Risky", 1), ("B: Ambiguous", 0)]:
    for sname, (sub, tv) in specs.items():
        r = fit(sub, tv, risky)
        res[(panel, sname)] = r
        col = {}
        for v, lab in [("T1", "T1"), ("T100", "T100"), ("T93_T107", "T93&T107"), ("T0_1", "T0.1")]:
            col[f"{panel} | {lab}"] = cell(r["b"][v], r["se"][v], r["p"][v]) if v in tv else ""
        b, se, p = lincom(r, {"T100": 1, "T1": -1})
        col[f"{panel} | T100 - T1"] = cell(b, se, p)
        col[f"{panel} | Observations"] = f"{r['N']:,}"
        col[f"{panel} | Participants"] = f"{r['G']}"
        rows.setdefault(sname, {}).update(col)
E1 = pd.DataFrame(rows)
E1.to_csv(OUT / "ext_E1_wave_fe.csv")

# consistency with the numbers quoted in the report (wave table)
chk = {("A: Risky", "(1) Baseline", "T1"): 0.040, ("A: Risky", "(2) 2023 + wave FE", "T1"): 0.038,
       ("A: Risky", "(3) June 2023", "T1"): 0.021, ("A: Risky", "(4) January 2023", "T1"): 0.069}
for (pn, sn, v), val in chk.items():
    assert abs(res[(pn, sn)]["b"][v] - val) < 6e-4, (pn, sn, v)
assert res[("A: Risky", "(2) 2023 + wave FE")]["N"] == 6589

# T1 effect June vs January (independent samples -> z-test on the difference)
rj, rn = res[("A: Risky", "(3) June 2023")], res[("A: Risky", "(4) January 2023")]
dif = rj["b"]["T1"] - rn["b"]["T1"]
dse = np.hypot(rj["se"]["T1"], rn["se"]["T1"])
p_wave = 2 * stats.norm.sf(abs(dif / dse))
pd.Series({"T1_June_minus_January": dif, "se": dse, "p_value": p_wave}).to_csv(
    OUT / "ext_E1b_wave_difference_T1.csv", header=["value"], float_format="%.4f")

# ------------------------------------------------ E2: attrition by arm
entries["treat"] = entries["treatment"].map(
    {1: "TC", 2: "T1", 3: "T100", 4: "T93", 5: "T107", 6: "T0.1"})
entries["non_completed"] = 1 - entries["completed"]
entries["early_exit"] = entries["non_completed"] - entries["screen_out"]
g = entries.groupby("treat")
E2 = pd.DataFrame({"entries": g.size(), "completed": g["completed"].sum(),
                   "failed_3x": g["screen_out"].sum(), "early_exit": g["early_exit"].sum()}).reindex(ORDER)
E2.loc["T93&T107"] = E2.loc[["T93", "T107"]].sum()
E2["share_non_completed"] = 1 - E2["completed"] / E2["entries"]
E2["share_failed_3x"] = E2["failed_3x"] / E2["entries"]
E2.to_csv(OUT / "ext_E2_attrition.csv", float_format="%.3f")
assert int(E2.loc[ORDER, "completed"].sum()) == 753


def fisher(a, b, col):
    t = [[a[col].sum(), len(a) - a[col].sum()], [b[col].sum(), len(b) - b[col].sum()]]
    return stats.fisher_exact(t)[1]


tc = entries[entries["treat"] == "TC"]
hard = entries[entries["treat"].isin(["T93", "T107"])]
t93, t107 = entries[entries["treat"] == "T93"], entries[entries["treat"] == "T107"]
pd.Series({
    "Fisher p: non-completion TC vs T93&T107": fisher(tc, hard, "non_completed"),
    "Fisher p: failed 3x TC vs T93&T107": fisher(tc, hard, "screen_out"),
    "Fisher p: non-completion T93 vs T107": fisher(t93, t107, "non_completed"),
    "non-completion share 2023": entries.loc[entries["wave"] != "2025-07", "non_completed"].mean(),
    "non-completion share 2025": entries.loc[entries["wave"] == "2025-07", "non_completed"].mean(),
}).to_csv(OUT / "ext_E2b_attrition_tests.csv", header=["value"], float_format="%.3f")

# re-entries: same Prolific ID more than once in the raw export
dup = entries[entries["Q_ID"].duplicated(keep=False)]
reent = dup.groupby("Q_ID").agg(rows=("treatment", "size"), n_treatments=("treatment", "nunique"),
                                n_completed=("completed", "sum"))
pd.Series({"ids_with_repeats": len(reent), "rows": int(reent["rows"].sum()),
           "ids_with_more_than_1_treatment": int((reent["n_treatments"] > 1).sum()),
           "ids_completed_more_than_once": int((reent["n_completed"] > 1).sum())}).to_csv(
    OUT / "ext_E2c_reentries.csv", header=["value"])

# ------------------------------------------------ E3: balance
BV = {"Age": "Age", "Female": "Female (0/1, missing excluded)", "Student": "Student",
      "Risk_Taking_Falk_1": "Risk preference", "Knowledge_Loot_box": "Loot box knowledge",
      "attention": "Passed both attention checks"}
y25 = wide["wave"] == "2025-07"
w23 = wide[~y25]
rows = []
for v, lab in BV.items():
    a, b = wide.loc[y25, v].dropna(), wide.loc[~y25, v].dropna()
    f = stats.f_oneway(*[w23.loc[w23["treat5"] == t, v].dropna() for t in ["TC", "T1", "T100", "T93&T107"]])
    rows.append({"variable": lab, "mean_2025": a.mean(), "mean_2023": b.mean(),
                 "p_2025_vs_2023 (Welch t)": stats.ttest_ind(a, b, equal_var=False).pvalue,
                 "p_across_2023_arms (ANOVA)": f.pvalue})
E3 = pd.DataFrame(rows)
E3.to_csv(OUT / "ext_E3_balance.csv", index=False, float_format="%.3f")

# ------------------------------------------------ E4: T93 and T107 separately
rows4 = []
for panel, risky in [("Risky", 1), ("Ambiguous", 0)]:
    r = fit(long, ["T1", "T100", "T93", "T107", "T0_1"], risky)
    for nm, w in {"T93 - T100": {"T93": 1, "T100": -1}, "T107 - T100": {"T107": 1, "T100": -1},
                  "T93 - T107": {"T93": 1, "T107": -1}}.items():
        b, se, p = lincom(r, w)
        rows4.append({"lotteries": panel, "contrast": nm, "estimate": b, "se": se, "p_value": p})
E4 = pd.DataFrame(rows4)
E4.to_csv(OUT / "ext_E4_t93_t107.csv", index=False, float_format="%.3f")

# ------------------------------------------------ E5: OVB for T0.1 (risk preference)
r2 = fit(long, ["T1", "T100", "T93_T107", "T0_1"], 1)
gamma = r2["b"]["Risk_Taking_Falk_1"]
delta = (wide.loc[wide["treat"] == "T0.1", "Risk_Taking_Falk_1"].mean()
         - wide.loc[wide["treat"] != "T0.1", "Risk_Taking_Falk_1"].mean())
d1 = long[long["risky"] == 1].dropna(subset=["wtp"])
r1 = cluster_ols(d1, ["T1", "T100", "T93_T107", "T0_1"])
pd.Series({"gamma (risk pref., model 2)": gamma, "delta (T0.1 minus other arms)": delta,
           "gamma*delta (approx. bias if omitted)": gamma * delta,
           "T0.1 coef. without controls (col 1)": r1["b"]["T0_1"],
           "T0.1 coef. with controls (col 2)": r2["b"]["T0_1"],
           "difference col1 - col2": r1["b"]["T0_1"] - r2["b"]["T0_1"]}).to_csv(
    OUT / "ext_E5_ovb_t01.csv", header=["value"], float_format="%.4f")

# ------------------------------------------------ E7: coding of Female (columns 2 and 4 of Table 1)
codings = {"Published Table 3: Prefer not to say = 3": long["Female_paper"],
           "Released data/code: Prefer not to say = 2": long["Female_authors"],
           "Dummy, Prefer not to say = 0": (long["Female_authors"] == 1).astype(float).where(long["Female_authors"].notna())}
rows7 = []
for cname, fem in codings.items():
    tmp = long.assign(Female_reg=fem)
    for panel, risky in [("Risky (col. 2)", 1), ("Ambiguous (col. 4)", 0)]:
        r = fit(tmp, ["T1", "T100", "T93_T107", "T0_1"], risky)
        rows7.append({"Female coding": cname, "lotteries": panel, **{k: r["b"][k] for k in ["T1", "T100", "T93_T107", "T0_1", "Female_reg"]},
                      "Constant": r["cons"]})
E7 = pd.DataFrame(rows7)
E7.to_csv(OUT / "ext_E7_female_coding.csv", index=False, float_format="%.3f")
_ref = E7[E7["Female coding"].str.startswith("Published")].set_index("lotteries")
_alt = E7[E7["Female coding"].str.startswith("Released")].set_index("lotteries")
max_diff_treat = float((_ref[["T1", "T100", "T93_T107", "T0_1"]] - _alt[["T1", "T100", "T93_T107", "T0_1"]]).abs().max().max())
assert max_diff_treat < 0.0035, max_diff_treat

# ------------------------------------------------ E6: attention by treatment
wide["treat"] = pd.Categorical(wide["treat"], ORDER)
wide.groupby("treat", observed=True)["attention"].agg(["mean", "count"]).to_csv(
    OUT / "ext_E6_attention.csv", float_format="%.3f")

pd.set_option("display.width", 220, "display.max_columns", 20)
print("E1 wave FE\n", E1.to_string())
print(f"  T1 effect June - January: {dif:.3f} (SE {dse:.3f}), p = {p_wave:.3f}")
print("\nE2 attrition\n", E2.round(3).to_string())
print("\nE3 balance\n", E3.round(3).to_string(index=False))
print("\nE4 T93 / T107\n", E4.round(3).to_string(index=False))
print("\nE7 Female coding\n", E7.round(3).to_string(index=False))
print(f"max |difference| in treatment coefficients, published vs. released coding: {max_diff_treat:.3f}")
print(f"\nE5 OVB T0.1: gamma={gamma:.4f}  delta={delta:.3f}  gamma*delta={gamma * delta:.4f}")
print("[extension] done")
