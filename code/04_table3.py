"""
04_table3.py — Replicate Table 3 (main result).

Paper, Eq. 2: WTP_il = b1 + sum_m b_m T_im + X_i'b + L_l + a_i + e_il,
described as a "linear random effect model" with lottery effects and SEs
clustered by participant, estimated separately for risky (models 1-2)
and ambiguous (models 3-4) lotteries.

NOTE: the authors' Stata code (Code2_DescriptiveStatisticsAndResults.do)
actually runs
    reghdfe C_ i.treatment_5 [controls] if risky==1, cluster(Q_ID) absorb(Lottery)
i.e. pooled OLS with lottery fixed effects and clustered SEs (no random
effect). We estimate both:
  (A) the authors' code: pooled OLS + lottery FE + cluster(Q_ID)  -> replication
  (B) the model described in the text: GLS random effects (linearmodels)
Because the treatment dummies and controls do not vary within a person, the
lottery dummies are the same for every participant and the panel is
balanced, (A) and (B) give identical point estimates; the clustered standard
errors agree to three decimals.

Controls: the "Female" variable is coded Male 0, Female 1, Prefer not to say 3
(Female_paper, see 01_clean.py). This is the coding that reproduces the published
Table 3 cell by cell; the authors' released data code "Prefer not to say" as 2, which
changes columns (2) and (4) by at most 0.003 (see 06_extension.py, table E7).
Observations with missing Age are dropped in the models with controls (as in Stata).

Outputs: output/tables/table3_panelA.csv, table3_panelB.csv,
         table3_random_effects.csv, table3.tex
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import RandomEffects
from scipy.stats import t as tdist

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "data" / "clean" / "long.csv")
df["Female_reg"] = df["Female_paper"]   # coding that reproduces the published Table 3 (see 01_clean.py)
TREATS = ["T1", "T100", "T93&T107", "T0.1"]
for t in TREATS:
    df[t] = (df["treat5"] == t).astype(int)
df = df.rename(columns={"T93&T107": "T93_T107", "T0.1": "T0_1"})
TV = ["T1", "T100", "T93_T107", "T0_1"]
CONTROLS = ["attention", "Age", "Female_reg", "Student", "Risk_Taking_Falk_1", "Knowledge_Loot_box"]

# Panel B contrasts (rows) as linear combinations of treatment coefficients
CONTRASTS = {
    "T100 - T1": {"T100": 1, "T1": -1},
    "T1 - T0.1": {"T1": 1, "T0_1": -1},
    "T100 - T0.1": {"T100": 1, "T0_1": -1},
    "T93&T107 - T1": {"T93_T107": 1, "T1": -1},
    "T93&T107 - T100": {"T93_T107": 1, "T100": -1},
}

MODELS = {
    "(1) Risky": (1, False),
    "(2) Risky + controls": (1, True),
    "(3) Ambiguous": (0, False),
    "(4) Ambiguous + controls": (0, True),
}


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def lincom(params, cov, w):
    names = list(w)
    b = sum(params[n] * w[n] for n in names)
    v = sum(w[i] * w[j] * cov.loc[i, j] for i in names for j in names)
    return b, np.sqrt(v)


def fit(risky, controls):
    d = df[df["risky"] == risky].copy()
    xs = TV + (CONTROLS if controls else [])
    d = d.dropna(subset=xs + ["wtp"])

    # (A) authors' code: pooled OLS, lottery FE, cluster by participant.
    # statsmodels' default cluster correction G/(G-1)*(N-1)/(N-K) with K
    # counting the lottery dummies is the same as reghdfe's when the
    # absorbed FE are not nested in the clusters.
    f = "wtp ~ " + " + ".join(xs) + " + C(lottery_id)"
    ols = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["Q_ID"])[0]},
                                 use_t=True)  # t(G-1) like Stata
    # reghdfe reports _cons = mean(y) - mean(X)'b (FE averaged out)
    cons = d["wtp"].mean() - (d[xs].mean() * ols.params[xs]).sum()

    # (B) GLS random effects with lottery dummies, clustered SEs
    p = d.set_index(["Q_ID", "lottery_id"])
    exog = pd.concat([pd.Series(1.0, index=p.index, name="const"), p[xs],
                      pd.get_dummies(p.index.get_level_values(1), prefix="L", drop_first=True,
                                     dtype=float).set_index(p.index)], axis=1)
    re = RandomEffects(p["wtp"], exog).fit(cov_type="clustered", cluster_entity=True)
    return d, xs, ols, cons, re


rowsA, rowsB, rowsRE = {}, {}, {}
for name, (risky, controls) in MODELS.items():
    d, xs, ols, cons, re = fit(risky, controls)
    colA, colRE = {}, {}
    for v in xs:
        colA[v] = f"{ols.params[v]:.3f}{stars(ols.pvalues[v])} ({ols.bse[v]:.3f})"
        colRE[v] = f"{re.params[v]:.3f}{stars(re.pvalues[v])} ({re.std_errors[v]:.3f})"
    colA["Constant (reghdfe-style)"] = f"{cons:.3f}"
    colA["Observations"] = f"{int(ols.nobs)}"
    colA["Participants"] = f"{d['Q_ID'].nunique()}"
    colA["R2"] = f"{ols.rsquared:.3f}"
    colRE["Observations"] = f"{int(re.nobs)}"
    rowsA[name], rowsRE[name] = colA, colRE

    G = d["Q_ID"].nunique()
    colB = {}
    cov = ols.cov_params()
    for cname, w in CONTRASTS.items():
        b, se = lincom(ols.params, cov, w)
        pval = 2 * tdist.sf(abs(b / se), G - 1)
        colB[cname] = f"{b:.3f}{stars(pval)} ({se:.3f})"
    rowsB[name] = colB

A = pd.DataFrame(rowsA)
B = pd.DataFrame(rowsB)
RE_ = pd.DataFrame(rowsRE)
rename = {"T93_T107": "T93&T107", "T0_1": "T0.1", "Female_reg": "Female",
          "Risk_Taking_Falk_1": "Risk preference", "Knowledge_Loot_box": "Loot box knowledge",
          "attention": "Attention checks"}
A, RE_ = A.rename(index=rename), RE_.rename(index=rename)
A.to_csv(OUT / "table3_panelA.csv")
B.to_csv(OUT / "table3_panelB.csv")
RE_.to_csv(OUT / "table3_random_effects.csv")
with open(OUT / "table3.tex", "w") as fh:
    fh.write(A.fillna("").to_latex())
    fh.write(B.fillna("").to_latex())

pd.set_option("display.width", 200)
print("PANEL A — authors' code (pooled OLS + lottery FE, SE clustered by participant)")
print(A.fillna("").to_string())
print("\nPANEL B — contrasts")
print(B.to_string())
print("\nSame models as GLS random effects (text of the paper)")
print(RE_.fillna("").to_string())
