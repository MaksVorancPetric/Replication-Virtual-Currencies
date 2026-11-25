"""
07_report_tables.py — Write the LaTeX tables used in the report and check them.

Table 1 (replication of Table 3 of the paper) is estimated here with the numpy
cluster-OLS in helpers.py (same estimator and standard errors as 04_table3.py).
Two automatic checks stop the script if they fail:
  (a) Table 1 equals the published Table 3 (data/paper/table3_published.csv) in every cell
      (coefficients, stars, standard errors, constants with SE, Panel B, N);
  (b) Table 1 equals the output of 04_table3.py (statsmodels / linearmodels), if it was run.
The wave-FE table and the balance table are formatted from the csv files written by 06_extension.py.

Outputs: output/report_tables/table1_replication.tex, table_wave.tex, table_balance.tex
"""
import re
from pathlib import Path

import pandas as pd

from helpers import cluster_ols, lincom, stars

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "report_tables"
OUT.mkdir(parents=True, exist_ok=True)
TAB = ROOT / "output" / "tables"


def num(x, nd=3):
    return f"{x:.{nd}f}".replace("-", "$-$")


def coef(b, p):
    sup = stars(p)
    return f"{num(b)}" + (f"$^{{{sup}}}$" if sup else "")


def cell(b, se, p):
    return f"{b:.3f}{stars(p)} ({se:.3f})"


# ------------------------------------------------------------ Table 1
df = pd.read_csv(ROOT / "data" / "clean" / "long.csv")
df["Female_reg"] = df["Female_paper"]       # coding that reproduces the published Table 3
for lab, name in [("T1", "T1"), ("T100", "T100"), ("T93&T107", "T93_T107"), ("T0.1", "T0_1")]:
    df[name] = (df["treat5"] == lab).astype(int)
TV = ["T1", "T100", "T93_T107", "T0_1"]
CT = ["attention", "Age", "Female_reg", "Student", "Risk_Taking_Falk_1", "Knowledge_Loot_box"]
MODELS = [(1, False), (1, True), (0, False), (0, True)]
res = []
for risky, ctrl in MODELS:
    xs = TV + (CT if ctrl else [])
    d = df[df["risky"] == risky].dropna(subset=xs + ["wtp"])
    res.append((cluster_ols(d, xs), ctrl))

LAB = {"T1": "T1", "T0_1": "T0.1", "T100": "T100", "T93_T107": "T93&T107", "attention": "Attention checks",
       "Age": "Age", "Female_reg": "Female", "Student": "Student", "Risk_Taking_Falk_1": "Risk preference",
       "Knowledge_Loot_box": "Loot box knowledge"}
CONTRASTS = {"T100 - T1": {"T100": 1, "T1": -1}, "T1 - T0.1": {"T1": 1, "T0_1": -1},
             "T100 - T0.1": {"T100": 1, "T0_1": -1}, "T93&T107 - T1": {"T93_T107": 1, "T1": -1},
             "T93&T107 - T100": {"T93_T107": 1, "T100": -1}}

# all cells of our table as strings in the format of the published table
ours = {}
for j, (r, ctrl) in enumerate(res):
    for v, l in LAB.items():
        if v in TV or ctrl:
            ours[(l, j)] = cell(r["b"][v], r["se"][v], r["p"][v])
    ours[("Constant", j)] = cell(r["cons"], r["cons_se"], r["cons_p"])
    for cname, w in CONTRASTS.items():
        ours[(cname, j)] = cell(*lincom(r, w))
    ours[("N", j)] = str(r["N"])

# (a) check against the published Table 3
pub = pd.read_csv(ROOT / "data" / "paper" / "table3_published.csv", index_col=0, dtype=str).fillna("")
n_cells = 0
for row in pub.index:
    for j, col in enumerate(pub.columns):
        if pub.loc[row, col] == "":
            assert (row, j) not in ours, ("cell should be empty", row, col)
            continue
        assert ours[(row, j)] == pub.loc[row, col], (row, col, ours[(row, j)], pub.loc[row, col])
        n_cells += 1
print(f"[report tables] OK: Table 1 equals the published Table 3 in all {n_cells} cells")

# (b) check against the output of 04_table3.py
_pa, _pb = TAB / "table3_panelA.csv", TAB / "table3_panelB.csv"
if _pa.exists() and _pb.exists():
    A04, B04 = pd.read_csv(_pa, index_col=0), pd.read_csv(_pb, index_col=0)
    for j, col in enumerate(A04.columns):
        for (l, jj), val in ours.items():
            if jj != j:
                continue
            if l in A04.index and l not in ("Constant", "N"):
                assert A04.loc[l, col] == val, ("04 vs 07", l, col)
            elif l in B04.index:
                assert B04.loc[l, col] == val, ("04 vs 07", l, col)
        assert A04.loc["Constant (reghdfe-style)", col] == f"{res[j][0]['cons']:.3f}", col
        assert int(A04.loc["Observations", col]) == res[j][0]["N"], col
        assert A04.loc["R2", col] == f"{res[j][0]['r2']:.3f}", col
    print("[report tables] OK: Table 1 equals the output of 04_table3.py cell by cell")
else:
    print("[report tables] note: output/tables/table3_panelA/B.csv not found (run 04_table3.py to compare)")

rows = []


def coefrow(label, var):
    a, b = [label], [""]
    for r, ctrl in res:
        if var in r["b"].index and (var in TV or ctrl):
            a.append(coef(r["b"][var], r["p"][var]))
            b.append(f"({num(r['se'][var])})")
        else:
            a.append("")
            b.append("")
    rows.append(" & ".join(a) + r" \\")
    rows.append(" & ".join(b) + r" \\")


rows.append(r"\multicolumn{5}{l}{\emph{Panel A: Effects relative to the control (GBP)}} \\")
coefrow("T1 (tokens 1:1)", "T1")
coefrow("T0.1 (tokens 0.1:1)", "T0_1")
coefrow("T100 (tokens 100:1)", "T100")
coefrow(r"T93\&T107 (tokens 93:1, 107:1)", "T93_T107")
coefrow("Attention checks", "attention")
coefrow("Age", "Age")
coefrow("Female", "Female_reg")
coefrow("Student", "Student")
coefrow("Risk preference", "Risk_Taking_Falk_1")
coefrow("Loot box knowledge", "Knowledge_Loot_box")
rows.append("Constant & " + " & ".join(coef(r["cons"], r["cons_p"]) for r, _ in res) + r" \\")
rows.append(" & " + " & ".join(f"({num(r['cons_se'])})" for r, _ in res) + r" \\")
rows.append(r"\midrule")
rows.append(r"\multicolumn{5}{l}{\emph{Panel B: Differences between treatments}} \\")
for lab, key in [("T100 $-$ T1", "T100 - T1"), ("T1 $-$ T0.1", "T1 - T0.1"), ("T100 $-$ T0.1", "T100 - T0.1"),
                 (r"T93\&T107 $-$ T1", "T93&T107 - T1"), (r"T93\&T107 $-$ T100", "T93&T107 - T100")]:
    a, b = [lab], [""]
    for r, _ in res:
        est, se, p = lincom(r, CONTRASTS[key])
        a.append(coef(est, p))
        b.append(f"({num(se)})")
    rows.append(" & ".join(a) + r" \\")
    rows.append(" & ".join(b) + r" \\")
rows.append(r"\midrule")
rows.append("Controls & No & Yes & No & Yes \\\\")
rows.append("Lottery fixed effects & Yes & Yes & Yes & Yes \\\\")
rows.append("Observations & " + " & ".join(f"{r['N']:,}" for r, _ in res) + r" \\")
rows.append("Participants & " + " & ".join(str(r["G"]) for r, _ in res) + r" \\")
rows.append("R$^2$ & " + " & ".join(f"{r['r2']:.3f}" for r, _ in res) + r" \\")

t1 = (r"\begin{table}[!t]" "\n" r"\centering\footnotesize\renewcommand{\arraystretch}{0.92}" "\n"
      r"\caption{Replication of Table 3: Effect of the payment currency on willingness to pay}" "\n"
      r"\label{tab:replication}" "\n"
      r"\begin{tabular}{lcccc}" "\n" r"\toprule" "\n"
      r" & \multicolumn{2}{c}{Risky lotteries} & \multicolumn{2}{c}{Ambiguous lotteries} \\" "\n"
      r" & (1) & (2) & (3) & (4) \\" "\n" r"\midrule" "\n"
      + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n\n"
      r"\smallskip" "\n" r"\begin{minipage}{0.97\textwidth}\footnotesize" "\n"
      r"\emph{Notes:} Replication of Table 3 in Toccafondi et al.\ (2025). Dependent variable: willingness to pay "
      r"to open a loot box, divided by the exchange rate (GBP, between 0 and 1). Columns (1)--(2): 11 risky lotteries; "
      r"columns (3)--(4): 5 ambiguous lotteries; one observation per participant and lottery. All models include "
      r"lottery fixed effects; columns (2) and (4) add the individual controls (participants with missing demographics "
      r"are dropped). Female is coded 0 (male), 1 (female) and 3 (prefer not to say), the coding that reproduces the "
      r"published table (Section~\ref{sec:replication}). Panel A: coefficients relative to the control group paying in "
      r"GBP (TC). Panel B: differences between treatment coefficients. Estimated as in the authors' code (pooled OLS "
      r"with lottery fixed effects); standard errors clustered at the participant level in parentheses. "
      r"$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$." "\n"
      r"\end{minipage}" "\n" r"\end{table}" "\n")
(OUT / "table1_replication.tex").write_text(t1, encoding="utf-8")

# ------------------------------------------------------------ wave table
E1 = pd.read_csv(TAB / "ext_E1_wave_fe.csv", index_col=0).fillna("")


def split(cellstr):
    cellstr = str(cellstr).strip()
    if not cellstr:
        return "", ""
    m = re.match(r"^(-?\d+\.\d+)(\**)\s*\((\d+\.\d+)\)$", cellstr)
    if not m:
        return cellstr, ""
    b, st, se = m.groups()
    return num(float(b)) + (f"$^{{{st}}}$" if st else ""), f"({se})"


wrows = []
for panel, title in [("A: Risky", "Panel A: Risky lotteries"), ("B: Ambiguous", "Panel B: Ambiguous lotteries")]:
    wrows.append(rf"\multicolumn{{5}}{{l}}{{\emph{{{title}}}}} \\")
    for lab, tex in [("T1", "T1 (tokens 1:1)"), ("T100", "T100 (tokens 100:1)"),
                     ("T93&T107", r"T93\&T107 (tokens 93:1, 107:1)"), ("T0.1", "T0.1 (tokens 0.1:1)"),
                     ("T100 - T1", "T100 $-$ T1")]:
        a, b = [tex], [""]
        for col in E1.columns:
            c1, c2 = split(E1.loc[f"{panel} | {lab}", col])
            a.append(c1)
            b.append(c2)
        wrows.append(" & ".join(a) + r" \\")
        wrows.append(" & ".join(b) + r" \\")
    for lab in ["Observations", "Participants"]:
        wrows.append(f"{lab} & " + " & ".join(str(E1.loc[f"{panel} | {lab}", c]) for c in E1.columns) + r" \\")
    if panel == "A: Risky":
        wrows.append(r"\midrule")
wrows.append(r"\midrule")
wrows.append("Controls \\& lottery FE & Yes & Yes & Yes & Yes \\\\")
wrows.append("Wave fixed effects & No & Yes & -- & -- \\\\")
tw = (r"\begin{table}[!t]" "\n" r"\centering\footnotesize\renewcommand{\arraystretch}{0.92}" "\n"
      r"\caption{Effect of the payment currency within data-collection waves}" "\n"
      r"\label{tab:wave}" "\n"
      r"\begin{tabular}{lcccc}" "\n" r"\toprule" "\n"
      r" & Baseline & 2023 waves & June 2023 & January 2023 \\" "\n"
      r" & & + wave FE & only & only \\" "\n"
      r" & (1) & (2) & (3) & (4) \\" "\n" r"\midrule" "\n"
      + "\n".join(wrows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n\n"
      r"\smallskip" "\n" r"\begin{minipage}{0.97\textwidth}\footnotesize" "\n"
      r"\emph{Notes:} Dependent variable: willingness to pay, divided by the exchange rate (GBP, 0 to 1). "
      r"Column (1) repeats columns (2) and (4) of Table~\ref{tab:replication}. Column (2) drops the July 2025 wave and adds "
      r"wave fixed effects. Columns (3) and (4) use only the June and January 2023 waves. T0.1 cannot be estimated in "
      r"columns (2)--(4) (it is the only treatment in 2025); T93 and T107 were only run in June 2023. All models include "
      r"individual controls and lottery fixed effects and are estimated by pooled OLS; standard errors clustered at the "
      r"participant level in parentheses. "
      r"$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$." "\n"
      r"\end{minipage}" "\n" r"\end{table}" "\n")
(OUT / "table_wave.tex").write_text(tw, encoding="utf-8")

# ------------------------------------------------------------ balance table
E3 = pd.read_csv(TAB / "ext_E3_balance.csv")
brows = []
for _, r in E3.iterrows():
    nd = 1 if r["variable"] in ("Age",) else 2
    brows.append(f"{r['variable']} & {r['mean_2025']:.{nd}f} & {r['mean_2023']:.{nd}f} & "
                 f"{r['p_2025_vs_2023 (Welch t)']:.3f} & {r['p_across_2023_arms (ANOVA)']:.2f} \\\\")
tb = (r"\begin{table}[!t]" "\n" r"\centering\footnotesize\renewcommand{\arraystretch}{0.92}" "\n"
      r"\caption{Balance of observables: 2025 wave (T0.1) vs.\ 2023 waves}" "\n"
      r"\label{tab:balance}" "\n"
      r"\begin{tabular}{lcccc}" "\n" r"\toprule" "\n"
      r" & 2025 (T0.1) & 2023 & $p$ (2025 vs.\ 2023) & $p$ (across 2023 arms) \\" "\n" r"\midrule" "\n"
      + "\n".join(brows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n\n"
      r"\smallskip" "\n" r"\begin{minipage}{0.97\textwidth}\footnotesize" "\n"
      r"\emph{Notes:} 753 participants who completed the study. Female excludes missing values. "
      r"Welch $t$-tests for 2025 vs.\ 2023; one-way ANOVA across TC, T1, T100 and T93\&T107 within 2023." "\n"
      r"\end{minipage}" "\n" r"\end{table}" "\n")
(OUT / "table_balance.tex").write_text(tb, encoding="utf-8")
print("[report tables] written to", OUT)
