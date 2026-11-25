"""
helpers.py — small shared functions used by 06_extension.py and 07_report_tables.py.

Cluster-robust OLS with lottery dummies, written with numpy so that the
extension does not depend on statsmodels. It uses the same small-sample
correction as 04_table3.py / Stata's reghdfe, G/(G-1) * (N-1)/(N-K) with K
counting the lottery dummies, and p-values from t(G-1). Its coefficients and
standard errors reproduce the ones from 04_table3.py (see output/tables).
"""
import numpy as np
import pandas as pd
from scipy import stats


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def cluster_ols(d, xs, y="wtp", fe="lottery_id", cl="Q_ID"):
    """OLS of y on xs + fixed-effect dummies; SEs clustered by `cl`."""
    X = pd.concat([d[xs].astype(float),
                   pd.get_dummies(d[fe], prefix="L", drop_first=True, dtype=float)], axis=1)
    X.insert(0, "const", 1.0)
    Y, Xm = d[y].values.astype(float), X.values
    XtXi = np.linalg.inv(Xm.T @ Xm)
    b = XtXi @ Xm.T @ Y
    e = Y - Xm @ b
    g = pd.factorize(d[cl])[0]
    G, (N, K) = g.max() + 1, Xm.shape
    S = np.zeros((K, K))
    for i in range(G):
        s = Xm[g == i].T @ e[g == i]
        S += np.outer(s, s)
    V = XtXi @ S @ XtXi * (G / (G - 1)) * ((N - 1) / (N - K))
    b = pd.Series(b, index=X.columns)
    V = pd.DataFrame(V, index=X.columns, columns=X.columns)
    se = pd.Series(np.sqrt(np.diag(V)), index=X.columns)
    p = pd.Series(2 * stats.t.sf(abs(b / se), G - 1), index=X.columns)
    r2 = 1 - (e ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
    # reghdfe-style constant: mean(y) - mean(X)'b
    cons = Y.mean() - (d[xs].astype(float).mean() * b[xs]).sum()
    # reghdfe's constant = intercept + average fixed effect: SE from the same covariance matrix
    w = pd.Series(0.0, index=X.columns)
    w["const"] = 1.0
    for c in X.columns:
        if c.startswith("L_"):
            w[c] = X[c].mean()
    cons_se = float(np.sqrt(w.values @ V.values @ w.values))
    cons_p = float(2 * stats.t.sf(abs(cons / cons_se), G - 1))
    return dict(b=b, se=se, p=p, V=V, G=G, N=N, r2=r2, cons=cons, cons_se=cons_se, cons_p=cons_p)


def lincom(r, w):
    """Linear combination of coefficients: returns (estimate, se, p)."""
    b = sum(r["b"][k] * v for k, v in w.items())
    se = np.sqrt(sum(w[i] * w[j] * r["V"].loc[i, j] for i in w for j in w))
    return b, se, 2 * stats.t.sf(abs(b / se), r["G"] - 1)
