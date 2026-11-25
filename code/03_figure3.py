"""
03_figure3.py — Replicate Figure 3: average WTP by treatment and lottery,
with 95% confidence intervals (mean ± 1.96·SE, as in the authors' code).

Output: output/figures/figure3.png, output/tables/figure3_means.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "output" / "figures"
TAB = ROOT / "output" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "data" / "clean" / "long.csv")

GROUPS = ["TC", "T1", "T0.1", "T100", "T93&T107"]
# Categorical palette in fixed order; colour is never the only cue: every
# group also has its own marker shape and a legend entry.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"

stats = (df.groupby(["treat5", "lottery"])["wtp"]
           .agg(mean="mean", se="sem", n="count").reset_index())
stats["lo"] = stats["mean"] - 1.96 * stats["se"]
stats["hi"] = stats["mean"] + 1.96 * stats["se"]
stats.to_csv(TAB / "figure3_means.csv", index=False, float_format="%.4f")

panels = [("a) Risky lotteries", [f"R{i}" for i in range(11)]),
          ("b) Ambiguous lotteries", ["A1", "A2", "A3", "A4", "A5"])]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [11, 5.5]},
                         sharey=True)
offsets = np.linspace(-0.24, 0.24, len(GROUPS))
for ax, (title, lots) in zip(axes, panels):
    x = np.arange(len(lots))
    for g, col, mk, off in zip(GROUPS, COLORS, MARKERS, offsets):
        s = stats[stats["treat5"] == g].set_index("lottery").reindex(lots)
        ax.vlines(x + off, s["lo"], s["hi"], color=col, lw=1.5, alpha=0.9)
        ax.scatter(x + off, s["mean"], marker=mk, s=40, color=col, edgecolor="white",
                   linewidth=1.0, zorder=3, label=g)
    ax.set_xticks(x, lots)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0)
axes[0].set_ylabel("Average WTP (GBP)", color=MUTED)
axes[0].legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
fig.tight_layout()
fig.savefig(FIG / "figure3.png", dpi=200)
print(f"[figure3] saved {FIG / 'figure3.png'}")
