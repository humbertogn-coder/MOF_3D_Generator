#!/usr/bin/env python3
"""Periodic vs finite-cluster GFN1-xTB gaps against QMOF PBE gaps (SI analysis).

Inputs
  data/xtb/qmof_validation_subset.csv        2,000 QMOF structures (PBE gaps, tiers)
  data/xtb/original_cluster_xtb_results.csv  original screen: unit cell as a finite cluster
                                             (also carries QMOF HSE06 gaps where available)
  results/xtb/xtb_gfn1_gaps.csv              scripts/13_xtb_validation.py --mode periodic

Groups (same convention as scripts/12_qmof_pbe_vs_hse06.py)
  closed-shell : no Ti-Cu and an even electron count (uhf = 0)
  open-shell 3d: contains Ti, V, Cr, Mn, Fe, Co, Ni or Cu
  other        : remaining structures (odd electron count without Ti-Cu)

Outputs
  results/xtb/xtb_merged.csv, results/xtb/xtb_summary.csv,
  results/figures/fig_xtb_periodic_vs_cluster.{pdf,png}

Usage:  python scripts/14_xtb_analysis.py
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OPEN_3D = {"Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"}
LOW = 1.0          # eV, "low PBE gap" threshold used in the paper
MM = 1 / 25.4
BLUE, ORANGE, INK, INK2 = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e6e5e1", "grid.linewidth": 0.5,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 300,
})


HS_UNPAIRED = {"Mn": 5, "Fe": 4, "Co": 3, "Ni": 2, "Cu": 1}   # as in 13_xtb_validation.py


def composition(formula: str) -> dict:
    out = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        out[el] = out.get(el, 0) + (int(n) if n else 1)
    return out


def group(formula: str) -> str:
    from ase.data import atomic_numbers
    comp = composition(formula)
    if set(comp) & OPEN_3D:
        return "open-shell 3d"
    n_el = sum(atomic_numbers[e] * n for e, n in comp.items())
    return "closed-shell" if n_el % 2 == 0 else "other"


def stats(x: pd.Series, y: pd.Series) -> dict:
    """x = xTB gap, y = reference DFT gap (both eV)."""
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return {"n": len(x)}
    low = (y < LOW).astype(int)
    auc = roc_auc_score(low, -x) if 0 < low.sum() < len(low) else np.nan
    return {
        "n": len(x), "MAE_eV": (x - y).abs().mean(), "RMSE_eV": np.sqrt(((x - y) ** 2).mean()),
        "bias_eV": (x - y).mean(), "pearson_r": pearsonr(x, y)[0], "spearman_rho": spearmanr(x, y)[0],
        "pct_xtb_gap_lt_0.1": 100 * (x < 0.1).mean(), "ROC_AUC_PBE_lt_1": auc,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--periodic", nargs="+", default=[str(ROOT / "results/xtb/xtb_gfn1_gaps.csv")],
                    help="one or more CSVs written by 13_xtb_validation.py")
    args = ap.parse_args()
    sub = pd.read_csv(ROOT / "data/xtb/qmof_validation_subset.csv")
    per = pd.concat([pd.read_csv(f) for f in args.periodic], ignore_index=True)
    per = per[per["mode"] == "periodic"].drop_duplicates("qmof_id", keep="last")
    clu = pd.read_csv(ROOT / "data/xtb/original_cluster_xtb_results.csv")
    clu = clu.rename(columns={"Name": "qmof_id", "name": "qmof_id", "gap_eV": "gap_cluster",
                              "gap_DFT_HSE06_eV": "gap_hse06"})[["qmof_id", "gap_cluster", "gap_hse06"]]
    df = (sub.merge(per[["qmof_id", "uhf", "gap_eV", "status", "seconds"]], on="qmof_id", how="left")
             .merge(clu, on="qmof_id", how="left")
             .rename(columns={"bandgap_pbe_eV": "gap_pbe", "gap_eV": "gap_periodic"}))
    df.loc[df.status != "ok", "gap_periodic"] = np.nan
    df["group"] = df.formula.map(group)
    out = ROOT / "results/xtb"
    df.to_csv(out / "xtb_merged.csv", index=False, lineterminator="\n")

    rows = []
    for grp, g in [("all", df)] + list(df.groupby("group")):
        for treat, col in [("cluster", "gap_cluster"), ("periodic", "gap_periodic")]:
            r = {"group": grp, "treatment": treat, "reference": "PBE"}
            r.update(stats(g[col], g.gap_pbe))
            rows.append(r)
            r = {"group": grp, "treatment": treat, "reference": "HSE06"}
            r.update(stats(g[col], g.gap_hse06))
            rows.append(r)
    summ = pd.DataFrame(rows)
    n_fail = int((df.status.notna() & (df.status != "ok")).sum())
    summ.to_csv(out / "xtb_summary.csv", index=False, float_format="%.3f", lineterminator="\n")
    print(summ[summ.reference == "PBE"].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"periodic runs: {df.status.notna().sum()} attempted, {n_fail} failed or skipped")
    figure(df)


def figure(df: pd.DataFrame):
    """2 x 2 parity plots: rows = treatment, columns = group."""
    fig, axes = plt.subplots(2, 2, figsize=(140 * MM, 132 * MM), sharex=True, sharey=True)
    lim = (0, 8.0)
    cols = [("closed-shell", "Closed-shell (no Ti–Cu)", BLUE),
            ("open-shell 3d", "Contains Ti–Cu (open-shell 3d)", ORANGE)]
    rows = [("gap_cluster", "Finite cluster (original screen)"),
            ("gap_periodic", "Periodic (this work)")]
    for i, (col, rlabel) in enumerate(rows):
        for j, (grp, clabel, color) in enumerate(cols):
            ax = axes[i, j]
            g = df[df.group == grp]
            s = stats(g[col], g.gap_pbe)
            ax.plot(lim, lim, color=INK2, lw=0.8, ls="--", zorder=1)
            ax.scatter(g.gap_pbe, g[col], s=6, color=color, alpha=0.45, linewidths=0, zorder=2)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_aspect("equal")
            if s.get("n", 0) >= 3:
                pos = (0.04, 0.96, "top", "left") if i == 0 else (0.96, 0.04, "bottom", "right")
                ax.text(pos[0], pos[1], f"n = {s['n']}\nMAE = {s['MAE_eV']:.2f} eV\n"
                        f"r = {s['pearson_r']:.2f}\n{s['pct_xtb_gap_lt_0.1']:.0f}% below 0.1 eV",
                        transform=ax.transAxes, va=pos[2], ha=pos[3], fontsize=7, color=INK,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85))
            if i == 0:
                ax.set_title(clabel, color=INK)
            if j == 0:
                ax.set_ylabel(f"{rlabel}\nGFN1-xTB gap (eV)")
            if i == 1:
                ax.set_xlabel("QMOF PBE gap (eV)")
    for k, ax in enumerate(axes.flat):
        ax.text(-0.02, 1.02, "abcd"[k], transform=ax.transAxes, fontweight="bold", fontsize=9,
                ha="right", va="bottom", color=INK)
    fig.tight_layout()
    fdir = ROOT / "results/figures"
    fig.savefig(fdir / "fig_xtb_periodic_vs_cluster.pdf")
    fig.savefig(fdir / "fig_xtb_periodic_vs_cluster.png")
    plt.close(fig)
    print("saved fig_xtb_periodic_vs_cluster")


if __name__ == "__main__":
    main()
