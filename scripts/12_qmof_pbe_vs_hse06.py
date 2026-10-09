#!/usr/bin/env python3
"""How reliable is a low PBE gap? PBE vs HSE06 band gaps in QMOF.

Uses only QMOF data (10,810 structures with both PBE and HSE06 gaps). Reports, per
primary metal and for open-shell (Ti-Cu) vs closed-shell frameworks, how many
structures with PBE gap < 1 eV keep an HSE06 gap < 1 eV, and the median
HSE06 - PBE difference. Supports the Discussion of the PBE screening criterion.

Usage:  python scripts/12_qmof_pbe_vs_hse06.py
Outputs: results/qmof_pbe_vs_hse06/{by_metal.csv, by_shell.csv}, figure in results/figures
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mofal.features import featurize_qmof  # noqa: E402

THR = 1.0


def stats(g):
    low = g[g.gap_PBE < THR]
    return pd.Series({
        "n": len(g), "median_PBE": g.gap_PBE.median(), "median_HSE06": g.hse.median(),
        "median_HSE06_minus_PBE": (g.hse - g.gap_PBE).median(),
        "n_PBE_lt_1": len(low),
        "pct_PBE_lt_1_with_HSE_lt_1": 100 * (low.hse < THR).mean() if len(low) else np.nan,
        "pct_PBE_lt_1_with_HSE_lt_2": 100 * (low.hse < 2).mean() if len(low) else np.nan,
    })


def main():
    out = ROOT / "results/qmof_pbe_vs_hse06"
    out.mkdir(parents=True, exist_ok=True)
    d = featurize_qmof(pd.read_csv(ROOT / "data/external/qmof.csv", low_memory=False))
    d["hse"] = pd.to_numeric(d["outputs.hse06.bandgap"], errors="coerce")
    h = d.dropna(subset=["hse"]).copy()
    h["shell"] = np.where(h.has_open_shell == 1, "open-shell 3d (Ti-Cu)", "other")
    by_shell = h.groupby("shell").apply(stats, include_groups=False).round(2)
    metals = ["Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Mg", "Zr", "Cd"]
    by_metal = h[h.primary_metal.isin(metals)].groupby("primary_metal").apply(stats, include_groups=False) \
        .reindex(metals).round(2)
    by_shell.to_csv(out / "by_shell.csv")
    by_metal.to_csv(out / "by_metal.csv")
    print(by_shell.to_string(), "\n\n", by_metal.to_string())

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig, axs = plt.subplots(1, 2, figsize=(190 / 25.4, 75 / 25.4), sharex=True, sharey=True)
    for ax, (lab, g) in zip(axs, h.groupby("shell")):
        hb = ax.hexbin(g.gap_PBE, g.hse, gridsize=40, norm=LogNorm(vmin=1, vmax=300), cmap="Blues",
                       mincnt=1, linewidths=0, extent=(0, 7, 0, 7))
        ax.plot([0, 7], [0, 7], color="#52514e", lw=0.8, ls="--")
        ax.axvline(THR, color="#eb6834", lw=0.8)
        ax.axhline(THR, color="#eb6834", lw=0.8)
        ax.set(xlim=(0, 7), ylim=(0, 7), xlabel="PBE gap (eV)", title=f"{lab} (n = {len(g):,})")
        ax.set_aspect("equal")
    axs[0].set_ylabel("HSE06 gap (eV)")
    fig.colorbar(hb, ax=axs, shrink=0.8, label="Structures per bin")
    figdir = ROOT / "results/figures"
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / "fig_qmof_pbe_vs_hse06.pdf", bbox_inches="tight")
    fig.savefig(figdir / "fig_qmof_pbe_vs_hse06.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
