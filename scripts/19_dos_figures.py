#!/usr/bin/env python3
"""Spin-resolved total and site-projected DOS of the DFT-labelled frameworks.

For every accepted label the static run of the spin solution that provided the label is
read (LORBIT = 11, NEDOS = 3001). The projected DOS is summed over the four site groups
of src/mofal/pdos.py (metal, mu4-O, carboxylate O, linker backbone; all orbitals) and
plotted with spin up positive and spin down negative, relative to the VBM. A Gaussian
broadening of --sigma eV (default 0.05) is applied to all curves for legibility, because
the statics with ALGO = All use SIGMA = 0.01 eV.

Outputs: results/figures/fig_dos_representative.{pdf,png}   one framework per family
         results/figures/fig_S_dos_all.{pdf,png}             all 18 labels
         results/band_edges/dos_curves/<name>.csv            the plotted curves

Usage:  python scripts/19_dos_figures.py [--sigma 0.05] [--window 3.0]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mofal.pdos import GROUPS, site_groups  # noqa: E402

warnings.filterwarnings("ignore")
FAMILY_ORDER = ["Zn4O pcu", "Co4 bcu", "Fe4 bcu", "Cu2 cds", "Co2 cds", "Fe2 cds"]
REPRESENTATIVE = ["pcu__Zn-N16__short-H", "bcu__Co-N35__short-H", "bcu__Fe-N350__short-F4",
                  "cds__Cu-N409__short-Nring1", "cds__Co-N317__short-F4", "cds__Fe-N355__short-F4"]
COL = {"metal": "#2a78d6", "oxo": "#eb6834", "Ocarb": "#1baf7a", "linker": "#eda100"}
LEG = {"metal": "metal", "oxo": "μ$_4$-O", "Ocarb": "carboxylate O", "linker": "linker backbone"}
INK, INK2 = "#0b0b0b", "#52514e"


def label_folder(name, iteration, spin_solution):
    base = ROOT / f"runs/iter{iteration}" / name
    sol = spin_solution if isinstance(spin_solution, str) else "fm-start"
    if sol == "fm-start":
        return base / "static"
    if sol == "afmbs-start":
        return base / "static_bs"
    return ROOT / f"runs/iter{iteration}" / f"{name}__{sol.replace('-start', '')}" / "static"


def family(r):
    return {"pcu": "Zn4O pcu", "bcu": f"{r.metal}4 bcu", "cds": f"{r.metal}2 cds"}[r.topology]


def short(r):
    sub = {"H": "BDC", "NH2": "NH$_2$-BDC", "F4": "F$_4$-BDC", "F8": "F$_8$-BPDC", "Nring1": "N-ring-BDC",
           "Nring2": "N$_2$-ring-BPDC"}[r.substituent]
    if "long" in r["name"] and r.substituent == "Nring1":
        sub = "N-ring-BPDC"
    known = {"pcu__Zn-N16__short-H": " (MOF-5)", "pcu__Zn-N16__short-NH2": " (IRMOF-3)"}.get(r["name"], "")
    return f"{family(r)}, {sub}{known}"


def curves(folder: Path, sigma: float, window: float):
    """Return energies (rel. to VBM) and {group: (up, down)} broadened densities, plus total."""
    from pymatgen.electronic_structure.core import Spin
    from pymatgen.io.vasp import Vasprun
    from scipy.ndimage import gaussian_filter1d

    vr = Vasprun(str(folder / "vasprun.xml"), parse_potcar_file=False)
    gap, cbm, vbm, _ = vr.eigenvalue_band_properties
    cdos = vr.complete_dos
    e = cdos.energies - vbm
    de = e[1] - e[0]
    s = sigma / de
    grp = np.array(site_groups(cdos.structure))
    out = {"total": tuple(gaussian_filter1d(cdos.densities[sp], s) for sp in (Spin.up, Spin.down))}
    for g in GROUPS:
        acc = {Spin.up: np.zeros_like(e), Spin.down: np.zeros_like(e)}
        for site, gname in zip(cdos.structure, grp):
            if gname != g:
                continue
            for orb, dd in cdos.pdos[site].items():
                for sp in (Spin.up, Spin.down):
                    acc[sp] += dd[sp]
        out[g] = tuple(gaussian_filter1d(acc[sp], s) for sp in (Spin.up, Spin.down))
    m = (e >= -window) & (e <= max(window, gap + 2.0))
    return e[m], {k: (v[0][m], v[1][m]) for k, v in out.items()}, float(gap), len(cdos.structure)


def panel(ax, e, c, gap, title, n_atoms, show_legend=False):
    scale = 1.0 / n_atoms                    # states / (eV atom)
    ax.fill_between(e, c["total"][0] * scale, 0, color="#e9e8e4", lw=0, zorder=1)
    ax.fill_between(e, -c["total"][1] * scale, 0, color="#e9e8e4", lw=0, zorder=1)
    for g in GROUPS:
        if c[g][0].max() * scale < 1e-3 and c[g][1].max() * scale < 1e-3:
            continue
        ax.plot(e, c[g][0] * scale, color=COL[g], lw=0.9, label=LEG[g], zorder=3)
        ax.plot(e, -c[g][1] * scale, color=COL[g], lw=0.9, zorder=3)
    ax.axhline(0, color=INK2, lw=0.4)
    ax.axvspan(0, gap, color="#ffffff", alpha=0.0)
    ax.axvline(0, color=INK2, lw=0.5, ls=":")
    ax.axvline(gap, color=INK2, lw=0.5, ls=":")
    ax.set_title(f"{title}   gap {gap:.2f} eV", loc="left", fontsize=6.5, color=INK)
    ax.set_xlim(e.min(), e.max())
    ymax = max(c["total"][0].max(), c["total"][1].max()) * scale * 1.05
    ax.set_ylim(-ymax, ymax)
    ax.set_yticks([])
    ax.grid(False)
    return ax.get_legend_handles_labels()


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--window", type=float, default=3.0)
    args = ap.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.labelsize": 7,
                         "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.edgecolor": INK2,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
                         "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 300})
    d = pd.read_csv(ROOT / "data/dft/dft_results.csv")
    d = d[~d.name.str.contains("__afm") & d.label_ok.astype(bool)].copy()
    d["family"] = pd.Categorical(d.apply(family, axis=1), FAMILY_ORDER, ordered=True)
    d = d.sort_values(["family", "gap_eV"]).reset_index(drop=True)
    out = ROOT / "results/band_edges/dos_curves"
    out.mkdir(parents=True, exist_ok=True)
    data = {}
    for _, r in d.iterrows():
        f = label_folder(r["name"], int(r.iteration), r.spin_solution)
        e, c, gap, n = curves(f, args.sigma, args.window)
        data[r["name"]] = (e, c, gap, n, short(r))
        df = pd.DataFrame({"E_minus_VBM_eV": e})
        for k, (up, dn) in c.items():
            df[f"{k}_up"] = up / n
            df[f"{k}_down"] = dn / n
        df.to_csv(out / f"{r['name']}.csv", index=False, float_format="%.5f", lineterminator="\n")
        print(r["name"], f"gap {gap:.3f}", flush=True)
    MM = 1 / 25.4
    # representative, 3 x 2
    fig, axes = plt.subplots(3, 2, figsize=(140 * MM, 120 * MM))
    handles = {}
    for ax, name in zip(axes.T.ravel(), REPRESENTATIVE):
        e, c, gap, n, title = data[name]
        h, l = panel(ax, e, c, gap, title, n)
        handles.update(dict(zip(l, h)))
    for ax in axes[-1]:
        ax.set_xlabel("E − E$_{VBM}$ (eV)")
    order = [LEG[g] for g in GROUPS if LEG[g] in handles]
    fig.legend([handles[k] for k in order], order, frameon=False, fontsize=6.5, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, -0.03))
    fig.text(0.0, 0.5, "DOS (states eV$^{-1}$ atom$^{-1}$), spin up ↑ / spin down ↓", rotation=90,
             va="center", ha="center", fontsize=7, color=INK2)
    fig.subplots_adjust(hspace=0.45, wspace=0.12, left=0.05)
    fig.savefig(ROOT / "results/figures/fig_dos_representative.pdf")
    fig.savefig(ROOT / "results/figures/fig_dos_representative.png")
    plt.close(fig)
    # all 18, 6 x 3
    fig, axes = plt.subplots(6, 3, figsize=(190 * MM, 230 * MM))
    handles = {}
    for ax, name in zip(axes.ravel(), d.name):
        e, c, gap, n, title = data[name]
        h, l = panel(ax, e, c, gap, title, n)
        handles.update(dict(zip(l, h)))
    for ax in axes[-1]:
        ax.set_xlabel("E − E$_{VBM}$ (eV)")
    order = [LEG[g] for g in GROUPS if LEG[g] in handles]
    fig.legend([handles[k] for k in order], order, frameon=False, fontsize=7, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, -0.015))
    fig.text(0.0, 0.5, "DOS (states eV$^{-1}$ atom$^{-1}$), spin up ↑ / spin down ↓", rotation=90,
             va="center", ha="center", fontsize=7, color=INK2)
    fig.subplots_adjust(hspace=0.5, wspace=0.1, left=0.04, bottom=0.06)
    fig.savefig(ROOT / "results/figures/fig_S_dos_all.pdf")
    fig.savefig(ROOT / "results/figures/fig_S_dos_all.png")
    print("saved fig_dos_representative, fig_S_dos_all")


if __name__ == "__main__":
    main()
