#!/usr/bin/env python3
"""Band-edge character of the DFT-labelled frameworks, and GFN1-xTB frontier orbitals
compared with DFT for the closed-shell Zn4O labels.

For every accepted label the static run of the spin solution that provided the label
(static, static_bs or the <name>__afm* sibling) is read with its site- and
orbital-projected eigenvalues (LORBIT = 11). States within --window eV of the VBM
(occupied) and of the CBM (empty) are summed, k-weighted, over both spins and assigned
to four site groups:

    metal   metal atoms (d fraction reported separately)
    oxo     O bound only to metals (mu4-O of Zn4O)
    Ocarb   carboxylate O
    linker  C, H, N, F of the organic part

The same partition is applied to the Mulliken populations of the GFN1-xTB frontier
orbitals (periodic, Gamma, DFT geometry) for the closed-shell Zn labels, where the
spin-restricted xTB Hamiltonian is meaningful.

Outputs (results/band_edges/):
    band_edge_character.csv       18 labels, DFT fractions (band projection) and the
                                  0.5 eV DOS-window metric of src/mofal/pdos.py
    xtb_vs_dft_band_edges.csv     Zn4O labels: xTB HOMO/LUMO vs DFT VBM/CBM fractions
    results/figures/fig_band_edge_character.{pdf,png}

Usage
-----
    python scripts/17_band_edge_character.py            # compute + figure
    python scripts/17_band_edge_character.py --figure   # figure only
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
from mofal.pdos import GROUPS, band_edge_character, band_edge_projection, site_groups  # noqa: E402

warnings.filterwarnings("ignore")
OUT = ROOT / "results/band_edges"
FAMILY_ORDER = ["Zn4O pcu", "Co4 bcu", "Fe4 bcu", "Cu2 cds", "Co2 cds", "Fe2 cds"]


def label_folder(name: str, iteration: int, spin_solution) -> Path:
    base = ROOT / f"runs/iter{iteration}" / name
    sol = spin_solution if isinstance(spin_solution, str) else "fm-start"
    if sol == "fm-start":
        return base / "static"
    if sol == "afmbs-start":
        return base / "static_bs"
    return ROOT / f"runs/iter{iteration}" / f"{name}__{sol.replace('-start', '')}" / "static"


def family(row) -> str:
    n = {"pcu": "Zn4O pcu", "bcu": f"{row.metal}4 bcu", "cds": f"{row.metal}2 cds"}
    return n.get(row.topology, f"{row.metal} {row.topology}")


def short(row) -> str:
    sub = {"H": "H", "NH2": "NH$_2$", "F4": "F$_4$", "F8": "F$_8$", "Nring1": "N-ring",
           "Nring2": "N$_2$-ring"}.get(row.substituent, row.substituent)
    lk = "long-" if "long" in row["name"] else ""
    known = {"pcu__Zn-N16__short-H": " (MOF-5)", "pcu__Zn-N16__short-NH2": " (IRMOF-3)"}
    return f"{lk}{sub}{known.get(row['name'], '')}"


def labels() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data/dft/dft_results.csv")
    d = d[~d.name.str.contains("__afm") & d.label_ok.astype(bool)].copy()
    d["family"] = d.apply(family, axis=1)
    d["label"] = d.apply(short, axis=1)
    d["family"] = pd.Categorical(d.family, FAMILY_ORDER, ordered=True)
    return d.sort_values(["family", "gap_eV"]).reset_index(drop=True)


def compute(window: float):
    OUT.mkdir(parents=True, exist_ok=True)
    d = labels()
    rows, xrows = [], []
    for _, r in d.iterrows():
        f = label_folder(r["name"], int(r.iteration), r.spin_solution)
        print(r["name"], f.relative_to(ROOT), flush=True)
        bp = band_edge_projection(f / "vasprun.xml", window=window)
        dw = band_edge_character(f / "vasprun.xml")       # 0.5 eV DOS window, element/orbital
        row = dict(name=r["name"], family=str(r.family), label=r.label, metal=r.metal,
                   topology=r.topology, substituent=r.substituent, iteration=int(r.iteration),
                   spin_solution=r.spin_solution if isinstance(r.spin_solution, str) else "fm-start",
                   source=str(f.relative_to(ROOT)), **bp)
        row.update({f"dos_{k}": v for k, v in dw.items() if k.startswith(("VB_", "CB_"))})
        rows.append(row)
        if r.metal in ("Zn", "Mg"):
            from ase.io import read
            from pymatgen.core import Structure
            from mofal.xtb import frontier_populations
            atoms = read(f / "POSCAR")
            groups = np.array(site_groups(Structure.from_file(f / "POSCAR")))
            ph, pl, eh, el, no, nv = frontier_populations(atoms, window=window)
            xr = dict(name=r["name"], label=r.label, gap_dft=r.gap_eV, gap_xtb=el - eh,
                      n_xtb_occ=no, n_xtb_virt=nv)
            for g in GROUPS:
                xr[f"VB_dft_{g}"] = bp[f"VB_{g}"]
                xr[f"VB_xtb_{g}"] = ph[groups == g].sum()
                xr[f"CB_dft_{g}"] = bp[f"CB_{g}"]
                xr[f"CB_xtb_{g}"] = pl[groups == g].sum()
            xrows.append(xr)
            print(f"   xTB gap {el - eh:.2f}  HOMO linker {xr['VB_xtb_linker']:.2f} Ocarb {xr['VB_xtb_Ocarb']:.2f} "
                  f"| LUMO linker {xr['CB_xtb_linker']:.2f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "band_edge_character.csv", index=False, float_format="%.4f",
                              lineterminator="\n")
    if xrows:
        pd.DataFrame(xrows).to_csv(OUT / "xtb_vs_dft_band_edges.csv", index=False,
                                   float_format="%.3f", lineterminator="\n")


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = pd.read_csv(OUT / "band_edge_character.csv")
    t["family"] = pd.Categorical(t.family, FAMILY_ORDER, ordered=True)
    t = t.sort_values(["family", "gap_eV"]).reset_index(drop=True)
    MM = 1 / 25.4
    INK, INK2 = "#0b0b0b", "#52514e"
    COL = {"metal": "#2a78d6", "oxo": "#eb6834", "Ocarb": "#1baf7a", "linker": "#eda100"}
    LEG = {"metal": "metal", "oxo": "μ$_4$-O", "Ocarb": "carboxylate O", "linker": "linker backbone"}
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 300})
    fig, axes = plt.subplots(1, 2, figsize=(140 * MM, 92 * MM), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    y = np.arange(len(t))[::-1]
    for ax, edge, title in zip(axes, ("VB", "CB"), ("Valence-band edge (VBM − 0.25 eV)",
                                                      "Conduction-band edge (CBM + 0.25 eV)")):
        left = np.zeros(len(t))
        for g in GROUPS:
            v = t[f"{edge}_{g}"].to_numpy()
            ax.barh(y, v, left=left, height=0.72, color=COL[g], edgecolor="white", linewidth=0.6,
                    label=LEG[g])
            left += v
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "", "0.5", "", "1"])
        ax.set_title(title, loc="left", fontsize=7.5, color=INK2)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{r.label}  ({r.gap_eV:.2f} eV)" for r in t.itertuples()], fontsize=6.5)
    axes[0].set_xlabel("fraction of projected weight")
    axes[1].set_xlabel("fraction of projected weight")
    # family separators and names
    fam = t.family.astype(str).to_numpy()
    for ax in axes:
        for i in range(1, len(t)):
            if fam[i] != fam[i - 1]:
                ax.axhline(y[i] + 0.5, color=INK2, lw=0.5, ls=":")
    starts = [0] + [i for i in range(1, len(t)) if fam[i] != fam[i - 1]] + [len(t)]
    for a, b in zip(starts[:-1], starts[1:]):
        axes[1].text(1.02, y[a:b].mean(), fam[a], transform=axes[1].get_yaxis_transform(),
                     va="center", ha="left", fontsize=6.5, color=INK2)
    axes[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(1.04, -0.12), frameon=False,
                   handlelength=1.0, columnspacing=1.2)
    fdir = ROOT / "results/figures"
    fig.savefig(fdir / "fig_band_edge_character.pdf")
    fig.savefig(fdir / "fig_band_edge_character.png")
    print("saved fig_band_edge_character")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=0.25, help="eV from each band edge")
    ap.add_argument("--figure", action="store_true", help="figure only")
    args = ap.parse_args()
    if not args.figure:
        compute(args.window)
    figure()


if __name__ == "__main__":
    main()
