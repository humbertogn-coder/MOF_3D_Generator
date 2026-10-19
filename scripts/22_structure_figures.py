#!/usr/bin/env python3
"""Ball-and-stick drawings of the frameworks, from the CIF/POSCAR files in the repository.

Top row: one DFT-relaxed framework per family that gave labels (the same six as Fig. 6),
hydrogens omitted, a block 1.6 cells wide and one cell deep viewed along the lattice vector
that shows the rings face-on. Bottom row: two frameworks that collapsed in DFT, as generated
and after the relaxation (Mn3 pcu, Mn2 cds), to show what the acceptance rules reject.
SI version: all 18 labelled frameworks.

Outputs: results/figures/fig_structures.{pdf,png}, fig_S_structures_all.{pdf,png}
Usage:   python scripts/22_structure_figures.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from ase.data import covalent_radii  # noqa: E402
from ase.io import read  # noqa: E402
from ase.neighborlist import neighbor_list  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
warnings.filterwarnings("ignore")
MM = 1 / 25.4
INK, INK2 = "#0b0b0b", "#52514e"
COLOR = {"C": "#6b6a66", "O": "#e34948", "N": "#4f8fe0", "F": "#1baf7a", "H": "#dddddd",
         "Zn": "#8a8bb5", "Co": "#2a78d6", "Fe": "#eb6834", "Cu": "#c47a2f", "Mn": "#8e6bd6", "Ni": "#3fae7a", "Mg": "#9ac35a"}
RADIUS = {"C": 0.33, "O": 0.36, "N": 0.35, "F": 0.34, "Zn": 0.62, "Co": 0.62, "Fe": 0.62, "Cu": 0.62, "Mn": 0.62, "Ni": 0.62, "Mg": 0.6}
METALS = {"Zn", "Co", "Fe", "Cu", "Mn", "Ni", "Mg"}
FAMILY_ORDER = ["Zn4O pcu", "Co4 bcu", "Fe4 bcu", "Cu2 cds", "Co2 cds", "Fe2 cds"]
REPRESENTATIVE = ["pcu__Zn-N16__short-H", "bcu__Co-N35__short-H", "bcu__Fe-N350__short-F4",
                  "cds__Cu-N409__short-Nring1", "cds__Co-N317__short-F4", "cds__Fe-N355__short-F4"]


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


def pretty(r):
    sub = {"H": "BDC", "NH2": "NH$_2$-BDC", "F4": "F$_4$-BDC", "F8": "F$_8$-BPDC", "Nring1": "N-ring-BDC",
           "Nring2": "N$_2$-ring-BPDC"}[r.substituent]
    if "long" in r["name"] and r.substituent == "Nring1":
        sub = "N-ring-BPDC"
    fam = family(r).replace("4 ", "$_4$ ").replace("2 ", "$_2$ ").replace("Zn4O", "Zn$_4$O")
    known = {"pcu__Zn-N16__short-H": " (MOF-5)", "pcu__Zn-N16__short-NH2": " (IRMOF-3)"}.get(r["name"], "")
    return f"{fam}\n{sub}{known}"


def frame(atoms, k):
    """Orthonormal frame with e3 along lattice vector k."""
    cell = atoms.cell.array
    e3 = cell[k] / np.linalg.norm(cell[k])
    v = cell[(k + 1) % 3]
    e1 = v - e3 * (v @ e3)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e3, e1)
    return np.vstack([e1, e2, e3])


def best_axis(atoms, topology=None):
    """Lattice vector to look along. cds nets: the long axis, which shows the square grid
    of paddlewheels and linkers face-on. Otherwise the vector with the fewest projected
    atom overlaps (rings face-on rather than edge-on); ties go to the shortest vector."""
    if topology == "cds":
        return int(np.argmax(np.linalg.norm(atoms.cell.array, axis=1)))
    scores = []
    for k in range(3):
        xy = (atoms.positions @ frame(atoms, k).T)[:, :2]
        d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
        n_over = (np.triu(d < 0.7, 1)).sum() / len(atoms)
        scores.append((n_over, np.linalg.norm(atoms.cell.array[k]), k))
    return sorted(scores)[0][2]


def draw(ax, atoms, axis=None, width=1.6, title="", note="", ring_metals=False, topology=None):
    """Draw a centred block `width` cells wide, one cell deep, viewed along lattice vector
    `axis`, with bonds between drawn atoms (H omitted)."""
    atoms = atoms[[a.symbol != "H" for a in atoms]]
    k = best_axis(atoms, topology) if axis is None else axis
    rep = [3, 3, 3]
    rep[k] = 1
    sup = atoms.repeat(rep)
    sup.pbc = False
    frac = sup.get_scaled_positions(wrap=False)
    lo, hi = (3 - width) / 6, (3 + width) / 6
    keep = np.ones(len(sup), bool)
    for i in range(3):
        if i != k:
            keep &= (frac[:, i] >= lo) & (frac[:, i] < hi)
    sup = sup[keep]
    R = frame(atoms, k)
    xyz = sup.positions @ R.T
    sym = sup.get_chemical_symbols()
    cut = np.array([covalent_radii[z] for z in sup.numbers])
    i, j = neighbor_list("ij", sup, cutoff=2.6, self_interaction=False)
    ok = (np.linalg.norm(sup.positions[i] - sup.positions[j], axis=1) <= 1.25 * (cut[i] + cut[j])) & (i < j)
    i, j = i[ok], j[ok]
    mm = np.array([sym[a] in METALS and sym[b] in METALS for a, b in zip(i, j)], bool)
    i, j = i[~mm], j[~mm]
    depth = xyz[:, 2]
    zmin, zmax = depth.min(), depth.max()
    shade = lambda z: 0.6 + 0.4 * (z - zmin) / max(zmax - zmin, 1e-6)
    segs, cols = [], []
    for a, b in zip(i, j):
        pa, pb = xyz[a, :2], xyz[b, :2]
        mid = (pa + pb) / 2
        zc = (depth[a] + depth[b]) / 2
        for p, q, s in ((pa, mid, sym[a]), (mid, pb, sym[b])):
            segs.append([p, q])
            cols.append(np.array(matplotlib.colors.to_rgb(COLOR.get(s, "#999999"))) * shade(zc))
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=1.0, zorder=1, capstyle="round"))
    for a in np.argsort(depth):
        s = sym[a]
        c = np.array(matplotlib.colors.to_rgb(COLOR.get(s, "#999999"))) * shade(depth[a])
        r = RADIUS.get(s, 0.35) * (0.85 + 0.15 * shade(depth[a]))
        ax.add_patch(Circle(xyz[a, :2], r, facecolor=c, edgecolor="white", linewidth=0.25, zorder=2 + shade(depth[a])))
        if ring_metals and s in METALS:
            ax.add_patch(Circle(xyz[a, :2], r + 0.22, facecolor="none", edgecolor=INK, linewidth=0.45, zorder=4))
    pad = 0.8
    cx, cy = (xyz[:, 0].min() + xyz[:, 0].max()) / 2, (xyz[:, 1].min() + xyz[:, 1].max()) / 2
    half = max(np.ptp(xyz[:, 0]), np.ptp(xyz[:, 1])) / 2 + pad
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=5.8, color=INK, loc="left", pad=2, linespacing=1.1)
    if note:
        ax.text(0.0, -0.01, note, transform=ax.transAxes, fontsize=5.6, color=INK2, va="top")
    return k


def labels():
    d = pd.read_csv(ROOT / "data/dft/dft_results.csv")
    d = d[~d.name.str.contains("__afm") & d.label_ok.astype(bool)].copy()
    d["family"] = pd.Categorical(d.apply(family, axis=1), FAMILY_ORDER, ordered=True)
    return d.sort_values(["family", "gap_eV"]).reset_index(drop=True)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 300})
    d = labels().set_index("name")
    # ---- main figure: 6 families + 2 collapses (before / after)
    fig = plt.figure(figsize=(190 * MM, 82 * MM))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.0], hspace=0.55, wspace=0.1)
    for col, name in enumerate(REPRESENTATIVE):
        r = d.loc[name]
        r = r.copy(); r["name"] = name
        atoms = read(label_folder(name, int(r.iteration), r.spin_solution) / "POSCAR")
        ax = fig.add_subplot(gs[0, col])
        draw(ax, atoms, title=f"{chr(97 + col)}  {pretty(r)}", note=f"PBE-D3 gap {r.gap_eV:.2f} eV", topology=r.topology)
    cases = [
        ("Mn$_3$ pcu, BDC\nas generated", ROOT / "data/catalogue/cifs/pcu__Mn-N548__short-H.cif", ""),
        ("Mn$_3$ pcu, BDC\nafter DFT", ROOT / "runs/iter0/pcu__Mn-N548__short-H/relax/CONTCAR", "ΔV −51 %, excluded"),
        ("Mn$_2$ cds, F$_4$-BDC\nas generated", ROOT / "data/catalogue/cifs/cds__Mn-N288__short-F4.cif", "Mn to Mn 4.0 Å"),
        ("Mn$_2$ cds, F$_4$-BDC\nafter 126 steps", ROOT / "runs/iter1/cds__Mn-N288__short-F4/relax/CONTCAR", "ΔV −21 %, excluded"),
    ]
    slots = [gs[1, 0], gs[1, 1], gs[1, 2], gs[1, 3]]
    k_prev = None
    for n, (title, path, note) in enumerate(cases):
        ax = fig.add_subplot(slots[n])
        k_prev = draw(ax, read(path), axis=None if n % 2 == 0 else k_prev, title=f"{chr(103 + n)}  {title}", note=note,
                      ring_metals=True, topology="cds" if "cds" in str(path) else "pcu")
    # legend panel
    ax = fig.add_subplot(gs[1, 4:6])
    ax.axis("off")
    items = [("Zn", "Zn"), ("Co", "Co"), ("Fe", "Fe"), ("Cu", "Cu"), ("Mn", "Mn"), ("O", "O"), ("N", "N"), ("F", "F"), ("C", "C")]
    for n, (s, lab) in enumerate(items):
        col, row = divmod(n, 3)
        x, y = 0.10 + 0.32 * col, 0.90 - row * 0.16
        ax.add_patch(Circle((x, y), 0.022, facecolor=COLOR[s], edgecolor="white", transform=ax.transAxes, clip_on=False))
        ax.text(x + 0.06, y, lab, transform=ax.transAxes, va="center", fontsize=6.5, color=INK)
    ax.text(0.04, 0.42, "Hydrogens omitted. Each panel shows a block 1.6 cells wide\nand one cell deep (cds nets viewed along the long axis).\nBottom row: metal atoms ringed in black.",
            transform=ax.transAxes, fontsize=5.6, color=INK2, va="top")
    fig.savefig(ROOT / "results/figures/fig_structures.pdf")
    fig.savefig(ROOT / "results/figures/fig_structures.png")
    plt.close(fig)
    # ---- SI: all 18
    fig, axes = plt.subplots(3, 6, figsize=(190 * MM, 118 * MM))
    for ax, (name, r) in zip(axes.ravel(), d.iterrows()):
        r = r.copy(); r["name"] = name
        atoms = read(label_folder(name, int(r.iteration), r.spin_solution) / "POSCAR")
        draw(ax, atoms, title=pretty(r), note=f"{r.gap_eV:.2f} eV", topology=r.topology)
    fig.subplots_adjust(hspace=0.45, wspace=0.08)
    fig.savefig(ROOT / "results/figures/fig_S_structures_all.pdf")
    fig.savefig(ROOT / "results/figures/fig_S_structures_all.png")
    print("saved fig_structures, fig_S_structures_all")


if __name__ == "__main__":
    main()
