#!/usr/bin/env python3
"""Schematic of the workflow (Figure 1), drawn with matplotlib from the numbers in
data/ and results/ (no image generator involved).

Usage:  python scripts/20_workflow_figure.py
Output: results/figures/fig_workflow.{pdf,png}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MM = 1 / 25.4
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, PAPER = "#0b0b0b", "#52514e", "#fcfcfb"


def numbers():
    cat = pd.read_csv(ROOT / "data/catalogue/catalogue.csv")
    p = pd.read_csv(ROOT / "data/catalogue/passed.csv")
    d = pd.read_csv(ROOT / "data/dft/dft_results.csv")
    d = d[~d.name.str.contains("__afm")]
    info = json.load(open(ROOT / "data/al/iter0/model_info.json"))
    return dict(built=len(cat), passed=len(p), open_shell=int(p.open_shell.sum()),
                nodes=p.node_id.nunique(), edges=p.edge_id.nunique(), topos=p.topology.nunique(),
                qmof=info["n_qmof"], computed=len(d), labels=int(d.label_ok.sum()),
                iters=int(d.iteration.nunique()))


def box(ax, xy, w, h, title, lines, color):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.0, edgecolor=color, facecolor=PAPER, zorder=2))
    ax.add_patch(FancyBboxPatch((x, y + h - 0.22), w, 0.22, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=0, facecolor=color, zorder=3))
    ax.text(x + w / 2, y + h - 0.11, title, ha="center", va="center", fontsize=7, color="white",
            fontweight="bold", zorder=4)
    ax.text(x + 0.06, y + h - 0.30, "\n".join(lines), ha="left", va="top", fontsize=5.3, color=INK,
            linespacing=1.3, zorder=4)


def arrow(ax, p0, p1, color=INK2, style="-|>", rad=0.0, lw=1.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=9, color=color, lw=lw,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=5))


def main():
    n = numbers()
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "savefig.bbox": "tight",
                         "savefig.dpi": 300})
    fig, ax = plt.subplots(figsize=(170 * MM, 72 * MM))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.45, 4.75)
    ax.axis("off")
    W, H = 2.25, 1.3
    # top row ----------------------------------------------------------------
    box(ax, (0.15, 3.3), W, H, "1  Catalogue", [
        "PORMAKE assembly from",
        "audited building blocks:",
        f"{n['nodes']} M(II) carboxylate nodes",
        f"× {n['edges']} dicarboxylate linkers",
        f"× {n['topos']} nets → {n['built']} built"], BLUE)
    box(ax, (2.65, 3.3), W, H, "2  Structural QC", [
        "overlaps, non-bonded",
        "contacts, fragments,",
        "H/C valence, node RMSD,",
        "density, ≤ 600 atoms",
        f"→ {n['passed']} pass ({n['open_shell']} open-shell)"], BLUE)
    box(ax, (5.15, 3.3), W, H, "3  Surrogate", [
        "64 descriptors (Zeo++",
        "pores, composition);",
        "5 × ExtraTrees trained on",
        f"QMOF (n = {n['qmof']:,})",
        "→ μ, σ for every candidate"], AQUA)
    box(ax, (7.65, 3.3), W, H, "4  Acquisition", [
        "μ − κσ − α/N_topo + βI_OOD",
        "eligibility rules from",
        "earlier iterations; caps per",
        "net, metal and substituent",
        "→ batch of 8"], AQUA)
    # bottom row ---------------------------------------------------------------
    box(ax, (7.65, 0.6), W, H, "5  Pre-relaxation", [
        "CHGNet universal potential:",
        "positions and cell,",
        "F ≤ 0.05 eV/Å; removes the",
        "compressed bonds of the",
        "assembled geometry"], ORANGE)
    box(ax, (5.15, 0.6), W, H, "6  DFT (VASP)", [
        "PBE-D3(BJ), spin-polarised",
        "full relaxation + static;",
        "broken-symmetry static",
        "for M₂ paddlewheels"], ORANGE)
    box(ax, (2.65, 0.6), W, H, "7  Acceptance", [
        "converged relaxation and",
        "SCF; QC of final structure;",
        "|ΔV| ≤ 20 %; no ligand",
        "radical; lowest-energy",
        "spin solution → label"], ORANGE)
    box(ax, (0.15, 0.6), W, H, "8  Labels", [
        f"{n['iters']} iterations, {n['computed']} frameworks",
        f"computed, {n['labels']} PBE-D3 labels;",
        "band-edge character;",
        "periodic xTB check;",
        "retrain the surrogate"], YELLOW)
    # arrows -------------------------------------------------------------------
    y_top, y_bot = 3.3 + H / 2, 0.6 + H / 2
    for x0 in (0.15 + W, 2.65 + W, 5.15 + W):
        arrow(ax, (x0 + 0.02, y_top), (x0 + 0.38, y_top))
    arrow(ax, (7.65 + W / 2, 3.3 - 0.02), (7.65 + W / 2, 0.6 + H + 0.08))
    for x0 in (7.65, 5.15, 2.65):
        arrow(ax, (x0 - 0.02, y_bot), (x0 - 0.38, y_bot))
    # feedback: labels -> surrogate
    arrow(ax, (0.15 + W * 0.6, 0.6 + H + 0.04), (5.15 + W * 0.3, 3.3 - 0.04), color=YELLOW, rad=-0.22,
          lw=1.2, ls="--")
    ax.text(3.0, 2.62, "new labels retrain the surrogate", fontsize=5.8, color=INK2, ha="center",
            va="center", style="italic",
            bbox=dict(boxstyle="round,pad=0.15", fc=PAPER, ec="none"))
    fig.savefig(ROOT / "results/figures/fig_workflow.pdf")
    fig.savefig(ROOT / "results/figures/fig_workflow.png")
    print("saved fig_workflow", n)


if __name__ == "__main__":
    main()
