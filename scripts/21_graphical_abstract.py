#!/usr/bin/env python3
"""Graphical abstract (Elsevier: >= 1328 x 531 px, here 13 x 5.2 cm at 300 dpi), drawn with
matplotlib from results/final/table_labels.csv. Usage: python scripts/21_graphical_abstract.py"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, AQUA, YELLOW, INK, INK2, PAPER = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#52514e", "#fcfcfb"
MM = 1 / 25.4
lab = pd.read_csv(ROOT / "results/final/table_labels.csv")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "savefig.dpi": 300, "pdf.fonttype": 42})
fig = plt.figure(figsize=(130 * MM, 52 * MM))
ax = fig.add_axes([0.0, 0.0, 0.62, 1.0]); ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")


def box(x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", lw=1, ec=color, fc=PAPER))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="center", fontsize=5.2, fontweight="bold", color=color)
    ax.text(x + w / 2, y + h / 2 - 0.18, body, ha="center", va="center", fontsize=4.4, color=INK, linespacing=1.3)


def arrow(p0, p1, color=INK2, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, color=color, lw=1, ls=ls, connectionstyle=f"arc3,rad={rad}"))


box(0.2, 2.25, 2.9, 1.5, "Audited catalogue", "131 hypothetical 3D MOFs\nM(II) carboxylate nodes\n5 nets, 8 linkers", BLUE)
box(3.55, 2.25, 2.9, 1.5, "QMOF surrogate", "ExtraTrees ensemble, μ ± σ\nLCB acquisition\nwith diversity caps", AQUA)
box(6.9, 2.25, 2.9, 1.5, "Prospective DFT", "ML potential pre-relaxation\nPBE-D3(BJ), spin states\n27 computed, 18 labels", ORANGE)
arrow((3.12, 3.0), (3.52, 3.0)); arrow((6.47, 3.0), (6.87, 3.0))
arrow((8.35, 2.22), (5.0, 2.22 - 0.0), color=YELLOW, rad=0.0, ls="--")
ax.text(5.0, 1.75, "labels retrain the surrogate · band-edge character · periodic GFN1-xTB check", ha="center", va="center",
        fontsize=4.6, color=INK2, style="italic")
ax.text(0.2, 1.15, "Open-shell Fe, Co, Cu: 13 of 13 below 1 eV; model error 0.23 eV; Fe and Co edges are metal d", fontsize=4.8, color=INK, va="center")
ax.text(0.2, 0.72, "Closed-shell Zn$_4$O: model 0.96 eV too low; organic band edges; periodic xTB within 0.23 eV", fontsize=4.8, color=INK, va="center")
ax.text(0.2, 0.29, "At this DFT budget: 3 to 4 times more low gap hits than random selection; retraining adds little", fontsize=4.8, color=INK, va="center")

ax2 = fig.add_axes([0.72, 0.2, 0.26, 0.72])
lim = (-0.15, 3.9)
ax2.plot(lim, lim, color=INK2, lw=0.7, ls="--")
for shell, col in ((True, BLUE), (False, ORANGE)):
    g = lab[lab.metal.eq("Zn") == shell]
    ax2.errorbar(g.gap_eV, g.mu_prospective, yerr=g.sigma_prospective, fmt="o", ms=3.5, color=col, elinewidth=0.6, capsize=0)
ax2.set_xlim(lim); ax2.set_ylim(lim)
ax2.set_xlabel("PBE-D3 gap of the label (eV)", fontsize=5.2)
ax2.set_ylabel("model μ ± σ before DFT (eV)", fontsize=5.2)
ax2.tick_params(labelsize=5)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
ax2.text(0.05, 3.55, "open-shell 3d", color=ORANGE, fontsize=5)
ax2.text(0.05, 3.15, "closed-shell Zn$_4$O", color=BLUE, fontsize=5)
fig.savefig(ROOT / "results/figures/graphical_abstract.png")
fig.savefig(ROOT / "results/figures/graphical_abstract.pdf")
print("saved graphical_abstract")
