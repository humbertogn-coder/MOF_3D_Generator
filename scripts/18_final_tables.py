#!/usr/bin/env python3
"""Final result tables and figures of the DFT campaign (iterations 0-2, 18 labels).

Inputs: data/dft/dft_results.csv, data/dft/spin_solutions.csv, data/dft/compute_log.csv,
data/al/iter*/predictions.csv, data/catalogue/passed.csv, results/band_edges/*,
results/validation_known_mofs.csv, results/retro_al/retro_al_headline.csv.

Outputs (results/final/):
    table_labels.csv          the 18 accepted labels (Table 1 / Table S-labels)
    table_al_evaluation.csv   per-iteration prospective evaluation of the surrogate
    table_excluded.csv        the 9 frameworks without a label and why
    table_family_series.csv   within-family substituent series (DFT vs surrogate)
    table_spin_solutions.csv  all spin solutions, with the resulting spin state
    table_compute.csv         core-hours per iteration and cluster
    results/figures/fig_al_prospective.{pdf,png}, fig_family_series.{pdf,png}

Usage:  python scripts/18_final_tables.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "results/final"
FAMILY = {("pcu", "Zn"): "Zn4O pcu", ("bcu", "Co"): "Co4 bcu", ("bcu", "Fe"): "Fe4 bcu",
          ("cds", "Cu"): "Cu2 cds", ("cds", "Co"): "Co2 cds", ("cds", "Fe"): "Fe2 cds",
          ("cds", "Mn"): "Mn2 cds", ("cds", "Zn"): "Zn2 cds", ("pcu", "Mn"): "Mn3 pcu",
          ("nbo", "Co"): "Co2 nbo", ("dia", "Ni"): "Ni2 dia"}
FAMILY_ORDER = ["Zn4O pcu", "Co4 bcu", "Fe4 bcu", "Cu2 cds", "Co2 cds", "Fe2 cds"]
SUB = {"H": "H", "NH2": "NH2", "F4": "F4", "F8": "F8", "Nring1": "N-ring", "Nring2": "N2-ring"}
KNOWN = {"pcu__Zn-N16__short-H": "MOF-5", "pcu__Zn-N16__short-NH2": "IRMOF-3"}


def family(r):
    return FAMILY.get((r.topology, r.metal), f"{r.metal} {r.topology}")


def linker(r):
    core = "BPDC" if "long" in r["name"] else "BDC"
    return core if r.substituent == "H" else f"{SUB.get(r.substituent, r.substituent)}-{core}"


def spin_state(r):
    if r.metal in ("Zn", "Mg"):
        return "closed shell"
    if pd.isna(r.total_magnetization):
        return ""
    if abs(r.total_magnetization) < 0.1:
        return "AFM" if r.mag_sum_abs_metal > 0.5 else "NM (low spin)"
    return "FM"


def prospective():
    """mu, sigma of the surrogate that selected each structure, and of the final model."""
    out = {}
    for it in sorted(ROOT.glob("data/al/iter[0-9]*")):
        k = int(it.name[4:])
        p = pd.read_csv(it / "predictions.csv").set_index("name")
        out[k] = p
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(ROOT / "data/dft/dft_results.csv")
    d = d[~d.name.str.contains("__afm")].copy()
    cat = pd.read_csv(ROOT / "data/catalogue/passed.csv").set_index("name")
    pred = prospective()
    final_k = max(pred)
    d["family"] = d.apply(family, axis=1)
    d["linker"] = d.apply(linker, axis=1)
    d["known_mof"] = d.name.map(KNOWN).fillna("")
    d["formula"] = d.name.map(cat.formula)
    d["n_atoms"] = d.name.map(cat.n_atoms)
    d["mu_prospective"] = [pred[int(k)].at[n, "mu"] for n, k in zip(d.name, d.iteration)]
    d["sigma_prospective"] = [pred[int(k)].at[n, "sigma_tree"] for n, k in zip(d.name, d.iteration)]
    d["mu_final"] = d.name.map(pred[final_k].mu)
    d["spin_state"] = d.apply(spin_state, axis=1)
    d["closed_shell"] = d.metal.isin(["Zn", "Mg"])
    d["error_prospective"] = d.mu_prospective - d.gap_eV
    comp = pd.read_csv(ROOT / "data/dft/compute_log.csv")
    comp["base"] = comp.structure.str.replace(r"__afm\d*$", "", regex=True)
    d["core_h"] = d.name.map(comp.groupby("base").core_h.sum()).fillna(0).round(0)
    be = pd.read_csv(ROOT / "results/band_edges/band_edge_character.csv").set_index("name")
    for e in ("VB", "CB"):
        grp = be[[f"{e}_metal", f"{e}_oxo", f"{e}_Ocarb", f"{e}_linker"]]
        d[f"{e}_dominant"] = d.name.map(grp.idxmax(axis=1).str.replace(f"{e}_", ""))
        d[f"{e}_metal_d"] = d.name.map(be[f"{e}_metal_d"])
    known = pd.read_csv(ROOT / "results/validation_known_mofs.csv").set_index("name")
    d["gap_PBE_QMOF"] = d.name.map(known.gap_PBE_qmof)

    lab = d[d.label_ok.astype(bool)].copy()
    lab["family"] = pd.Categorical(lab.family, FAMILY_ORDER, ordered=True)
    lab = lab.sort_values(["family", "gap_eV"])
    cols = ["name", "known_mof", "family", "topology", "node_family", "metal", "linker", "formula", "n_atoms",
            "iteration", "gap_eV", "is_direct", "spin_solution", "spin_state", "spin_solutions_dE_eV",
            "total_magnetization", "mu_prospective", "sigma_prospective", "error_prospective", "mu_final",
            "VB_dominant", "VB_metal_d", "CB_dominant", "CB_metal_d", "gap_PBE_QMOF", "relax_stopped",
            "relax_fmax_final", "vol_change_pct", "core_h"]
    lab[cols].to_csv(OUT / "table_labels.csv", index=False, float_format="%.3f", lineterminator="\n")

    # ---- per-iteration evaluation
    rows = []
    for k, g in d.groupby("iteration"):
        ok = g[g.label_ok.astype(bool)]
        err = ok.error_prospective
        info = json.load(open(ROOT / f"data/al/iter{k}/model_info.json"))
        rows.append(dict(
            iteration=k, n_train_labels=info.get("n_dft", 0), n_submitted=len(g), n_accepted=len(ok),
            n_collapsed=int(g.label_reasons.fillna("").str.contains("collapse|shear").sum()),
            n_other=len(g) - len(ok) - int(g.label_reasons.fillna("").str.contains("collapse|shear").sum()),
            MAE_eV=err.abs().mean(), bias_eV=err.mean(),
            n_within_2sigma=int((err.abs() <= 2 * ok.sigma_prospective).sum()),
            n_gap_lt_1eV=int((ok.gap_eV < 1).sum()), n_gap_lt_0p5eV=int((ok.gap_eV < 0.5).sum()),
            n_open_shell_labels=int((~ok.closed_shell).sum()),
            mean_sigma_batch=g.sigma_prospective.mean(),
            core_h=comp[comp.iteration == k].core_h.sum()))
    ok = lab
    rows.append(dict(iteration="all", n_train_labels="", n_submitted=len(d), n_accepted=len(ok),
                     n_collapsed=int(d.label_reasons.fillna("").str.contains("collapse|shear").sum()),
                     n_other=len(d) - len(ok) - int(d.label_reasons.fillna("").str.contains("collapse|shear").sum()),
                     MAE_eV=ok.error_prospective.abs().mean(), bias_eV=ok.error_prospective.mean(),
                     n_within_2sigma=int((ok.error_prospective.abs() <= 2 * ok.sigma_prospective).sum()),
                     n_gap_lt_1eV=int((ok.gap_eV < 1).sum()), n_gap_lt_0p5eV=int((ok.gap_eV < 0.5).sum()),
                     n_open_shell_labels=int((~ok.closed_shell).sum()),
                     mean_sigma_batch=d.sigma_prospective.mean(), core_h=comp.core_h.sum()))
    ev = pd.DataFrame(rows)
    for shell, g in ok.groupby("closed_shell"):
        ev.loc[len(ev)] = dict(iteration="closed-shell" if shell else "open-shell", n_train_labels="",
                               n_submitted="", n_accepted=len(g), n_collapsed="", n_other="",
                               MAE_eV=g.error_prospective.abs().mean(), bias_eV=g.error_prospective.mean(),
                               n_within_2sigma=int((g.error_prospective.abs() <= 2 * g.sigma_prospective).sum()),
                               n_gap_lt_1eV=int((g.gap_eV < 1).sum()), n_gap_lt_0p5eV=int((g.gap_eV < 0.5).sum()),
                               n_open_shell_labels="", mean_sigma_batch="", core_h="")
    ev.to_csv(OUT / "table_al_evaluation.csv", index=False, float_format="%.3f", lineterminator="\n")

    # ---- excluded / not accepted
    ex = d[~d.label_ok.astype(bool)].copy()

    def outcome(r):
        t = str(r.label_reasons)
        if "collapse" in t or "shear" in t:
            return "collapse/shear"
        if "static SCF" in t or r.spin_solution == "undetermined":
            return "SCF not converged (static)"
        if "SCF did not converge" in t:
            return "SCF not converged (relaxation)"
        if "ZBRENT" in t:
            return "unstable relaxation"
        return "relaxation unfinished"
    ex["outcome"] = ex.apply(outcome, axis=1)
    # volume change of collapsed runs whose vasprun.xml is unreadable: from the EXCLUDED note
    miss = ex.vol_change_pct.isna()
    ex.loc[miss, "vol_change_pct"] = ex.loc[miss, "label_reasons"].str.extract(r"volume (-?\d+) %")[0].astype(float)
    ex[["name", "family", "topology", "metal", "linker", "n_atoms", "iteration", "outcome", "vol_change_pct",
        "max_angle_change_deg", "relax_ionic_steps", "mu_prospective", "sigma_prospective", "core_h",
        "label_reasons"]].to_csv(OUT / "table_excluded.csv", index=False, float_format="%.2f", lineterminator="\n")

    # ---- family series
    fs = lab[["family", "name", "known_mof", "linker", "gap_eV", "spin_state", "mu_prospective",
              "sigma_prospective", "mu_final", "VB_dominant", "CB_dominant"]]
    fs.to_csv(OUT / "table_family_series.csv", index=False, float_format="%.3f", lineterminator="\n")

    # ---- spin solutions
    sp = pd.read_csv(ROOT / "data/dft/spin_solutions.csv")
    sp["metal"] = sp.name.map(cat.metal)
    sp["spin_state"] = sp.apply(spin_state, axis=1)
    sp["dE_vs_fm_start_eV"] = sp.energy_eV - sp.name.map(sp[sp.start == "fm-start"].set_index("name").energy_eV)
    sp["label"] = [n in set(lab.name) and (lab.set_index("name").spin_solution.get(n) == s)
                   for n, s in zip(sp.name, sp.start)]
    sp.to_csv(OUT / "table_spin_solutions.csv", index=False, float_format="%.4f", lineterminator="\n")

    # ---- compute
    cs = comp.groupby(["iteration", "cluster"]).agg(jobs=("job_id", "count"), wall_h=("wall_h", "sum"),
                                                     core_h=("core_h", "sum")).reset_index()
    cs.to_csv(OUT / "table_compute.csv", index=False, float_format="%.0f", lineterminator="\n")

    pd.set_option("display.width", 250)
    print(lab[["known_mof", "family", "linker", "gap_eV", "spin_state", "mu_prospective", "sigma_prospective",
               "mu_final", "VB_dominant", "CB_dominant", "core_h"]].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(ev.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(ex[["name", "iteration", "outcome", "vol_change_pct", "core_h"]].to_string(index=False))
    print(cs.to_string(index=False))
    figures(lab, d)


def figures(lab, d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MM = 1 / 25.4
    BLUE, ORANGE, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#e6e5e1", "grid.linewidth": 0.5, "pdf.fonttype": 42, "savefig.bbox": "tight",
        "savefig.dpi": 300})
    fdir = ROOT / "results/figures"

    # (1) prospective parity -------------------------------------------------
    fig, ax = plt.subplots(figsize=(90 * MM, 85 * MM))
    lim = (-0.15, 3.9)
    ax.plot(lim, lim, color=INK2, lw=0.8, ls="--", zorder=1)
    ax.fill_between(lim, [l - 0.5 for l in lim], [l + 0.5 for l in lim], color="#e9e8e4", zorder=0, lw=0)
    mk = {0: "o", 1: "s", 2: "^"}
    for shell, col, lab_s in ((True, BLUE, "closed-shell Zn4O"), (False, ORANGE, "open-shell 3d")):
        for k in (0, 1, 2):
            g = lab[(lab.closed_shell == shell) & (lab.iteration == k)]
            if len(g) == 0:
                continue
            ax.errorbar(g.gap_eV, g.mu_prospective, yerr=g.sigma_prospective, fmt=mk[k], ms=5, color=col,
                        ecolor=col, elinewidth=0.7, capsize=0, alpha=0.95, zorder=3,
                        label=f"{lab_s}, iteration {k}" if True else None)
    for r in lab[lab.known_mof != ""].itertuples():
        ax.annotate(r.known_mof, (r.gap_eV, r.mu_prospective), xytext=(-4, -9), textcoords="offset points",
                    fontsize=6, color=INK2, ha="right")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("PBE-D3 band gap of the DFT label (eV)")
    ax.set_ylabel("surrogate prediction μ ± σ before the calculation (eV)")
    ax.legend(frameon=False, loc="upper left", handletextpad=0.3)
    ax.text(3.8, 3.2, "±0.5 eV", fontsize=6, color=INK2, ha="right")
    fig.savefig(fdir / "fig_al_prospective.pdf")
    fig.savefig(fdir / "fig_al_prospective.png")
    plt.close(fig)

    # (2) within-family series ------------------------------------------------
    fams = [f for f in FAMILY_ORDER if f in set(lab.family.astype(str))]
    rows = lab.copy()
    rows["family"] = rows.family.astype(str)
    rows["ypos"] = 0.0
    order = []
    y = 0
    for f in fams:
        g = rows[rows.family == f].sort_values("gap_eV", ascending=False)
        for n in g.name:
            order.append((n, y))
            y += 1
        y += 0.6
    pos = dict(order)
    rows["ypos"] = rows.name.map(pos)
    fig, ax = plt.subplots(figsize=(90 * MM, 95 * MM))
    ax.hlines(rows.ypos, rows.mu_prospective, rows.gap_eV, color="#d9d8d3", lw=1.2, zorder=1)
    ax.scatter(rows.mu_prospective, rows.ypos, marker="o", s=20, facecolor="white", edgecolor=GRAY, lw=1.0,
               zorder=2, label="surrogate μ before DFT")
    for shell, col, lab_s in ((True, BLUE, "PBE-D3 label, closed-shell Zn4O"), (False, ORANGE, "PBE-D3 label, open-shell 3d")):
        g = rows[rows.closed_shell == shell]
        ax.scatter(g.gap_eV, g.ypos, marker="o", s=22, color=col, zorder=3, label=lab_s)
    ax.set_yticks(rows.ypos)
    ax.set_yticklabels([f"{r.linker}{' (' + r.known_mof + ')' if r.known_mof else ''}"
                        + (f"  [{r.spin_state}]" if r.spin_state in ("AFM", "NM (low spin)") else "")
                        for r in rows.itertuples()], fontsize=6.5)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    for f in fams:
        g = rows[rows.family == f]
        ax.text(4.05, g.ypos.mean(), f, va="center", ha="left", fontsize=6.5, color=INK2)
        ax.axhline(g.ypos.max() + 0.8, color=INK2, lw=0.4, ls=":")
    ax.set_xlim(-0.1, 4.0)
    ax.set_xlabel("band gap (eV)")
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, frameon=False, loc="upper right", bbox_to_anchor=(0.99, 0.70))
    fig.savefig(fdir / "fig_family_series.pdf")
    fig.savefig(fdir / "fig_family_series.png")
    plt.close(fig)
    print("saved fig_al_prospective, fig_family_series")


if __name__ == "__main__":
    main()
