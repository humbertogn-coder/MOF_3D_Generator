#!/usr/bin/env python3
"""Build tables.json (header/rows for every manuscript and SI table) and image_sizes.json
from the repository outputs, so that no number is typed by hand."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
T = {}
SUB = {"H": "H", "NH2": "NH_{2}", "F4": "F_{4}", "F8": "F_{8}", "Nring1": "N-ring", "Nring2": "N_{2}-ring"}
FAM = {"Zn4O pcu": "Zn_{4}O pcu", "Co4 bcu": "Co_{4} bcu", "Fe4 bcu": "Fe_{4} bcu", "Cu2 cds": "Cu_{2} cds",
       "Co2 cds": "Co_{2} cds", "Fe2 cds": "Fe_{2} cds", "Mn2 cds": "Mn_{2} cds", "Zn2 cds": "Zn_{2} cds",
       "Mn3 pcu": "Mn_{3} pcu", "Co2 nbo": "Co_{2} nbo", "Ni2 dia": "Ni_{2} dia"}
GRP = {"metal": "metal", "oxo": "μ_{4}-O", "Ocarb": "O_{carb}", "linker": "linker"}


def f(x, nd=2):
    return "" if pd.isna(x) else f"{x:.{nd}f}"


def formula(s):
    return re.sub(r"(\d+)", r"_{\1}", s)


def linker_fmt(s):
    s = s.replace("NH2", "NH_{2}").replace("F4", "F_{4}").replace("F8", "F_{8}").replace("N2-ring", "N_{2}-ring")
    return s


lab = pd.read_csv(REPO / "results/final/table_labels.csv")
ev = pd.read_csv(REPO / "results/final/table_al_evaluation.csv")
ex = pd.read_csv(REPO / "results/final/table_excluded.csv")
sp = pd.read_csv(REPO / "results/final/table_spin_solutions.csv")
comp = pd.read_csv(REPO / "results/final/table_compute.csv")
be = pd.read_csv(REPO / "results/band_edges/band_edge_character.csv")
xbe = pd.read_csv(REPO / "results/band_edges/xtb_vs_dft_band_edges.csv")
xs = pd.read_csv(REPO / "results/xtb_candidates/xtb_candidates_summary.csv")
xc = pd.read_csv(REPO / "results/xtb_candidates/xtb_vs_dft_vs_surrogate.csv")
xq = pd.read_csv(REPO / "results/xtb/xtb_summary.csv")
_ra_path = REPO / "results/retro_al/retro_al_headline.csv"
if not _ra_path.exists():
    _ra_path = REPO / "results/retro_al_v1/retro_al_headline.csv"
ra = pd.read_csv(_ra_path)
RETRO_BUDGET, RETRO_END = int(ra.labels.min()), int(ra.labels.max())
bm = pd.read_csv(REPO / "results/benchmark/table_S1_feature_set_C.csv")
bc = pd.read_csv(REPO / "results/benchmark/benchmark_combined.csv")
cat = pd.read_csv(REPO / "data/catalogue/catalogue.csv")
pas = pd.read_csv(REPO / "data/catalogue/passed.csv")
known = pd.read_csv(REPO / "results/validation_known_mofs.csv")
hse_shell = pd.read_csv(REPO / "results/qmof_pbe_vs_hse06/by_shell.csv")
hse_metal = pd.read_csv(REPO / "results/qmof_pbe_vs_hse06/by_metal.csv")

# ---------------------------------------------------------------- Table 1: labels
rows = []
for r in lab.itertuples():
    name = f"{FAM[r.family]}, {linker_fmt(r.linker)}" + (f" ({r.known_mof})" if isinstance(r.known_mof, str) and r.known_mof else "")
    edges = f"{GRP[r.VB_dominant]} / {GRP[r.CB_dominant]}"
    rows.append([name, formula(r.formula), str(int(r.n_atoms)), str(int(r.iteration)),
                 f"{r.mu_prospective:.2f} ± {r.sigma_prospective:.2f}", f"{r.gap_eV:.2f}", r.spin_state.replace("NM (low spin)", "NM"), edges])
T["labels"] = dict(header=["Framework", "Formula (cell)", "Atoms", "Iter.", "μ ± σ before DFT (eV)", "PBE-D3 gap (eV)",
                           "Spin state", "VBM / CBM"], rows=rows, widths=[2.6, 1.9, 0.7, 0.6, 1.5, 1.1, 1.3, 1.4])

# ---------------------------------------------------------------- Table 2: evaluation
rows = []
for r in ev.itertuples():
    it = str(r.iteration)
    if it in ("open-shell", "closed-shell"):
        continue
    rows.append([("All" if it == "all" else it), ("" if it == "all" else str(int(r.n_train_labels))), str(int(r.n_submitted)),
                 str(int(r.n_accepted)), str(int(r.n_collapsed)), str(int(r.n_other)), f"{r.MAE_eV:.2f}", f"{r.bias_eV:+.2f}",
                 f"{int(r.n_within_2sigma)}/{int(r.n_accepted)}", str(int(r.n_gap_lt_1eV)), str(int(r.n_gap_lt_0p5eV)),
                 f"{r.core_h:,.0f}"])
T["al_eval"] = dict(header=["Iteration", "Labels in training", "Submitted", "Accepted", "Collapsed / sheared", "Other failures",
                            "MAE (eV)", "Bias (eV)", "Within 2σ", "Gap < 1 eV", "Gap < 0.5 eV", "Core-hours"],
                    rows=rows, widths=[0.9, 0.9, 0.9, 0.9, 1.0, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8, 1.0])
shell = {str(r.iteration): r for r in ev.itertuples() if str(r.iteration) in ("open-shell", "closed-shell")}

# ---------------------------------------------------------------- Table 3: xTB
lab_ = {"mu_prospective": "Model μ before DFT", "dft": "GFN1-xTB, DFT geometry (Γ)", "dft_kmesh": "GFN1-xTB, DFT geometry, k-mesh supercell",
        "chgnet": "GFN1-xTB, pre-relaxed geometry (ML potential)", "pormake": "GFN1-xTB, as-assembled geometry"}
rows = []
for est in ["mu_prospective", "dft", "dft_kmesh", "chgnet", "pormake"]:
    c = xs[(xs.group == "closed-shell (Zn)") & (xs.estimate == est)].iloc[0]
    o = xs[(xs.group == "open-shell 3d") & (xs.estimate == est)].iloc[0]
    rows.append([lab_[est], f"{c.MAE_eV:.2f} ({c.bias_eV:+.2f}), n = {int(c.n)}", f"{o.MAE_eV:.2f} ({o.bias_eV:+.2f}), n = {int(o.n)}"])
T["xtb"] = dict(header=["Estimate", "Closed-shell Zn_{4}O: MAE (bias)", "Open-shell 3d: MAE (bias)"], rows=rows, widths=[2.4, 1.6, 1.6])

# ---------------------------------------------------------------- SI tables
# S: benchmark
rows = [[r.model, f"{r.R2:.2f}", f"{r.MAE_eV:.2f}", f"{r.RMSE_eV:.2f}", f"{r.Spearman:.2f}" if pd.notna(r.Spearman) else "–",
         r.classifier_used, f"{r.ROC_AUC:.2f}", f"{r.F1:.2f}"] for r in bm.itertuples()]
T["benchmark"] = dict(header=["Regressor", "R^{2}", "MAE (eV)", "RMSE (eV)", "Spearman ρ", "Classifier", "ROC AUC", "F_{1}"], rows=rows,
                      widths=[1.6, 0.6, 0.7, 0.7, 0.8, 1.6, 0.7, 0.6])
# S: feature sets with ExtraTrees performance
desc = {"A_basic_geometry": ("A", "8", "n_atoms, density, PLD, LCD, cell volume, LCD − PLD, LCD/PLD, volume per atom"),
        "B_geometry_plus_composition": ("B", "30", "A + aggregate composition (metal counts and fractions, open-shell 3d flag and fraction, C/N/O/H per atom, ratios, heteroatom fraction, mean and std of Z and electronegativity)"),
        "C_geometry_composition_elements": ("C", "64", "B + fractions of 34 common elements (selected)"),
        "D_all_structural_plus_categories": ("D", "64 + 4", "C + categorical topology, primary metal, crystal system and node SMILES (one-hot)"),
        "E_all_plus_PBE_auxiliary_no_gap": ("E", "68 + 4", "D + four PBE-derived quantities per atom (energy, vdW, electronic, |magmom|); reference only, leakage")}
rows = []
for fs_, (lbl, n, d) in desc.items():
    r = bc[(bc.task == "regression") & (bc.model == "ExtraTrees") & (bc.feature_set == fs_)].iloc[0]
    c = bc[(bc.task == "classification") & (bc.model == "ExtraTrees") & (bc.feature_set == fs_)].iloc[0]
    rows.append([lbl, n, d, f"{r.R2:.2f}", f"{r.MAE_eV:.2f}", f"{c.ROC_AUC:.2f}"])
T["feature_sets"] = dict(header=["Set", "Features", "Content", "R^{2} (ET)", "MAE (eV)", "ROC AUC"], rows=rows, widths=[0.4, 0.7, 4.2, 0.7, 0.7, 0.7], align=["l", "l", "l"])
# S: building blocks
nodes = pas.groupby(["node_id", "node_desc", "node_family", "metal"]).topology.agg(lambda s: ", ".join(sorted(set(s)))).reset_index()
nodes["n"] = pas.groupby(["node_id"]).size().reindex(nodes.node_id).values
rows = [[r.node_id, formula(r.node_desc.split(",")[0]).replace("(CO_{2})", "(CO_{2})"), r.node_family.replace("M2(CO2)4", "M_{2}(CO_{2})_{4}").replace("M3(CO2)6", "M_{3}(CO_{2})_{6}").replace("M4(CO2)8", "M_{4}(CO_{2})_{8}").replace("Zn4O(CO2)6", "Zn_{4}O(CO_{2})_{6}"),
         r.metal, r.topology, str(int(r.n))] for r in nodes.sort_values(["node_family", "metal"]).itertuples()]
T["nodes"] = dict(header=["PORMAKE node", "Composition", "Family", "Metal", "Nets", "Frameworks passing QC"], rows=rows, widths=[1.0, 1.6, 1.8, 0.6, 0.9, 1.2])
edges = pas.groupby(["edge_id", "linker_desc", "linker_label"]).size().reset_index(name="n")
rows = [[r.edge_id, r.linker_desc.replace("tetrafluoro-BDC", "2,3,5,6-tetrafluoro-BDC").replace("octafluoro-BPDC", "octafluoro-BPDC"),
         linker_fmt(r.linker_label.replace("short-", "BDC-type, ").replace("long-", "BPDC-type, ")), str(int(r.n))] for r in edges.itertuples()]
T["linkers"] = dict(header=["PORMAKE edge", "Assembled dicarboxylate linker", "Label", "Frameworks passing QC"], rows=rows, widths=[1.0, 3.0, 2.0, 1.2], align=["l", "l", "l"])
# S: catalogue composition
cc = pas.groupby(["node_family", "topology", "metal"]).size().unstack(fill_value=0)
rows = []
for (famy, topo), r in cc.iterrows():
    rows.append([famy.replace("M2(CO2)4", "M_{2}(CO_{2})_{4}").replace("M3(CO2)6", "M_{3}(CO_{2})_{6}").replace("M4(CO2)8", "M_{4}(CO_{2})_{8}").replace("Zn4O(CO2)6", "Zn_{4}O(CO_{2})_{6}"), topo] + [str(int(v)) if v else "–" for v in r.values] + [str(int(r.sum()))])
T["catalogue"] = dict(header=["Node family", "Net"] + list(cc.columns) + ["Total"], rows=rows, widths=[2.2, 0.7] + [0.6] * len(cc.columns) + [0.7])
# S: QC rejections
rej = cat[~cat.passed].qc_reasons.str.split(";").explode().str.strip()
pat = {"density": "density outside 0.20–3.0 g cm^{−3}", "heavy/H-H": "heavy–heavy or H–H pair closer than 1.00 Å", "connected components": "more than one connected component",
       "H with CN": "H atom without exactly one neighbour", "C with organic CN": "C atom with organic coordination number outside 2–4",
       "non-bonded contacts": "non-bonded contact shorter than 0.8 × Σ r_{cov}", "pairs <": "any pair closer than 0.70 Å"}
cnt = {v: int(rej.str.contains(k, regex=False).sum()) for k, v in pat.items()}
T["qc"] = dict(header=["QC criterion", "Frameworks failing it (of 37 rejected)"], rows=[[k, str(v)] for k, v in cnt.items()], widths=[3.5, 2.0], align=["l"])
# S: VASP parameters (static text)
T["vasp"] = dict(header=["Parameter", "Relaxation", "Static", "Broken-symmetry static"], rows=[
    ["Functional / dispersion", "PBE + D3(BJ)", "PBE + D3(BJ)", "PBE + D3(BJ)"],
    ["PAW set", "potpaw_PBE.52", "potpaw_PBE.52", "potpaw_PBE.52"],
    ["ENCUT (eV)", "520", "520", "520"],
    ["EDIFF (eV)", "1 × 10^{−5}", "1 × 10^{−6}", "1 × 10^{−6}"],
    ["NELM", "120", "300", "300"],
    ["ISPIN / MAGMOM", "2 / high-spin FM", "2 / high-spin FM", "2 / ±max(HS, 2) per M_{2} unit"],
    ["ISYM", "default", "default", "0"],
    ["LMAXMIX", "4", "4", "4"],
    ["ALGO", "Normal (All for Fe, Co, Ni)", "Normal (All for Fe, Co, Ni)", "Normal (All for Fe, Co, Ni)"],
    ["ISMEAR / SIGMA", "0 / 0.01", "−5 if ≥ 4 k-points and ALGO ≠ All, else 0 / 0.01", "as static"],
    ["IBRION / ISIF / NSW", "2 / 3 / 600", "−1 / – / 0", "−1 / – / 0"],
    ["EDIFFG (eV Å^{−1})", "−0.05", "–", "–"],
    ["k-points", "Γ if all a_{i} ≥ 12 Å, else Γ-centred n_{i} = round(25 Å / a_{i})", "Γ-centred n_{i} = round(25 Å / a_{i})", "as static"],
    ["LORBIT / NEDOS", "–", "11 / 3001", "11 / 3001"],
    ["KPAR", "2 if ≥ 2 k-points", "2 if ≥ 2 k-points", "2 if ≥ 2 k-points"],
], widths=[1.6, 1.9, 2.0, 1.9], align=["l", "l", "l", "l"])
# S: full labels
rows = []
for r in lab.itertuples():
    name = f"{FAM[r.family]}, {linker_fmt(r.linker)}"
    rows.append([name, str(int(r.iteration)), f"{r.gap_eV:.3f}", "yes" if r.is_direct else "no", r.spin_solution,
                 f(r.spin_solutions_dE_eV, 2) if pd.notna(r.spin_solutions_dE_eV) else "–", f(r.total_magnetization, 1),
                 "yes" if r.relax_stopped else "no", f(r.relax_fmax_final, 3), f(r.vol_change_pct, 1), f(r.VB_metal_d, 2), f(r.CB_metal_d, 2),
                 f(r.gap_PBE_QMOF, 2) if pd.notna(r.gap_PBE_QMOF) else "–", f"{r.core_h:,.0f}"])
T["labels_full"] = dict(header=["Framework", "Iter.", "Gap (eV)", "Direct", "Start of label run", "ΔE vs FM start (eV)", "M_{tot} (μ_{B})",
                                "Relax. stopped", "F_{max} (eV Å^{−1})", "ΔV (%)", "VB metal d", "CB metal d", "QMOF PBE (eV)", "Core-h"],
                        rows=rows, widths=[2.2, 0.5, 0.7, 0.7, 0.9, 0.8, 0.7, 0.7, 0.8, 0.6, 0.6, 0.6, 0.7, 0.7])
# S: spin solutions
rows = []
for r in sp.itertuples():
    nm = lab[lab.name == r.name]
    if len(nm):
        fam_ = f"{FAM[nm.family.iloc[0]]}, {linker_fmt(nm.linker.iloc[0])}"
    else:
        e_ = ex[ex.name == r.name].iloc[0]
        fam_ = f"{FAM.get(e_.family, e_.family)}, {linker_fmt(e_.linker)} (no label)"
    rows.append([fam_, r.start.replace("-start", ""), f"{r.energy_eV:.3f}", f"{r.dE_vs_fm_start_eV:+.3f}", f"{r.gap_eV:.3f}" if pd.notna(r.gap_eV) else "–",
                 f(r.mag_sum_abs_metal, 2), f(r.total_magnetization, 2), r.spin_state, "yes" if r.scf_converged else "no", "label" if r.label else ""])
T["spin"] = dict(header=["Framework", "Initial moments", "E (eV)", "ΔE (eV)", "Gap (eV)", "Σ|m_{M}| (μ_{B})", "M_{tot} (μ_{B})", "State", "SCF conv.", ""],
                 rows=rows, widths=[2.0, 0.9, 0.9, 0.7, 0.7, 0.8, 0.8, 0.9, 0.6, 0.5])
# S: excluded
rows = []
for r in ex.itertuples():
    rows.append([f"{FAM.get(r.family, r.family)}, {linker_fmt(r.linker)}", str(int(r.iteration)), str(int(r.n_atoms)), f"{r.mu_prospective:.2f} ± {r.sigma_prospective:.2f}",
                 r.outcome, f(r.vol_change_pct, 0) if pd.notna(r.vol_change_pct) else "–", f(r.max_angle_change_deg, 0) if pd.notna(r.max_angle_change_deg) else "–",
                 str(int(r.relax_ionic_steps)) if pd.notna(r.relax_ionic_steps) else "–", f"{r.core_h:,.0f}"])
T["excluded"] = dict(header=["Framework", "Iter.", "Atoms", "μ ± σ (eV)", "Outcome", "ΔV (%)", "Δangle (°)", "Ionic steps", "Core-h"],
                     rows=rows, widths=[2.2, 0.5, 0.6, 1.1, 1.9, 0.6, 0.7, 0.7, 0.7])
# S: band edges all groups
rows = []
for r in be.itertuples():
    rows.append([f"{FAM[r.family]}, {r.label.replace('$', '').replace('_2', '_{2}').replace('_4', '_{4}').replace('_8', '_{8}')}", f"{r.gap_eV:.2f}"] +
                [f"{getattr(r, c):.2f}" for c in ["VB_metal", "VB_metal_d", "VB_oxo", "VB_Ocarb", "VB_linker", "CB_metal", "CB_metal_d", "CB_oxo", "CB_Ocarb", "CB_linker"]])
T["band_edges"] = dict(header=["Framework", "Gap (eV)", "VB metal", "VB metal d", "VB μ_{4}-O", "VB O_{carb}", "VB linker", "CB metal", "CB metal d", "CB μ_{4}-O", "CB O_{carb}", "CB linker"],
                       rows=rows, widths=[2.4, 0.6] + [0.6] * 10)
# S: xTB vs DFT band edges
rows = []
for r in xbe.itertuples():
    rows.append([r.label.replace("$", "").replace("_2", "_{2}").replace("_4", "_{4}"), f"{r.gap_dft:.2f}", f"{r.gap_xtb:.2f}"] +
                [f"{getattr(r, c):.2f}" for c in ["VB_dft_linker", "VB_xtb_linker", "VB_dft_Ocarb", "VB_xtb_Ocarb", "VB_dft_oxo", "VB_xtb_oxo", "CB_dft_linker", "CB_xtb_linker", "CB_dft_Ocarb", "CB_xtb_Ocarb"]])
T["xtb_edges"] = dict(header=["Zn_{4}O framework", "Gap DFT", "Gap xTB", "VB linker DFT", "VB linker xTB", "VB O_{carb} DFT", "VB O_{carb} xTB", "VB μ_{4}-O DFT", "VB μ_{4}-O xTB",
                              "CB linker DFT", "CB linker xTB", "CB O_{carb} DFT", "CB O_{carb} xTB"], rows=rows, widths=[1.8, 0.6, 0.6] + [0.6] * 10)
# S: xTB QMOF benchmark
rows = []
for r in xq[xq.group.isin(["all", "closed-shell", "open-shell 3d"])].itertuples():
    rows.append([r.group, r.treatment, r.reference, str(int(r.n)), f"{r.MAE_eV:.2f}", f"{r.bias_eV:+.2f}", f"{r.spearman_rho:.2f}", f"{r._10:.1f}" if hasattr(r, "_10") else f"{getattr(r, 'pct_xtb_gap_lt_0.1', float('nan')):.1f}", f"{r.ROC_AUC_PBE_lt_1:.2f}"])
T["xtb_qmof"] = dict(header=["Group", "Treatment", "Reference", "n", "MAE (eV)", "Bias (eV)", "Spearman ρ", "xTB gap < 0.1 eV (%)", "ROC AUC (ref. < 1 eV)"],
                     rows=rows, widths=[1.2, 0.9, 0.9, 0.6, 0.7, 0.7, 0.8, 1.0, 1.0])
# S: xTB candidates per structure
rows = []
for r in xc.itertuples():
    nm = lab[lab.name == r.name]
    rows.append([f"{FAM[nm.family.iloc[0]]}, {linker_fmt(nm.linker.iloc[0])}", f"{r.gap_dft:.2f}", f"{r.mu_prospective:.2f}", f(r.dft), f(r.dft_kmesh) or "–", f(r.chgnet) or "–", f(r.pormake)])
T["xtb_cand"] = dict(header=["Framework", "PBE-D3 label (eV)", "Model μ (eV)", "xTB, DFT geom. (Γ)", "xTB, DFT geom., k-mesh", "xTB, pre-relaxed geom.", "xTB, as assembled"],
                     rows=rows, widths=[2.2, 0.9, 0.9, 0.9, 1.0, 0.9, 0.9])
# S: retrospective AL
rows = []
name_ = {"random": "random", "greedy": "greedy (lowest μ)", "lcb": "LCB (μ − σ)", "paper": "this work (LCB + novelty + caps)", "paper_static": "this work, no retraining", "uncertainty": "uncertainty (highest σ)"}
for n_ in (RETRO_BUDGET, RETRO_END):
    for s_ in ["random", "greedy", "lcb", "paper", "paper_static", "uncertainty"]:
        r = ra[(ra.labels == n_) & (ra.strategy == s_)].iloc[0]
        rows.append([str(n_), name_[s_], f"{r.hits_lt_0p5:.1f} ± {r.hits_lt_0p5_sd:.1f}", f"{r.hits_lt_1p0:.1f}", f"{r.recall_top10:.2f}", f"{r.test_MAE:.3f}", f"{r.test_MAE_lowgap:.3f}"])
T["retro"] = dict(header=["Labels", "Strategy", "Hits < 0.5 eV", "Hits < 1 eV", "Recall of 10 lowest", "OOD test MAE (eV)", "OOD test MAE, gap < 1 eV"],
                  rows=rows, widths=[0.6, 2.2, 1.1, 0.9, 1.0, 1.0, 1.1])
# S: compute
rows = [[str(int(r.iteration)), r.cluster, str(int(r.jobs)), f"{r.wall_h:.0f}", f"{r.core_h:,.0f}"] for r in comp.itertuples()]
rows.append(["All", "", str(int(comp.jobs.sum())), f"{comp.wall_h.sum():.0f}", f"{comp.core_h.sum():,.0f}"])
T["compute"] = dict(header=["Iteration", "Cluster (cores per node)", "Jobs", "Wall-clock hours", "Core-hours"], rows=[[a, b + (" (48)" if b == "Grace" else " (64)" if b else ""), c, d, e] for a, b, c, d, e in rows], widths=[0.8, 1.6, 0.6, 1.0, 1.0])
# S: PBE vs HSE06
rows = [[r.shell, str(int(r.n)), f"{r.median_PBE:.2f}", f"{r.median_HSE06:.2f}", f"{r.median_HSE06_minus_PBE:.2f}", str(int(r.n_PBE_lt_1)), f"{r.pct_PBE_lt_1_with_HSE_lt_1:.1f}", f"{r.pct_PBE_lt_1_with_HSE_lt_2:.1f}"] for r in hse_shell.itertuples()]
for r in hse_metal[hse_metal.primary_metal.isin(["Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Mg"])].itertuples():
    rows.append([f"  {r.primary_metal}", str(int(r.n)), f"{r.median_PBE:.2f}", f"{r.median_HSE06:.2f}", f"{r.median_HSE06_minus_PBE:.2f}", str(int(r.n_PBE_lt_1)),
                 f"{r.pct_PBE_lt_1_with_HSE_lt_1:.1f}" if pd.notna(r.pct_PBE_lt_1_with_HSE_lt_1) else "–", f"{r.pct_PBE_lt_1_with_HSE_lt_2:.1f}" if pd.notna(r.pct_PBE_lt_1_with_HSE_lt_2) else "–"])
T["hse"] = dict(header=["QMOF subset (primary metal)", "n", "Median PBE (eV)", "Median HSE06 (eV)", "Median HSE06 − PBE (eV)", "n with PBE < 1 eV", "of which HSE06 < 1 eV (%)", "of which HSE06 < 2 eV (%)"],
                rows=rows, widths=[1.8, 0.6, 0.8, 0.8, 0.9, 0.8, 0.9, 0.9])
# S: known MOFs
rows = [[r.mof, formula(r.expected_formula), "yes" if r.composition_matches else "no", "yes" if r.qc_passed else "no", f"{r.density_built:.2f}", f"{r.density_qmof:.2f}", f"{r.gap_PBE_qmof:.2f}", str(int(r.n_qmof))] for r in known.itertuples()]
T["known"] = dict(header=["Known MOF", "Formula (cell)", "Composition reproduced", "Passes QC", "Density built (g cm^{−3})", "Density QMOF", "QMOF PBE gap (eV)", "QMOF entries"],
                  rows=rows, widths=[1.3, 1.5, 0.9, 0.6, 0.9, 0.8, 0.9, 0.7])


# S: acquisition parameters per iteration
import json as _json
rows = [["0", "0", "131", "11", "2 per net, 2 per metal", "none", "MOF-5, Zn_{2} cds (closed-shell references)", "1.0 / 0.3 / 0.5"]]
for k in (1, 2):
    s = _json.load(open(REPO / f"data/al/iter{k}/selection_rules.json"))
    excl = "; ".join([f"family {x}" for x in s["exclude_family"]] + [x.replace("M2(CO2)4-paddlewheel", "paddlewheel").replace("M4(CO2)8", "M4") for x in s["exclude_combo"]]) + f"; ≤ {s['max_atoms']} atoms"
    excl = excl.replace("M3(CO2)6", "M_{3}(CO_{2})_{6}").replace("M4:", "M_{4}(CO_{2})_{8}:")
    n_train = {1: 6, 2: 11}[k]
    rows.append([str(k), str(n_train), str(s["n_eligible"]), str(s["n"]), f"{s['max_per_topology']} per net, {s['max_per_metal']} per metal, {s['max_per_substituent']} per substituent", excl,
                 ", ".join(s["forced"]).replace("cds__Mn-N288__short-F4", "Mn_{2} cds F_{4}-BDC (anchor of the metal series)") or "none", f"{s['kappa']} / {s['alpha']} / {s['beta']}"])
T["acquisition"] = dict(header=["Iteration", "Labels in training", "Eligible candidates", "Batch", "Diversity caps", "Eligibility rules", "Forced", "κ / α / β"],
                        rows=rows, widths=[0.6, 0.7, 0.8, 0.5, 1.6, 2.4, 1.6, 0.8], align=["l", "l", "l", "l", "l", "l", "l", "l"])
T["hyper"] = dict(header=["Parameter", "Value", "Note"], rows=[
    ["Regressor", "ExtraTreesRegressor (scikit-learn 1.9)", "feature set C, 64 descriptors"],
    ["n_estimators", "300", "per ensemble member"],
    ["max_features", "sqrt", ""],
    ["min_samples_leaf", "2", ""],
    ["Ensemble", "5 members, seeds 42, 123, 456, 789, 2024", "μ = mean of member means"],
    ["Uncertainty σ", "standard deviation over the 1,500 trees of the five members", "spread of member means (0.03 eV) also reported"],
    ["Benchmark split", "80/20, stratified by gap < 1 eV, random_state = 42", "same split for all models and feature sets"],
    ["Missing values", "median imputation", ""],
    ["OOD flag", "density, PLD, n_atoms, metal fraction or C per atom outside the 5–95 % range of QMOF", "89 of 131 frameworks flagged"],
], widths=[1.4, 3.0, 2.4], align=["l", "l", "l"])




# S: robustness of the headline numbers to the six convergence exceptions
dft_ = pd.read_csv(REPO / "data/dft/dft_results.csv").set_index("name")
lab2 = lab.copy()
lab2["scf_rule"] = lab2.name.map(dft_.relax_scf_checked_by_static).fillna(False).astype(bool)
lab2["exception"] = lab2.relax_stopped.astype(bool) | lab2.scf_rule
lab2["open"] = ~lab2.metal.isin(["Zn", "Mg"])
rows = []
for name_, g in (("All 18 labels", lab2), ("Without the six exceptions (five stopped relaxations, one SCF checked by the static run)", lab2[~lab2.exception])):
    o, c = g[g.open], g[~g.open]
    rows.append([name_, str(len(g)), str(len(o)), f"{o.error_prospective.abs().mean():.2f}", f"{o.error_prospective.mean():+.2f}",
                 f"{int((o.gap_eV < 1).sum())}/{len(o)}", f"{int((o.gap_eV < 0.5).sum())}/{len(o)}", str(len(c)), f"{c.error_prospective.abs().mean():.2f}", f"{c.error_prospective.mean():+.2f}"])
T["robust"] = dict(header=["Label set", "n", "Open-shell n", "MAE open (eV)", "Bias open (eV)", "Open < 1 eV", "Open < 0.5 eV", "Zn_{4}O n", "MAE Zn_{4}O (eV)", "Bias Zn_{4}O (eV)"],
                   rows=rows, widths=[2.6, 0.4, 0.7, 0.8, 0.8, 0.8, 0.8, 0.6, 0.8, 0.8], align=["l"])
S_extra = {}
def _h(s_, n_, col):
    return float(ra[(ra.labels == n_) & (ra.strategy == s_)][col].iloc[0])
S_extra["retro_budget"], S_extra["retro_end"] = RETRO_BUDGET, RETRO_END
for s_ in ["random", "greedy", "lcb", "paper", "paper_static", "uncertainty", "expected_random"]:
    S_extra[f"hits_{s_}"] = _h(s_, RETRO_BUDGET, "hits_lt_0p5")
    S_extra[f"hits1_{s_}"] = _h(s_, RETRO_BUDGET, "hits_lt_1p0")
    S_extra[f"recall_{s_}"] = _h(s_, RETRO_BUDGET, "recall_top10")
    S_extra[f"hitsend_{s_}"] = _h(s_, RETRO_END, "hits_lt_0p5")
S_extra["hits_paper_sd"] = _h("paper", RETRO_BUDGET, "hits_lt_0p5_sd")
S_extra["hits_random_sd"] = _h("random", RETRO_BUDGET, "hits_lt_0p5_sd")
acc_ = pd.read_csv(_ra_path.with_name("retro_al_accuracy_summary.csv"))
base_ = float(acc_[(acc_.strategy == "paper") & (acc_.n_before == 0)].test_MAE_mean.iloc[0])
S_extra["dmae_paper_budget_meV"] = 1000 * (float(acc_[(acc_.strategy == "paper") & (acc_.n_before == RETRO_BUDGET)].test_MAE_mean.iloc[0]) - base_)
S_extra["dmae_paper_end_meV"] = 1000 * (float(acc_[(acc_.strategy == "paper") & (acc_.n_before == RETRO_END)].test_MAE_mean.iloc[0]) - base_)
S_extra["dmae_random_end_meV"] = 1000 * (float(acc_[(acc_.strategy == "random") & (acc_.n_before == RETRO_END)].test_MAE_mean.iloc[0]) - base_)
S_extra["retro_source"] = str(_ra_path.parent.name)

def _fix(x):
    if isinstance(x, list): return [_fix(v) for v in x]
    if isinstance(x, float) and np.isnan(x): return "–"
    if x is None: return "–"
    return x
for _k in T: T[_k]["rows"] = _fix(T[_k]["rows"])
json.dump(T, open(OUT / "tables.json", "w"), indent=1, ensure_ascii=False)

# a few scalar numbers used in the text
S = {}
S["n_built"], S["n_passed"] = len(cat), len(pas)
S["n_open"], S["n_closed"] = int(pas.open_shell.sum()), int((~pas.open_shell).sum())
S["qc_rejected"] = int((~cat.passed).sum())
S["mae_open"], S["mae_closed"] = shell["open-shell"].MAE_eV, shell["closed-shell"].MAE_eV
S["bias_open"], S["bias_closed"] = shell["open-shell"].bias_eV, shell["closed-shell"].bias_eV
S["core_h_total"] = float(comp.core_h.sum())
S["core_h_failed"] = float(ex.core_h.sum())
S.update(S_extra)
json.dump(S, open(OUT / "scalars.json", "w"), indent=1)
print(json.dumps(S, indent=1))

# image sizes
sizes = {}
for p in (REPO / "results/figures").glob("*.png"):
    with Image.open(p) as im:
        sizes[f"../results/figures/{p.name}"] = im.size
json.dump(sizes, open(OUT / "image_sizes.json", "w"), indent=1)
print(len(sizes), "images")
