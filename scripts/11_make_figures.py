#!/usr/bin/env python3
"""Publication figures (vector PDF, Elsevier column widths) generated from data files.

Every figure is produced from CSVs in data/ and results/ by this script, so the
numbers in the figures are traceable to the pipeline outputs.

Usage:  python scripts/11_make_figures.py --iteration 0 [--only parity shap ...]
Output: results/figures/*.pdf (+ .png previews at 300 dpi)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MM = 1 / 25.4
SINGLE, ONEHALF, DOUBLE = 90 * MM, 140 * MM, 190 * MM
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2 = "#0b0b0b", "#52514e"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#e8f1fc", "#9cc3f0", BLUE, "#123f7a"])
DIV = LinearSegmentedColormap.from_list("div", ["#123f7a", BLUE, "#e9e8e4", ORANGE, "#7a2a0b"])
METAL_ORDER = ["Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Mg"]
TOPO_MARKERS = {"pcu": "o", "nbo": "s", "cds": "D", "dia": "^", "bcu": "v"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e6e5e1", "grid.linewidth": 0.5, "lines.linewidth": 1.5,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 300,
})


def save(fig, name):
    out = ROOT / "results/figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png")
    plt.close(fig)
    print("saved", name)


# ------------------------------------------------------------------ QMOF model
def qmof_holdout():
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from mofal.features import FEATURE_SET_C, featurize_qmof

    df = featurize_qmof(pd.read_csv(ROOT / "data/external/qmof.csv", low_memory=False))
    y = df.gap_PBE.values
    tr, te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42, stratify=(y < 1).astype(int))
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", ExtraTreesRegressor(n_estimators=500, max_features="sqrt", n_jobs=-1, random_state=42))])
    X = df[FEATURE_SET_C]
    pipe.fit(X.iloc[tr], y[tr])
    return df, X, y, tr, te, pipe


def fig_parity(state):
    from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
    df, X, y, tr, te, pipe = state
    p = pipe.predict(X.iloc[te])
    yt = y[te]
    pd.DataFrame({"qmof_id": df.iloc[te].qmof_id.values, "gap_PBE": yt, "pred": p}) \
        .to_csv(ROOT / "results/surrogate/qmof_holdout_predictions.csv", index=False)
    fig, ax = plt.subplots(figsize=(SINGLE, SINGLE * 0.9))
    hb = ax.hexbin(yt, p, gridsize=45, bins="log", cmap=SEQ, mincnt=1, linewidths=0)
    ax.plot([0, 6.5], [0, 6.5], color=INK2, lw=1, ls="--")
    ax.set(xlim=(0, 6.5), ylim=(0, 6.5), xlabel="QMOF PBE band gap (eV)", ylabel="Predicted band gap (eV)")
    ax.set_aspect("equal")
    ax.text(0.04, 0.96, f"$R^2$ = {r2_score(yt, p):.2f}\nMAE = {mean_absolute_error(yt, p):.2f} eV\n"
            f"RMSE = {root_mean_squared_error(yt, p):.2f} eV\nn = {len(yt):,}",
            transform=ax.transAxes, va="top", color=INK)
    cb = fig.colorbar(hb, ax=ax, shrink=0.8)
    cb.set_label("Structures per bin")
    save(fig, "fig_parity_qmof_holdout")


def fig_shap(state, n_sample=300):
    """Exact TreeSHAP for one surrogate member (300 trees, min_samples_leaf=2, seed 42)
    trained on the benchmark training split; n_sample random holdout structures."""
    import shap
    from sklearn.ensemble import ExtraTreesRegressor
    from mofal.surrogate import ET_PARAMS
    df, X, y, tr, te, pipe = state
    imp = pipe.named_steps["imp"]
    member = ExtraTreesRegressor(random_state=42, n_jobs=-1, **ET_PARAMS).fit(imp.transform(X.iloc[tr]), y[tr])
    Xs = X.iloc[te].sample(min(n_sample, len(te)), random_state=42)
    Xi = pd.DataFrame(imp.transform(Xs), columns=X.columns)
    sv = shap.TreeExplainer(member).shap_values(Xi, check_additivity=False)
    imp = pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values(ascending=False)
    imp.to_csv(ROOT / "results/surrogate/shap_mean_abs.csv", header=["mean_abs_shap_eV"])
    top = imp.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(SINGLE, SINGLE * 0.95))
    ax.barh(top.index, top.values, color=BLUE, height=0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Mean |SHAP| (eV), n = {len(Xs)} holdout structures")
    for i, v in enumerate(top.values):
        ax.text(v + top.max() * 0.01, i, f"{v:.3f}", va="center", color=INK2, fontsize=6)
    save(fig, "fig_shap_feature_importance")


# ------------------------------------------------------------------ catalogue
def fig_distributions(pred, qmof):
    cols = [("density", "Density (g cm$^{-3}$)"), ("PLD_A", "Pore-limiting diameter (Å)"),
            ("n_atoms", "Atoms per cell"), ("metal_frac", "Metal fraction")]
    from mofal.features import composition_features, parse_formula
    pred = pred.copy()
    pred["metal_frac"] = [composition_features(parse_formula(f), n)["metal_frac"]
                          for f, n in zip(pred.formula, pred.n_atoms)]
    fig, axs = plt.subplots(2, 2, figsize=(ONEHALF, ONEHALF * 0.62))
    for ax, (c, lab) in zip(axs.flat, cols):
        q = qmof[c].dropna()
        lo, hi = np.nanpercentile(np.r_[q, pred[c]], [0.5, 99.5])
        bins = np.linspace(lo, hi, 40)
        ax.hist(q.clip(lo, hi), bins=bins, density=True, color=BLUE, alpha=0.55, label=f"QMOF (n = {len(q):,})")
        ax.hist(pred[c].clip(lo, hi), bins=bins, density=True, color=ORANGE, alpha=0.75,
                label=f"Catalogue (n = {len(pred)})")
        ax.set_xlabel(lab)
        ax.set_ylabel("Density of structures")
    axs.flat[0].legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig_feature_distributions")


def _box(ax, groups, labels, color):
    bp = ax.boxplot(groups, tick_labels=labels, widths=0.55, patch_artist=True, showfliers=True,
                    medianprops=dict(color=INK, lw=1), flierprops=dict(marker="o", ms=2.5, mfc=GRAY, mec="none"))
    for b in bp["boxes"]:
        b.set(facecolor=color, alpha=0.75, edgecolor=INK2, lw=0.6)
    for i, g in enumerate(groups, 1):
        ax.text(i, ax.get_ylim()[0], f"n={len(g)}", ha="center", va="bottom", fontsize=6, color=INK2)


def fig_axes_effects():
    d = ROOT / "results/surrogate"
    pm, ps = pd.read_csv(d / "pairs_metal.csv"), pd.read_csv(d / "pairs_substituent.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(DOUBLE, DOUBLE * 0.36))
    mets = [m for m in METAL_ORDER if m in set(pm.value)]
    a1.axhline(0, color=INK2, lw=0.8)
    _box(a1, [pm[pm.value == m].delta_mu.values for m in mets], mets, BLUE)
    a1.set(ylabel=r"$\Delta\mu$ vs Zn analogue (eV)", xlabel="Metal (same node family, topology, linker)")
    a1.set_title("(a) Metal, surrogate prediction", loc="left")
    subs = [s for s in ["NH2", "F4", "Nring1", "F8", "Nring2"] if s in set(ps.value)]
    a2.axhline(0, color=INK2, lw=0.8)
    _box(a2, [ps[ps.value == s].delta_mu.values for s in subs], subs, ORANGE)
    a2.set(ylabel=r"$\Delta\mu$ vs H (eV)", xlabel="Substituent (same node, topology, linker length)")
    a2.set_title("(b) Linker substituent, surrogate prediction", loc="left")
    fig.tight_layout()
    save(fig, "fig_surrogate_axis_effects")


def fig_heatmap(pred):
    lab_order = ["short-H", "short-NH2", "short-F4", "short-Nring1", "long-H", "long-F8", "long-Nring1", "long-Nring2"]
    rows = []
    for fam, g in pred.groupby("node_family"):
        for m in [x for x in METAL_ORDER if x in set(g.metal)]:
            rows.append((f"{m} | {fam}", g[g.metal == m].groupby("linker_label").mu.mean().reindex(lab_order)))
    mat = pd.DataFrame({k: v for k, v in rows}).T
    fig, ax = plt.subplots(figsize=(DOUBLE, 0.2 * len(mat) + 0.9))
    norm = TwoSlopeNorm(vcenter=1.0, vmin=min(0.3, np.nanmin(mat.values)), vmax=max(2.0, np.nanmax(mat.values)))
    im = ax.imshow(mat.values, cmap=DIV, norm=norm, aspect="auto")
    ax.set_xticks(range(len(lab_order)), [l.replace("short-", "BDC ").replace("long-", "BPDC ") for l in lab_order],
                  rotation=30, ha="right")
    ax.set_yticks(range(len(mat)), mat.index)
    ax.grid(False)
    for (i, j), v in np.ndenumerate(mat.values):
        if np.isfinite(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(norm(v) - 0.5) > 0.3 else INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label("Mean predicted PBE gap (eV); centre = 1 eV")
    save(fig, "fig_surrogate_heatmap")


def fig_topology(pred):
    fams = list(pred.node_family.unique())
    fig, ax = plt.subplots(figsize=(DOUBLE, DOUBLE * 0.33))
    pos, ticks, groups, colors = 0, [], [], []
    for fam in fams:
        g = pred[pred.node_family == fam]
        for t in sorted(g.topology.unique()):
            for m in [x for x in METAL_ORDER if x in set(g.metal)]:
                v = g[(g.topology == t) & (g.metal == m)].mu.values
                if len(v):
                    groups.append(v)
                    ticks.append(f"{m}\n{t}")
                    colors.append(GRAY if m in ("Zn", "Mg") else BLUE)
        groups.append(np.array([]))
        ticks.append("")
        colors.append("none")
    bp = ax.boxplot(groups, widths=0.6, patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1))
    for b, c in zip(bp["boxes"], colors):
        b.set(facecolor=c, alpha=0.8, edgecolor=INK2, lw=0.6)
    ax.set_xticks(range(1, len(ticks) + 1), ticks, fontsize=6)
    ax.axhline(1.0, color=ORANGE, lw=0.8, ls="--")
    ax.set_ylabel("Predicted PBE gap (eV)")
    ax.set_title("Grey: closed-shell (Zn, Mg); blue: open-shell 3d metals. Dashed: 1 eV.", loc="left", color=INK2)
    save(fig, "fig_surrogate_topology_metal")


def fig_acquisition(iteration):
    d = ROOT / f"data/al/iter{iteration}"
    r = pd.read_csv(d / "acquisition_ranking.csv")
    sel = pd.read_csv(d / "selected_batch.csv")
    fig, ax = plt.subplots(figsize=(ONEHALF, ONEHALF * 0.55))
    for t, mk in TOPO_MARKERS.items():
        g = r[r.topology == t]
        ax.scatter(g.mu, g.sigma_tree, marker=mk, s=14, facecolor="none", edgecolor=GRAY, lw=0.7, label=t)
    s = r[r.name.isin(sel.name)]
    ax.scatter(s.mu, s.sigma_tree, marker="*", s=90, color=BLUE, edgecolor=INK, lw=0.5, label="selected", zorder=3)
    ax.axvline(1.0, color=ORANGE, lw=0.8, ls="--")
    ax.set(xlabel=r"Ensemble mean $\mu$ (eV)", ylabel=r"Ensemble spread $\sigma$ (eV)")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    save(fig, f"fig_acquisition_landscape_iter{iteration}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    want = lambda k: args.only is None or k in args.only  # noqa: E731
    (ROOT / "results/surrogate").mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(ROOT / f"data/al/iter{args.iteration}/predictions.csv")
    if want("parity") or want("shap") or want("dist"):
        state = qmof_holdout()
        if want("parity"):
            fig_parity(state)
        if want("shap"):
            fig_shap(state)
        if want("dist"):
            fig_distributions(pred, state[0])
    if want("axes"):
        fig_axes_effects()
    if want("heatmap"):
        fig_heatmap(pred)
    if want("topology"):
        fig_topology(pred)
    if want("acq") and (ROOT / f"data/al/iter{args.iteration}/acquisition_ranking.csv").exists():
        fig_acquisition(args.iteration)


if __name__ == "__main__":
    main()
