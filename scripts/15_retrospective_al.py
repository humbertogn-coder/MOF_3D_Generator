#!/usr/bin/env python3
"""Retrospective active-learning benchmark on QMOF (no new DFT).

Question: with the paper's surrogate and acquisition, and a DFT budget of a few small
batches, does active learning find low-gap frameworks faster than random selection,
and does retraining on the new labels improve the predictions for out-of-distribution
candidates?

Set-up (mirrors the prospective workflow)
  * Training set: QMOF structures inside the 5-95 % range of all five OOD descriptors
    (density, PLD, n_atoms, metal fraction, C per atom), i.e. the in-distribution part.
  * Candidate pool: QMOF structures outside that range (out of distribution, as 89/131 of
    the generated candidates are) with the candidates' chemistry: one metal from
    Mn, Fe, Co, Ni, Cu, Zn, Mg and otherwise only C, H, N, O, F.
    Each repeat draws a pool of 150 structures (the real catalogue has 131).
  * Labels: the QMOF PBE gap plays the role of the DFT result.
  * Batches: 11, then 6 per iteration (as in the prospective run; 23 labels after three
    batches), continued to 8 batches (53 labels) to show the trend.
  * Surrogate: ExtraTrees with the paper's settings (300 trees, max_features='sqrt',
    min_samples_leaf=2, feature set C). One member per repeat (seed varies) instead of
    five, to keep the cost down; sigma is the spread over its trees. New labels are added
    to the training set without weights, as in scripts/05_train_predict.py.

Strategies
  random       uniform random batch
  greedy       lowest predicted gap (mu)
  lcb          lowest mu - sigma
  paper        mu - kappa*sigma - alpha*novelty with the batch caps of mofal.acquisition
               (2 per metal, 2 per topology; QMOF topology where known, otherwise unique)
  paper_static the same ranking with the initial model only (no retraining): isolates the
               effect of retraining
  uncertainty  highest sigma (pure exploration)

Metrics per batch: low-gap hits (PBE < 0.5 and < 1.0 eV) among the labelled structures,
recall of the pool's 10 lowest gaps, MAE on the still-unlabelled pool, MAE on a fixed
external OOD test set (all other candidate-chemistry OOD structures; all and PBE < 1 eV),
and the prospective MAE of the batch itself (the metric reported per iteration in the
earlier manuscript).

Usage:  python scripts/15_retrospective_al.py [--repeats 20]
Output: results/retro_al/retro_al_runs.csv (appended per repeat; the run can be resumed)
        then python scripts/15_retrospective_al.py --summarize  -> summary CSV + figure
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mofal.acquisition import ALPHA, KAPPA, select_batch  # noqa: E402
from mofal.features import FEATURE_SET_C, featurize_qmof  # noqa: E402
from mofal.surrogate import ood_ranges, train_ensemble  # noqa: E402

METALS = {"Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Mg"}
LIGHT = {"C", "H", "N", "O", "F"}
SCHEDULE = (11, 6, 6, 6, 6, 6, 6, 6)
POOL_SIZE = 150
STRATEGIES = ("random", "greedy", "lcb", "paper", "paper_static", "uncertainty")
RETRAIN = {"random", "greedy", "lcb", "paper", "uncertainty"}
OUT = ROOT / "results/retro_al"


def elements(formula: str) -> set:
    return set(re.findall(r"([A-Z][a-z]?)", formula))


def split_qmof(q: pd.DataFrame):
    ranges = ood_ranges(q)
    ood = np.zeros(len(q), bool)
    for c, (lo, hi) in ranges.items():
        ood |= ~q[c].between(lo, hi).to_numpy()
    chem = q["info.formula"].map(lambda f: (elements(f) - LIGHT) <= METALS
                                 and len(elements(f) & METALS) == 1).to_numpy()
    train = q[~ood].copy()
    cand = q[ood & chem].copy()
    cand["metal"] = cand["info.formula"].map(lambda f: next(iter(elements(f) & METALS)))
    cand["topology"] = [t if t != "unknown" else f"unk_{i}" for i, t in zip(cand.index, cand["cat_topology"])]
    cand["name"] = cand["qmof_id"] if "qmof_id" in cand else cand.index.astype(str)
    return train, cand


def fit(train: pd.DataFrame, extra: pd.DataFrame | None, seed: int):
    data = train if extra is None or extra.empty else pd.concat([train, extra])
    return train_ensemble(data[FEATURE_SET_C + ["gap_PBE"]], seeds=(seed,), n_jobs=-1)


def choose(strategy, pred, labelled, n, rng):
    unl = pred.drop(index=labelled)
    if strategy == "random":
        return list(rng.choice(unl.index.to_numpy(), size=n, replace=False))
    if strategy == "greedy":
        return list(unl.sort_values("mu").index[:n])
    if strategy == "lcb":
        return list((unl.mu - KAPPA * unl.sigma_tree).sort_values().index[:n])
    if strategy == "uncertainty":
        return list(unl.sort_values("sigma_tree", ascending=False).index[:n])
    # paper acquisition: novelty from labels per topology, caps from mofal.acquisition
    n_topo = Counter(pred.loc[labelled, "topology"])
    novelty = unl.topology.map(lambda t: 1.0 / max(1, n_topo.get(t, 0)))
    unl = unl.assign(acquisition_score=unl.mu - KAPPA * unl.sigma_tree - ALPHA * novelty)
    picked = list(select_batch(unl, n).index)
    if len(picked) < n:   # caps exhausted (small pools): fill by score
        rest = unl.drop(index=picked).sort_values("acquisition_score").index
        picked += list(rest[: n - len(picked)])
    return picked


def evaluate(pred, pool, labelled, tpred, test):
    """Accuracy on the still-unlabelled pool and on a fixed external OOD test set
    (all other OOD structures with the target chemistry), which is not affected by
    which structures a strategy removed from the pool."""
    unl = pool.index.difference(labelled)
    err = (pred.loc[unl, "mu"] - pool.loc[unl, "gap_PBE"]).abs()
    low = pool.loc[unl, "gap_PBE"] < 1.0
    rho = spearmanr(pred.loc[unl, "mu"], pool.loc[unl, "gap_PBE"])[0] if len(unl) > 2 else np.nan
    terr = (tpred - test.gap_PBE).abs()
    tlow = test.gap_PBE < 1.0
    return {"pool_MAE": err.mean(), "pool_MAE_lowgap": err[low].mean() if low.any() else np.nan,
            "pool_spearman": rho, "test_MAE": terr.mean(), "test_MAE_lowgap": terr[tlow].mean(),
            "test_spearman": spearmanr(tpred, test.gap_PBE)[0]}


def run_repeat(rep: int, train: pd.DataFrame, cand: pd.DataFrame) -> pd.DataFrame:
    seed = 1000 + rep
    pool = cand.sample(POOL_SIZE, random_state=seed)
    test = cand.drop(index=pool.index)
    top10 = set(pool.nsmallest(10, "gap_PBE").index)
    m0 = fit(train, None, seed)
    pred0 = pd.concat([pool[["topology", "metal", "name"]], m0.predict(pool)], axis=1)
    tpred0 = m0.predict(test)["mu"]
    rows = []
    for strategy in STRATEGIES:
        rng = np.random.default_rng(seed)
        model, pred, tpred, labelled = m0, pred0, tpred0, []
        for it, n in enumerate(SCHEDULE):
            row = {"repeat": rep, "strategy": strategy, "iteration": it, "n_before": len(labelled)}
            row.update(evaluate(pred, pool, labelled, tpred, test))
            batch = choose(strategy, pred, labelled, n, rng)
            row["batch_MAE"] = (pred.loc[batch, "mu"] - pool.loc[batch, "gap_PBE"]).abs().mean()
            labelled += batch
            lab_gap = pool.loc[labelled, "gap_PBE"]
            row.update(n_after=len(labelled), hits_lt_0p5=int((lab_gap < 0.5).sum()),
                       hits_lt_1p0=int((lab_gap < 1.0).sum()),
                       recall_top10=len(top10 & set(labelled)) / 10,
                       mean_gap_selected=lab_gap.mean())
            rows.append(row)
            if strategy in RETRAIN:
                model = fit(train, pool.loc[labelled], seed)
                pred = pd.concat([pool[["topology", "metal", "name"]], model.predict(pool)], axis=1)
                tpred = model.predict(test)["mu"]
        final = {"repeat": rep, "strategy": strategy, "iteration": len(SCHEDULE), "n_before": len(labelled)}
        final.update(evaluate(pred, pool, labelled, tpred, test))
        rows.append(final)
    out = pd.DataFrame(rows)
    out["pool_n_lt_0p5"] = int((pool.gap_PBE < 0.5).sum())
    out["pool_n_lt_1p0"] = int((pool.gap_PBE < 1.0).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qmof", default=str(ROOT / "data/external/qmof.csv"))
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    runs = OUT / "retro_al_runs.csv"
    if args.summarize:
        return summarize(pd.read_csv(runs))
    q = featurize_qmof(pd.read_csv(args.qmof, low_memory=False))
    if "qmof_id" not in q and "name" in q:
        q["qmof_id"] = q["name"]
    train, cand = split_qmof(q)
    print(f"training {len(train):,} in-distribution QMOF; candidate set {len(cand):,} OOD, "
          f"{(cand.gap_PBE < 1).mean():.0%} with PBE < 1 eV", flush=True)
    done = set(pd.read_csv(runs).repeat) if runs.exists() else set()
    for rep in range(args.repeats):
        if rep in done:
            continue
        t = time.time()
        res = run_repeat(rep, train, cand)
        res.to_csv(runs, mode="a", header=not runs.exists(), index=False, lineterminator="\n")
        print(f"repeat {rep} done in {time.time() - t:.0f} s", flush=True)


# ------------------------------------------------------------------ summary + figure
LABELS = {"random": "Random", "greedy": "Greedy (lowest μ)", "lcb": "LCB (μ − σ)",
          "paper": "Paper acquisition (LCB + caps)", "paper_static": "Paper acquisition, no retraining",
          "uncertainty": "Uncertainty (highest σ)"}


def summarize(df: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel = df[df.iteration < len(SCHEDULE)]
    agg = (sel.groupby(["strategy", "n_after"])
              [["hits_lt_0p5", "hits_lt_1p0", "recall_top10", "mean_gap_selected", "batch_MAE"]]
              .agg(["mean", "std"]))
    agg.columns = ["_".join(c) for c in agg.columns]
    mae = (df.groupby(["strategy", "n_before"])[["pool_MAE", "pool_MAE_lowgap", "pool_spearman",
                                                 "test_MAE", "test_MAE_lowgap", "test_spearman"]]
             .agg(["mean", "std"]))
    mae.columns = ["_".join(c) for c in mae.columns]
    agg.reset_index().to_csv(OUT / "retro_al_selection_summary.csv", index=False,
                             float_format="%.3f", lineterminator="\n")
    mae.reset_index().to_csv(OUT / "retro_al_accuracy_summary.csv", index=False,
                             float_format="%.3f", lineterminator="\n")
    # headline table at the prospective budget (23 labels) and at 53 labels
    pool_rate = df.groupby("repeat")[["pool_n_lt_0p5", "pool_n_lt_1p0"]].first().mean() / POOL_SIZE
    head = []
    for budget in (23, 53):
        for s in STRATEGIES:
            a = sel[(sel.strategy == s) & (sel.n_after == budget)]
            b = df[(df.strategy == s) & (df.n_before == budget)]
            head.append({"labels": budget, "strategy": s,
                         "hits_lt_0p5": a.hits_lt_0p5.mean(), "hits_lt_0p5_sd": a.hits_lt_0p5.std(),
                         "hits_lt_1p0": a.hits_lt_1p0.mean(), "recall_top10": a.recall_top10.mean(),
                         "test_MAE": b.test_MAE.mean(), "test_MAE_lowgap": b.test_MAE_lowgap.mean(),
                         "pool_MAE": b.pool_MAE.mean(), "pool_MAE_lowgap": b.pool_MAE_lowgap.mean()})
        head.append({"labels": budget, "strategy": "expected_random",
                     "hits_lt_0p5": budget * pool_rate.pool_n_lt_0p5,
                     "hits_lt_1p0": budget * pool_rate.pool_n_lt_1p0, "recall_top10": budget / POOL_SIZE})
    head = pd.DataFrame(head)
    head.to_csv(OUT / "retro_al_headline.csv", index=False, float_format="%.3f", lineterminator="\n")
    base = df[df.n_before == 0].groupby("repeat")[["pool_MAE", "test_MAE", "test_MAE_lowgap"]].first()
    print(head.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"initial model ({df.repeat.nunique()} repeats): pool MAE {base.pool_MAE.mean():.3f}, "
          f"external OOD test MAE {base.test_MAE.mean():.3f} (PBE < 1 eV: {base.test_MAE_lowgap.mean():.3f}) eV")
    figure(df, plt)


def figure(df, plt):
    MM = 1 / 25.4
    INK, INK2 = "#0b0b0b", "#52514e"
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#e6e5e1", "grid.linewidth": 0.5, "pdf.fonttype": 42,
        "savefig.bbox": "tight", "savefig.dpi": 300,
    })
    style = {"paper": ("#2a78d6", "-", "o"), "paper_static": ("#2a78d6", ":", None),
             "greedy": ("#1baf7a", "-", "s"), "lcb": ("#7a4fc9", "-", "^"),
             "uncertainty": ("#eb6834", "-", "D"), "random": ("#8a8984", "--", None)}
    order = ("paper", "paper_static", "greedy", "lcb", "uncertainty", "random")
    sel = df[df.iteration < len(SCHEDULE)]
    base = df[df.n_before == 0].set_index(["strategy", "repeat"]).test_MAE
    acc = df.assign(dMAE_meV=1000 * (df.test_MAE - base.reindex(
        pd.MultiIndex.from_frame(df[["strategy", "repeat"]])).to_numpy()))
    fig, axes = plt.subplots(1, 3, figsize=(190 * MM, 64 * MM))
    panels = [(axes[0], sel, "n_after", "hits_lt_0p5", "Low-gap hits (PBE < 0.5 eV)"),
              (axes[1], sel, "n_after", "recall_top10", "Recall of the 10 lowest gaps"),
              (axes[2], acc, "n_before", "dMAE_meV", "Change in OOD test-set MAE (meV)")]
    for ax, data, x, y, ylab in panels:
        for s in order:
            g = data[data.strategy == s].groupby(x)[y]
            m, sd = g.mean(), g.std()
            c, ls, mk = style[s]
            ax.plot(m.index, m.values, color=c, ls=ls, marker=mk, ms=3, lw=1.5, label=LABELS[s])
            if s in ("paper", "random"):   # spread (1 s.d. over repeats) for the two reference curves
                ax.fill_between(m.index, m - sd, m + sd, color=c, alpha=0.12, lw=0)
        ax.axvline(23, color=INK2, lw=0.8, ls=(0, (2, 2)))
        ax.set_xlabel("DFT labels acquired")
        ax.set_ylabel(ylab)
    axes[2].axhline(0, color=INK2, lw=0.8)

    axes[0].text(23.5, axes[0].get_ylim()[1] * 0.98, "budget of\nthis work", va="top", fontsize=6.5, color=INK2)
    for k, ax in enumerate(axes):
        ax.text(-0.02, 1.03, "abc"[k], transform=ax.transAxes, fontweight="bold", fontsize=9, ha="right")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fdir = ROOT / "results/figures"
    fig.savefig(fdir / "fig_retrospective_al.pdf")
    fig.savefig(fdir / "fig_retrospective_al.png")
    print("saved fig_retrospective_al")


if __name__ == "__main__":
    main()
