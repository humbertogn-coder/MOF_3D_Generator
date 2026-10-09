#!/usr/bin/env python3
"""Train the ExtraTrees ensemble (QMOF + accepted DFT labels) and predict the catalogue.

Usage
-----
    # iteration 0 (QMOF only)
    python scripts/05_train_predict.py --iteration 0
    # iteration k: add every DFT label accepted so far (label_ok == True)
    python scripts/05_train_predict.py --iteration 1 --dft data/dft/dft_results.csv

Outputs: data/al/iter<k>/predictions.csv, data/al/iter<k>/model_info.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mofal.features import FEATURE_SET_C, featurize_candidates, featurize_qmof  # noqa: E402
from mofal.surrogate import SEEDS, ET_PARAMS, flag_ood, ood_ranges, train_ensemble  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qmof", default=str(ROOT / "data/external/qmof.csv"))
    ap.add_argument("--candidates", default=str(ROOT / "data/catalogue/passed.csv"))
    ap.add_argument("--zeopp", default=str(ROOT / "data/catalogue/zeopp.csv"))
    ap.add_argument("--dft", default=None, help="dft_results.csv with label_ok column")
    ap.add_argument("--iteration", type=int, required=True)
    ap.add_argument("--out", default=str(ROOT / "data/al"))
    args = ap.parse_args()
    out = Path(args.out) / f"iter{args.iteration}"
    out.mkdir(parents=True, exist_ok=True)

    qmof = featurize_qmof(pd.read_csv(args.qmof, low_memory=False))
    qmof["source"] = "QMOF"
    cand = featurize_candidates(pd.read_csv(args.candidates), pd.read_csv(args.zeopp))
    missing = cand[FEATURE_SET_C].isna().any(axis=1)
    if missing.any():
        print(f"WARNING: {missing.sum()} candidates without full descriptors (no Zeo++?) are skipped")
        cand = cand[~missing].copy()

    train = qmof[FEATURE_SET_C + ["gap_PBE", "source"]].copy()
    n_dft = 0
    if args.dft:
        dft = pd.read_csv(args.dft)
        dft = dft[dft["label_ok"].astype(bool) & (dft["iteration"] < args.iteration)]
        lab = cand.merge(dft[["name", "gap_eV"]], on="name")
        lab = lab.assign(gap_PBE=lab.gap_eV, source="DFT")[FEATURE_SET_C + ["gap_PBE", "source"]]
        n_dft = len(lab)
        train = pd.concat([train, lab], ignore_index=True)
    print(f"Training rows: QMOF {len(qmof):,} + DFT {n_dft}")

    ens = train_ensemble(train)
    pred = ens.predict(cand)
    ood = flag_ood(cand, ood_ranges(qmof))
    res = pd.concat([cand, pred, ood], axis=1)
    res["iteration"] = args.iteration
    keep = ["name", "node_family", "topology", "node_id", "metal", "open_shell", "linker_label",
            "linker_family", "n_rings", "substituent", "formula", "n_atoms", "density", "PLD_A",
            "LCD_A", "mu", "sigma_seed", "sigma_tree", "is_ood", "ood_reasons", "iteration"]
    res[keep].to_csv(out / "predictions.csv", index=False)
    info = dict(iteration=args.iteration, n_qmof=len(qmof), n_dft=n_dft, seeds=list(SEEDS),
                et_params=ET_PARAMS, features="C (64)", sklearn=sklearn.__version__,
                n_candidates=len(res), n_ood=int(res.is_ood.sum()),
                mean_sigma_seed=float(res.sigma_seed.mean()), mean_sigma_tree=float(res.sigma_tree.mean()))
    (out / "model_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
