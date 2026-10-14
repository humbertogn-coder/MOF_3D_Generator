#!/usr/bin/env python3
"""Select the next DFT batch with the multi-objective LCB acquisition function.

Usage
-----
    python scripts/06_select_batch.py --iteration 0 --n 10 \
        --forced pcu__Zn-N16__short-H pcu__Mg-N406__short-H
    python scripts/06_select_batch.py --iteration 1 --n 6 --dft data/dft/dft_results.csv

Candidates already submitted to DFT (any status) are excluded; only accepted labels
(label_ok) count toward N_topo.

Eligibility rules learned from earlier iterations (written to selection_rules.json):
    --exclude_family "M3(CO2)6"                 node family whose frameworks collapsed in DFT
    --exclude_combo "M2(CO2)4-paddlewheel:cds:Zn"  node family:topology:metal that sheared
    --max_atoms 160                             DFT cost/stability limit
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mofal.acquisition import (ALPHA, BETA, KAPPA, MAX_PER_METAL,  # noqa: E402
                               MAX_PER_SUBSTITUENT, MAX_PER_TOPOLOGY, acquisition_score,
                               select_batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iteration", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--dft", default=None)
    ap.add_argument("--forced", nargs="*", default=[])
    ap.add_argument("--sigma", choices=["tree", "seed"], default="tree")
    ap.add_argument("--kappa", type=float, default=KAPPA)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--beta", type=float, default=BETA)
    ap.add_argument("--max_per_topology", type=int, default=MAX_PER_TOPOLOGY)
    ap.add_argument("--max_per_metal", type=int, default=MAX_PER_METAL)
    ap.add_argument("--max_per_substituent", type=int, default=MAX_PER_SUBSTITUENT)
    ap.add_argument("--exclude_family", nargs="*", default=[])
    ap.add_argument("--exclude_combo", nargs="*", default=[], help="node_family:topology:metal")
    ap.add_argument("--max_atoms", type=int, default=None)
    args = ap.parse_args()
    d = ROOT / "data/al" / f"iter{args.iteration}"
    pred = pd.read_csv(d / "predictions.csv")

    n_topo, done = {}, set()
    if args.dft:
        dft = pd.read_csv(args.dft)
        done = set(dft.name)
        ok = dft[dft.label_ok.astype(bool)]
        n_topo = ok.topology.value_counts().to_dict()
    pool = pred[~pred.name.isin(done)].copy()
    n0 = len(pool)
    if args.exclude_family:
        pool = pool[~pool.node_family.isin(args.exclude_family)]
    for combo in args.exclude_combo:
        fam, topo, metal = combo.split(":")
        pool = pool[~((pool.node_family == fam) & (pool.topology == topo) & (pool.metal == metal))]
    if args.max_atoms:
        pool = pool[pool.n_atoms <= args.max_atoms]
    rules = dict(iteration=args.iteration, n=args.n, exclude_family=args.exclude_family,
                 exclude_combo=args.exclude_combo, max_atoms=args.max_atoms,
                 max_per_topology=args.max_per_topology, max_per_metal=args.max_per_metal,
                 max_per_substituent=args.max_per_substituent, kappa=args.kappa, alpha=args.alpha,
                 beta=args.beta, sigma=args.sigma, forced=args.forced, n_not_yet_computed=n0, n_eligible=len(pool),
                 n_topo_labels=n_topo)
    (d / "selection_rules.json").write_text(json.dumps(rules, indent=2) + "\n")
    pool["acquisition_score"] = acquisition_score(pool, n_topo, sigma_col=f"sigma_{args.sigma}",
                                                  kappa=args.kappa, alpha=args.alpha, beta=args.beta)
    pool.sort_values("acquisition_score").to_csv(d / "acquisition_ranking.csv", index=False)
    batch = select_batch(pool, args.n, args.max_per_topology, args.max_per_metal, forced=args.forced,
                         max_per_substituent=args.max_per_substituent)
    batch.to_csv(d / "selected_batch.csv", index=False)
    print(f"pool={len(pool)}  N_topo={n_topo}  sigma={args.sigma} kappa={args.kappa} alpha={args.alpha} beta={args.beta}")
    print(batch[["selection_rank", "name", "mu", f"sigma_{args.sigma}", "is_ood", "acquisition_score"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
