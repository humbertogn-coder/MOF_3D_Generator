#!/usr/bin/env python3
"""Surrogate-level chemistry analysis of the catalogue (no DFT involved).

All effects are paired comparisons that change ONE axis and hold the others fixed:

* metal      : same node family + topology + linker, metal vs the Zn member of the
               same family (closed-shell reference)
* substituent: same node + topology + linker length, substituent vs H
* length     : same node + topology + substituent-free linker, BPDC-H vs BDC-H
* topology   : same paddlewheel node + linker, cds vs nbo

These are trends of the surrogate model, not DFT results, and are labelled as such.

Usage:  python scripts/10_surrogate_analysis.py --iteration 0
Outputs (results/surrogate/): pairs_<axis>.csv, summary_<axis>.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def paired(df, key_cols, axis_col, ref_value, label):
    rows = []
    for key, g in df.groupby(key_cols):
        ref = g[g[axis_col] == ref_value]
        if ref.empty:
            continue
        r = ref.iloc[0]
        for _, x in g[g[axis_col] != ref_value].iterrows():
            rows.append({**dict(zip(key_cols, key if isinstance(key, tuple) else (key,))),
                         "axis": label, "value": x[axis_col], "reference": ref_value,
                         "name": x["name"], "ref_name": r["name"],
                         "mu": x["mu"], "mu_ref": r["mu"], "delta_mu": x["mu"] - r["mu"]})
    return pd.DataFrame(rows)


def summarise(p):
    return (p.groupby("value").delta_mu.agg(n="count", mean="mean", std="std", median="median",
                                            min="min", max="max").round(3).reset_index())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results/surrogate"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / f"data/al/iter{args.iteration}/predictions.csv")

    sets = {
        "metal": paired(df, ["node_family", "topology", "linker_label"], "metal", "Zn", "metal"),
        "substituent": paired(df, ["node_id", "topology", "linker_family"], "substituent", "H", "substituent"),
        "length": paired(df[df.substituent == "H"], ["node_id", "topology"], "linker_family", "BDC", "length"),
        "topology": paired(df[df.node_family == "M2(CO2)4-paddlewheel"], ["node_id", "linker_label"],
                           "topology", "nbo", "topology"),
    }
    for k, p in sets.items():
        p.to_csv(out / f"pairs_{k}.csv", index=False)
        s = summarise(p) if len(p) else pd.DataFrame()
        s.to_csv(out / f"summary_{k}.csv", index=False)
        print(f"\n== {k} (surrogate mu, eV)\n{s.to_string(index=False)}")

    by = df.groupby(["node_family", "metal"]).mu.agg(["count", "mean", "std", "min", "max"]).round(3)
    by.to_csv(out / "mu_by_family_metal.csv")
    print("\n", by)
    print(f"\nOOD fraction: {df.is_ood.mean():.2f}; predicted gap <= 1 eV: {(df.mu <= 1).sum()} / {len(df)}")


if __name__ == "__main__":
    main()
