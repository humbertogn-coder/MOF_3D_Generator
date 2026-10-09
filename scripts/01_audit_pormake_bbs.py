#!/usr/bin/env python3
"""Audit of the PORMAKE building-block library used to design the catalogue.

For every node: metals, connection number, number of carboxylate groups, extra
O/OH/H2O, net anionic charge and implied average metal oxidation state.
For every 2-c edge: composition, whether it carries its own carboxylate groups
(which would be duplicated by carboxylate-containing nodes), and the angle
between the two connection vectors (180 deg = para/linear).

Usage:  python scripts/01_audit_pormake_bbs.py --out data/pormake_audit
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pormake as pm
from ase.neighborlist import natural_cutoffs, neighbor_list

NONMETALS = {"C", "H", "N", "O", "F", "S", "Cl", "Br", "I", "P", "B", "Se", "Si", "X"}


def quiet_bb(db, name):
    with contextlib.redirect_stdout(io.StringIO()):
        return db.get_bb(name)


def audit_node(bb):
    a = bb.atoms
    s = np.array(a.get_chemical_symbols())
    i, j = neighbor_list("ij", a, cutoff=natural_cutoffs(a, mult=1.2))
    nb = {k: s[j[i == k]].tolist() for k in range(len(a))}
    metals = [x for x in s if x not in NONMETALS]
    n_carbox = sum(1 for k in np.where(s == "C")[0] if nb[k].count("O") == 2 and "X" in nb[k])
    q, extra = -n_carbox, []
    for k in np.where(s == "O")[0]:
        nonmet = [x for x in nb[k] if x in NONMETALS and x != "X"]
        if any(x in ("C", "N", "S", "P") for x in nonmet):
            continue
        nh = nonmet.count("H")
        q -= {0: 2, 1: 1}.get(nh, 0)
        extra.append({0: "O", 1: "OH"}.get(nh, "H2O"))
    return dict(metals="/".join(sorted(set(metals))), n_metal=len(metals),
                n_carboxylate=n_carbox, extra_O=" ".join(f"{extra.count(t)}{t}" for t in sorted(set(extra))),
                anion_charge=q, implied_ox=round(-q / len(metals), 2) if metals else np.nan,
                carboxylate_only=n_carbox == bb.n_connection_points)


def audit_edge(bb):
    a = bb.atoms
    s = np.array(a.get_chemical_symbols())
    p = a.positions
    xi = np.where(s == "X")[0]
    i, j = neighbor_list("ij", a, cutoff=natural_cutoffs(a, mult=1.2))
    nb = {k: s[j[i == k]].tolist() for k in range(len(a))}
    carbox = sum(1 for k in np.where(s == "C")[0] if nb[k].count("O") == 2)
    ang = np.nan
    if len(xi) == 2:
        anchors = [int(np.argmin(np.where(s != "X", np.linalg.norm(p - p[x], axis=1), 1e9))) for x in xi]
        v1, v2 = p[xi[0]] - p[anchors[0]], p[xi[1]] - p[anchors[1]]
        ang = float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2), -1, 1))))
    return dict(has_carboxylate=carbox > 0, n_carboxylate=carbox, angle_X_vectors_deg=round(ang, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/pormake_audit")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    db = pm.Database()
    bbdir = Path(pm.__file__).parent / "database" / "bbs"
    nodes, edges = [], []
    for f in sorted(os.listdir(bbdir)):
        name = f[:-4]
        try:
            bb = quiet_bb(db, name)
        except Exception:  # noqa: BLE001
            continue
        comp = pd.Series([x for x in bb.atoms.get_chemical_symbols() if x != "X"]).value_counts()
        formula = "".join(f"{k}{v}" for k, v in sorted(comp.items()))
        base = dict(name=name, n_conn=bb.n_connection_points, formula=formula,
                    n_atoms=int(comp.sum()))
        if name.startswith("N"):
            nodes.append({**base, **audit_node(bb)})
        else:
            edges.append({**base, **audit_edge(bb)})
    pd.DataFrame(nodes).to_csv(out / "nodes.csv", index=False)
    pd.DataFrame(edges).to_csv(out / "edges.csv", index=False)
    print(f"nodes: {len(nodes)}  edges: {len(edges)}  -> {out}")


if __name__ == "__main__":
    main()
