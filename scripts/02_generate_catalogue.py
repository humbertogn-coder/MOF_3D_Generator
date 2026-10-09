#!/usr/bin/env python3
"""Generate the v3 candidate catalogue with PORMAKE and apply structural QC.

Usage
-----
    python scripts/02_generate_catalogue.py --out data/catalogue

Outputs (in --out)
------------------
    cifs_all/<name>.cif        every structure that PORMAKE could build
    cifs/<name>.cif            structures that pass all QC filters
    catalogue.csv              one row per enumerated candidate with QC results
    passed.csv                 subset that passed QC
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mofal.catalogue import enumerate_candidates  # noqa: E402
from mofal.qc import check_atoms  # noqa: E402

import pormake as pm  # noqa: E402

RHO_MIN, RHO_MAX = 0.20, 3.0     # g cm-3
A_MIN = 5.0                      # A
N_ATOMS_MAX = 600                # DFT tractability
RMSD_MAX = 0.30                  # PORMAKE node-vertex RMSD (A)


def bb_composition(bb) -> Counter:
    return Counter(s for s in bb.atoms.get_chemical_symbols() if s != "X")


def density(atoms) -> float:
    return float(atoms.get_masses().sum() / 6.02214076e23 / (atoms.cell.volume * 1e-24))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/catalogue")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "cifs_all").mkdir(parents=True, exist_ok=True)
    (out / "cifs").mkdir(parents=True, exist_ok=True)

    db = pm.Database()
    builder = pm.Builder()
    locator = pm.Locator()
    quiet = contextlib.redirect_stdout(io.StringIO())
    rows = []
    for cand in enumerate_candidates():
        row = dict(cand)
        with contextlib.redirect_stdout(io.StringIO()):
            topo = db.get_topo(cand["topology"])
            node = db.get_bb(cand["node_id"])
            edge = db.get_bb(cand["edge_id"])
        rmsd = float(locator.calculate_rmsd(topo.unique_local_structures[0], node))
        row["node_rmsd_A"] = round(rmsd, 3)
        expected = Counter()
        for k, v in bb_composition(node).items():
            expected[k] += v * topo.n_nodes
        for k, v in bb_composition(edge).items():
            expected[k] += v * topo.n_edges
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                mof = builder.build_by_type(topology=topo, node_bbs={0: node},
                                            edge_bbs={(0, 0): edge})
        except Exception as exc:  # noqa: BLE001
            row.update(built=False, passed=False, qc_reasons=f"build failed: {type(exc).__name__}")
            rows.append(row)
            print(f"[build-fail] {cand['name']}", flush=True)
            continue
        atoms = mof.atoms
        q = check_atoms(atoms, pormake_bonds=mof.bonds, expected_formula=expected)
        rho = density(atoms)
        a, b, c = atoms.cell.cellpar()[:3]
        reasons = list(q.reasons)
        if rmsd > RMSD_MAX:
            reasons.append(f"node RMSD {rmsd:.2f} > {RMSD_MAX}")
        if not (RHO_MIN <= rho <= RHO_MAX):
            reasons.append(f"density {rho:.2f} outside [{RHO_MIN},{RHO_MAX}]")
        if min(a, b, c) < A_MIN:
            reasons.append("cell length < 5 A")
        if len(atoms) > N_ATOMS_MAX:
            reasons.append(f"n_atoms {len(atoms)} > {N_ATOMS_MAX}")
        row.update(built=True, formula=q.formula, n_atoms=q.n_atoms, density=round(rho, 4),
                   a=round(a, 3), b=round(b, 3), c=round(c, 3),
                   min_distance=q.min_distance, n_overlaps=q.n_overlaps,
                   n_nonbonded_contacts=q.n_nonbonded_contacts,
                   n_components=q.n_components, n_bad_H=q.n_bad_H, n_bad_C=q.n_bad_C,
                   composition_ok=q.composition_ok,
                   passed=not reasons, qc_reasons=";".join(reasons))
        mof.write_cif(str(out / "cifs_all" / f"{cand['name']}.cif"))
        if not reasons:
            mof.write_cif(str(out / "cifs" / f"{cand['name']}.cif"))
        rows.append(row)
        print(f"[{'PASS' if not reasons else 'fail'}] {cand['name']} {';'.join(reasons)[:90]}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "catalogue.csv", index=False)
    df[df.passed].to_csv(out / "passed.csv", index=False)
    print(f"\nEnumerated {len(df)}, built {int(df.built.sum())}, passed QC {int(df.passed.sum())}")
    print(df.groupby(["topology", "metal"]).passed.agg(["sum", "count"]).unstack(fill_value=0))


if __name__ == "__main__":
    main()
