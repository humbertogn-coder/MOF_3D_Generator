#!/usr/bin/env python3
"""GFN1-xTB band gaps of QMOF structures: periodic vs. finite-cluster treatment.

Re-examines the exploratory xTB screen of the original workflow. That screen ran
xtb on the unit-cell atoms written as an .xyz file, i.e. as a *finite cluster*:
every bond crossing the cell boundary is cut, producing dangling-bond states
and near-zero HOMO-LUMO gaps. Here the same 2,000 QMOF structures (tiered subset,
seed 42; data/xtb/qmof_validation_subset.csv) are computed with tblite GFN1-xTB

  * periodic   : 3D periodic boundary conditions (Gamma point, Ewald electrostatics)
  * molecular  : the unit-cell atoms as an isolated cluster (reproduces the old setup)

Settings as in the original screen: neutral, electronic temperature 500 K,
500 SCF iterations. Unpaired electrons: high-spin count of the 3d metals
(Mn 5, Fe 4, Co 3, Ni 2, Cu 1 per atom), adjusted by one if the parity does not
match the electron count.

Usage:
    python scripts/13_xtb_validation.py --cifs data/external/qmof_xtb_subset/cifs \
        --out results/xtb/xtb_gfn1_gaps.csv --mode both
The output is appended row by row, so the script can be interrupted and resumed.
"""
from __future__ import annotations

import argparse
import csv
import time
import warnings
from pathlib import Path

import numpy as np
from ase.io import read

warnings.filterwarnings("ignore")
ANG2BOHR = 1 / 0.52917721067
KB_HA = 3.166811563e-6
HA2EV = 27.211386245988
HS_UNPAIRED = {"Mn": 5, "Fe": 4, "Co": 3, "Ni": 2, "Cu": 1}
FIELDS = ["qmof_id", "mode", "n_atoms", "uhf", "gap_eV", "homo_eV", "lumo_eV", "seconds", "status"]


def unpaired(atoms) -> int:
    u = sum(HS_UNPAIRED.get(s, 0) for s in atoms.get_chemical_symbols())
    if (sum(atoms.numbers) - u) % 2:
        u = u + 1 if u == 0 else u - 1
    return u


def gap(res):
    e = np.atleast_2d(res.get("orbital-energies"))
    occ = np.atleast_2d(res.get("orbital-occupations")).reshape(e.shape)
    thr = 0.5 * occ.max()
    homo = max(ei[oi > thr].max() for ei, oi in zip(e, occ) if (oi > thr).any())
    lumo = min(ei[oi <= thr].min() for ei, oi in zip(e, occ) if (oi <= thr).any())
    return (lumo - homo) * HA2EV, homo * HA2EV, lumo * HA2EV


def run(atoms, periodic: bool, uhf: int, etemp=500.0, max_iter=500):
    from tblite.interface import Calculator
    kw = {"uhf": uhf}
    if periodic:
        kw.update(lattice=atoms.cell.array * ANG2BOHR, periodic=np.array([True, True, True]))
    calc = Calculator("GFN1-xTB", atoms.numbers, atoms.positions * ANG2BOHR, **kw)
    calc.set("temperature", etemp * KB_HA)
    calc.set("max-iter", max_iter)
    calc.set("verbosity", 0)
    return gap(calc.singlepoint())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cifs", required=True)
    ap.add_argument("--out", default="results/xtb/xtb_gfn1_gaps.csv")
    ap.add_argument("--mode", choices=["periodic", "molecular", "both"], default="both")
    ap.add_argument("--max_atoms", type=int, default=800)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(r["qmof_id"], r["mode"]) for r in csv.DictReader(open(out))}
    modes = ["periodic", "molecular"] if args.mode == "both" else [args.mode]
    new = not out.exists()
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        if new:
            w.writeheader()
        for cif in sorted(Path(args.cifs).glob("*.cif")):
            qid = cif.stem
            atoms = None
            for mode in modes:
                if (qid, mode) in done:
                    continue
                if atoms is None:
                    atoms = read(cif)
                    atoms.wrap()
                row = {"qmof_id": qid, "mode": mode, "n_atoms": len(atoms), "uhf": unpaired(atoms)}
                if len(atoms) > args.max_atoms:
                    row.update(status="skipped_size")
                else:
                    t = time.time()
                    try:
                        g, h, l = run(atoms, mode == "periodic", row["uhf"])
                        row.update(gap_eV=round(g, 4), homo_eV=round(h, 4), lumo_eV=round(l, 4), status="ok")
                    except Exception as exc:  # noqa: BLE001
                        row.update(status=f"error: {str(exc)[:80]}")
                    row["seconds"] = round(time.time() - t, 1)
                w.writerow(row)
                fh.flush()
                print(qid, mode, row.get("gap_eV"), row["status"], row.get("seconds"), flush=True)


if __name__ == "__main__":
    main()
