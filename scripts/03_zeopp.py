#!/usr/bin/env python3
"""Geometric pore descriptors (Zeo++ via the ``pyzeo`` bindings).

LCD (Di) and PLD (Df) are computed with Zeo++ default (CCDC) radii and the
high-accuracy Voronoi decomposition (-ha, mode HA_MODE below), i.e. the
equivalent of ``network -ha OCC -res``. Density and volume come from the cell.

Usage:  python scripts/03_zeopp.py --cifs data/catalogue/cifs --out data/catalogue/zeopp.csv
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pandas as pd
from ase.io import read
from pyzeo.high_accuracy import high_accuracy_atomnet
from pyzeo.netstorage import AtomNetwork

# Zeo++ high-accuracy setting. "DEF" (-ha) is ~10x slower than "OCC"; on test
# structures OCC reproduces DEF LCD within 0.02 A and PLD within 0.15 A.
HA_MODE = "OCC"


def res(cif: Path):
    atmnet = AtomNetwork.read_from_CIF(str(cif))
    high_accuracy_atomnet(atmnet, HA_MODE)
    with tempfile.NamedTemporaryFile(suffix=".res") as fh:
        atmnet.calculate_free_sphere_parameters(fh.name)
        vals = open(fh.name).read().split()
    # file: <name> Di Df Dif
    return float(vals[1]), float(vals[2]), float(vals[3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cifs", default="data/catalogue/cifs")
    ap.add_argument("--out", default="data/catalogue/zeopp.csv")
    args = ap.parse_args()
    rows = []
    for cif in sorted(Path(args.cifs).glob("*.cif")):
        at = read(cif)
        di, df_, dif = res(cif)
        vol = at.cell.volume
        rho = at.get_masses().sum() / 6.02214076e23 / (vol * 1e-24)
        rows.append(dict(name=cif.stem, LCD_A=di, PLD_A=df_, LCD_along_path_A=dif,
                         vol_A3=vol, density_zeo=rho, n_atoms_zeo=len(at)))
        print(f"{cif.stem:45s} LCD={di:6.2f} PLD={df_:6.2f} rho={rho:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
