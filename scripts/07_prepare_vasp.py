#!/usr/bin/env python3
"""Write VASP relaxation inputs for a selected batch.

Usage
-----
    python scripts/07_prepare_vasp.py --batch data/al/iter0/selected_batch.csv \
        --cifs data/catalogue/cifs --out runs/iter0

Creates runs/iter0/<name>/relax/{POSCAR,INCAR,KPOINTS,POTCAR.spec}, static/{INCAR,KPOINTS}
and copies the
SLURM driver (hpc/run_relax_static.slrm). On the cluster, run hpc/make_potcar.sh in
each folder (needs your licensed PAW_PBE 52/54 set) and submit.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mofal.vasp import write_inputs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--cifs", default=str(ROOT / "data/catalogue/cifs"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--afm", action="store_true", help="alternate signs of initial moments")
    args = ap.parse_args()
    from pymatgen.core import Structure

    batch = pd.read_csv(args.batch)
    out = Path(args.out)
    for name in batch.name:
        s = Structure.from_file(Path(args.cifs) / f"{name}.cif")
        folder = out / name
        write_inputs(s, folder / "relax", stage="relax", afm=args.afm)
        write_inputs(s, folder / "static", stage="static", afm=args.afm)
        (folder / "static" / "POSCAR").unlink()   # replaced by relax/CONTCAR at run time
        shutil.copy(ROOT / "hpc/run_relax_static.slrm", folder / "run.slrm")
        mesh = open(folder / 'relax' / 'KPOINTS').read().splitlines()[3].split()
        print(f"{name}: {len(s)} atoms, k-mesh {'x'.join(mesh)}")
    batch.to_csv(out / "batch.csv", index=False)


if __name__ == "__main__":
    main()
