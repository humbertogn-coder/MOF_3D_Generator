#!/usr/bin/env python3
"""Write VASP relaxation inputs for a selected batch.

Usage
-----
    python scripts/07_prepare_vasp.py --batch data/al/iter0/selected_batch.csv \
        --cifs data/catalogue/cifs --out runs/iter0 [--prerelaxed data/prerelaxed]

With --prerelaxed, <dir>/<name>.vasp (scripts/09_prerelax_chgnet.py) is used as the
starting geometry when present (protocol v2). A walltime file per structure sets the
SLURM time request from the system size, so fewer SUs are reserved per job.

Creates runs/iter0/<name>/relax/{POSCAR,INCAR,KPOINTS,POTCAR.spec}, static/{INCAR,KPOINTS},
a SLURM driver per structure (run.slrm) and the helper scripts in runs/iter0/
(submit_all.sh, make_potcar.sh, check_status.sh, collect_results.sh). See hpc/README.md.
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


def walltime(n_atoms: int) -> str:
    """SLURM time request (iteration-0 timings on Grace, 48 cores, protocol v2)."""
    return "12:00:00" if n_atoms <= 160 else "24:00:00"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--cifs", default=str(ROOT / "data/catalogue/cifs"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--afm", action="store_true", help="alternate signs of initial moments")
    ap.add_argument("--prerelaxed", default=None, help="folder with CHGNet-pre-relaxed <name>.vasp")
    args = ap.parse_args()
    from pymatgen.core import Structure

    batch = pd.read_csv(args.batch)
    out = Path(args.out)
    for name in batch.name:
        pre = Path(args.prerelaxed) / f"{name}.vasp" if args.prerelaxed else None
        s = Structure.from_file(pre if pre is not None and pre.exists() else Path(args.cifs) / f"{name}.cif")
        folder = out / name
        write_inputs(s, folder / "relax", stage="relax", afm=args.afm)
        write_inputs(s, folder / "static", stage="static", afm=args.afm)
        (folder / "static" / "POSCAR").unlink()   # replaced by relax/CONTCAR at run time
        shutil.copy(ROOT / "hpc/run_relax_static.slrm", folder / "run.slrm")
        (folder / "walltime").write_text(walltime(len(s)) + "\n")
        mesh = open(folder / 'relax' / 'KPOINTS').read().splitlines()[3].split()
        smesh = open(folder / 'static' / 'KPOINTS').read().splitlines()[3].split()
        print(f"{name}: {len(s)} atoms, relax k-mesh {'x'.join(mesh)}, static {'x'.join(smesh)}, "
              f"walltime {walltime(len(s))}")
    batch.to_csv(out / "batch.csv", index=False)
    for helper in ("make_potcar.sh", "make_potcars_all.sh", "submit_all.sh", "check_status.sh",
                   "collect_results.sh"):
        shutil.copy(ROOT / "hpc" / helper, out / helper)


if __name__ == "__main__":
    main()
