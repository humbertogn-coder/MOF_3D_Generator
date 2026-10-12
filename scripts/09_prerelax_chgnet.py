#!/usr/bin/env python3
"""Pre-relax PORMAKE structures with CHGNet before DFT (positions and cell).

Why: in iteration 0, several PORMAKE frameworks started with compressed bonds at the
node-linker junction (C-C down to 1.10 A) or in substituents (N-H 0.86 A). Initial DFT
forces reached 12-35 eV/A, the spin-polarised SCF did not converge (NELM reached at
every ionic step) and five relaxations made 3-6 ionic steps in 24 h. A universal
machine-learned potential removes these distortions in seconds to minutes per
structure; DFT then starts close to a minimum.

The pre-relaxed geometry is only a starting point: every label still comes from the
full PBE-D3(BJ) relaxation and static run.

Environment: CHGNet needs PyTorch, so it runs in a separate environment
(requirements-prerelax.txt: chgnet 0.4.2, model 0.3.0).

Usage
-----
    python scripts/09_prerelax_chgnet.py --batch data/al/iter1/selected_batch.csv \
        --cifs data/catalogue/cifs --out data/prerelaxed
    python scripts/09_prerelax_chgnet.py --names bcu__Co-N35__short-F4 ... --cifs ...

Writes <out>/<name>.vasp and appends one row per structure to <out>/prerelax_log.csv
(initial/final CHGNet max force, steps, volumes, compressed-bond counts before/after).
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mofal.qc import compressed_bonds  # noqa: E402

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--batch", help="CSV with a 'name' column")
    g.add_argument("--names", nargs="+")
    ap.add_argument("--cifs", default=str(ROOT / "data/catalogue/cifs"),
                    help="folder with <name>.cif (or <name>.vasp / POSCAR files)")
    ap.add_argument("--out", default=str(ROOT / "data/prerelaxed"))
    ap.add_argument("--fmax", type=float, default=0.05, help="eV/A")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    import torch
    from ase.io import read
    from chgnet.model import CHGNet, StructOptimizer
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    torch.set_num_threads(args.threads)
    names = list(pd.read_csv(args.batch)["name"]) if args.batch else args.names
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "prerelax_log.csv"
    model = CHGNet.load(verbose=False)
    opt = StructOptimizer(model=model, optimizer_class="FIRE", use_device="cpu")
    for name in names:
        src = next((p for p in (Path(args.cifs) / f"{name}.cif", Path(args.cifs) / f"{name}.vasp",
                                Path(args.cifs) / name / "relax" / "POSCAR") if p.exists()), None)
        if src is None:
            print(f"{name}: no structure found, skipped")
            continue
        s0 = Structure.from_file(src)
        t = time.time()
        f0 = np.linalg.norm(model.predict_structure(s0)["f"], axis=1).max()
        res = opt.relax(s0, fmax=args.fmax, steps=args.steps, relax_cell=True, verbose=False)
        s1 = res["final_structure"]
        f1 = np.linalg.norm(model.predict_structure(s1)["f"], axis=1).max()
        s1.to(filename=str(out / f"{name}.vasp"), fmt="poscar")
        cb0 = compressed_bonds(read(src) if src.suffix == ".cif" else AseAtomsAdaptor.get_atoms(s0))
        cb1 = compressed_bonds(AseAtomsAdaptor.get_atoms(s1))
        row = dict(name=name, n_atoms=len(s0), source=str(src.relative_to(ROOT) if src.is_relative_to(ROOT) else src),
                   fmax_initial=round(float(f0), 3), fmax_final=round(float(f1), 3),
                   steps=len(res["trajectory"].energies), converged=bool(f1 <= args.fmax + 1e-3),
                   vol_initial=round(s0.volume, 1), vol_final=round(s1.volume, 1),
                   vol_change_pct=round(100 * (s1.volume / s0.volume - 1), 2),
                   compressed_bonds_initial=len(cb0), compressed_bonds_final=len(cb1),
                   seconds=round(time.time() - t, 1))
        pd.DataFrame([row]).to_csv(log, mode="a", header=not log.exists(), index=False, lineterminator="\n")
        print(row, flush=True)


if __name__ == "__main__":
    main()
