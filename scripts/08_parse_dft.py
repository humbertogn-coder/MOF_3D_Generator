#!/usr/bin/env python3
"""Parse finished VASP runs and decide which band gaps are accepted as labels.

A label is accepted (label_ok) only if
  * the relaxation reached ionic and electronic convergence,
  * the static run reached electronic convergence,
  * the final structure passes the same structural QC as the catalogue
    (no overlaps, no isolated atoms, a single connected framework), and
  * no non-metal atom carries a magnetic moment > 0.3 muB (radical flag);
    flagged runs are kept for manual inspection but not used for training.

Usage
-----
    python scripts/08_parse_dft.py --runs runs/iter0 --iteration 0 \
        --out data/dft/dft_results.csv          # appends / updates rows
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mofal.qc import check_atoms, magnetization_report  # noqa: E402
from mofal.vasp import parse_run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--iteration", type=int, required=True)
    ap.add_argument("--catalogue", default=str(ROOT / "data/catalogue/passed.csv"))
    ap.add_argument("--out", default=str(ROOT / "data/dft/dft_results.csv"))
    args = ap.parse_args()
    from pymatgen.io.ase import AseAtomsAdaptor

    meta = pd.read_csv(args.catalogue).set_index("name")
    rows = []
    for folder in sorted(p for p in Path(args.runs).iterdir() if p.is_dir()):
        name = folder.name
        row = {"name": name, "iteration": args.iteration}
        reasons = []
        try:
            rel = parse_run(folder / "relax")
            row.update(relax_converged_ionic=rel["converged_ionic"],
                       relax_converged_electronic=rel["converged_electronic"],
                       relax_ionic_steps=rel["n_ionic_steps"])
            if not (rel["converged_ionic"] and rel["converged_electronic"]):
                reasons.append("relaxation not converged")
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"relax unreadable: {type(exc).__name__}")
        try:
            st = parse_run(folder / "static")
            atoms = AseAtomsAdaptor.get_atoms(st["_final_structure"])
            q = check_atoms(atoms)
            row.update(gap_eV=st["gap_eV"], vbm_eV=st["vbm_eV"], cbm_eV=st["cbm_eV"],
                       is_direct=st["is_direct"], energy_eV=st["energy_eV"],
                       static_converged_electronic=st["converged_electronic"],
                       total_magnetization=st.get("total_magnetization"),
                       final_qc_passed=q.passed, final_qc_reasons=";".join(q.reasons))
            if not st["converged_electronic"]:
                reasons.append("static SCF not converged")
            if not q.passed:
                reasons.append("final structure fails QC: " + ";".join(q.reasons))
            if st["_site_moments"] is not None:
                mr = magnetization_report(st["_site_moments"], atoms.get_chemical_symbols())
                row.update({f"mag_{k}": v for k, v in mr.items()})
                if mr["radical_flag"]:
                    reasons.append("radical flag (ligand moment > 0.3 muB)")
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"static unreadable: {type(exc).__name__}")
        row["label_ok"] = not reasons
        row["label_reasons"] = ";".join(reasons)
        if name in meta.index:
            for c in ("topology", "metal", "node_id", "node_family", "linker_label", "substituent", "n_rings"):
                row[c] = meta.at[name, c]
        rows.append(row)
        print(f"{name:45s} gap={row.get('gap_eV', float('nan')):.3f}  ok={row['label_ok']}  {row['label_reasons']}")

    new = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        old = pd.read_csv(out)
        new = pd.concat([old[~old.name.isin(new.name)], new], ignore_index=True)
    new.to_csv(out, index=False)


if __name__ == "__main__":
    main()
