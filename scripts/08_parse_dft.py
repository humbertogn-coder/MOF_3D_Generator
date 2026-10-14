#!/usr/bin/env python3
"""Parse finished VASP runs and decide which band gaps are accepted as labels.

A label is accepted (label_ok) only if
  * the relaxation reached ionic and electronic convergence,
  * the static run reached electronic convergence,
  * the final structure passes the same structural QC as the catalogue
    (no overlaps, no isolated atoms, a single connected framework), and
  * no non-metal atom carries a magnetic moment > 0.3 muB (radical flag);
    flagged runs are kept for manual inspection but not used for training, and
  * relaxations stopped by hand (relax/RELAX_DONE starting with "stopped") are accepted
    when the final maximum force is <= 0.25 eV/A; they are marked relax_stopped, and the
    note in RELAX_DONE records why (iteration 0: allocation limit, gap stable to within
    0.05 eV over the last 50 ionic steps);
  * for a structure with sibling runs <name>__afm, <name>__afm2, ... (other initial
    magnetic moments, same geometry), the lowest-energy spin solution provides the label
    (spin_solution; all solutions are listed in data/dft/spin_solutions.csv);
  * the framework did not collapse: cell volume within +/-20 % of the generated
    structure and no cell angle changed by more than 10 degrees. In iteration 0 two
    M3(CO2)6 pcu frameworks lost ~50 % of their volume and the Zn-paddlewheel cds
    framework sheared by 25 degrees (dense phases, not the designed framework), while
    intact soft frameworks contracted by up to 8.5 % (PBE-D3 vs PORMAKE geometry); the
    volume limit was 10 % during iteration 0 and set to 20 % before iteration 1 so that
    such contraction is not mistaken for collapse (no iteration-0 decision changes).

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

SPIN_KEYS = ("gap_eV", "vbm_eV", "cbm_eV", "is_direct", "energy_eV", "total_magnetization",
             "mag_sum_abs_metal", "mag_sum_abs_ligand", "mag_max_abs_ligand",
             "mag_n_ligand_above_threshold", "mag_radical_flag")


def last_fmax(outcar: Path) -> float:
    import re
    import numpy as np
    txt = outcar.read_text(errors="ignore")
    blocks = re.findall(r"TOTAL-FORCE \(eV/Angst\)\n -+\n(.*?)\n -+", txt, re.S)
    if not blocks:
        return float("nan")
    f = np.array([[float(x) for x in ln.split()[3:6]] for ln in blocks[-1].strip().splitlines()])
    return float(np.linalg.norm(f, axis=1).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--iteration", type=int, required=True)
    ap.add_argument("--catalogue", default=str(ROOT / "data/catalogue/passed.csv"))
    ap.add_argument("--out", default=str(ROOT / "data/dft/dft_results.csv"))
    ap.add_argument("--cifs", default=str(ROOT / "data/catalogue/cifs"),
                    help="generated structures, reference for the collapse check")
    ap.add_argument("--max_dv", type=float, default=20.0, help="percent")
    ap.add_argument("--max_dangle", type=float, default=10.0, help="degrees")
    args = ap.parse_args()
    from pymatgen.io.ase import AseAtomsAdaptor

    meta = pd.read_csv(args.catalogue).set_index("name")
    rows, extra_rows = [], []
    for folder in sorted(p for p in Path(args.runs).iterdir() if p.is_dir()):
        name = folder.name
        row = {"name": name, "iteration": args.iteration}
        reasons = []
        is_spin_sibling = "__" in name and name.split("__")[-1].startswith("afm")
        if is_spin_sibling:
            reasons.append("spin-state sensitivity run (not a training label)")
        if (folder / "EXCLUDED").exists():
            reasons.append("excluded: " + (folder / "EXCLUDED").read_text().strip())
        final = None
        sibling = is_spin_sibling
        done = folder / "relax" / "RELAX_DONE"
        stopped = done.exists() and done.read_text().lstrip().lower().startswith("stopped")
        try:
            if sibling:
                raise FileNotFoundError("sibling runs reuse the parent geometry")
            rel = parse_run(folder / "relax")
            final = rel["_final_structure"]
            row.update(relax_converged_ionic=rel["converged_ionic"],
                       relax_converged_electronic=rel["converged_electronic"],
                       relax_ionic_steps=rel["n_ionic_steps"], relax_stopped=stopped,
                       relax_fmax_final=round(last_fmax(folder / "relax" / "OUTCAR"), 3))
            ionic_ok = rel["converged_ionic"] or (stopped and row["relax_fmax_final"] <= 0.25)
            if not (ionic_ok and rel["converged_electronic"]):
                reasons.append("relaxation not converged")
        except Exception as exc:  # noqa: BLE001
            if not sibling:
                reasons.append(f"relax unreadable: {type(exc).__name__}")
        try:
            st = parse_run(folder / "static")
            final = st["_final_structure"]
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
        ref = Path(args.cifs) / f"{name}.cif"
        if final is not None and ref.exists():
            from pymatgen.core import Structure
            s0 = Structure.from_file(ref)
            dv = 100 * (final.volume / s0.volume - 1)
            dang = max(abs(a - b) for a, b in zip(final.lattice.angles, s0.lattice.angles))
            row.update(vol_change_pct=round(dv, 1), max_angle_change_deg=round(dang, 1))
            if abs(dv) > args.max_dv or dang > args.max_dangle:
                reasons.append(f"framework collapse/shear (dV {dv:.0f} %, dangle {dang:.0f} deg)")
        row["label_ok"] = not reasons
        if (folder / "static_bs" / "vasprun.xml").exists():   # broken-symmetry static, same folder
            try:
                bs = parse_run(folder / "static_bs")
                bs_row = {"name": f"{name}__afmbs", "iteration": args.iteration,
                          "gap_eV": bs["gap_eV"], "vbm_eV": bs["vbm_eV"], "cbm_eV": bs["cbm_eV"],
                          "is_direct": bs["is_direct"], "energy_eV": bs["energy_eV"],
                          "static_converged_electronic": bs["converged_electronic"],
                          "total_magnetization": bs.get("total_magnetization"),
                          "label_ok": False, "label_reasons": "spin-state sensitivity run (not a training label)"}
                if bs["_site_moments"] is not None:
                    from pymatgen.io.ase import AseAtomsAdaptor as _A
                    mr = magnetization_report(bs["_site_moments"], _A.get_atoms(bs["_final_structure"]).get_chemical_symbols())
                    bs_row.update({f"mag_{k}": v for k, v in mr.items()})
                extra_rows.append(bs_row)
            except Exception as exc:  # noqa: BLE001
                print(f"{name}: static_bs unreadable ({type(exc).__name__})")
        row["label_reasons"] = ";".join(reasons)
        if name in meta.index:
            for c in ("topology", "metal", "node_id", "node_family", "linker_label", "substituent", "n_rings"):
                row[c] = meta.at[name, c]
        rows.append(row)
        print(f"{name:45s} gap={row.get('gap_eV', float('nan')):.3f}  ok={row['label_ok']}  {row['label_reasons']}")

    rows += extra_rows
    # spin solutions: <name>__afm* runs on the same geometry; the lowest energy gives the label
    spin_rows = []
    for r in rows:
        if "__" in r["name"] and r["name"].split("__")[-1].startswith("afm"):
            continue
        sibs = [x for x in rows if x["name"].startswith(r["name"] + "__afm")
                and x.get("static_converged_electronic") and pd.notna(x.get("energy_eV", float("nan")))]
        if not sibs or pd.isna(r.get("energy_eV", float("nan"))):
            continue
        cands = [("fm-start", r)] + [(x["name"].split("__")[-1] + "-start", x) for x in sibs]
        for lab, x in cands:
            spin_rows.append(dict(name=r["name"], start=lab, energy_eV=x["energy_eV"], gap_eV=x.get("gap_eV"),
                                  mag_sum_abs_metal=x.get("mag_sum_abs_metal"),
                                  total_magnetization=x.get("total_magnetization")))
        e0 = r["energy_eV"]
        best_lab, best = min(cands, key=lambda c: c[1]["energy_eV"])
        r["spin_solutions_dE_eV"] = round(best["energy_eV"] - e0, 4)
        r["gap_fm_start_eV"] = r.get("gap_eV")
        r["spin_solutions_gap_range_eV"] = f"{min(c[1]['gap_eV'] for c in cands):.3f}-{max(c[1]['gap_eV'] for c in cands):.3f}"
        if best is not r and best["energy_eV"] < e0 - 1e-3:
            for k in SPIN_KEYS:
                if k in best:
                    r[k] = best[k]
            r["spin_solution"] = best_lab
        else:
            r["spin_solution"] = "fm-start"
    if spin_rows:
        sp = Path(args.out).with_name("spin_solutions.csv")
        sp.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(spin_rows).to_csv(sp, index=False)

    new = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        old = pd.read_csv(out)
        new = pd.concat([old[~old.name.isin(new.name)], new], ignore_index=True)
    new.to_csv(out, index=False)


if __name__ == "__main__":
    main()
