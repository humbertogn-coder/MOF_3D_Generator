"""VASP input generation and output parsing (VASP 6.x, PAW-PBE).

Inputs follow a QMOF-compatible protocol (PBE-D3(BJ), 520 eV, spin polarised with a
high-spin ferromagnetic initial guess, full cell relaxation) followed by a static
calculation for gaps and projected DOS. POTCAR files are *not* generated here
(VASP licence); ``hpc/make_potcar.sh`` assembles them on the cluster.

Protocol v2 (after iteration 0, see hpc/README.md):
  * starting geometries are pre-relaxed with CHGNet (scripts/09_prerelax_chgnet.py);
  * relaxations use the Gamma point when every lattice vector is >= 12 A (the static
    run keeps the full k-mesh), EDIFFG = -0.05 eV/A and NELM = 120;
  * LMAXMIX = 4, and ALGO = All for Fe, Co and Ni frameworks, whose SCF did not
    converge with ALGO = Normal in iteration 0.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np

HIGH_SPIN_MAGMOM = {"Mn": 5.0, "Fe": 4.0, "Co": 3.0, "Ni": 2.0, "Cu": 1.0,
                    "Zn": 0.0, "Mg": 0.0, "Zr": 0.0}
K_LENGTH = 25.0   # A; n_k = max(1, round(K_LENGTH / a))
GAMMA_RELAX_MIN_A = 12.0   # relax at Gamma if all lattice vectors are at least this long
ALGO_ALL_METALS = {"Fe", "Co", "Ni"}

COMMON = {
    "PREC": "Accurate", "ENCUT": 520, "LASPH": ".TRUE.", "ISPIN": 2,
    "IVDW": 12, "LREAL": "Auto", "NCORE": 4, "LWAVE": ".FALSE.", "LMAXMIX": 4,
}
RELAX = {**COMMON, "EDIFF": "1E-5", "NELM": 120, "ALGO": "Normal",
         "ISMEAR": 0, "SIGMA": 0.01, "IBRION": 2, "ISIF": 3, "NSW": 300,
         "EDIFFG": -0.05, "LCHARG": ".FALSE.", "LORBIT": 11}
STATIC = {**COMMON, "EDIFF": "1E-6", "NELM": 300, "ALGO": "Normal",
          "ISMEAR": 0, "SIGMA": 0.01, "IBRION": -1, "NSW": 0,
          "LCHARG": ".FALSE.", "LORBIT": 11, "NEDOS": 3001}


def kpoint_mesh(lattice_abc) -> tuple:
    return tuple(max(1, int(round(K_LENGTH / a))) for a in lattice_abc)


def magmom_string(symbols, afm: bool = False) -> str:
    vals, sign = [], 1.0
    for s in symbols:
        m = HIGH_SPIN_MAGMOM.get(s, 0.0)
        if afm and m:
            m *= sign
            sign *= -1
        vals.append(m)
    out, prev, count = [], None, 0
    for v in vals + [None]:
        if v == prev:
            count += 1
            continue
        if prev is not None:
            out.append(f"{count}*{prev:g}")
        prev, count = v, 1
    return " ".join(out)


def write_inputs(structure, folder: Path, stage: str = "relax", afm: bool = False,
                 extra: Optional[Dict] = None, gamma_relax: bool = True) -> None:
    """Write POSCAR/INCAR/KPOINTS for ``stage`` in {'relax', 'static'}.

    ``structure`` is a pymatgen Structure; species are sorted so that POTCAR order
    matches POSCAR order.
    """
    from pymatgen.io.vasp import Kpoints, Poscar

    folder.mkdir(parents=True, exist_ok=True)
    s = structure.get_sorted_structure()
    Poscar(s).write_file(folder / "POSCAR")
    params = dict(RELAX if stage == "relax" else STATIC)
    symbols = [site.specie.symbol for site in s]
    params["MAGMOM"] = magmom_string(symbols, afm=afm)
    if set(symbols) & ALGO_ALL_METALS:
        params["ALGO"] = "All"
    mesh = kpoint_mesh(s.lattice.abc)
    if stage == "relax" and gamma_relax and min(s.lattice.abc) >= GAMMA_RELAX_MIN_A:
        mesh = (1, 1, 1)
    nk = int(np.prod(mesh))
    if nk >= 2:   # k-point parallelisation (48 ranks: 4 groups x 12, NCORE=4); no effect on results
        params["KPAR"] = 4 if nk >= 4 else 2
    if stage == "static" and nk >= 4 and params["ALGO"] != "All":   # no tetrahedra with ALGO = All
        params.update(ISMEAR=-5)
        params.pop("SIGMA", None)
    if extra:
        params.update(extra)
    lines = [f"SYSTEM = {folder.parent.name} ({stage})"] + [f"{k} = {v}" for k, v in params.items()]
    (folder / "INCAR").write_text("\n".join(lines) + "\n")
    Kpoints.gamma_automatic(mesh).write_file(folder / "KPOINTS")
    (folder / "POTCAR.spec").write_text("\n".join(dict.fromkeys(site.specie.symbol for site in s)) + "\n")


def parse_run(folder: Path) -> dict:
    """Parse a finished VASP run (vasprun.xml + OUTCAR)."""
    from pymatgen.io.vasp import Outcar, Vasprun

    out = {"folder": str(folder)}
    vr = Vasprun(str(folder / "vasprun.xml"), parse_potcar_file=False,
                 parse_dos=False, exception_on_bad_xml=False)
    out["converged_electronic"] = bool(vr.converged_electronic)
    # pymatgen's converged_ionic only checks n_steps < NSW and is True for runs killed
    # by the walltime; use the explicit VASP message instead (relaxations only).
    if vr.parameters.get("IBRION", -1) in (1, 2, 3) and vr.parameters.get("NSW", 0) > 0:
        outcar_txt = (folder / "OUTCAR").read_text(errors="ignore") if (folder / "OUTCAR").exists() else ""
        out["converged_ionic"] = "reached required accuracy" in outcar_txt
    else:
        out["converged_ionic"] = True
    out["n_ionic_steps"] = len(vr.ionic_steps)
    try:
        gap, cbm, vbm, direct = vr.eigenvalue_band_properties
        out.update(gap_eV=float(gap), cbm_eV=float(cbm), vbm_eV=float(vbm), is_direct=bool(direct))
    except (ValueError, TypeError):  # truncated run without final eigenvalues
        out.update(gap_eV=math.nan, cbm_eV=math.nan, vbm_eV=math.nan, is_direct=None)
    out.update(efermi_eV=float(vr.efermi) if vr.efermi is not None else math.nan,
               energy_eV=float(vr.final_energy))
    try:
        oc = Outcar(str(folder / "OUTCAR"))
        mag = [m.get("tot", 0.0) for m in oc.magnetization]
        out["total_magnetization"] = float(oc.total_mag) if oc.total_mag is not None else math.nan
        out["_site_moments"] = mag
    except Exception:  # noqa: BLE001
        out["_site_moments"] = None
    out["_final_structure"] = vr.final_structure
    return out
