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


def metal_pairs(structure, max_dist: float = 4.5, isolation: float = 1.5):
    """Isolated M2 units (paddlewheels): pairs of open-shell metals that are mutual nearest
    neighbours closer than ``max_dist`` and whose next-nearest metal is at least
    ``isolation`` times farther away. Indices refer to the sorted structure (POSCAR order).
    Returns [] if any open-shell metal is not in such a pair (e.g. M4 clusters in bcu)."""
    s = structure.get_sorted_structure()
    metals = [i for i, site in enumerate(s) if HIGH_SPIN_MAGMOM.get(site.specie.symbol, 0) > 0]
    if len(metals) < 2:
        return []
    near = {}
    for a in metals:
        d = sorted((s.get_distance(a, b), b) for b in metals if b != a)
        if d[0][0] > max_dist or (len(d) > 1 and d[1][0] < isolation * d[0][0]):
            return []
        near[a] = d[0][1]
    if any(near[near[a]] != a for a in metals):
        return []
    return sorted({tuple(sorted((a, near[a]))) for a in metals})


def bs_magmom_string(structure) -> Optional[str]:
    """Broken-symmetry initial moments for frameworks built from isolated M2 units
    (paddlewheels): +m/-m on the two metals of each unit, m = max(high-spin moment, 2). In iteration 0 this start found the
    lowest-energy solution of both Cu-paddlewheel frameworks (0.41-0.43 eV below the
    non-magnetic solution reached from the ferromagnetic start)."""
    s = structure.get_sorted_structure()
    pairs = metal_pairs(s)
    if not pairs:
        return None
    vals = [0.0] * len(s)
    for a, b in pairs:
        m = max(HIGH_SPIN_MAGMOM[s[a].specie.symbol], 2.0)
        vals[a], vals[b] = m, -m
    out, prev, count = [], None, 0
    for v in vals + [None]:
        if v == prev:
            count += 1
            continue
        if prev is not None:
            out.append(f"{count}*{prev:g}" if count > 1 else f"{prev:g}")
        prev, count = v, 1
    return " ".join(out)


def write_inputs(structure, folder: Path, stage: str = "relax", afm: bool = False,
                 extra: Optional[Dict] = None, gamma_relax: bool = True) -> None:
    """Write POSCAR/INCAR/KPOINTS for ``stage`` in {'relax', 'static', 'static_bs'}.

    'static_bs' is the static run with broken-symmetry moments (paddlewheels only; see
    bs_magmom_string) and ISYM = 0; it is run after 'static' on the same geometry.

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
    if stage == "static_bs":
        params["MAGMOM"] = bs_magmom_string(s)
        params["ISYM"] = 0
    if set(symbols) & ALGO_ALL_METALS:
        params["ALGO"] = "All"
    mesh = kpoint_mesh(s.lattice.abc)
    if stage == "relax" and gamma_relax and min(s.lattice.abc) >= GAMMA_RELAX_MIN_A:
        mesh = (1, 1, 1)
    nk = int(np.prod(mesh))
    if nk >= 2:   # k-point parallelisation; no effect on results. KPAR = 4 ran out of memory
        params["KPAR"] = 2   # (120 GB, 48 ranks) for the 136-152-atom bcu statics with 8 k-points

    if stage.startswith("static") and nk >= 4 and params["ALGO"] != "All":   # no tetrahedra with ALGO = All
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
    # A relaxation killed by the walltime has no final positions in vasprun.xml, and
    # pymatgen then returns the initial structure; CONTCAR holds the last geometry.
    contcar = folder / "CONTCAR"
    if out["n_ionic_steps"] > 1 and contcar.exists() and len(contcar.read_text().splitlines()) > 8:
        from pymatgen.core import Structure
        out["_final_structure"] = Structure.from_file(contcar)
    return out
