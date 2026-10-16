"""Periodic GFN1-xTB band gaps with tblite (shared by scripts 13 and 16).

Settings follow the original exploratory screen: neutral, electronic temperature 500 K,
500 SCF iterations. Unpaired electrons: high-spin count of the 3d metals (Mn 5, Fe 4,
Co 3, Ni 2, Cu 1 per atom), adjusted by one if the parity does not match the electron
count. tblite works at the Gamma point only; a Gamma-point calculation on an
n1 x n2 x n3 supercell samples the same k-points as a Gamma-centred n1 x n2 x n3 mesh of
the unit cell.
"""
from __future__ import annotations

import numpy as np

ANG2BOHR = 1 / 0.52917721067
KB_HA = 3.166811563e-6
HA2EV = 27.211386245988
HS_UNPAIRED = {"Mn": 5, "Fe": 4, "Co": 3, "Ni": 2, "Cu": 1}


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


def run(atoms, periodic: bool = True, uhf: int | None = None, etemp: float = 500.0,
        max_iter: int = 500):
    """Return (gap, HOMO, LUMO) in eV for an ASE Atoms object."""
    from tblite.interface import Calculator
    kw = {"uhf": unpaired(atoms) if uhf is None else uhf}
    if periodic:
        kw.update(lattice=atoms.cell.array * ANG2BOHR, periodic=np.array([True, True, True]))
    calc = Calculator("GFN1-xTB", atoms.numbers, atoms.positions * ANG2BOHR, **kw)
    calc.set("temperature", etemp * KB_HA)
    calc.set("max-iter", max_iter)
    calc.set("verbosity", 0)
    return gap(calc.singlepoint())
