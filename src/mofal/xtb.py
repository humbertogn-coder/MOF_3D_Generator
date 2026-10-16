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


# GFN1-xTB atomic orbitals per element (shells: H 1s+2s; 2p block s+p; Zn 4s+4p;
# other 3d metals 3d+4s+4p)
N_AO = {"H": 2, "B": 4, "C": 4, "N": 4, "O": 4, "F": 4, "Zn": 4,
        "Mn": 9, "Fe": 9, "Co": 9, "Ni": 9, "Cu": 9}


def frontier_populations(atoms, window: float = 0.25, uhf: int | None = None,
                         etemp: float = 500.0, max_iter: int = 500):
    """Per-atom Mulliken populations of the frontier orbitals (periodic GFN1-xTB, Gamma).

    Returns (pop_homo, pop_lumo, homo_eV, lumo_eV, n_occ, n_virt): occupied orbitals within
    `window` eV of the HOMO and empty orbitals within `window` eV of the LUMO, summed and
    normalised to 1 (spin-restricted Hamiltonian, fixed number of unpaired electrons).
    """
    from tblite.interface import Calculator
    kw = {"uhf": unpaired(atoms) if uhf is None else uhf,
          "lattice": atoms.cell.array * ANG2BOHR, "periodic": np.array([True, True, True])}
    calc = Calculator("GFN1-xTB", atoms.numbers, atoms.positions * ANG2BOHR, **kw)
    calc.set("temperature", etemp * KB_HA)
    calc.set("max-iter", max_iter)
    calc.set("verbosity", 0)
    calc.set("save-integrals", 1)
    res = calc.singlepoint()
    e = np.asarray(res.get("orbital-energies")) * HA2EV
    occ = np.asarray(res.get("orbital-occupations"))
    S = np.asarray(res.get("overlap-matrix"))
    C = np.asarray(res.get("orbital-coefficients"))
    nao = len(e)
    counts = [N_AO[s] for s in atoms.get_chemical_symbols()]
    if sum(counts) != nao:
        raise ValueError(f"AO count {sum(counts)} != {nao}; update N_AO")
    atom_of_ao = np.repeat(np.arange(len(atoms)), counts)
    thr = 0.5 * occ.max()
    homo = e[occ > thr].max()
    lumo = e[occ <= thr].min()
    # orientation of C: Mulliken population of an occupied MO must sum to 1
    i0 = int(np.argmax(e == homo))
    pop_col = C[:, i0] * (S @ C[:, i0])
    Cm = C if abs(pop_col.sum() - 1) < 1e-3 else C.T
    if abs((Cm[:, i0] * (S @ Cm[:, i0])).sum() - 1) > 1e-3:
        raise RuntimeError("could not orient the coefficient matrix")
    SC = S @ Cm
    out = []
    for sel in ((occ > thr) & (e >= homo - window), (occ <= thr) & (e <= lumo + window)):
        pop = (Cm[:, sel] * SC[:, sel]).sum(axis=1)
        per_atom = np.bincount(atom_of_ao, weights=pop, minlength=len(atoms))
        out.append(per_atom / per_atom.sum())
    n_occ, n_virt = int(((occ > thr) & (e >= homo - window)).sum()), int(((occ <= thr) & (e <= lumo + window)).sum())
    return out[0], out[1], float(homo), float(lumo), n_occ, n_virt
