"""Structural quality control for generated (PORMAKE) frameworks.

The checks are deliberately simple and fully geometric/compositional so that they
can be applied to any CIF/POSCAR, with or without the PORMAKE bond list:

1. ``min_distance``      : no pair of atoms closer than ``d_hard`` (default 0.70 A).
2. ``overlaps``          : no pair of heavy atoms (non-H) closer than 1.00 A and no
                           H-H pair closer than 1.00 A.
3. ``nonbonded_contacts``: (only when a PORMAKE bond list is available) no pair of
                           atoms that is *not* bonded in the building-block topology
                           closer than ``f_nb`` x (sum of covalent radii).
4. ``fragments``         : after construction, every atom must belong to the single
                           periodic framework (no isolated atoms / molecules).
5. ``valence``           : every H has exactly one neighbour; every C has 2-4
                           neighbours (organic part).
6. ``composition``       : the framework formula equals n_nodes*node + n_edges*edge.
7. ``charge``            : the node building block is charge balanced for the
                           declared metal oxidation state (checked upstream, recorded
                           here).

All thresholds are module-level constants so that they are reported in the paper/SI.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict, field
from typing import Iterable, Optional

import numpy as np
from ase import Atoms
from ase.data import covalent_radii, atomic_numbers
from ase.neighborlist import neighbor_list, natural_cutoffs

D_HARD = 0.70          # A, any pair
D_OVERLAP = 1.00       # A, heavy-heavy or H-H pairs
F_NONBONDED = 0.80     # fraction of covalent-radius sum for non-bonded contacts
BOND_MULT = 1.20       # multiplier on covalent radii to detect bonds
METALS = {
    "Li", "Na", "K", "Mg", "Ca", "Sr", "Ba", "Al", "Ga", "In", "Sc", "Y", "Ti", "Zr",
    "Hf", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Cd", "Ag", "Pb", "Sn", "Bi",
    "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
}


@dataclass
class QCResult:
    n_atoms: int
    formula: str
    min_distance: float
    n_pairs_below_hard: int
    n_overlaps: int
    n_nonbonded_contacts: int
    n_isolated_atoms: int
    n_components: int
    n_bad_H: int
    n_bad_C: int
    composition_ok: Optional[bool] = None
    passed: bool = False
    reasons: list = field(default_factory=list)

    def as_dict(self):
        d = asdict(self)
        d["reasons"] = ";".join(self.reasons)
        return d


def _components(n: int, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(i, j):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return np.array([find(k) for k in range(n)])


def check_atoms(atoms: Atoms,
                pormake_bonds: Optional[np.ndarray] = None,
                expected_formula: Optional[Counter] = None) -> QCResult:
    """Run all geometric checks on an ASE Atoms object (periodic)."""
    s = np.array(atoms.get_chemical_symbols())
    n = len(atoms)
    reasons = []

    # pair distances up to 1.2 A
    i, j, d = neighbor_list("ijd", atoms, cutoff=1.2)
    m = i < j
    i, j, d = i[m], j[m], d[m]
    min_d = float(d.min()) if len(d) else float("inf")
    n_hard = int((d < D_HARD).sum())
    heavy = (s[i] != "H") & (s[j] != "H")
    hh = (s[i] == "H") & (s[j] == "H")
    n_overlap = int(((heavy | hh) & (d < D_OVERLAP)).sum())

    # bonds / connectivity
    bi, bj = neighbor_list("ij", atoms, cutoff=natural_cutoffs(atoms, mult=BOND_MULT))
    cn = np.bincount(bi, minlength=n)
    comp = _components(n, bi, bj)
    n_comp = len(set(comp))
    n_iso = int((cn == 0).sum())
    org = ~np.isin(s[bj], list(METALS))
    cn_org = np.bincount(bi[org], minlength=n)
    n_bad_H = int(((s == "H") & (cn != 1)).sum())
    n_bad_C = int(((s == "C") & ((cn_org < 2) | (cn_org > 4))).sum())

    # non-bonded contacts relative to the PORMAKE bond graph
    n_nb = 0
    if pormake_bonds is not None and len(pormake_bonds):
        bonded = set(map(tuple, np.sort(np.asarray(pormake_bonds), axis=1)))
        ci, cj, cd = neighbor_list("ijd", atoms, cutoff=2.0)
        mm = ci < cj
        for a, b, dist in zip(ci[mm], cj[mm], cd[mm]):
            if (a, b) in bonded:
                continue
            rsum = covalent_radii[atomic_numbers[s[a]]] + covalent_radii[atomic_numbers[s[b]]]
            if dist < F_NONBONDED * rsum:
                n_nb += 1

    comp_ok = None
    if expected_formula is not None:
        comp_ok = Counter(s.tolist()) == expected_formula

    if n_hard:
        reasons.append(f"{n_hard} pairs < {D_HARD} A")
    if n_overlap:
        reasons.append(f"{n_overlap} heavy/H-H pairs < {D_OVERLAP} A")
    if n_nb:
        reasons.append(f"{n_nb} non-bonded contacts < {F_NONBONDED}*sum(r_cov)")
    if n_comp != 1:
        reasons.append(f"{n_comp} connected components")
    if n_iso:
        reasons.append(f"{n_iso} isolated atoms")
    if n_bad_H:
        reasons.append(f"{n_bad_H} H with CN != 1")
    if n_bad_C:
        reasons.append(f"{n_bad_C} C with organic CN outside 2-4")
    if comp_ok is False:
        reasons.append("composition mismatch")

    formula = atoms.get_chemical_formula()
    return QCResult(n_atoms=n, formula=formula, min_distance=round(min_d, 3),
                    n_pairs_below_hard=n_hard, n_overlaps=n_overlap,
                    n_nonbonded_contacts=n_nb, n_isolated_atoms=n_iso,
                    n_components=n_comp, n_bad_H=n_bad_H, n_bad_C=n_bad_C,
                    composition_ok=comp_ok, passed=not reasons, reasons=reasons)


def magnetization_report(moments: Iterable[float], symbols: Iterable[str],
                         ligand_threshold: float = 0.3) -> dict:
    """Summarise site magnetic moments (e.g. from OUTCAR 'magnetization (x)').

    Large moments on non-metal atoms indicate radical / dangling-bond states, which
    in this project were the signature of structurally broken frameworks.
    """
    m = np.asarray(list(moments), float)
    s = np.asarray(list(symbols))
    is_m = np.isin(s, list(METALS))
    lig = np.abs(m[~is_m])
    return {
        "sum_abs_metal": float(np.abs(m[is_m]).sum()),
        "sum_abs_ligand": float(lig.sum()),
        "max_abs_ligand": float(lig.max()) if lig.size else 0.0,
        "n_ligand_above_threshold": int((lig > ligand_threshold).sum()),
        "radical_flag": bool(lig.size and (lig.max() > ligand_threshold)),
    }


# Shortest acceptable length (A) for common covalent bonds; about 0.88-0.92 of the
# usual value (aromatic C-C 1.39, C-O(carboxylate) 1.26, C-N 1.34-1.47, C-H 1.09,
# N-H 1.01, C-F 1.35). Shorter bonds indicate a strained PORMAKE junction or a
# distorted substituent, which the overlap tests above do not catch.
MIN_BOND = {("C", "C"): 1.30, ("C", "O"): 1.18, ("C", "N"): 1.25, ("C", "H"): 0.98,
            ("H", "N"): 0.95, ("C", "F"): 1.28, ("H", "O"): 0.90}


def compressed_bonds(atoms: Atoms, cutoff: float = 1.75) -> list:
    """Return (i, j, symbol pair, distance) for covalent bonds shorter than MIN_BOND.

    Reported, not used to reject structures: CHGNet pre-relaxation
    (scripts/09_prerelax_chgnet.py) removes these distortions before DFT.
    """
    sym = atoms.get_chemical_symbols()
    i, j, d = neighbor_list("ijd", atoms, cutoff)
    out = []
    for a, b, dist in zip(i, j, d):
        if a >= b:
            continue
        key = tuple(sorted((sym[a], sym[b])))
        lo = MIN_BOND.get(key)
        if lo is not None and dist < lo:
            out.append((int(a), int(b), f"{key[0]}-{key[1]}", round(float(dist), 3)))
    return out
