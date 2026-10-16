"""Band-edge orbital character from projected DOS (LORBIT = 11).

Replaces the earlier "character at E_F (+/-0.1 eV)" metric, which for gapped
systems only samples smearing tails. Here the valence- and conduction-band edges
are characterised separately by integrating element/orbital-projected DOS over a
window below the VBM and above the CBM.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

LIGAND = {"C", "N", "O", "F", "H"}
WINDOW_EV = 0.5


def band_edge_character(vasprun_path: Path, window: float = WINDOW_EV) -> dict:
    from pymatgen.electronic_structure.core import OrbitalType
    from pymatgen.io.vasp import Vasprun

    vr = Vasprun(str(vasprun_path), parse_potcar_file=False)
    gap, cbm, vbm, _ = vr.eigenvalue_band_properties
    cdos = vr.complete_dos
    energies = cdos.energies
    res = {"gap_eV": float(gap), "vbm_eV": float(vbm), "cbm_eV": float(cbm)}
    for edge, lo, hi in (("VB", vbm - window, vbm + 1e-3), ("CB", cbm - 1e-3, cbm + window)):
        mask = (energies >= lo) & (energies <= hi)
        tot = {"metal_d": 0.0, "ligand_p": 0.0, "other": 0.0}
        for el, spd in _element_spd(cdos).items():
            for orb, dos in spd.items():
                dens = sum(np.asarray(v) for v in dos.densities.values())
                w = float(np.trapezoid(dens[mask], energies[mask])) if mask.sum() > 1 else 0.0
                if el not in LIGAND and orb == OrbitalType.d:
                    tot["metal_d"] += w
                elif el in LIGAND and orb == OrbitalType.p:
                    tot["ligand_p"] += w
                else:
                    tot["other"] += w
        s = sum(tot.values()) or np.nan
        for k, v in tot.items():
            res[f"{edge}_{k}_frac"] = v / s
        res[f"{edge}_states"] = s
    return res


def _element_spd(cdos):
    """{element symbol: {OrbitalType: Dos}} summed over sites of that element."""
    return {e.symbol: cdos.get_element_spd_dos(e) for e in cdos.structure.composition.elements}


# ----------------------------------------------------------------- site groups
METALS = {"Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Mg"}
GROUPS = ("metal", "oxo", "Ocarb", "linker")


def site_groups(structure) -> list[str]:
    """Assign every site to one of four groups.

    metal   metal atoms
    oxo     O (or OH) bound only to metals (mu4-O of Zn4O, mu3-O of M3 nodes)
    Ocarb   carboxylate O (bound to a C and to a metal)
    linker  everything else (C, H, N, F of the organic part)
    """
    syms = [s.specie.symbol for s in structure]
    groups = []
    for i, s in enumerate(syms):
        if s in METALS:
            groups.append("metal")
        elif s == "O":
            nn = structure.get_neighbors(structure[i], 1.75)
            groups.append("Ocarb" if any(n.specie.symbol == "C" for n in nn) else "oxo")
        else:
            groups.append("linker")
    for i, s in enumerate(syms):      # H of a hydroxide belongs to the node
        if s == "H":
            nn = structure.get_neighbors(structure[i], 1.2)
            if nn and groups[nn[0].index] == "oxo":
                groups[i] = "oxo"
    return groups


def band_edge_projection(vasprun_path: Path, window: float = 0.25) -> dict:
    """Band-edge character from the site- and orbital-projected eigenvalues (LORBIT = 11).

    VB: occupied states within `window` of the VBM; CB: empty states within `window` of
    the CBM, both spins, weighted by k-point weight. Fractions are normalised to the
    projected weight inside the PAW spheres. Robust for very small gaps, where a DOS
    window would mix the two edges.
    """
    from pymatgen.io.vasp import Vasprun

    vr = Vasprun(str(vasprun_path), parse_projected_eigen=True, parse_potcar_file=False)
    gap, cbm, vbm, direct = vr.eigenvalue_band_properties
    groups = np.array(site_groups(vr.final_structure))
    w = np.array(vr.actual_kpoints_weights)
    res = {"gap_eV": float(gap), "vbm_eV": float(vbm), "cbm_eV": float(cbm)}
    acc = {"VB": np.zeros((len(groups), 9)), "CB": np.zeros((len(groups), 9))}
    nst = {"VB": 0.0, "CB": 0.0}
    for spin, ev in vr.eigenvalues.items():
        pe = vr.projected_eigenvalues[spin]            # (nk, nb, nions, 9)
        e, occ = ev[..., 0], ev[..., 1]
        vb = (occ > 0.5) & (e >= vbm - window)
        cb = (occ <= 0.5) & (e <= cbm + window)
        for edge, m in (("VB", vb), ("CB", cb)):
            for k in range(len(w)):
                sel = pe[k][m[k]]                      # (nsel, nions, 9)
                acc[edge] += w[k] * sel.sum(axis=0)
                nst[edge] += w[k] * m[k].sum()
    for edge in ("VB", "CB"):
        tot = acc[edge].sum() or np.nan
        for g in GROUPS:
            res[f"{edge}_{g}"] = acc[edge][groups == g].sum() / tot
        res[f"{edge}_metal_d"] = acc[edge][groups == "metal"][:, 4:].sum() / tot
        res[f"{edge}_n_states"] = nst[edge]
    return res
