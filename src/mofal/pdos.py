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
