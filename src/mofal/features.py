"""Tabular descriptors used by the surrogate models (feature sets A-E).

The definitions reproduce the original project code (``phase1_v2_ml_inference.py``)
so that QMOF training data and generated candidates are featurised identically.

Feature sets
------------
A  basic geometry (8)
B  A + aggregate composition (30)
C  B + element fractions of 34 common elements (64)          <- surrogate
D  C + 4 categorical descriptors (topology, primary metal, crystal system,
       metal-containing node SMILES)                         (64 numeric + 4 cat.)
E  D + 4 PBE-derived auxiliary quantities (leakage, reference only)
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict

import numpy as np
import pandas as pd

ELEMENT_RE = re.compile(r"([A-Z][a-z]?)(\d*)")

METALS = ["Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba",
          "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
          "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
          "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
          "Al", "Ga", "In", "Tl", "Sn", "Pb", "Sb", "Bi", "Ge", "Si"]
TRANSITION_METALS = ["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
                     "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
                     "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"]
# NOTE: "open shell" here means "primary metal is a 3d element Ti-Cu"; it is a
# composition flag, not a computed spin state.
OPEN_SHELL_3D = ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"]
MAIN_GROUP_METALS = ["Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba",
                     "Al", "Ga", "In", "Tl", "Sn", "Pb", "Sb", "Bi", "Ge", "Si"]
COMMON_ELEMENTS = ["H", "C", "N", "O", "F", "S", "Cl", "Br", "I", "P", "B",
                   "Li", "Na", "K", "Mg", "Ca", "Ba",
                   "Al", "Si", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
                   "Zr", "Cd", "Ag", "Sn", "Pb", "Hg"]
ATOMIC_NUMBER = {"H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16, "Cl": 17,
                 "K": 19, "Ca": 20, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26,
                 "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Br": 35, "Ag": 47, "Cd": 48,
                 "I": 53, "Ba": 56, "Hg": 80, "Pb": 82, "Zr": 40}
ELECTRONEGATIVITY = {"H": 2.20, "B": 2.04, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98,
                     "P": 2.19, "S": 2.58, "Cl": 3.16, "K": 0.82, "Ca": 1.00,
                     "Ti": 1.54, "V": 1.63, "Cr": 1.66, "Mn": 1.55, "Fe": 1.83,
                     "Co": 1.88, "Ni": 1.91, "Cu": 1.90, "Zn": 1.65, "Br": 2.96,
                     "Ag": 1.93, "Cd": 1.69, "I": 2.66, "Ba": 0.89, "Hg": 2.00, "Pb": 2.33,
                     "Zr": 1.33}

BASIC_GEOMETRY = ["n_atoms", "density", "PLD_A", "LCD_A", "vol_A3",
                  "LCD_minus_PLD", "LCD_over_PLD", "vol_per_atom"]
COMPOSITION = ["n_metals_total", "metal_frac", "n_transition_metals", "transition_metal_frac",
               "n_open_shell_3d", "open_shell_3d_frac", "has_transition_metal",
               "has_open_shell", "has_main_group_metal", "C_per_atom", "N_per_atom",
               "O_per_atom", "H_per_atom", "N_O_ratio", "C_metal_ratio", "O_metal_ratio",
               "hetero_atoms", "hetero_frac", "mean_Z", "std_Z", "mean_eneg", "std_eneg"]
ELEMENT_FRACTIONS = [f"frac_{el}" for el in COMMON_ELEMENTS]
CATEGORICAL = ["cat_topology", "cat_primary_metal", "cat_crystal_system", "cat_node_smiles"]
PBE_AUX = ["pbe_energy_per_atom", "pbe_vdw_per_atom", "pbe_elec_per_atom", "pbe_abs_magmom_per_atom"]

FEATURE_SETS = {
    "A_basic_geometry": (BASIC_GEOMETRY, []),
    "B_geometry_plus_composition": (BASIC_GEOMETRY + COMPOSITION, []),
    "C_geometry_composition_elements": (BASIC_GEOMETRY + COMPOSITION + ELEMENT_FRACTIONS, []),
    "D_all_structural_plus_categories": (BASIC_GEOMETRY + COMPOSITION + ELEMENT_FRACTIONS, CATEGORICAL),
    "E_all_plus_PBE_auxiliary_no_gap": (BASIC_GEOMETRY + COMPOSITION + ELEMENT_FRACTIONS + PBE_AUX, CATEGORICAL),
}
FEATURE_SET_C = FEATURE_SETS["C_geometry_composition_elements"][0]


def parse_formula(formula) -> Dict[str, int]:
    out: Counter = Counter()
    if not isinstance(formula, str):
        return dict(out)
    for el, n in ELEMENT_RE.findall(formula.replace(" ", "")):
        out[el] += int(n) if n else 1
    return dict(out)


def primary_metal(comp: Dict[str, int]) -> str:
    mets = {k: v for k, v in comp.items() if k in METALS}
    return max(mets, key=mets.get) if mets else "none"


def _wmean(comp, prop):
    num = sum(prop[e] * n for e, n in comp.items() if e in prop)
    den = sum(n for e, n in comp.items() if e in prop)
    return num / den if den else np.nan


def _wstd(comp, prop):
    mu = _wmean(comp, prop)
    num = sum(n * (prop[e] - mu) ** 2 for e, n in comp.items() if e in prop)
    den = sum(n for e, n in comp.items() if e in prop)
    return float(np.sqrt(num / den)) if den else np.nan


def composition_features(comp: Dict[str, int], n_atoms: int) -> Dict[str, float]:
    pm = primary_metal(comp)
    n = max(n_atoms, 1)
    n_metals = sum(comp.get(m, 0) for m in METALS)
    n_tm = sum(comp.get(m, 0) for m in TRANSITION_METALS)
    n_os = sum(comp.get(m, 0) for m in OPEN_SHELL_3D)
    hetero = sum(comp.get(e, 0) for e in ("N", "O", "S", "F", "Cl", "Br", "I"))
    f = {
        "primary_metal": pm,
        "n_metals_total": n_metals, "metal_frac": n_metals / n,
        "n_transition_metals": n_tm, "transition_metal_frac": n_tm / n,
        "n_open_shell_3d": n_os, "open_shell_3d_frac": n_os / n,
        "has_transition_metal": int(n_tm > 0),
        "has_open_shell": int(pm in OPEN_SHELL_3D),
        "has_main_group_metal": int(pm in MAIN_GROUP_METALS),
        "hetero_atoms": hetero,
        "mean_Z": _wmean(comp, ATOMIC_NUMBER), "std_Z": _wstd(comp, ATOMIC_NUMBER),
        "mean_eneg": _wmean(comp, ELECTRONEGATIVITY), "std_eneg": _wstd(comp, ELECTRONEGATIVITY),
        "C_per_atom": comp.get("C", 0) / n, "N_per_atom": comp.get("N", 0) / n,
        "O_per_atom": comp.get("O", 0) / n, "H_per_atom": comp.get("H", 0) / n,
        "N_O_ratio": comp.get("N", 0) / (comp.get("O", 0) + 1),
        "C_metal_ratio": comp.get("C", 0) / (n_metals + 1),
        "O_metal_ratio": comp.get("O", 0) / (n_metals + 1),
        "hetero_frac": hetero / n,
    }
    for el in COMMON_ELEMENTS:
        f[f"frac_{el}"] = comp.get(el, 0) / n
    return f


def add_geometry_derivatives(df: pd.DataFrame) -> pd.DataFrame:
    df["LCD_minus_PLD"] = df["LCD_A"] - df["PLD_A"]
    df["LCD_over_PLD"] = df["LCD_A"] / (df["PLD_A"] + 1e-6)
    df["vol_per_atom"] = df["vol_A3"] / df["n_atoms"]
    return df


def featurize_qmof(qmof: pd.DataFrame) -> pd.DataFrame:
    """QMOF csv -> feature table with target ``gap_PBE`` (eV)."""
    df = qmof.copy()
    df.columns = [c.strip() for c in df.columns]
    num = ["info.natoms", "info.pld", "info.lcd", "info.density", "info.volume",
           "outputs.pbe.bandgap", "outputs.pbe.energy_total", "outputs.pbe.energy_vdw",
           "outputs.pbe.energy_elec", "outputs.pbe.net_magmom"]
    for c in num:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["outputs.pbe.bandgap", "info.natoms"])
    df = df[df["outputs.pbe.bandgap"] >= 0].copy()
    df = df.rename(columns={"outputs.pbe.bandgap": "gap_PBE", "info.natoms": "n_atoms",
                            "info.pld": "PLD_A", "info.lcd": "LCD_A",
                            "info.density": "density", "info.volume": "vol_A3"})
    comp = df["info.formula"].apply(parse_formula)
    feats = pd.DataFrame([composition_features(c, n) for c, n in zip(comp, df["n_atoms"])],
                         index=df.index)
    df = pd.concat([df, feats], axis=1)
    df = add_geometry_derivatives(df)
    # categorical (set D/E)
    df["cat_topology"] = df.get("info.mofid.topology", pd.Series("unknown", index=df.index)).fillna("unknown")
    df["cat_primary_metal"] = df["primary_metal"]
    df["cat_crystal_system"] = df.get("info.symmetry.spacegroup_crystal", "unknown")
    df["cat_node_smiles"] = df.get("info.mofid.smiles_nodes", pd.Series("unknown", index=df.index)).fillna("unknown")
    # PBE auxiliary (set E only)
    n = df["n_atoms"]
    df["pbe_energy_per_atom"] = df["outputs.pbe.energy_total"] / n
    df["pbe_vdw_per_atom"] = df["outputs.pbe.energy_vdw"] / n
    df["pbe_elec_per_atom"] = df["outputs.pbe.energy_elec"] / n
    df["pbe_abs_magmom_per_atom"] = df["outputs.pbe.net_magmom"].abs() / n
    return df.dropna(subset=FEATURE_SET_C + ["gap_PBE"]).copy()


def featurize_candidates(cat: pd.DataFrame, geo: pd.DataFrame) -> pd.DataFrame:
    """Generated candidates: ``cat`` has name/formula/n_atoms, ``geo`` has Zeo++ output
    (name, PLD_A, LCD_A, density, vol_A3)."""
    df = cat.merge(geo, on="name", how="left", suffixes=("", "_zeo"))
    if "density_zeo" in df:
        df["density"] = df["density_zeo"].fillna(df["density"])
    comp = df["formula"].apply(parse_formula)
    feats = pd.DataFrame([composition_features(c, n) for c, n in zip(comp, df["n_atoms"])],
                         index=df.index)
    df = pd.concat([df, feats], axis=1)
    return add_geometry_derivatives(df)
