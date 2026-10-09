import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ase import Atoms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mofal.acquisition import acquisition_score, select_batch  # noqa: E402
from mofal.features import composition_features, parse_formula  # noqa: E402
from mofal.qc import check_atoms, magnetization_report  # noqa: E402
from mofal.vasp import kpoint_mesh, magmom_string  # noqa: E402


def test_parse_formula():
    assert parse_formula("C24H12O13Zn4") == {"C": 24, "H": 12, "O": 13, "Zn": 4}


def test_open_shell_flag():
    f = composition_features(parse_formula("C24H12O12Co3"), 51)
    assert f["has_open_shell"] == 1 and f["primary_metal"] == "Co"
    assert composition_features(parse_formula("C24H12O13Zn4"), 53)["has_open_shell"] == 0


def test_qc_detects_overlap():
    ok = Atoms("CH4", positions=[[5, 5, 5], [5.63, 5.63, 5.63], [4.37, 4.37, 5.63],
                                 [4.37, 5.63, 4.37], [5.63, 4.37, 4.37]], cell=[10, 10, 10], pbc=True)
    assert check_atoms(ok).n_overlaps == 0
    bad = ok.copy()
    bad.append("C")
    bad.positions[-1] = [5.3, 5.0, 5.0]
    r = check_atoms(bad)
    assert not r.passed and r.n_pairs_below_hard >= 1


def test_radical_flag():
    rep = magnetization_report([4.5, -4.5, 0.02, 1.2], ["Mn", "Mn", "C", "O"])
    assert rep["radical_flag"] and rep["n_ligand_above_threshold"] == 1


def test_acquisition_and_caps():
    df = pd.DataFrame({"name": list("abcdef"), "topology": ["pcu"] * 3 + ["nbo"] * 3,
                       "metal": ["Co", "Co", "Co", "Fe", "Mn", "Zn"],
                       "mu": [0.1, 0.2, 0.3, 0.4, 0.5, 2.0], "sigma_tree": [0.1] * 6,
                       "is_ood": [False] * 6})
    df["acquisition_score"] = acquisition_score(df, {"pcu": 2})
    b = select_batch(df, 4, max_per_topology=2, max_per_metal=2, forced=["f"])
    assert list(b.name)[0] == "f"
    assert (b.topology.value_counts() <= 2).all() and (b.metal.value_counts() <= 2).all()


def test_vasp_helpers():
    assert kpoint_mesh([13.0, 22.0, 30.0]) == (2, 1, 1)
    assert magmom_string(["Co", "Co", "C", "H"]) == "2*3 2*0"
    assert magmom_string(["Mn", "Mn"], afm=True) == "1*5 1*-5"
