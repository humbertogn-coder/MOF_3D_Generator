#!/usr/bin/env python3
"""Sanity check of the generator against known IRMOF-series MOFs in QMOF.

PORMAKE builds of Zn4O(CO2)6 (N16) in pcu with BDC, BPDC and NH2-BDC must
reproduce the composition of MOF-5, IRMOF-10 and IRMOF-3 exactly; density and
pore descriptors are compared with the QMOF entries of the same composition.

Usage:  python scripts/04_validate_known_mofs.py
Output: results/validation_known_mofs.csv
"""
from __future__ import annotations

import sys
from functools import reduce
from math import gcd
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mofal.features import parse_formula  # noqa: E402

KNOWN = {  # catalogue name -> (MOF, expected formula per Zn4O unit)
    "pcu__Zn-N16__short-H": ("MOF-5 (IRMOF-1)", "C24H12O13Zn4"),
    "pcu__Zn-N16__long-H": ("IRMOF-10", "C42H24O13Zn4"),
    "pcu__Zn-N16__short-NH2": ("IRMOF-3", "C24H15N3O13Zn4"),
}


def reduced(f):
    c = parse_formula(f)
    g = reduce(gcd, c.values())
    return tuple(sorted((k, v // g) for k, v in c.items()))


def main():
    cat = pd.read_csv(ROOT / "data/catalogue/catalogue.csv").set_index("name")
    zeo = pd.read_csv(ROOT / "data/catalogue/zeopp.csv").set_index("name")
    q = pd.read_csv(ROOT / "data/external/qmof.csv", low_memory=False)
    q["red"] = q["info.formula"].apply(reduced)
    rows = []
    for name, (mof, formula) in KNOWN.items():
        r = cat.loc[name]
        m = q[q.red == reduced(formula)]
        row = dict(mof=mof, name=name, expected_formula=formula, built_formula=r.formula,
                   composition_matches=reduced(r.formula) == reduced(formula), qc_passed=bool(r.passed),
                   density_built=r.density, n_qmof=len(m),
                   density_qmof=m["info.density"].mean() if len(m) else None,
                   gap_PBE_qmof=m["outputs.pbe.bandgap"].mean() if len(m) else None,
                   qmof_ids=";".join(m.qmof_id))
        if name in zeo.index:
            row.update(LCD_built=zeo.at[name, "LCD_A"], PLD_built=zeo.at[name, "PLD_A"],
                       LCD_qmof=m["info.lcd"].mean() if len(m) else None,
                       PLD_qmof=m["info.pld"].mean() if len(m) else None)
        rows.append(row)
    df = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results/validation_known_mofs.csv", index=False)
    print(df.round(3).T.to_string())


if __name__ == "__main__":
    main()
