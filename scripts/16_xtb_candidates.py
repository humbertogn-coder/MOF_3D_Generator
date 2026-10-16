#!/usr/bin/env python3
"""Periodic GFN1-xTB gaps of the generated candidates, compared with DFT labels and the
ML surrogate.

Question: can a cheap physics-based estimate (CHGNet geometry + periodic GFN1-xTB, no
training data) complement the surrogate, in particular for out-of-distribution
closed-shell frameworks where the surrogate under-predicts (Zn4O pcu)?

Geometry sources (column ``geometry``):
  pormake   generated structure as built (data/catalogue/cifs)
  chgnet    CHGNet pre-relaxed (data/prerelaxed; scripts/09_prerelax_chgnet.py)
  dft       DFT-relaxed geometry of a labelled structure (runs/iter*/<name>/static/POSCAR)
For dft geometries a second row ``dft_kmesh`` uses the supercell that matches the k-mesh
of the static DFT run (tblite is Gamma-only), when it has <= --max_super atoms.

Usage
-----
    python scripts/16_xtb_candidates.py --geometry dft             # labelled structures
    python scripts/16_xtb_candidates.py --geometry chgnet          # all with a CHGNet geometry
    python scripts/16_xtb_candidates.py --geometry pormake --names ...
    python scripts/16_xtb_candidates.py --summarize                # tables + figure
Output: results/xtb_candidates/xtb_candidates.csv (appended row by row; resumable)
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mofal.xtb import run, unpaired  # noqa: E402

warnings.filterwarnings("ignore")
OUT = ROOT / "results/xtb_candidates"
FIELDS = ["name", "geometry", "supercell", "n_atoms", "uhf", "gap_eV", "homo_eV", "lumo_eV",
          "seconds", "status"]


def dft_folder(name: str):
    for it in sorted(ROOT.glob("runs/iter[0-9]*")):
        f = it / name / "static"
        if (f / "STATIC_DONE").exists() and (f / "POSCAR").exists():
            return f
    return None


def structures(geometry: str, names=None):
    from ase.io import read
    if geometry == "dft":
        dft = pd.read_csv(ROOT / "data/dft/dft_results.csv")
        names = names or list(dft[dft.label_ok.astype(bool)].name)
        for n in names:
            f = dft_folder(n)
            if f is None:
                continue
            a = read(f / "POSCAR")
            yield n, "dft", (1, 1, 1), a
            mesh = tuple(int(x) for x in (f / "KPOINTS").read_text().splitlines()[3].split())
            yield n, "dft_kmesh", mesh, a
    elif geometry == "chgnet":
        names = names or sorted(p.stem for p in (ROOT / "data/prerelaxed").glob("*.vasp"))
        for n in names:
            p = ROOT / "data/prerelaxed" / f"{n}.vasp"
            if p.exists():
                yield n, "chgnet", (1, 1, 1), read(p)
    elif geometry == "pormake":
        names = names or list(pd.read_csv(ROOT / "data/catalogue/passed.csv").name)
        for n in names:
            yield n, "pormake", (1, 1, 1), read(ROOT / "data/catalogue/cifs" / f"{n}.cif")


def compute(args):
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "xtb_candidates.csv"
    done = set()
    if out.exists():
        done = {(r["name"], r["geometry"]) for r in csv.DictReader(open(out))}
    new = not out.exists()
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        if new:
            w.writeheader()
        for name, geo, sc, atoms in structures(args.geometry, args.names):
            if (name, geo) in done:
                continue
            a = atoms.repeat(sc) if sc != (1, 1, 1) else atoms
            row = {"name": name, "geometry": geo, "supercell": "x".join(map(str, sc)),
                   "n_atoms": len(a), "uhf": unpaired(a)}
            if geo == "dft_kmesh" and (sc == (1, 1, 1) or len(a) > args.max_super):
                row.update(status="skipped" if sc != (1, 1, 1) else "same_as_gamma")
            else:
                t = time.time()
                try:
                    g, h, l = run(a)
                    row.update(gap_eV=round(g, 4), homo_eV=round(h, 4), lumo_eV=round(l, 4), status="ok")
                except Exception as exc:  # noqa: BLE001
                    row.update(status=f"error: {str(exc)[:80]}")
                row["seconds"] = round(time.time() - t, 1)
            w.writerow(row)
            fh.flush()
            print(name, geo, row.get("gap_eV"), row["status"], row.get("seconds"), flush=True)


def surrogate_prospective() -> pd.DataFrame:
    """mu, sigma of the surrogate that selected each structure (prospective) and of the
    first model (iteration 0) for all candidates."""
    dft = pd.read_csv(ROOT / "data/dft/dft_results.csv")
    rows = []
    for it in sorted(ROOT.glob("data/al/iter[0-9]*")):
        k = int(it.name[4:])
        p = pd.read_csv(it / "predictions.csv").set_index("name")
        for n in dft[dft.iteration == k].name:
            if n in p.index:
                rows.append({"name": n, "mu_prospective": p.at[n, "mu"],
                             "sigma_prospective": p.at[n, "sigma_tree"]})
    return pd.DataFrame(rows).drop_duplicates("name")


def summarize():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = pd.read_csv(OUT / "xtb_candidates.csv")
    x = x[x.status == "ok"]
    dft = pd.read_csv(ROOT / "data/dft/dft_results.csv")
    lab = dft[dft.label_ok.astype(bool)][["name", "gap_eV", "metal", "topology", "substituent"]] \
        .rename(columns={"gap_eV": "gap_dft"})
    lab["closed_shell"] = lab.metal.isin(["Zn", "Mg"])
    sur = surrogate_prospective()
    wide = x.pivot_table(index="name", columns="geometry", values="gap_eV")
    cmp_ = lab.merge(sur, on="name", how="left").merge(wide, left_on="name", right_index=True, how="left")
    cmp_.to_csv(OUT / "xtb_vs_dft_vs_surrogate.csv", index=False, float_format="%.3f", lineterminator="\n")
    rows = []
    for grp, g in [("closed-shell (Zn)", cmp_[cmp_.closed_shell]), ("open-shell 3d", cmp_[~cmp_.closed_shell])]:
        for col in ["mu_prospective", "dft", "dft_kmesh", "chgnet", "pormake"]:
            if col not in g:
                continue
            ok = g[col].notna()
            if ok.sum() == 0:
                continue
            err = g.loc[ok, col] - g.loc[ok, "gap_dft"]
            rows.append({"group": grp, "estimate": col, "n": int(ok.sum()), "MAE_eV": err.abs().mean(),
                         "bias_eV": err.mean(), "max_abs_err_eV": err.abs().max()})
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "xtb_candidates_summary.csv", index=False, float_format="%.3f", lineterminator="\n")
    print(cmp_.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    # catalogue-wide comparison: surrogate (iteration 0) vs xTB on CHGNet geometries
    p0 = pd.read_csv(ROOT / "data/al/iter0/predictions.csv")[["name", "mu", "sigma_tree", "metal", "open_shell"]]
    cat = p0.merge(wide[["chgnet"]] if "chgnet" in wide else pd.DataFrame(), left_on="name",
                   right_index=True, how="inner")
    cat.to_csv(OUT / "catalogue_xtb_vs_surrogate.csv", index=False, float_format="%.3f", lineterminator="\n")
    if len(cat):
        from scipy.stats import spearmanr
        for shell, g in cat.dropna(subset=["mu", "chgnet"]).groupby("open_shell"):
            print(f"catalogue, open_shell={shell}: n={len(g)}, Spearman(mu, xTB) = "
                  f"{spearmanr(g.mu, g.chgnet)[0]:.2f}, median xTB - mu = {np.median(g.chgnet - g.mu):.2f} eV")
    # figure: DFT label vs surrogate and vs xTB
    MM = 1 / 25.4
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1",
                         "grid.linewidth": 0.5, "pdf.fonttype": 42, "savefig.bbox": "tight",
                         "savefig.dpi": 300})
    fig, ax = plt.subplots(figsize=(90 * MM, 85 * MM))
    lim = (-0.1, 4.0)
    ax.plot(lim, lim, color="#52514e", lw=0.8, ls="--")
    best_xtb = cmp_["dft_kmesh"].fillna(cmp_["dft"]) if "dft_kmesh" in cmp_ else cmp_.get("dft")
    for shell, mk in ((True, "o"), (False, "s")):
        g = cmp_.closed_shell == shell
        lab_s = "closed-shell Zn" if shell else "open-shell 3d"
        ax.scatter(cmp_.loc[g, "gap_dft"], cmp_.loc[g, "mu_prospective"], marker=mk, s=22,
                   color="#8a8984", label=f"surrogate, {lab_s}", zorder=3)
        if best_xtb is not None:
            ax.scatter(cmp_.loc[g, "gap_dft"], best_xtb[g], marker=mk, s=22, color="#2a78d6",
                       label=f"GFN1-xTB (DFT geometry), {lab_s}", zorder=4)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("DFT (PBE-D3) gap of the label (eV)")
    ax.set_ylabel("Estimated gap (eV)")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    fdir = ROOT / "results/figures"
    fig.savefig(fdir / "fig_xtb_candidates.pdf")
    fig.savefig(fdir / "fig_xtb_candidates.png")
    print("saved fig_xtb_candidates")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", choices=["dft", "chgnet", "pormake"])
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--max_super", type=int, default=700)
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        return summarize()
    if not args.geometry:
        ap.error("--geometry is required unless --summarize")
    compute(args)


if __name__ == "__main__":
    main()
