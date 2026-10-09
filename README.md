# Active learning for low-band-gap 3D metal–organic frameworks

Code and data for the manuscript *"Active learning workflow for the systematic
identification of three-dimensional metal–organic framework candidates with low
band gap"* (C. H. Guerrero Navarro, M. Ghotbi, P. B. Balbuena; Texas A&M University).

The workflow:

1. assemble a chemically audited catalogue of hypothetical 3D MOFs with PORMAKE;
2. apply structural quality control (QC);
3. compute Zeo++ pore descriptors;
4. predict PBE band gaps with an ExtraTrees ensemble trained on QMOF;
5. select DFT candidates with a multi-objective acquisition function;
6. validate them with VASP (PBE-D3(BJ));
7. feed accepted labels back to the surrogate.

> **Status:** catalogue v3 and the iteration-0 selection are complete. DFT
> iterations are in progress; `data/dft/` is filled as runs finish.

## Repository layout

```
src/mofal/             library code
  catalogue.py         building-block design (nodes, linkers, topologies)
  qc.py                structural QC (overlaps, non-bonded contacts, fragments, valence)
  features.py          descriptor sets A-E (set C = surrogate features)
  surrogate.py         ExtraTrees ensemble, uncertainty, out-of-distribution flags
  acquisition.py       LCB + topology novelty + OOD penalty, diversity caps
  vasp.py              VASP inputs (no POTCAR) and output parsing
  pdos.py              band-edge (VBM/CBM) orbital character from PDOS
scripts/               numbered pipeline steps (see below)
hpc/                   SLURM driver and POTCAR helper for the cluster
data/
  external/            QMOF csv (not redistributed; see data/external/README.md)
  pormake_audit/       audit of all PORMAKE nodes/edges (charge balance, carboxylates)
  catalogue/           catalogue.csv (all), passed.csv (QC-passed), cifs/, zeopp.csv
  al/iter<k>/          predictions, acquisition ranking, selected batch per iteration
  dft/                 dft_results.csv (gaps, convergence, QC, label_ok), band edges
results/               benchmark tables, surrogate analysis, figures (PDF)
```

## Installation

```bash
conda env create -f environment.yml      # or: pip install -r requirements.txt
conda activate mofal
```

Tested with Python 3.13. Key packages: PORMAKE 0.2.3, ASE, pymatgen, scikit-learn,
pyzeo (Zeo++ bindings), xgboost, lightgbm, shap.

## Reproducing the results

| Step | Command | Produces |
|---|---|---|
| QMOF benchmark (Table S1) | `python scripts/00_qmof_benchmark.py --qmof data/external/qmof.csv --out results/benchmark` | `results/benchmark/*.csv` |
| PORMAKE audit (Table S2) | `python scripts/01_audit_pormake_bbs.py --out data/pormake_audit` | `nodes.csv`, `edges.csv` |
| Catalogue + QC | `python scripts/02_generate_catalogue.py --out data/catalogue` | `catalogue.csv`, `passed.csv`, `cifs/` |
| Pore descriptors | `python scripts/03_zeopp.py` | `data/catalogue/zeopp.csv` |
| Surrogate, iteration k | `python scripts/05_train_predict.py --iteration k [--dft data/dft/dft_results.csv]` | `data/al/iter<k>/predictions.csv` |
| Batch selection | `python scripts/06_select_batch.py --iteration k --n N [--dft ...]` | `selected_batch.csv` |
| VASP inputs | `python scripts/07_prepare_vasp.py --batch data/al/iter<k>/selected_batch.csv --out runs/iter<k>` | `runs/iter<k>/<name>/` |
| Parse DFT | `python scripts/08_parse_dft.py --runs runs/iter<k> --iteration k` | `data/dft/dft_results.csv` |
| Surrogate trends | `python scripts/10_surrogate_analysis.py --iteration 0` | `results/surrogate/*.csv` |
| Figures | `python scripts/11_make_figures.py --iteration k` | `results/figures/*.pdf` |
| QMOF PBE vs HSE06 | `python scripts/12_qmof_pbe_vs_hse06.py` | `results/qmof_pbe_vs_hse06/*.csv` |
| GFN1-xTB gaps, 2,000 QMOF MOFs (~13 CPU h) | `python scripts/13_xtb_validation.py --cifs data/external/qmof_xtb_subset/cifs --mode periodic` | `results/xtb/xtb_gfn1_gaps.csv` |
| Periodic vs cluster xTB analysis | `python scripts/14_xtb_analysis.py` | `results/xtb/xtb_summary.csv`, `fig_xtb_periodic_vs_cluster.pdf` |
| Retrospective AL benchmark on QMOF (20 repeats, ~45 min on 2 cores) | `python scripts/15_retrospective_al.py` then `--summarize` | `results/retro_al/*.csv`, `fig_retrospective_al.pdf` |

All random seeds are fixed. The benchmark split is `train_test_split(test_size=0.2,
stratify=gap<1 eV, random_state=42)`; the surrogate ensemble uses seeds
42, 123, 456, 789 and 2024. Trained models are not distributed (≈1.5 GB each); they
are rebuilt deterministically by the scripts in a few minutes.

## Methods summary (defaults in code)

- **Catalogue.**
  - Nodes are native PORMAKE carboxylate clusters, charge balanced for M²⁺. The metal label is read from the building block.
  - Edges are carboxylate-free spacers, so the assembled linkers are BDC, NH₂-BDC, F₄-BDC, pyridine-2,5-dicarboxylate, BPDC, F₈-BPDC, phenylpyridine- and bipyridine-dicarboxylate.
  - Node families and topologies:

    | Node family | Topology | Metals |
    |---|---|---|
    | M₃(CO₂)₆ | pcu | Mn, Fe, Co, Zn, Mg |
    | Zn₄O(CO₂)₆ | pcu | Zn |
    | M₂(CO₂)₄ paddlewheel | nbo, cds | Mn, Fe, Co, Cu, Zn |
    | M₂(CO₂)₄ tetrahedral | dia | Co, Ni, Zn |
    | M₄(CO₂)₈ | bcu | Fe, Co |

- **QC** (`src/mofal/qc.py`). A structure is rejected for any of:
  - a pair of atoms closer than 0.70 Å;
  - a heavy–heavy or H–H pair closer than 1.00 Å;
  - a non-bonded contact shorter than 0.8 × the sum of covalent radii;
  - more than one connected component, or any isolated atom;
  - an H atom that does not have exactly one neighbour, or a C atom with an organic coordination number outside 2–4;
  - a composition different from the building-block sum;
  - density outside 0.20–3.0 g cm⁻³, a cell length below 5 Å, more than 600 atoms, or a node RMSD above 0.30 Å.
- **Descriptors.** Feature set C (64 descriptors): geometry plus composition plus element fractions. PLD and LCD come from Zeo++ (`-ha OCC -res`, CCDC radii).
- **Surrogate.** Five ExtraTrees regressors (300 trees, `max_features="sqrt"`, `min_samples_leaf=2`). μ is the ensemble mean; σ is the spread over all trees of all members (`sigma_tree`). The spread of the five member means (`sigma_seed`) is also reported.
- **Acquisition.**
  - Score: `mu - kappa*sigma - alpha/max(1, N_topo) + beta*I_ood`, with κ = 1.0, α = 0.3 and β = 0.5.
  - Selection keeps at most 2 candidates per topology and 2 per metal in each batch.
  - I_ood = 1 if density, PLD, n_atoms, metal fraction or C/atom lies outside the 5–95 % range of QMOF.
- **DFT** (VASP 6.3.2).
  - Settings: PBE-D3(BJ), 520 eV cutoff, spin polarised with a high-spin ferromagnetic initial guess, Γ-centred k-mesh with n_i = round(25 Å / a_i).
  - Relaxation: full cell (ISIF = 3, EDIFFG = −0.03 eV/Å), restarted until VASP reports "reached required accuracy".
  - Static run: EDIFF = 10⁻⁶ eV, LORBIT = 11.
  - A gap is used as a training label only if the relaxation and static SCF converged, the final structure passes QC, and no non-metal atom carries a moment above 0.3 μB.

## Data availability

The CIFs, QC tables, predictions and DFT summary tables are in this repository.
Raw VASP outputs (`vasprun.xml`, `OUTCAR`, `DOSCAR`; POTCARs excluded for licence
reasons) are deposited on Zenodo: DOI to be added. QMOF is available from
https://doi.org/10.6084/m9.figshare.13147324 (cite Rosen et al., Matter 2021).

## License

Code: MIT. Data produced in this work: CC-BY-4.0.
