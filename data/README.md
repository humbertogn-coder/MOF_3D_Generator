# Data files

## `pormake_audit/`
| file | content |
|---|---|
| `nodes.csv` | every PORMAKE node: metals, connection number, number of carboxylates, extra O/OH/H2O, anionic charge, implied mean metal oxidation state |
| `edges.csv` | every PORMAKE edge: composition, whether it carries carboxylate groups, angle between connection vectors |

## `catalogue/`
| file | content |
|---|---|
| `catalogue.csv` | all 168 enumerated candidates with design metadata and QC results (`passed`, `qc_reasons`) |
| `passed.csv` | the 131 QC-passed candidates |
| `cifs/` | CIFs of the QC-passed candidates (PORMAKE output, P1) |
| `zeopp.csv` | Zeo++ LCD/PLD (Å), cell volume (Å^3), density (g cm^-3) |

Naming: `<topology>__<metal>-<node_id>__<length>-<substituent>`, e.g.
`pcu__Zn-N16__short-H` = Zn4O(BDC)3 (MOF-5) in pcu.

Key columns: `node_family`, `node_id`, `metal`, `metal_ox_state`, `open_shell`,
`linker_label`, `edge_id`, `linker_family` (BDC/BPDC), `substituent`,
`node_rmsd_A`, `min_distance`, `n_overlaps`, `n_nonbonded_contacts`,
`n_components`, `composition_ok`.

## `al/iter<k>/`
| file | content |
|---|---|
| `predictions.csv` | surrogate mean `mu`, `sigma_tree` (spread over all trees), `sigma_seed` (spread of the 5 member means), OOD flag and reasons |
| `acquisition_ranking.csv` | pool ranked by acquisition score |
| `selected_batch.csv` | candidates sent to DFT in iteration k |
| `model_info.json` | training-set size, seeds, hyperparameters, package versions |

## `dft/`
`dft_results.csv`: one row per DFT run with PBE gap (eV), VBM/CBM, convergence flags
(`relax_converged_ionic`, `relax_converged_electronic`, `static_converged_electronic`),
final-structure QC, magnetisation summary (`mag_*`), `label_ok` and `label_reasons`.
Only rows with `label_ok == True` are used to retrain the surrogate.
