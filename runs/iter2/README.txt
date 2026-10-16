Iteration 2 (protocol v2, TAMU FASTER), prepared 2026-10-15.

Surrogate retrained with the 11 accepted labels of iterations 0-1 (data/al/iter2). The cds Co-F4
label of iteration 1 was still pending (static run) at selection time.
Eligibility (data/al/iter2/selection_rules.json): no M3(CO2)6 pcu, no Zn- or Mn-paddlewheel cds,
no Fe4 bcu (collapsed or sheared in DFT), <= 160 atoms; caps 4 per topology, 2 per metal,
4 per substituent. 26 eligible candidates.
Batch (8): bcu Co4-H, Co4-NH2 | cds Fe-NH2, Fe-Nring1, Cu-long-Nring1, Cu-long-F8 |
  pcu Zn4O-short-Nring1, Zn4O-long-Nring1. CHGNet pre-relaxed; paddlewheels also run static_bs.

1) On Grace:   cd /scratch/user/$USER && tar xzf iter2.tar.gz && cd iter2 && bash make_potcars_all.sh
2) On FASTER:  cd /scratch/user/$USER/iter2 && echo 142737597217 > .account && bash submit_all.sh
SUs reserved at submission: 4 x 1536 + 4 x 768 = 9,216 (charged: time actually used).

Outcome (closed 2026-10-16; FASTER balance after the run 11,206 SUs): 6 labels accepted, 2 not.
  bcu Co-H 0.379 (last relaxation SCF stopped at NELM 120 with dE 1.5e-5 eV; the converged static
  on the same geometry gives Fmax 0.048 eV/A, so the relaxation is accepted) | bcu Co-NH2 0.332 |
  cds Cu-long-F8 0.890 (broken symmetry, 0.45 eV below FM start) |
  cds Fe-Nring1 0.035 (FM; broken-symmetry static not converged in 300 steps, 1.3 eV above FM
  and drifting by only ~0.02 eV over the last 60 steps) |
  Zn4O-short-Nring1 3.114 | Zn4O-long-Nring1 3.029.
  Not accepted:
  cds Fe-NH2: relaxation hit the 12 h walltime after 235 ionic steps; energy and volume still
    decreasing (-0.14 eV and -1.4 % over the last 35 steps, Fmax 0.15-0.27 eV/A), so it was not
    stopped by hand; no static run.
  cds Cu-long-Nring1: FM-start static SCF did not converge in 300 steps (energy oscillating by
    eV); the broken-symmetry start converged to a mixed state (one Cu2 unit antiferromagnetic,
    the other ferromagnetic; total moment -2 muB, gap 0.62 eV). Spin solution undetermined.
  Iteration 2 is the last DFT iteration; data/al/iter3 holds the final surrogate (18 labels).
