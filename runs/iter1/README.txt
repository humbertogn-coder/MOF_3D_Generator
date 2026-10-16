Iteration 1 (protocol v2, TAMU FASTER), prepared 2026-10-14.

Batch (data/al/iter1/selected_batch.csv; rules in data/al/iter1/selection_rules.json):
  cds  Mn-, Fe-, Co-, Cu-paddlewheel + F4-BDC  (metal series at fixed node, linker and topology;
       Mn-F4 forced as anchor of the series)
  bcu  Fe-H, Co-Nring1
  pcu  Zn4O-NH2 (IRMOF-3; QMOF PBE 2.35 eV, second protocol check), Zn4O-long-Nring2
Eligibility (from iteration 0): no M3(CO2)6 pcu (collapsed), no Zn-paddlewheel cds (sheared),
  <= 160 atoms; caps 4 per topology, 2 per metal, 4 per substituent.
Starting geometries: CHGNet pre-relaxed (data/prerelaxed). Paddlewheel frameworks also run a
  broken-symmetry static (static_bs); the lower-energy spin solution gives the label.

1) On Grace (PAW library; /scratch is shared with FASTER):
     cd /scratch/user/$USER && tar xzf iter1.tar.gz && cd iter1 && bash make_potcars_all.sh
2) On FASTER:
     cd /scratch/user/$USER/iter1 && echo 142737597217 > .account && bash submit_all.sh
     bash check_status.sh
SUs reserved at submission: 6 x 768 + 2 x 1536 = 7,680 (charged: time actually used).

Outcome (closed 2026-10-16; data/dft/dft_results.csv): 6 labels accepted, 2 excluded.
  bcu Co-Nring1 0.372 | cds Co-F4 0.668 (relaxation stopped at Fmax 0.105 eV/A, runs/iter1_finish;
  broken-symmetry start converged to a non-magnetic state 0.26 eV below the FM state) |
  cds Cu-F4 0.339 (broken symmetry) | cds Fe-F4 0.046 (FM, 0.84 eV below broken symmetry) |
  IRMOF-3 2.514 | Zn4O-long-Nring2 2.750.
  Excluded: bcu Fe-H (volume -25 %), cds Mn-F4 (volume -21 %, still decreasing).
