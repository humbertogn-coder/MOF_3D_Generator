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
