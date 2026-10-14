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
