Iteration-0 restart (protocol v2), prepared 2026-10-12.

On Grace:
  cd /scratch/user/$USER
  tar xzf iter0_restart.tar.gz
  bash iter0_restart/apply_restart.sh /scratch/user/$USER/iter0
  cd iter0 && bash submit_all.sh && bash check_status.sh

Restarted from CHGNet-pre-relaxed geometries (old outputs kept in <name>/attempt1/):
  bcu__Co-N35__short-F4, bcu__Fe-N350__short-F4, bcu__Fe-N350__short-NH2 (12 h each)
  cds__Cu-N409__short-NH2 (12 h), nbo__Co-N317__short-F4 (24 h), dia__Ni-N649__short-NH2 (24 h)
Excluded (framework collapse/shear in DFT): pcu__Mn-N548__short-H, pcu__Mn-N548__short-Nring1,
  cds__Zn-N153__short-H
New: cds__Cu-N409__short-Nring1__afm (antiferromagnetic static check, 4 h)
Finished, untouched: pcu__Zn-N16__short-H (MOF-5), cds__Cu-N409__short-Nring1

SUs reserved at submission: 4 x 576 + 2 x 1152 + 192 = 4,800 (balance 6,739). Charged SUs are
only the time actually used.
