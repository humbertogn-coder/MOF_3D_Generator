#!/bin/bash
# Pack the outputs needed for analysis (no WAVECAR/CHG*/POTCAR) into one archive.
# Usage (on Grace):  cd /scratch/user/$USER/iter0 && bash collect_results.sh
# Produces iter0_results.tar.gz next to the folders; copy it to your PC.
set -euo pipefail
cd "$(dirname "$0")"
tag=$(basename "$PWD")
files=$(for d in */; do n=${d%/}; [ -d "$n/relax" ] || continue
  for f in relax/OUTCAR relax/vasprun.xml relax/CONTCAR relax/OSZICAR relax/INCAR relax/KPOINTS \
           relax/RELAX_DONE static/OUTCAR static/vasprun.xml static/DOSCAR static/EIGENVAL \
           static/OSZICAR static/INCAR static/KPOINTS static/POSCAR static/STATIC_DONE; do
    [ -f "$n/$f" ] && echo "$n/$f"
  done; done)
tar czf "${tag}_results.tar.gz" $files
ls -lh "${tag}_results.tar.gz"
