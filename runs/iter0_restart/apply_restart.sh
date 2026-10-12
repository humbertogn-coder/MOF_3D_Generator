#!/bin/bash
# Apply the iteration-0 restart (protocol v2) to the iteration folder on Grace.
# Usage:  bash apply_restart.sh /scratch/user/$USER/iter0
#  * restart.txt : old relax/static outputs move to <name>/attempt1/, new inputs start
#                  from the CHGNet-pre-relaxed geometry (Gamma relaxation for cells >= 12 A,
#                  ALGO = All for Fe/Co/Ni, LMAXMIX = 4, EDIFFG = -0.05, shorter walltime)
#  * excluded.txt: frameworks that collapsed in DFT get an EXCLUDED file (never resubmitted)
#  * new.txt     : extra runs (AFM spin-state check of the Cu paddlewheel, static only)
# Nothing is deleted.
set -euo pipefail
SRC=$(cd "$(dirname "$0")" && pwd)
DST=$(cd "${1:?usage: bash apply_restart.sh <iteration folder>}" && pwd)
cd "$DST"
while read -r n; do
  [ -z "$n" ] && continue
  [ -d "$n" ] || { echo "missing $n - skipped"; continue; }
  if [ -d "$n/attempt1" ]; then echo "[skip]     $n (already restarted)"; continue; fi
  mkdir "$n/attempt1"
  for x in "$n"/relax "$n"/static "$n"/run.slrm "$n"/walltime "$n"/slurm_*; do
    [ -e "$x" ] && mv "$x" "$n/attempt1/"
  done
  cp -r "$SRC/$n/relax" "$SRC/$n/static" "$SRC/$n/run.slrm" "$SRC/$n/walltime" "$n/"
  echo "[restart]  $n (walltime $(cat "$n/walltime"))"
done < "$SRC/restart.txt"
while read -r n reason; do
  [ -z "$n" ] && continue
  [ -d "$n" ] && echo "$reason" > "$n/EXCLUDED" && echo "[excluded] $n"
done < "$SRC/excluded.txt"
while read -r n; do
  [ -z "$n" ] && continue
  if [ -d "$n" ]; then echo "[skip]     $n (exists)"; else cp -r "$SRC/$n" . && echo "[new]      $n"; fi
done < "$SRC/new.txt"
cp "$SRC"/submit_all.sh "$SRC"/check_status.sh "$SRC"/collect_results.sh "$SRC"/make_potcar.sh .
echo "Done. Next: cd $DST && bash submit_all.sh"
