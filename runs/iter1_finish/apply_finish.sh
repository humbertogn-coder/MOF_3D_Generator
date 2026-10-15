#!/bin/bash
# Finish iteration 1: stop the near-converged cds Co-F4 relaxation and run its statics
# (static + broken-symmetry static_bs); mark the two collapsed frameworks as EXCLUDED.
# Usage (on FASTER):  bash apply_finish.sh /scratch/user/$USER/iter1
# Nothing is deleted.
set -euo pipefail
SRC=$(cd "$(dirname "$0")" && pwd)
DST=$(cd "${1:?usage: bash apply_finish.sh <iteration folder>}" && pwd)
cd "$DST"
while read -r n; do
  [ -z "$n" ] && continue
  if [ "$(wc -l < "$n/relax/CONTCAR" 2>/dev/null || echo 0)" -le 8 ]; then echo "[skip]     $n: no usable relax/CONTCAR"; continue; fi
  cp "$SRC/$n/RELAX_DONE" "$n/relax/RELAX_DONE"
  cp "$SRC/$n/walltime" "$n/walltime"
  echo "[static]   $n (static + static_bs, walltime $(cat "$n/walltime"))"
done < "$SRC/stopped.txt"
while read -r n reason; do
  [ -z "$n" ] && continue
  [ -d "$n" ] && echo "$reason" > "$n/EXCLUDED" && echo "[excluded] $n"
done < "$SRC/excluded.txt"
echo "Done. Next: cd $DST && bash submit_all.sh"
