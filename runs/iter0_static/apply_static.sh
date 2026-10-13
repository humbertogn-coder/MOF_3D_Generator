#!/bin/bash
# Static-only finish of iteration 0 (after the protocol-v2 relaxations hit the walltime).
# Usage:  bash apply_static.sh /scratch/user/$USER/iter0
#  * stopped.txt : relaxation stopped here (RELAX_DONE with a note); the static run starts
#                  from relax/CONTCAR; shorter walltime
#  * new.txt     : AFM-start static for the second Cu paddlewheel (same geometry)
#  * excluded.txt: structures not worth more SUs (EXCLUDED file, never resubmitted)
# Nothing is deleted.
set -euo pipefail
SRC=$(cd "$(dirname "$0")" && pwd)
DST=$(cd "${1:?usage: bash apply_static.sh <iteration folder>}" && pwd)
cd "$DST"
while read -r n; do
  [ -z "$n" ] && continue
  if [ "$(wc -l < "$n/relax/CONTCAR" 2>/dev/null || echo 0)" -le 8 ]; then echo "[skip]     $n: no usable relax/CONTCAR"; continue; fi
  cp "$SRC/$n/RELAX_DONE" "$n/relax/RELAX_DONE"
  cp "$SRC/$n/walltime" "$n/walltime"
  echo "[static]   $n (walltime $(cat "$n/walltime"))"
done < "$SRC/stopped.txt"
while read -r n; do
  [ -z "$n" ] && continue
  if [ -d "$n" ]; then echo "[skip]     $n (exists)"; else cp -r "$SRC/$n" . && echo "[new]      $n"; fi
done < "$SRC/new.txt"
while read -r n reason; do
  [ -z "$n" ] && continue
  [ -d "$n" ] && echo "$reason" > "$n/EXCLUDED" && echo "[excluded] $n"
done < "$SRC/excluded.txt"
echo "Done. Next: cd $DST && bash submit_all.sh"
