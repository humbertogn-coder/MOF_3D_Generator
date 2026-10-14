#!/bin/bash
# Finish iteration 0 on TAMU FASTER (the iteration folder on /scratch is shared with Grace).
# Usage (on FASTER):  bash apply_faster.sh /scratch/user/$USER/iter0
#  * static_rerun.txt: bcu statics that ran out of memory on Grace (48 ranks, 120 GB, KPAR = 4):
#                      KPAR -> 2, walltime 12 h; the failed log is kept as static/vasp_oom_grace.log
#  * new.txt         : broken-symmetry spin checks of the two Cu paddlewheels (static only);
#                      relax/POTCAR is copied from the parent folder (built on Grace)
#  * sbatch_extra    : FASTER options (64 cores, 240 GB); .account is not changed, so submit with
#                      ACCOUNT=<FASTER account> bash submit_all.sh
# Nothing is deleted.
set -euo pipefail
SRC=$(cd "$(dirname "$0")" && pwd)
DST=$(cd "${1:?usage: bash apply_faster.sh <iteration folder>}" && pwd)
cd "$DST"
while read -r n; do
  [ -z "$n" ] && continue
  [ -f "$n/static/STATIC_DONE" ] && { echo "[done]     $n"; continue; }
  [ -f "$n/static/vasp.log" ] && [ ! -f "$n/static/vasp_oom_grace.log" ] && mv "$n/static/vasp.log" "$n/static/vasp_oom_grace.log"
  sed -i 's/^KPAR = .*/KPAR = 2/' "$n/static/INCAR"
  echo 12:00:00 > "$n/walltime"
  echo "[static]   $n ($(grep '^KPAR' "$n/static/INCAR"), walltime 12 h)"
done < "$SRC/static_rerun.txt"
while read -r n; do
  [ -z "$n" ] && continue
  if [ -d "$n" ]; then echo "[skip]     $n (exists)"; continue; fi
  p=$(cat "$SRC/$n/parent")
  [ -s "$p/relax/POTCAR" ] || { echo "[error]    $p/relax/POTCAR missing"; exit 1; }
  cp -r "$SRC/$n" . && cp "$p/relax/POTCAR" "$n/relax/POTCAR" && echo "[new]      $n"
done < "$SRC/new.txt"
cp "$SRC/sbatch_extra.faster" sbatch_extra
cp "$SRC/submit_all.sh" .
echo "Done. Next (on FASTER): cd $DST && ACCOUNT=142737597217 bash submit_all.sh"
