#!/bin/bash
# Build relax/POTCAR for every structure of an iteration folder (run on Grace, where the PAW
# library is; /scratch is shared with FASTER, so the jobs can then be submitted from FASTER).
# Usage:  cd /scratch/user/$USER/iter1 && bash make_potcars_all.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
for d in */; do
  n=${d%/}
  [ -f "$n/relax/POTCAR.spec" ] || continue
  [ -s "$n/relax/POTCAR" ] && { echo "[ok]  $n"; continue; }
  bash "$ROOT/make_potcar.sh" "$n/relax"
done
