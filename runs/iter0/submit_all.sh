#!/bin/bash
# Build POTCARs and submit every structure of an iteration folder.
# Usage (on Grace):  cd /scratch/user/$USER/iter0 && bash submit_all.sh
# Re-running is safe: folders with static/STATIC_DONE are skipped, and folders that
# are queued/running (job name match) are skipped.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
running=$(squeue -u "$USER" -h -o "%j" 2>/dev/null || true)
for d in */; do
  name=${d%/}
  [ -d "$name/relax" ] || continue
  if [ -f "$name/static/STATIC_DONE" ]; then echo "[done]    $name"; continue; fi
  if echo "$running" | grep -qx "${name:0:60}"; then echo "[queued]  $name"; continue; fi
  [ -s "$name/relax/POTCAR" ] || bash "$ROOT/make_potcar.sh" "$name/relax" > /dev/null
  (cd "$name" && sbatch --job-name="${name:0:60}" run.slrm)
done
