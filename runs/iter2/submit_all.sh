#!/bin/bash
# Build POTCARs and submit every structure of an iteration folder.
# Usage (on Grace):  cd /scratch/user/$USER/iter0 && bash submit_all.sh [name ...]
#   no arguments -> every structure; with names -> only those folders (e.g. a test job)
# Re-running is safe: folders with static/STATIC_DONE are skipped, and folders that
# are queued/running (job name match) are skipped. A file <name>/walltime (e.g. 08:00:00)
# overrides the 24 h default; shorter requests reserve fewer SUs.
# Slurm account (TAMU HPRC: list yours with `myproject`), first match wins:
#   ACCOUNT=123456789 bash submit_all.sh      or      echo 123456789 > .account
# Cluster: an optional file sbatch_extra overrides the #SBATCH defaults of run.slrm (Grace:
# medium, 48 cores), e.g. for TAMU FASTER:  echo "--partition=cpu --ntasks=64 --mem=240G" > sbatch_extra
# POTCARs: built with make_potcar.sh where the PAW library exists (Grace); on a cluster without
# it, build them first on Grace (same /scratch):  bash make_potcars_all.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
ACCOUNT=${ACCOUNT:-$(cat "$ROOT/.account" 2>/dev/null || true)}
acct=(); [ -n "$ACCOUNT" ] && acct=(--account="$ACCOUNT")
echo "Slurm account: ${ACCOUNT:-<default of your user>}"
extra=(); [ -s "$ROOT/sbatch_extra" ] && read -r -a extra < "$ROOT/sbatch_extra"
[ ${#extra[@]} -gt 0 ] && echo "sbatch options: ${extra[*]}"
running=$(squeue -u "${USER:-$(whoami)}" -h -o "%j" 2>/dev/null || true)
targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=(*/)
for d in "${targets[@]}"; do
  name=${d%/}
  [ -d "$name/relax" ] || continue
  if [ -f "$name/static/STATIC_DONE" ] && { [ ! -d "$name/static_bs" ] || [ -f "$name/static_bs/STATIC_DONE" ]; }; then
    echo "[done]    $name"; continue; fi
  if [ -f "$name/EXCLUDED" ]; then echo "[excluded] $name: $(cat "$name/EXCLUDED")"; continue; fi
  if echo "$running" | grep -qx "${name:0:60}"; then echo "[queued]  $name"; continue; fi
  if [ ! -s "$name/relax/POTCAR" ]; then
    bash "$ROOT/make_potcar.sh" "$name/relax" > /dev/null || { echo "[no POTCAR] $name: run 'bash make_potcars_all.sh' on Grace first"; continue; }
  fi
  wt=$(cat "$name/walltime" 2>/dev/null || echo 24:00:00)   # optional per-structure walltime
  (cd "$name" && sbatch "${acct[@]}" "${extra[@]}" --time="$wt" --job-name="${name:0:60}" run.slrm)
done
