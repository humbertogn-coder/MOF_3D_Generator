#!/bin/bash
# Status of every structure in an iteration folder.
# Usage:  cd /scratch/user/$USER/iter0 && bash check_status.sh
cd "$(dirname "$0")"
printf "%-36s %-10s %-7s %-8s %-14s %s\n" NAME STAGE ROUNDS STEPS "LAST_E(eV)" NOTE
for d in */; do
  n=${d%/}; [ -d "$n/relax" ] || continue
  r=$(ls "$n"/relax/OUTCAR.round* 2>/dev/null | wc -l)
  steps=$(grep -c "F=" "$n/relax/OSZICAR" 2>/dev/null); steps=${steps:-0}
  e=$(grep "F=" "$n/relax/OSZICAR" 2>/dev/null | tail -1 | awk '{print $5}')
  if   [ -f "$n/static/STATIC_DONE" ]; then st=done
  elif [ -f "$n/relax/RELAX_DONE" ];   then st=static
  elif [ -f "$n/relax/OUTCAR" ];       then st=relaxing
  else st=pending; fi
  note=""
  last_log=$(ls -t "$n"/relax/vasp_round*.log 2>/dev/null | head -1)
  [ -n "$last_log" ] && grep -q "ZBRENT\|EDDDAV\|BRMIX\|Error" "$last_log" && note="check log"
  printf "%-36s %-10s %-7s %-8s %-14s %s\n" "${n:0:36}" "$st" "$r" "$steps" "${e:-}" "$note"
done
echo; squeue -u "$USER" -o "%.10i %.40j %.8T %.10M %.10L" 2>/dev/null
