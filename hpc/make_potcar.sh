#!/bin/bash
# Assemble POTCAR from POTCAR.spec (one element per line, POSCAR order).
# Usage: hpc/make_potcar.sh <run_folder>/relax  [PAW_PBE directory]
# POTCARs are licensed by the VASP Software GmbH and are NOT distributed in this repo.
set -euo pipefail
dir=${1:?folder}
PAW=${2:-${VASP_PP_PATH:-$HOME/potpaw_PBE.52}}
# Materials-Project-like choices for the elements used here
declare -A PV=( [Mn]=Mn_pv [Fe]=Fe_pv [Co]=Co [Ni]=Ni_pv [Cu]=Cu_pv [Zn]=Zn [Mg]=Mg_pv [Zr]=Zr_sv
                [C]=C [H]=H [N]=N [O]=O [F]=F )
: > "$dir/POTCAR"
while read -r el; do
  [ -z "$el" ] && continue
  cat "$PAW/${PV[$el]:-$el}/POTCAR" >> "$dir/POTCAR"
done < "$dir/POTCAR.spec"
grep TITEL "$dir/POTCAR"
