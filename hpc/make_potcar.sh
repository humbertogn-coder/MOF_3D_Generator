#!/bin/bash
# Assemble relax/POTCAR from relax/POTCAR.spec (one element per line, POSCAR order).
# Usage:  bash make_potcar.sh <run_folder>/relax  [PAW_PBE directory]
# POTCARs are licensed by VASP Software GmbH and are NOT distributed in this repo.
set -euo pipefail
dir=${1:?usage: make_potcar.sh <folder>/relax [paw_dir]}
PAW=${2:-${VASP_PP:-/sw/restricted/vasp/sw/potentials/potpaw_PBE.52}}
[ -d "$PAW" ] || { echo "PAW library not found: $PAW (on FASTER build POTCARs on Grace: bash make_potcars_all.sh)" >&2; exit 1; }
# Materials-Project / QMOF (pymatgen MPRelaxSet) choices for the elements used here
declare -A PV=( [Mn]=Mn_pv [Fe]=Fe_pv [Co]=Co [Ni]=Ni_pv [Cu]=Cu_pv [Zn]=Zn [Mg]=Mg_pv [Zr]=Zr_sv
                [C]=C [H]=H [N]=N [O]=O [F]=F )
: > "$dir/POTCAR"
while read -r el; do
  [ -z "$el" ] && continue
  f="$PAW/${PV[$el]:-$el}/POTCAR"
  [ -f "$f" ] || { echo "missing $f - falling back to $PAW/$el/POTCAR" >&2; f="$PAW/$el/POTCAR"; }
  [ -f "$f" ] || { echo "no POTCAR for $el" >&2; rm -f "$dir/POTCAR"; exit 1; }
  cat "$f" >> "$dir/POTCAR"
done < "$dir/POTCAR.spec"
echo "$dir: $(grep -c TITEL "$dir/POTCAR") POTCARs:" $(grep TITEL "$dir/POTCAR" | awk '{print $4}')
