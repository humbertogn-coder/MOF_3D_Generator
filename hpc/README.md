# Running the DFT step on the cluster

1. Copy `runs/iter<k>/` to the cluster (scratch).
2. In every `<name>/relax` folder: `bash hpc/make_potcar.sh <name>/relax $VASP_PP_PATH/potpaw_PBE.52`
   (POTCAR order follows `POTCAR.spec`, i.e. POSCAR order; Materials-Project PAW choices).
3. `cd <name> && sbatch run.slrm` - relaxation (ISIF=3) restarted from CONTCAR until VASP
   prints "reached required accuracy" within <= 3 ionic steps (max 4 rounds), then a static
   run in `static/` (CONTCAR of the relaxation, EDIFF=1e-6, LORBIT=11).
4. If the job hits the walltime: `cp relax/CONTCAR relax/POSCAR` and resubmit.
5. Copy back `relax/{OUTCAR,vasprun.xml,CONTCAR,OSZICAR}` and
   `static/{OUTCAR,vasprun.xml,DOSCAR,EIGENVAL}` and run
   `python scripts/08_parse_dft.py --runs runs/iter<k> --iteration <k>`.
