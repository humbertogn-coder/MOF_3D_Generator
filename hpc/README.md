# Running a DFT iteration on TAMU Grace

Every iteration folder written by `scripts/07_prepare_vasp.py` (e.g. `runs/iter0`) is
self-contained: one sub-folder per structure (`relax/`, `static/`, `run.slrm`) plus the
helper scripts `submit_all.sh`, `make_potcar.sh`, `check_status.sh`, `collect_results.sh`.

Defaults in `run.slrm` match the earlier Grace runs:
- partition `medium`, 24 h, 48 cores, 120 GB;
- Slurm account: not hard-coded. Run `myproject` on Grace to see your active
  allocations (they renew each fiscal year), then
  `echo <account> > .account` in the iteration folder (or `ACCOUNT=<account> bash submit_all.sh`);
- modules `intel/2022a` and `vasp/6.3.2`;
- `vasp_gam` for Γ-only meshes, `vasp_std` otherwise.

POTCARs come from `/sw/restricted/vasp/sw/potentials/potpaw_PBE.52`. To use another set,
export `VASP_PP=/path/to/potpaw` before `submit_all.sh`.

```bash
# 1. copy the folder to scratch (from your PC), then on Grace:
cd /scratch/user/$USER/iter0
bash submit_all.sh          # builds POTCARs and submits one job per structure
bash check_status.sh        # stage per structure + queue
# 2. jobs stop at 24 h; when a job ends without static/STATIC_DONE, resubmit:
bash submit_all.sh          # skips finished and queued structures, resumes the rest
# 3. when every structure shows "done":
bash collect_results.sh     # -> iter0_results.tar.gz (no WAVECAR/CHG/POTCAR)
```

The relaxation (ISIF = 3) is restarted from CONTCAR until VASP prints "reached required
accuracy" within ≤ 3 ionic steps; a static run then gives the gap and PDOS. A structure
that is not converged after 6 rounds stops with a message in `slurm_<id>.out`; inspect
`relax/vasp_round*.log` before resubmitting.

Back on the PC, extract `iter0_results.tar.gz` into `runs/iter0/` of the repository and run:

```bash
python scripts/08_parse_dft.py --runs runs/iter0 --iteration 0
```
