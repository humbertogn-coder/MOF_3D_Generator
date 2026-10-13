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

## Protocol v2 (from the iteration-0 restart on)

Iteration 0 showed three problems with the first protocol:

1. **Strained starting geometries.** 48 of the 131 generated frameworks have a compressed
   C–C bond (< 1.30 Å, down to 1.10 Å) at the node–linker junction, and more have short
   C–H/N–H bonds (`data/catalogue/compressed_bonds_report.csv`). Initial DFT forces reached
   12–35 eV/Å, the spin-polarised SCF hit NELM at nearly every ionic step, and five
   relaxations made only 3–6 ionic steps in 24 h. **Fix:** CHGNet pre-relaxation
   (`scripts/09_prerelax_chgnet.py`, separate environment `requirements-prerelax.txt`);
   `07_prepare_vasp.py --prerelaxed data/prerelaxed` uses the result. CHGNet volumes were
   within 1 % of the converged DFT volumes for MOF-5 and the Cu-paddlewheel cds framework.
2. **Cost.** Relaxations now use the Γ point when every lattice vector is ≥ 12 Å (static
   runs keep the full mesh), EDIFFG = −0.05 eV/Å, NELM = 120, LMAXMIX = 4, ALGO = All for
   Fe/Co/Ni (no tetrahedron smearing with ALGO = All), and a per-structure `walltime` file
   (12 h up to 160 atoms, 24 h above) so that fewer SUs are reserved per job.
3. **Convergence rule and collapse.** Soft frameworks kept drifting along flat modes, so a
   relaxation is now accepted after a restarted round (round ≥ 2) that reaches the force
   criterion. `08_parse_dft.py` rejects frameworks whose volume changed by more than 10 % or
   whose cell angles changed by more than 10° relative to the generated structure (two
   M3(CO2)6 pcu frameworks lost ~50 % of their volume, the Zn-paddlewheel cds framework
   sheared by 25°). A folder with an `EXCLUDED` file is never resubmitted.

The restart package for iteration 0 is `runs/iter0_restart/` (`apply_restart.sh`).

### Iteration-0 finish (static-only, `runs/iter0_static/`)

The protocol-v2 relaxations of the bcu and cds Cu-NH2 frameworks reached the 12 h walltime
after 97–240 ionic steps with final forces of 0.08–0.22 eV/Å; the gap along the last 50 steps
varied by ≤ 0.06 eV. With the allocation nearly spent, these relaxations are stopped
(`relax/RELAX_DONE` starting with "stopped" plus a note) and only the static run is done;
`08_parse_dft.py` accepts such runs if the final force is ≤ 0.25 eV/Å and marks them
`relax_stopped`. dia Ni-NH2 (SCF failures) and nbo Co-F4 (unstable relaxation) are excluded.
Cu paddlewheels get a second static with antiferromagnetic initial moments (`<name>__afm`);
the lower-energy spin solution provides the label (for cds Cu-Nring1 the FM start collapsed
to a non-magnetic state 0.32 eV above the magnetic solution; gap 0.26 vs 0.53 eV).
