# `export/` benchmarks

This directory holds the timing harness and the recorded results for the exported
`pair_style ace` code path.

The *measurement protocol* below was fixed by Task 0 and must not be rewritten. The
Results section at the bottom holds Task 4's baseline rows and Task 5's B1 rows, all taken
on this host. Do not add numbers here that were not measured.

## Measurement protocol

Every timing block in this plan is taken the same way. A number quoted without these
conditions is not a result.

### Correctness first

A configuration is **verified before it is timed**. Nothing in this directory may quote a
time for a build that has not passed its accuracy gate:

| comparison | tolerance |
|---|---|
| generated Julia code vs `ETACEPotential` / `StackedCalculator` | `1e-12` |
| compiled library via the Python C API vs Julia | `1e-12` |
| `pair_style ace` in LAMMPS vs Julia, 1 and 2 MPI ranks | `1e-10` |
| `OMP_NUM_THREADS=4` vs serial (once the workspace API exists) | `1e-12` |
| a performance step vs the previous step's exported model (forces) | `1e-13` relative |

Tolerances are never loosened. A force differing by more than `1e-13` relative between two
performance steps is a bug, not a speed-up. `:hermite_spline` is compared to the
**splinified** model at `1e-12`; its (much larger) error against the **fitted** model is
*reported*, not asserted. Every recorded result states which reference it used.

### Timing conditions

* Single MPI rank, pinned to one idle core: `taskset -c <N> ... `, `OMP_NUM_THREADS=1`.
* LAMMPS input: `timestep 0.0`, **100 steps**.
* Two runs must agree within **3 %**; otherwise take a third run and report the median.
* The `pace recursive` (ML-PACE) comparator is re-run **the same day on the same core** as
  the ACE numbers it is compared with. A cross-day or cross-core comparison is not reported.
* `cat /proc/loadavg` is recorded immediately before each timing block and quoted with the
  result. This is a shared host: yield to other users, and never time on a contended core.
  (`nproc` reports 1 here because `OMP_NUM_THREADS=1` is inherited; the host really has 32
  cores, affinity 0-31.)

### What is recorded per block

| field | example |
|---|---|
| date | 2026-09-15 |
| core (`taskset -c`) | 7 |
| `/proc/loadavg` before the block | `0.03 0.21 1.24` |
| binary / library under test | `verify_cantor/lib/cantor_poly.so` |
| reference the accuracy gate used | `E0+many-body ETACE stack` |
| measured accuracy | `max\|dF\| = 2.1e-14 eV/Å` |
| run 1 / run 2 (/ run 3) wall time | `12.41 s / 12.55 s` |
| reported value | median, with the spread |

## Reusable pieces

* `export/test/fixtures/cantor_fixture.jl` — the Cantor reference model rebuilt from saved
  parameters (never refit: the fit is 43 min), the 10 rotated held-out geometries, and the
  17-digit references `verify_cantor/ref_{1..10}.txt`.
* `export/test/check_export.jl` — `check_export` / `check_export_report`, the accuracy gate
  above applied to a generated `.jl` model file.
* `verify_cantor/compare_common.py`, `compare_pylib.py`, `compare_lammps.py` — Python-side
  comparators.
* `verify_cantor/profile/profile_export.jl` — Julia-side profiling.

Added by Task 4 (all in this directory unless stated):

* `bench_parity.sh` — one timed row; the stable interface Tasks 5-8 call.
* `in.bench_ace`, `in.bench_pace`, `box_cantor.lmp`, `box_tial.lmp` — the LAMMPS timing
  inputs. The box is an `include ${box}`, so the two pair styles provably run the identical
  geometry; `in.bench_pace` says `pace recursive` explicitly. (Supersede
  `verify_cantor/in.bench_ace` / `in.bench_pace`, which hard-code the Cantor box and leave the
  `pace` evaluator to the build default.)
* `verify_bench_models.jl` — exports and gates the generated SOURCE at 1e-12, and writes the
  gate manifest `bench_parity/<tag>.gated`.
* `compile_bench_libs.sh` — juliac each gated export into `bench_parity/libace_<tag>.so`.
* `test_assert_ranks.jl` — a standalone check that gate C's rank assertion actually fires
  (no LAMMPS, no MPI needed): `julia export/bench/test_assert_ranks.jl`.
* `gate_bench_libs.jl` — the LIBRARY-level gates (LAMMPS vs Julia 1e-10, library via the
  Python C API vs Julia 1e-12, 2 ranks vs 1 rank 1e-12), appended to the same manifest
  together with the library's sha256. `bench_parity.sh` refuses to time a library whose
  manifest is missing, stale, or says `library_gates=FAIL`.
* `fit_tial_order4.jl` — the TiAl order-4 reference fit (`--probe` prints basis size vs
  `max_level`).
* `make_pace_basis.py` — the size-matched `pair_style pace` comparator (multi-species).
* `export/test/fixtures/tial_fixture.jl` — `load_tial_fixture()` and friends.

## Results

### Baseline rows — 2026-09-16, moriarty.scrtp.warwick.ac.uk, core 31

Raw rows: `bench_parity/rows_task4.txt` (the protocol block) and
`bench_parity/rows_task4_repeat.txt` (repeats and diagnostics). Every row was produced by
`export/bench/bench_parity.sh`, which writes the date, time, core, `/proc/loadavg`, box,
step count and both binaries into the row itself.

Common to every row below: 1 MPI rank, `taskset -c 31`, `OMP_NUM_THREADS=1`,
`timestep 0.0`, 100 steps, `pair_style pace recursive` re-run in the same block on the same
core, `/proc/loadavg` between 0.06 and 1.16 (the ~1.0 is this benchmark's own single pinned
process).

**Every row's generated SOURCE was gated at 1e-12 before it was timed**
(`export/bench/verify_bench_models.jl`, finished 14:43:05; the libraries were compiled
14:44-14:45 and the rows taken 14:54-15:11; full output in `bench_parity/verify.log`):

| tag | accuracy reference | max&#124;dE&#124;/atom | max&#124;dF&#124; | max&#124;dV&#124;/atom | gate |
|---|---|---|---|---|---|
| `cantor_poly` | `E0 + pair + ETACE` (the fitted stack) | 1.599e-14 | 1.867e-14 | 1.614e-13 | **PASS** @ 1e-12 |
| `cantor_h50`  | `E0 + pair + splinified(Nspl=50) ETACE` | 1.421e-14 | 1.982e-14 | 1.800e-13 | **PASS** @ 1e-12 |
| `tial_poly`   | `E0 + pair + ETACE` (the fitted stack) | 2.695e-13 | 5.511e-13 | 6.626e-13 | **PASS** @ 1e-12 |
| `tial_h50`    | `E0 + pair + splinified(Nspl=50) ETACE` | 2.695e-13 | 6.240e-13 | 8.600e-13 | **PASS** @ 1e-12 |

The Hermite rows' error against the **fitted** stack is reported, never asserted (it is model
error, not export error): Cantor `max|dF| = 2.68e-04` eV/Å, TiAl `max|dF| = 1.65e-02` eV/Å.

**The LIBRARY-level gates** (`export/bench/gate_bench_libs.jl`, full output in
`bench_parity/gate_libs.log`; recorded per tag in `bench_parity/<tag>.gated`).

*Ordering, stated exactly:* these three were run at **17:08**, i.e. **after** the 14:54-15:11
timing block — the script did not exist until it was added in response to a review finding that
they had never been run at all. The guarantee still holds for the four rows above because the
`.so` files were not rebuilt in between: their mtimes are the 14:44-14:45 compile times and
each sha256 matches its manifest, so the binaries that were timed are bit-for-bit the binaries
that were gated. For every row taken from that point on the ordering is *enforced*:
`bench_parity.sh` refuses a library whose manifest is missing, stale or failing. The source gate
above says nothing about the compiled `.so` that is actually timed: a juliac or `cpu_target`
miscompilation would change what the library computes, and therefore what it costs, without
touching the generated `.jl`. All three run on the benchmark's own box at `-var cells 5`
(500 / 250 atoms, same lattice and neighbour count per atom), built once serially and handed to
every run through one `write_data` file, so the atom ids are identical across rank counts.

Gate C's two-rank run is **verified to have happened**, from LAMMPS' own output, before its
numbers are used: `Loop time of … on 2 procs` and a `1 by 1 by 2 MPI processor grid` are both
asserted (and recorded in the manifest as `mpi2_decomposition_confirmed=1x1x2 procs=2`). Without
that, a `mpirun` belonging to a different MPI than LAMMPS was linked against launches two
*independent serial* jobs, each computing the whole cell and printing the same energy — and gate
C then agrees perfectly while proving nothing about ghost atoms or the neighbour list. The
assertion throws; it never skips. `export/bench/test_assert_ranks.jl` exercises it on synthetic
output, including that exact hazard.

Gate C compares the **energy relatively** (`|dE| / |E_total|`, tol **1e-13**) and the **forces
absolutely** (`max|dF|`, tol **1e-12**) — the same extensive/intensive split
`export/test/check_export.jl` draws for the virial, and for the same reason. See "Why gate C's
energy is relative" below; the per-atom and total energy deviations are printed and recorded
too, but are not the gate.

| tag | N | A: LAMMPS vs Julia (abs, 1e-10) | B: library via Python C API vs Julia (abs, 1e-12) | C energy: &#124;dE&#124;/&#124;E&#124; (rel, 1e-13) | C forces: max&#124;dF&#124; (abs, 1e-12) | C reported-only: &#124;dE&#124;/atom, &#124;dE&#124; total | verdict |
|---|---|---|---|---|---|---|---|
| `cantor_poly` | 500 | dE/at 9.10e-15, dF 1.73e-14 | dE/at 0.00e+00, dF 4.20e-14 | **1.478e-15** (68× inside) | **2.70e-15** | 2.00e-14 eV/atom, 1.00e-11 eV | **PASS** |
| `cantor_h50`  | 500 | dE/at 3.64e-15, dF 1.72e-14 | dE/at 9.10e-15, dF 3.89e-14 | **0.000e+00** | **3.62e-15** | 0, 0 | **PASS** |
| `tial_poly`   | 250 | dE/at 1.63e-12, dF 5.24e-13 | dE/at 5.82e-13, dF 5.27e-13 | **4.581e-15** (22× inside) | **7.63e-14** | 3.96e-12 eV/atom, 9.90e-10 eV | **PASS** |
| `tial_h50`    | 250 | dE/at 1.86e-12, dF 6.52e-13 | dE/at 1.16e-13, dF 5.25e-13 | **0.000e+00** | **4.52e-14** | 0, 0 | **PASS** |

**Why gate C's energy is relative.** It was absolute-per-atom at first (`|dE|/natoms` ≤ 1e-12
eV/atom, the convention `export/lammps/test/compare_dump.py` uses), and `tial_poly` "failed" it
at 3.96e-12. It was the gate that was wrong, not the library:

* `E0(Ti) = −1586.02` eV/atom against `E0(Cr) ≈ −14.4` makes the 250-atom TiAl cell total
  −216027 eV and the 500-atom Cantor cell −6768 eV, so a flat 1e-12 eV/atom gate is **8.6 ulp**
  of the total on TiAl and **550 ulp** on Cantor — two orders of magnitude tighter in relative
  terms on one model than on the other, purely because of the reference energies.
* TiAl measured **34 ulp**, where summing ~2000 terms in a different order is expected to drift
  O(√N) ≈ 45 ulp. A tolerance no correct implementation can meet is not a gate.
* Every run is deterministic (1-rank twice and 2-rank twice: each bit-identical); **2 ranks and
  4 ranks produce the identical total**, so it is the serial-vs-parallel summation path, not
  accumulation across domains; and the forces from the same runs agree to 7.6e-14. A real
  rank-to-rank fault — a ghost atom missing from one domain — is an eV-scale effect, ~1e-5
  relative here, ten orders of magnitude above the gate.

**ΣE0 is deliberately NOT subtracted before the comparison.** That would remove a term in order
to make the check pass, which the plan's constraints forbid; keeping E0 in and changing the
metric's *dimension* is the honest fix. **Residual risk, stated rather than discovered later:**
an absolute energy error below `1e-13·|E|` passes — 2.2e-8 eV total (8.6e-11 eV/atom) for the
250-atom TiAl cell, 6.8e-10 eV total (1.4e-12 eV/atom) for the 500-atom Cantor cell. Nothing
else re-checks the rank-to-rank energy, so this gate is the only thing behind it.

#### The rows

| tag | box | ace ms/step | runs (ms/step) | spread | pace recursive ms/step | spread | **ratio** |
|---|---|---|---|---|---|---|---|
| `cantor_poly` | 2048-atom fcc CrMnFeCoNi | **1064.3** | 1077.5 / 1051.1 | 2.5 % | 130.55 | 0.2 % | **8.15** |
| `cantor_h50`  | 2048-atom fcc CrMnFeCoNi | **419.1**  | 421.9 / 416.3   | 1.3 % | 131.09 | 1.3 % | **3.20** |
| `tial_poly`   | 2000-atom bcc TiAl       | **841.4**† | 808.8 / 841.4 / 892.6 | 10.4 % | 371.31 | 0.4 % | **2.27**† |
| `tial_h50`    | 2000-atom bcc TiAl       | **530.7**  | 537.8 / 523.6   | 2.7 % | 369.88 | 0.0 % | **1.43** |

**Row format note.** The rows above were taken with the pre-fix-round row format. The reported
statistic is unchanged — it was, and is, the SAMPLE MEDIAN (for n = 2 the midpoint of the two,
for n = 3 the middle one), so the numbers are directly comparable with anything Tasks 5-8
produce. What changed is that a row now (a) prints the runs in EXECUTION ORDER rather than
sorted, (b) names the statistic, (c) carries the library's resolved path, sha256 and mtime,
(d) carries `natoms` and a `us/site` column, (e) carries its gate verdict, and (f) keeps every
LAMMPS screen log beside it. One casualty: the three `tial_poly` runs recorded in the first
block were printed sorted, so their execution order is lost and cannot be recovered.

† `tial_poly` is the one configuration that does **not** settle inside 3 %: its first two runs
disagreed by 10.4 %, so a third was taken and the median reported, as the protocol requires.
Four further blocks were then taken on the same core (`bench_parity/rows_task4_repeat.txt`).
Across all 14 runs the values are 797.8 … 925.2 ms/step, median **868.3** (ratio **2.35**),
i.e. **±7 % block-to-block**, while the `pace` side of the same blocks is stable to 0.4 % and
every other ACE configuration is stable to ≤ 2.7 %. **Tasks 5-7 must not read a `tial_poly`
speed-up smaller than ~10 % out of two or three runs.**

**It is irreducible noise, not a measurement artefact that could be fixed.** Once the runs were
printed in execution order (they used to be sorted), fresh `tial_poly` series come out
830.90 / 918.72 / 870.32 and 856.60 / 897.34 / 869.32 ms/step — the slowest run is in the
*middle* both times. That rules out monotonic drift, i.e. thermal throttling or any warm-up
effect, which would show the runs getting steadily slower. What is left is run-to-run scatter,
so the response is more samples (≥ 5 runs for any `tial_poly` comparison), not a change to how
the runs are taken.

#### What the two models and the two comparators are

| | Cantor | TiAl |
|---|---|---|
| species | Cr Mn Fe Co Ni (5) | Ti Al (2) |
| correlation order | 3 | 4 |
| many-body basis / species | 1348 | **2369** |
| pair basis / species | 30 | 20 |
| rcut | 6.25 Å | 5.5 Å |
| box | fcc a = 3.59, 8³ cells, 2048 atoms, 201.3 neighbours/atom | bcc a = 3.19, 10³ cells, 2000 atoms, 112.1 neighbours/atom |
| fit | `verify_cantor/chain_cantor.jl` (params in `verify_cantor/cantor_v010_params.jld2`) | `export/bench/fit_tial_order4.jl` (params in `bench_parity/tial_o4_params.jld2`) |
| `pace` comparator | `~/si-ace/spike_yace/cantor/cantor_n10000_exact.yace` | `bench_parity/tial_o4_pace.yace` |
| comparator provenance | the **same physical model**, an ACEpotentials v0.6 fit exported exactly by `spike_yace/cantor_all.jl` | **size-matched only**, random coefficients, built by `export/bench/make_pace_basis.py` |
| comparator C-tilde functions / element | 1065 (60 rank-1 + 1005 rank>1) — **21 % fewer** than our 1348 | 2299 (44 + 2255) — **3.0 % fewer** than our 2369 |
| `init_Wradial` | `:onehot` | `:onehot` |

**Neither benchmark model exercises a dense radial weight matrix.** Both are built with
`init_Wradial = :onehot` and neither fit touches `Wnlq`, so `RBASIS_W_k` is one-hot in both
exported libraries and Task 5's DENSE `RBASIS_W_k` code path has **no benchmark coverage here
at all**. Task 5 must construct its own dense case (and gate it) rather than assuming these
rows cover it.

The Cantor box is 2048 atoms, above the brief's "1728-2000" window. That is deliberate: it is
`verify_cantor/in.bench_ace`'s box verbatim, which is what makes these rows comparable with the
pre-plan ≈7x / ≈2.6x finding. The TiAl box (2000) is inside the window.

The comparator binary is
`/storage/eng/essswb/lammps-jax-build/lammps/build-SKX-AMPERE86-acejl/lmp` (ACE version
2023.11.25), **not** the `-mlpace` build named in the plan's environment notes: that build's
newer `ace-evaluator` rejects both comparator files with `Exception: bad conversion`. Using one
binary for both rows keeps them comparable.

#### Reading the ratios

* The Cantor `:polynomial` ratio is **8.15**, against the finding's ≈ 7. The difference is
  accounted for, not waved away. The finding's library was produced by the pre-Task-1
  generator, which **silently dropped the pair term**. Timing that same old library today on
  the same core (`cantor_poly_NOPAIR_diag`, a DIAGNOSTIC — it fails its accuracy gate at
  `max|dF| = 6.88` eV/Å and is not a parity row) gives **999.8 ms/step** against the correct
  library's 1064.3: the pair term the generator now exports costs **+6.5 %**, and the pair-less
  ratio is **7.66**, within ~9 % of the finding's round "≈7".
* Same for Hermite: `cantor_h50_NOPAIR_diag` gives **360.3 ms/step** (ratio **2.75**) against
  the correct library's 419.1 (ratio 3.20) — the pair term costs **+16.3 %** here, and 2.75 is
  within 6 % of the finding's ≈ 2.6.
* The Cantor ratios are pessimistic by a further factor the table above names: our model has
  1348 many-body functions per species and the comparator 1065, i.e. **27 % more work**. The
  TiAl comparator is matched to 3 %, so the TiAl ratios are the cleaner cost comparison.
* The TiAl ratios (**2.27** polynomial, **1.43** Hermite) are new. That the order-4 /
  2369-function model is only 2.3× `pace` while the order-3 / 1348-function Cantor model is
  8.2× says the exported code's overhead is not proportional to basis size — the fixed
  per-edge cost (radial basis, spherical harmonics, neighbour handling) dominates at Cantor's
  size and is amortised at TiAl's. This is the single most useful number Tasks 5-7 have to
  work with: whatever they optimise has to show up on **both** boxes.

#### Diagnostic rows (NOT parity rows — these libraries fail their accuracy gate)

| tag | ace ms/step | runs | spread | note |
|---|---|---|---|---|
| `cantor_poly_NOPAIR_diag` | 999.8 | 1007.2 / 992.4 | 1.5 % | pre-Task-1 `verify_cantor/lib/libace_cantor_poly.so`; no pair term; `max\|dF\| = 6.88` eV/Å vs the fitted stack |
| `cantor_h50_NOPAIR_diag`  | 360.3 | 360.31 / 360.32 | 0.0 % | pre-Task-1 `verify_cantor/lib/libace_cantor_hermite50.so`; same defect |

They exist only to attribute the gap between this table and the finding's ≈7× / ≈2.6×. They
must never be quoted as performance results.

### How to reproduce

```bash
cd ~/ace-potentials-julia-1.2/ACEpotentials.jl
mkdir -p bench_parity

# 1. the TiAl reference model (~6 min; the Cantor one is NEVER refit)
nohup julia --project=export export/bench/fit_tial_order4.jl > bench_parity/fit_tial.log 2>&1 &
julia --project=export export/bench/fit_tial_order4.jl --probe      # basis size vs max_level

# 2. the matched pace comparator (needs the cp39 pyace venv; see make_pace_basis.py)
/storage/eng/essswb/venvs/pyacenv/bin/python export/bench/make_pace_basis.py \
    --target 2369 --order 4 --lmax 7 --rcut 5.5 --elements Ti,Al \
    --nradmax-by-orders 22,6,2,1 --out bench_parity/tial_o4_pace.yace

# 3. export + gate the SOURCE at 1e-12, compile, then gate the LIBRARIES
julia --project=export export/bench/verify_bench_models.jl
export/bench/compile_bench_libs.sh
julia --project=export export/bench/gate_bench_libs.jl

# 4. the rows (check /proc/loadavg first; never time on a contended core)
for tag in cantor_poly cantor_h50 tial_poly tial_h50; do
  case $tag in cantor*) pace=~/si-ace/spike_yace/cantor/cantor_n10000_exact.yace ;;
               *)       pace=$PWD/bench_parity/tial_o4_pace.yace ;; esac
  CORE=31 OUT=$PWD/bench_parity/rows_task4.txt \
    export/bench/bench_parity.sh $tag bench_parity/libace_$tag.so $pace 100
done
```

`bench_parity.sh <tag> <lib.so> <pace-file|none> [steps]` is the stable interface Tasks 5-8
call. It refuses any library without a valid gate manifest; `ALLOW_UNGATED="<reason>"` (no
manifest at all) and `ALLOW_PARTIAL_GATE=1` (gated, but not every gate passed) are the two
explicit escape hatches, and the row itself then reads `gate=UNGATED(...)` or `gate=PARTIAL[...]`
so an ungated number cannot be mistaken for a gated one. As of 2026-09-16 all four benchmark
libraries pass every gate, so neither hatch is needed for any row in this file; the only use of
`ALLOW_UNGATED` was the two pre-Task-1 NOPAIR diagnostics. `none` skips the `pace` half, which is what an ACE-vs-ACE comparison between two
generator versions wants; re-run the `pace` half in the same block on the same core whenever a
ratio is quoted. `BOX` (cantor|tial|path), `CORE`, `PLUGIN`, `OUT`, `LMP_ACE` and `LMP_PACE`
are the environment knobs.

### Task 5 rows (B1: radial mixing from `W`'s sparsity, pruning, integer powers) — 2026-09-16, core 31

Raw rows: `bench_parity/rows_task5.txt` (the four `*_b1` rows with their `pace` comparator) and
`bench_parity/rows_task5_repeat.txt` (the ACE-vs-ACE repeat blocks, `pace=none`). Conditions
are the ones above: 1 MPI rank, `taskset -c 31`, `OMP_NUM_THREADS=1`, `timestep 0.0`, 100 steps,
`/proc/loadavg` in the row, `pace recursive` re-run in the same block on the same core. Every
LAMMPS screen log shows `Pair | … | 99.98 %` of the loop and `Neighbor list builds = 0`, so the
numbers are pair-style cost, and `Ave neighs/atom` is identical to the baseline blocks
(201.33789 Cantor, 112.13 TiAl) — the same geometry is being timed.

The libraries are `bench_parity/libace_<tag>_b1.so`, exported by
`verify_bench_models.jl cantor_poly_b1 cantor_h50_b1 tial_poly_b1 tial_h50_b1`
(source gate, `bench_parity/verify_b1.log`) and gated by
`gate_bench_libs.jl` on the same tags (library gates, `bench_parity/gate_libs_b1.log`),
**both before any of them was timed** — `bench_parity.sh`'s interlock enforces it and every row
below reads `gate=OK[src,lib,lammps,mpi2]`.

Every gate figure is **identical to the Task 4 baseline to every printed digit**, which is what
`export/test/test_generator_parity.jl` predicts: on these four models the B1 generator's output
is bit-identical to `3570eb8e`'s.

| tag | source gate max&#124;dE&#124;/at / max&#124;dF&#124; / max&#124;dV&#124;/at | A LAMMPS vs Julia | B lib vs Julia | C 2-vs-1 rank (rel E / abs F) | verdict |
|---|---|---|---|---|---|
| `cantor_poly_b1` | 1.599e-14 / 1.867e-14 / 1.614e-13 | 9.10e-15, 1.73e-14 | 0.00e+00, 4.20e-14 | 1.478e-15 / 2.70e-15 | **PASS** |
| `cantor_h50_b1`  | 1.421e-14 / 1.982e-14 / 1.800e-13 | 3.64e-15, 1.72e-14 | 9.10e-15, 3.89e-14 | 0.000e+00 / 3.62e-15 | **PASS** |
| `tial_poly_b1`   | 2.695e-13 / 5.511e-13 / 6.626e-13 | 1.63e-12, 5.24e-13 | 5.82e-13, 5.27e-13 | 4.581e-15 / 7.63e-14 | **PASS** |
| `tial_h50_b1`    | 2.695e-13 / 6.240e-13 / 8.600e-13 | 1.86e-12, 6.52e-13 | 1.16e-13, 5.25e-13 | 0.000e+00 / 4.52e-14 | **PASS** |

#### The rows

`pace recursive` was re-run in each of the four blocks below: 129.98 / 131.01 ms/step (Cantor)
and 369.65 / 370.52 ms/step (TiAl), i.e. within 0.8 % of Task 4's 130.55 / 131.09 / 371.31 /
369.88. The comparator has not moved, so the ratio column is comparable with the baseline table.

| tag | box | ace ms/step (median) | runs (exec order) | spread | µs/site | pace recursive | **ratio** | baseline ratio |
|---|---|---|---|---|---|---|---|---|
| `cantor_poly_b1` | 2048-atom fcc CrMnFeCoNi | **456.21** | 455.96 / 456.47 | 0.1 % | 222.8 | 129.98 | **3.51** | 8.15 |
| `cantor_h50_b1`  | 2048-atom fcc CrMnFeCoNi | **421.90** | 417.74 / 426.05 | 2.0 % | 206.0 | 131.01 | **3.22** | 3.20 |
| `tial_poly_b1`   | 2000-atom bcc TiAl       | **567.96**† | see below (n = 7) | 4.5 % | 284.0 | 369.65 | **1.54** | 2.27 |
| `tial_h50_b1`    | 2000-atom bcc TiAl       | **528.89** | 533.09 / 524.69 | 1.6 % | 264.4 | 370.52 | **1.43** | 1.43 |

† `tial_poly` has the ±7 % block-to-block scatter Task 4 established, so the B1 figure is the
median of **7** runs taken in three separate blocks, in execution order:
569.06 / 546.73 / 565.10 (18:39), 569.92 / 567.96 (18:52), 571.06 / 557.02 (18:57).

#### Same-session re-measurement of the PREVIOUS libraries

Because `tial_poly` scatters by ±7 % block to block, a cross-day comparison of medians is not
enough to claim a speed-up on it. The Task 4 libraries were therefore re-timed on the same core
in the same session, interleaved with the new ones, with `pace=none`:

| library | Task 4 row | re-measured 2026-09-16 19:0x | runs (exec order) | agreement |
|---|---|---|---|---|
| `libace_cantor_poly.so` | 1064.3 | **1039.91** (n = 3) | 1055.68 / 985.57 / 1039.91 | 2.3 % |
| `libace_cantor_h50.so`  | 419.1  | **415.83** (n = 2) | 414.38 / 417.27 | 0.8 % |
| `libace_tial_poly.so`   | 841.4 (868.3 over 14 runs) | **861.66** (n = 5) | 855.14 / 888.00 / 884.40 (18:47), 860.13 / 861.66 (18:54) | 0.8 % of the 14-run median |
| `libace_tial_h50.so`    | 530.7  | **528.89** (n = 3) | 551.15 / 518.55 / 528.89 | 0.3 % |

The baselines reproduce. The honest same-day, same-core, same-session speed-ups are therefore

| model | previous generator | B1 | speed-up |
|---|---|---|---|
| Cantor `:polynomial` | 1039.91 | 456.21 | **2.28x** |
| Cantor `:hermite_spline` | 415.83 | 421.90 | 0.99x (**no change**) |
| TiAl `:polynomial` | 861.66 (n = 5) | 567.96 (n = 7) | **1.52x** |
| TiAl `:hermite_spline` | 528.89 | 528.89 | 1.00x (**no change**) |

#### What the numbers say, including what they do not

* **B1 is a `:polynomial`-mode change and the rows say exactly that.** It closes essentially the
  whole gap between the polynomial and the Hermite path: on Cantor that gap was
  1039.91 − 415.83 = **624 ms/step** and is now 456.21 − 421.90 = **34 ms/step**; on TiAl it was
  861.66 − 528.89 = **333 ms** and is now 567.96 − 528.89 = **39 ms**. 94 % and 88 % of it is
  gone. It shows on **both** boxes, which is what Task 4 required of any optimisation.
* **The Hermite rows do not move at all**, even though the B1 knot tables are 4.5x smaller
  (`cantor_h50` 4.72 → 1.05 MB) and the cubic is evaluated 8x narrower (9 rows instead of 74).
  That is not a failed change, it is a measurement of where the Hermite cost is *not*: a Hermite
  edge touches only **4 knots**, i.e. 4 × 74 × 8 = 2368 bytes before and 288 after, so neither
  the table size nor the SIMD-wide cubic was ever the bottleneck. What B1 removed from the
  polynomial path — per edge, **two 74x45 dense matrix-vector products reading 26 KB of weights
  out of a 666 KB table** — has no counterpart in the Hermite path.
* **The plan's ~150 µs/site target for Cantor `:polynomial` is not reached: the row is 222.8**
  (from 507.8 today / 519.7 at the baseline). It cannot be reached by any further work on the
  radial basis, and the Hermite row is the evidence: `cantor_h50_b1` runs at **206.0 µs/site**
  with a 9-wide radial evaluation, so ~206 µs/site is the cost of everything that is *not* the
  radial basis — neighbour handling, the `N_RNL`-wide embedding copies, `abasis`/`aabasis`, the
  A2B contraction, and the force/virial assembly, which still loops `for t in 1:N_RNL` (74) per
  edge and applies a rank-1 virial update inside that loop. The polynomial row now sits 8 %
  above that floor. Getting to 150 µs/site is a Task 6/7 problem, not a B1 one.
* **The largest remaining per-edge item that B1 deliberately did not touch** is the polynomial
  recurrence itself. `eval_polys_ed` evaluates `N_POLYS` terms — 45 on Cantor, 33 on TiAl — but
  the emitted `RBASIS_SEL_k` tables show that **only polynomials 1..6 are ever read on Cantor and
  1..11 on TiAl**. The brief scopes the width change to Task 6's kernel, so this was left alone
  and is recorded here as the next lever.
