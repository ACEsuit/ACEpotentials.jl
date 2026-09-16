# `export/` benchmarks

This directory holds the timing harness and the recorded results for the exported
`pair_style ace` code path.

The *measurement protocol* below was fixed by Task 0 and must not be rewritten. The
Results section at the bottom holds Task 4's baseline rows, all taken on this host.
Do not add numbers here that were not measured.

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
* `verify_bench_models.jl` — exports and **gates** the four benchmark models at 1e-12.
* `compile_bench_libs.sh` — juliac each gated export into `bench_parity/libace_<tag>.so`.
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

**Every row was gated at 1e-12 before it was timed** (`export/bench/verify_bench_models.jl`,
full output in `bench_parity/verify.log`):

| tag | accuracy reference | max&#124;dE&#124;/atom | max&#124;dF&#124; | max&#124;dV&#124;/atom | gate |
|---|---|---|---|---|---|
| `cantor_poly` | `E0 + pair + ETACE` (the fitted stack) | 1.599e-14 | 1.867e-14 | 1.614e-13 | **PASS** @ 1e-12 |
| `cantor_h50`  | `E0 + pair + splinified(Nspl=50) ETACE` | 1.421e-14 | 1.982e-14 | 1.800e-13 | **PASS** @ 1e-12 |
| `tial_poly`   | `E0 + pair + ETACE` (the fitted stack) | 2.695e-13 | 5.511e-13 | 6.626e-13 | **PASS** @ 1e-12 |
| `tial_h50`    | `E0 + pair + splinified(Nspl=50) ETACE` | 2.695e-13 | 6.240e-13 | 8.600e-13 | **PASS** @ 1e-12 |

The Hermite rows' error against the **fitted** stack is reported, never asserted (it is model
error, not export error): Cantor `max|dF| = 2.68e-04` eV/Å, TiAl `max|dF| = 1.65e-02` eV/Å.

#### The rows

| tag | box | ace ms/step | runs (ms/step) | spread | pace recursive ms/step | spread | **ratio** |
|---|---|---|---|---|---|---|---|
| `cantor_poly` | 2048-atom fcc CrMnFeCoNi | **1064.3** | 1077.5 / 1051.1 | 2.5 % | 130.55 | 0.2 % | **8.15** |
| `cantor_h50`  | 2048-atom fcc CrMnFeCoNi | **419.1**  | 421.9 / 416.3   | 1.3 % | 131.09 | 1.3 % | **3.20** |
| `tial_poly`   | 2000-atom bcc TiAl       | **841.4**† | 808.8 / 841.4 / 892.6 | 10.4 % | 371.31 | 0.4 % | **2.27**† |
| `tial_h50`    | 2000-atom bcc TiAl       | **530.7**  | 537.8 / 523.6   | 2.7 % | 369.88 | 0.0 % | **1.43** |

† `tial_poly` is the one configuration that does **not** settle inside 3 %: its first two runs
disagreed by 10.4 %, so a third was taken and the median reported, as the protocol requires.
Four further blocks were then taken on the same core (`bench_parity/rows_task4_repeat.txt`).
Across all 14 runs the values are 797.8 … 925.2 ms/step, median **868.3** (ratio **2.35**),
i.e. **±7 % block-to-block**, while the `pace` side of the same blocks is stable to 0.4 % and
every other ACE configuration is stable to ≤ 2.7 %. **Tasks 5-7 must not read a `tial_poly`
speed-up smaller than ~10 % out of two or three runs.**

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

# 3. export + GATE at 1e-12, then compile
julia --project=export export/bench/verify_bench_models.jl
export/bench/compile_bench_libs.sh

# 4. the rows (check /proc/loadavg first; never time on a contended core)
for tag in cantor_poly cantor_h50 tial_poly tial_h50; do
  case $tag in cantor*) pace=~/si-ace/spike_yace/cantor/cantor_n10000_exact.yace ;;
               *)       pace=$PWD/bench_parity/tial_o4_pace.yace ;; esac
  CORE=31 OUT=$PWD/bench_parity/rows_task4.txt \
    export/bench/bench_parity.sh $tag bench_parity/libace_$tag.so $pace 100
done
```

`bench_parity.sh <tag> <lib.so> <pace-file|none> [steps]` is the stable interface Tasks 5-8
call. `none` skips the `pace` half, which is what an ACE-vs-ACE comparison between two
generator versions wants; re-run the `pace` half in the same block on the same core whenever a
ratio is quoted. `BOX` (cantor|tial|path), `CORE`, `PLUGIN`, `OUT`, `LMP_ACE` and `LMP_PACE`
are the environment knobs.
