# `export/` benchmarks

This directory holds the timing harness and the recorded results for the exported
`pair_style ace` code path.

**Status: stub.** Task 0 fixes the *measurement protocol* only. The result tables are
filled in by Task 4 from measurements actually taken on this host. Do not add numbers
here that were not measured.

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
* `verify_cantor/in.bench_ace`, `in.bench_pace` — the LAMMPS timing inputs.
* `verify_cantor/profile/profile_export.jl` — Julia-side profiling.

## Results

*(Task 4)*
