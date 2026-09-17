# ACE Model Export for LAMMPS and Python

This directory contains tools for exporting fitted ACE potentials to standalone shared libraries that can be used with LAMMPS and Python/ASE **without requiring a Julia installation**.

## Overview

The export workflow:
1. **Fit** an ACE model using ACEpotentials.jl
2. **Export** to trim-compatible Julia code
3. **Compile** to a native shared library using `juliac --trim`
4. **Deploy** with bundled runtime libraries

## Quick Start

```julia
using ACEpotentials

# Fit your model (example)
model = ACEpotentials.ace1_model(elements=[:Si], order=3, totaldegree=10)
ACEpotentials.acefit!(data, model)

# Create deployment package
include("export/scripts/build_deployment.jl")
build_deployment(model, "silicon_ace"; output_dir="deployments/")
```

This creates a self-contained deployment in `deployments/silicon_ace/` containing:
- Compiled shared library
- Julia runtime libraries (no Julia installation needed)
- LAMMPS plugin and examples
- Python/ASE calculator and examples

## Choosing ACE vs ETACE

ACEpotentials supports two evaluation backends. Choose based on your needs:

| Feature | Standard ACE | ETACE |
|---------|-------------|-------|
| **Evaluation speed** | not measured on this branch | not measured on this branch |
| **Export complexity** | Simple (`ace1_model`) | Requires conversion step |
| **Use case** | Development, small MD | Production MD, HPC |

There is no measured speed comparison between the two backends in this repository. Any
figure quoted here previously was unsourced; benchmark numbers belong in
`export/bench/README.md`, whose measurement protocol is fixed and whose "## Results" section
is filled in by the benchmarking step of the export-parity plan.

### Standard ACE Export (Simpler)

```julia
# Fit model
model = ace1_model(elements=[:Si], order=3, totaldegree=10)
acefit!(data, model)

# Export directly
include("export/scripts/build_deployment.jl")
build_deployment(model, "silicon_ace")
```

### ETACE Export (Recommended for Production)

```julia
using ACEpotentials.Models, ACEpotentials.ETModels

# Create model with learnable radial basis (required for ETACE)
ace_model = Models.ace_model(elements=(:Si,), order=3, ...)
acefit!(data, ACEPotential(ace_model, ps, st))

# Convert to ETACE.  Do NOT splinify: the default :polynomial mode exports the recurrence
# itself, exactly, and splinify() removes the recurrence it needs.
et_model = ETModels.convert2et(ace_model)

# Export (default: :polynomial, exact to 1e-12 against the fitted model)
export_ace_model(et_calc, "model.jl")
```

`splinify()` is **not** a step in this workflow, and exporting a splinified model with the
default mode is a hard error rather than a silent substitution — see
[Radial Basis Export Options](#radial-basis-export-options). It belongs only to the
`:hermite_spline` path:

```julia
# The :hermite_spline path -- approximate, slower, and needed only if the model was FITTED
# after splinification.  See the mode table below before choosing it.
et_model_splined = ETModels.splinify(et_model, et_ps, et_st; Nspl=50)
export_ace_model(et_calc_splined, "model.jl"; radial_basis=:hermite_spline)
```

See [`examples/etace_lammps_tutorial.jl`](examples/etace_lammps_tutorial.jl) for a complete walkthrough.

## Radial Basis Export Options

When exporting, choose the radial basis representation:

| Mode | Accuracy | Reference it reproduces | Speed, µs/site (Cantor / TiAl) | Status | Use case |
|------|----------|-------------------------|---|--------|----------|
| `:polynomial` | **exact** (1e-12 in energy, forces and virial) | the **fitted** model | **58.1 / 94.0** | **default** | any model that has not been splinified |
| `:hermite_spline` | approximate: 2.7e-4 eV/Å at `Nspl=50`, 3e-6 eV/Å at `Nspl=200` on the Cantor model; **1.6e-2 eV/Å** on the TiAl order-4 model | the **splinified** model | 62.5 / 152.5 (**7 % / 62 % slower**) | opt-in | a model that was *fitted after* `splinify()` — the only case it can be exported at all |

> **Do not choose `:hermite_spline` for speed.** As of the per-neighbour kernel it is the
> **slower** mode on both reference models *and* the approximate one. The speed column is
> measured — one pinned core, 2048- and 2000-atom boxes, `timestep 0.0`, 100 steps; the
> protocol, the rows and the artefacts are in [`bench/README.md`](bench/README.md). Its old
> justification, "learned radials with a small `N_POLYS`", no longer holds either:
> `:polynomial` now emits an arbitrary dense (i.e. genuinely learned) radial-mixing tensor
> exactly, and emits the recurrence only at the width a model actually reads. What remains is
> that `:polynomial` **cannot** export an already-splinified model — `splinify()` leaves no
> recurrence to emit — so `:hermite_spline` is the only route for one. See
> [`bench/FINDINGS_parity.md`](bench/FINDINGS_parity.md) §7, which puts the question of
> retiring this mode to the maintainer.

`:polynomial` re-evaluates the polynomial recurrence at runtime and reproduces the model it
was exported from to double-precision roundoff.

`:hermite_spline` emits the knot tables of a model that was already splinified with
`ETModels.splinify`, and evaluates a piecewise cubic.  It reproduces *that splinified model*
to 1e-12 — but the splinified model is not the fitted one, and the numbers in the table above
are that model error (from `verify_cantor/log.chain`), not export error.  Splinify **before**
fitting if you intend to deploy this mode, so the fit absorbs the spline discretisation into
its coefficients — the recipe is written out in the header of `export/src/splinify.jl`.  Note
that no runnable example here follows it: both `examples/etace_lammps_tutorial.jl` and
`verify_cantor/chain_cantor.jl` fit first and splinify afterwards, so the Hermite models they
produce do carry the error quoted above (the tutorial says so at its Step 6).  Two further
constraints:

* every species pair must share one cutoff — with per-pair cutoffs the splinified model
  itself throws a `BoundsError` at `y = 1` (upstream `EquivariantTensors._spl_grid`);
* never compare a `:hermite_spline` export against the fitted model and call the difference
  an export error.

`:hermite_spline` is an **explicit opt-in**.  Exporting a model that has already been
splinified without asking for it is an error, not a substitution: `splinify()` leaves no
polynomial recurrence to emit, so the exporter cannot honour `:polynomial` — and because
`:polynomial` is the default, silently substituting the spline mode would hand a caller who
never chose it an approximate model.  Either export the model as it was *before* `splinify()`
(exact, gated at 1e-12), or pass `radial_basis=:hermite_spline` to say that the approximate
mode is what you want.  The reverse mismatch, `:hermite_spline` on an unsplinified model, is
only warned about: substituting the exact mode there cannot make a result wrong.

```julia
# Polynomial (default, exact) — the model must NOT be splinified
export_ace_model(calc, "model.jl")

# Hermite cubic splines (opt-in; the model must already be splinified)
export_ace_model(calc, "model.jl"; radial_basis=:hermite_spline)
```

## Directory Structure

```
export/
├── src/                          # the generator
│   ├── export_ace_model.jl       # entry point: model -> trim-compatible code
│   ├── write_radial.jl           #   radial basis + pair term (:polynomial)
│   ├── codegen.jl                #   solid harmonics + Hermite spline tables
│   ├── splinify.jl               #   spline extraction and shared pair-index helpers
│   ├── write_evaluation.jl       #   the per-neighbour evaluation kernel and Workspace
│   ├── write_c_interface.jl      #   the @ccallable C ABI (workspace pool, handles)
│   └── build_stamp.jl            #   EXPORT_BUILD_ID / ace_build_id provenance
│
├── lammps/                       # LAMMPS integration
│   ├── plugin/                   # pair_style ace plugin source
│   │   ├── cmake/
│   │   └── src/
│   └── examples/
│
├── ase-ace/                      # Python/ASE integration (pip installable)
│   ├── src/ase_ace/              # ASE calculators
│   │   ├── calculator.py         # ACECalculator (socket-based)
│   │   ├── julia_calculator.py   # ACEJuliaCalculator (JuliaCall)
│   │   └── library_calculator.py # ACELibraryCalculator (compiled .so)
│   └── tests/
│
├── scripts/                      # Convenience scripts
│   └── build_deployment.jl       # Export + compile + package
│
└── examples/                     # Complete example workflows
    ├── silicon/                  # Basic ACE workflow (simple)
    │   └── fit_and_export.jl
    └── etace_lammps_tutorial.jl  # ETACE workflow (production)
```

## LAMMPS Usage

After deployment, use in LAMMPS:

```lammps
plugin load /path/to/aceplugin.so
pair_style ace
pair_coeff * * /path/to/libace_mymodel.so Si O ...
```

See [lammps/](lammps/) for plugin build instructions and examples.

## Python/ASE Usage

The [`ase-ace`](ase-ace/) package provides three ASE-compatible calculators:

| Calculator | Backend | Threading | Startup | Julia Required |
|------------|---------|-----------|---------|----------------|
| `ACECalculator` | Socket/i-PI | Multi-threaded | 5-10s | Yes (runtime) |
| `ACEJuliaCalculator` | JuliaCall | Multi-threaded | 10-30s | Yes (managed) |
| `ACELibraryCalculator` | Compiled .so | Single-threaded | Instant | No |

### Installation

```bash
pip install ./ase-ace              # Base package
pip install "./ase-ace[julia]"     # With JuliaCall support
pip install "./ase-ace[lib]"       # With library support (matscipy)
pip install "./ase-ace[all]"       # All backends
```

### Examples

```python
from ase.build import bulk
from ase_ace import ACELibraryCalculator, ACEJuliaCalculator

atoms = bulk('Si', 'diamond', a=5.43)

# Option 1: Compiled library (instant startup, no Julia at runtime)
calc = ACELibraryCalculator("/path/to/libace_mymodel.so")
atoms.calc = calc
energy = atoms.get_potential_energy()

# Option 2: JuliaCall (multi-threaded, auto-manages Julia)
calc = ACEJuliaCalculator("/path/to/model.json", num_threads=4)
atoms.calc = calc
energy = atoms.get_potential_energy()
descriptors = calc.get_descriptors(atoms)  # ACE basis values
```

See [ase-ace/](ase-ace/) for full documentation and examples.

## Parallelization

### LAMMPS with MPI + OpenMP

The LAMMPS plugin supports both MPI domain decomposition and OpenMP threading:

```bash
# Build plugin (from the export/lammps/plugin directory)
mkdir build && cd build
cmake ../cmake -DLAMMPS_HEADER_DIR=/path/to/lammps/src
make

# Run with MPI + OpenMP
OMP_NUM_THREADS=4 mpirun -np 4 lmp -in input.lmp
```

**Recommended**: Use MPI for domain decomposition with moderate OpenMP threading per rank.

### Python Parallelization Options

For parallel Python calculations:

1. **`ACEJuliaCalculator`**: Multi-threaded via `JULIA_NUM_THREADS`, uses JuliaCall for direct Julia integration
2. **`ACECalculator`**: Multi-threaded via socket protocol to Julia subprocess
3. **LAMMPS + LAMMPSlib**: Use ASE's LAMMPS interface for production MD with MPI parallelization

The `ACELibraryCalculator` is single-threaded due to Julia's `--trim=safe` compilation limitations.

## Requirements

### For Export (Julia side)
- Julia 1.12+ with juliac support
- ACEpotentials.jl and dependencies

### For Deployment (End users)
- **No Julia required** for `ACELibraryCalculator` - runtime libraries are bundled
- LAMMPS: Any version with plugin support
- Python: 3.9+, see [ase-ace/](ase-ace/) for calculator-specific dependencies

## C API Reference

Summarised here; **[`docs/C_INTERFACE_API.md`](docs/C_INTERFACE_API.md) is the authority**, and
it also explains why each rule below exists.

### Workspaces (every evaluation entry point takes one)

The library holds **no mutable global state**. All scratch lives in a *workspace*, which the
caller obtains once and passes as the **first argument** to every evaluation call:

```c
void *ace_workspace_new(void);   /* NULL when the pool is exhausted -- CHECK IT */
void  ace_workspace_free(void *ws);
int   ace_max_workspaces(void);  /* pool size, fixed when the library was built (32) */
```

- **Re-entrant with one workspace per concurrent caller.** Not thread-safe with a shared one,
  and there is no internal locking that would make it so.
- The pool is **fixed size and built into the library image**; size your thread pool with
  `ace_max_workspaces()` and check `ace_workspace_new` for `NULL`. (It is in the image because
  a runtime-allocated Julia object is not reliably rooted in a `juliac --trim` library — that
  was measured twice, and both variants passed every accuracy gate before dying in a real run.)
- A workspace is sized from the **model**, never from the neighbour count, so there is **no
  maximum neighbour count**: the old `MAX_NEIGHBORS = 256` cap is gone.
- Handles are **tagged opaque integers** and are validated on every call; a stale, freed or
  never-allocated handle is rejected rather than served.

A library compiled before this API does not export `ace_workspace_new`. Both the LAMMPS plugin
and `ACELibrary` resolve that symbol first and **refuse to load** such a library — re-export
and re-compile rather than working around it.

### Model information (no workspace, no state)

```c
double ace_get_cutoff(void);      /* maximum cutoff radius, Å */
int    ace_get_n_species(void);
int    ace_get_species(int idx);  /* atomic number for 1-based species index */
int    ace_get_n_basis(void);
unsigned long long ace_build_id(void);   /* 64-bit hash of the generated source */
```

### Site-level evaluation

One site (atom *i*) given its neighbours. LAMMPS and Python both use this API with their own
neighbour lists.

```c
double ace_site_energy(void *ws, int z0, int nneigh,
                       const int *neighbor_z, const double *neighbor_Rij);

double ace_site_energy_forces(void *ws, int z0, int nneigh,
                              const int *neighbor_z, const double *neighbor_Rij,
                              double *forces);

double ace_site_energy_forces_virial(void *ws, int z0, int nneigh,
                                     const int *neighbor_z, const double *neighbor_Rij,
                                     double *forces, double *virial);

int    ace_site_basis(void *ws, int z0, int nneigh,
                      const int *neighbor_z, const double *neighbor_Rij,
                      double *basis_out);

void   ace_batch_energy_forces_virial(void *ws, int natoms, const int *z,
                                      const int *neighbor_counts,
                                      const int *neighbor_offsets,
                                      const int *neighbor_z, const double *neighbor_Rij,
                                      double *energies, double *forces, double *virials);
```

### Important conventions

- `neighbor_Rij` is `nneigh * 3` doubles of **displacement vectors** `R_j - R_i` in Å — not
  positions.
- **Forces are on neighbours:** `forces[j]` is `-dE_i/dR_j`. For totals,
  `F[j] += forces[j]`, `F[i] -= sum(forces)`.
- **Virial** is 6 doubles in Voigt order `[xx, yy, zz, yz, xz, xy]`.
- The returned energy is the **site energy including `E0`** of the centre species, and the pair
  term when the model has one. `nneigh == 0` returns `E0` and writes zeros.
- `ace_batch_*` is sequential within one call and uses the single workspace it is given; to go
  parallel, split the atom range and give each thread its own.

## Technical Details

### Julia --trim Compilation

The export uses Julia 1.12's `juliac --trim=safe` feature to create standalone libraries:
- Type-stable code paths (no dynamic dispatch)
- Pre-computed tensor structures
- Manual pullback for analytic forces (avoids Zygote allocations)
- A per-neighbour kernel: one pass builds the `A` basis edge by edge, one pass turns `∂A` back
  into forces, and the radial recurrence is emitted only for the `(n,l)` rows and species pairs
  a given model actually reads

### Library Size

Typical deployment sizes (measured on `libace_cantor_poly_b2.so`, a 5-species order-3 model):
- Model library: ~4 MB
- Julia runtime: ~20 MB
- LAMMPS plugin: ~75 KB
- **Total: ~25 MB**

## Known limitations

1. **The Julia runtime ships beside LAMMPS.** A compiled model is a `juliac --trim=safe`
   shared library and needs `libjulia` and its support libraries on `LD_LIBRARY_PATH` at run
   time — about 20 MB, and the reason `setup_env.sh` exists. No Julia *process* is started and
   no Julia code is interpreted, but the runtime is not removable.
2. **LAMMPS copies the neighbour list.** `pair_style ace` requests a FULL neighbour list, filters each
   atom's list to `r < rcut` (LAMMPS' list carries the skin, so this is a real reduction —
   201 listed neighbours become 82 on the Cantor box) and packs displacement vectors into a
   contiguous buffer before calling the library. That copy is per-site and unavoidable across the C boundary in the
   current ABI; it is inside every timing row in [`bench/README.md`](bench/README.md), so the
   published throughput already includes it.
3. **`:hermite_spline` is approximate by construction, and since B2 it is also the slower
   mode.** See the table above. It cannot be used at all when species pairs have different
   cutoffs (an upstream `EquivariantTensors._spl_grid` `BoundsError`), and the export refuses
   that combination rather than emitting it.
4. **One workspace per concurrent caller, from a pool of 32.** Not a physical limit — see the
   C API section — but it is fixed when the library is built.
5. **The minimal-export path is inoperable.** `pair_ace_minimal.cpp` looks for
   `ace_c_interface_minimal.jl` through `ACE_C_INTERFACE_PATH`; that file was deleted in
   `0372f90d`. Use the compiled-library path. Whether the minimal path is restored against the
   current ABI or removed is an open question for the maintainer — see
   [`bench/FINDINGS_parity.md`](bench/FINDINGS_parity.md).

## Verification

Every claim of exactness or speed in this file is gated by a test that fails when it stops
being true, and the protocol and the full measurement table are in
**[`bench/README.md`](bench/README.md)**. In brief:

| level | reference | tolerance |
|---|---|---|
| generated Julia vs the fitted `ETACEPotential`/`StackedCalculator` | the model as fitted (or as splinified, for `:hermite_spline`) | 1e-12 energies, forces, virial |
| compiled `.so` via the Python C API vs Julia | the generated Julia | 1e-12 |
| `pair_style ace` in LAMMPS vs Julia | the compiled library | 1e-10 |
| the LAMMPS virial, all six Voigt components, rattled cell — **the Si test model only**, since that is the model the LAMMPS group builds; the multi-species virial is gated at the Julia and compiled-library levels above, not through LAMMPS | the Julia reference | 1e-10 relative |
| 2 MPI ranks vs 1 | the serial run | 1e-13 relative in energy, 1e-12 absolute in forces |
| `OMP_NUM_THREADS=4` vs serial | the serial run | bitwise |
| generator vs the previous generator | the previous commit's exported model | 1e-13 relative |

```bash
cd export/test && julia --project=.. runtests.jl        # the default groups
ACE_REQUIRE_GROUPS=all julia --project=.. runtests.jl   # fail, don't skip, on a missing fixture
EXPORT_REF_SHA=<commit> julia --project=.. runtests.jl parity
```

## Troubleshooting

### Library not found
```bash
source setup_env.sh  # Sets LD_LIBRARY_PATH
```

### LAMMPS plugin build fails
Ensure LAMMPS headers are accessible:
```bash
cmake ../cmake -DLAMMPS_HEADER_DIR=/path/to/lammps/src
```

### Python import error
Install the ase-ace package with appropriate dependencies:
```bash
pip install "./ase-ace[all]"  # All calculators
# Or specific backends:
pip install "./ase-ace[lib]"    # ACELibraryCalculator
pip install "./ase-ace[julia]"  # ACEJuliaCalculator
```
