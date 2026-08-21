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
| **Evaluation speed** | Baseline | ~2x faster |
| **Export complexity** | Simple (`ace1_model`) | Requires conversion step |
| **Use case** | Development, small MD | Production MD, HPC |

### Standard ACE Export (Simpler)

```julia
# Fit model
model = ace1_model(elements=[:Si], order=3, totaldegree=10)
acefit!(data, model)

# Export directly
include("export/scripts/build_deployment.jl")
build_deployment(model, "silicon_ace")
```

### ETACE Export (Faster, Recommended for Production)

```julia
using ACEpotentials.Models, ACEpotentials.ETModels

# Create model with learnable radial basis (required for ETACE)
ace_model = Models.ace_model(elements=(:Si,), order=3, ...)
acefit!(data, ACEPotential(ace_model, ps, st))

# Convert to ETACE and splinify (CRITICAL: before export)
et_model = ETModels.convert2et(ace_model)
et_model_splined = ETModels.splinify(et_model, et_ps, et_st; Nspl=50)

# Export with Hermite splines
export_ace_model(et_calc, "model.jl"; radial_basis=:hermite_spline)
```

See [`examples/etace_lammps_tutorial.jl`](examples/etace_lammps_tutorial.jl) for a complete walkthrough.

## Radial Basis Export Options

When exporting, choose the radial basis representation:

| Mode | Accuracy | File Size | Speed | Use Case |
|------|----------|-----------|-------|----------|
| `:hermite_spline` | Machine precision | ~1 MB | Fast | **Recommended** |
| `:polynomial` | Exact | ~100 KB | Medium | Debugging, verification |

**Recommendation**: Use `:hermite_spline` for all production deployments.

```julia
# Hermite cubic splines (recommended)
export_ace_model(calc, "model.jl"; radial_basis=:hermite_spline)

# Polynomial (for debugging)
export_ace_model(calc, "model.jl"; radial_basis=:polynomial)
```

## Directory Structure

```
export/
├── src/                          # Core export functionality
│   ├── export_ace_model.jl       # Model → trim-compatible code
│   └── ace_c_interface.jl        # C API implementation
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

The compiled library exports these C functions:

### Model Information Functions

```c
double ace_get_cutoff(void);      // Returns maximum cutoff radius (Angstroms)
int ace_get_n_species(void);       // Returns number of supported species
int ace_get_species(int idx);      // Returns atomic number for species index (1-indexed)
```

### Site-Level Evaluation

These functions compute the contribution from a single site (atom i) given its neighbors.
Both LAMMPS and Python use this API (with their respective neighbor list implementations).

```c
// Energy only
double ace_site_energy(
    int z0,           // Atomic number of center atom
    int nneigh,       // Number of neighbors
    int* neighbor_z,  // Array[nneigh]: atomic numbers of neighbors
    double* Rij       // Array[nneigh*3]: displacement vectors R_j - R_i (row-major)
);

// Energy and forces
double ace_site_energy_forces(
    int z0,           // Atomic number of center atom
    int nneigh,       // Number of neighbors
    int* neighbor_z,  // Array[nneigh]: atomic numbers of neighbors
    double* Rij,      // Array[nneigh*3]: displacement vectors (input)
    double* forces    // Array[nneigh*3]: forces ON neighbors (output)
);

// Energy, forces, and virial
double ace_site_energy_forces_virial(
    int z0,           // Atomic number of center atom
    int nneigh,       // Number of neighbors
    int* neighbor_z,  // Array[nneigh]: atomic numbers of neighbors
    double* Rij,      // Array[nneigh*3]: displacement vectors (input)
    double* forces,   // Array[nneigh*3]: forces ON neighbors (output)
    double* virial    // Array[6]: site virial in Voigt notation (xx,yy,zz,yz,xz,xy) (output)
);
```

### Important Conventions

**Force Convention:**
The site-level API returns forces **on neighbors**:
- `forces[j]` = force on neighbor j due to center atom
- For total forces: `F[j] += forces[j]`, `F[i] -= sum(forces)`

**Virial Format:**
6 elements in Voigt notation: `[xx, yy, zz, yz, xz, xy]`

## Technical Details

### Julia --trim Compilation

The export uses Julia 1.12's `juliac --trim=safe` feature to create standalone libraries:
- Type-stable code paths (no dynamic dispatch)
- Pre-computed tensor structures
- Manual pullback for analytic forces (avoids Zygote allocations)
- Spline-based radial basis for fast evaluation

### Library Size

Typical deployment sizes:
- Model library: ~3 MB
- Julia runtime: ~20 MB
- LAMMPS plugin: ~70 KB
- **Total: ~25 MB**

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
