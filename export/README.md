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
├── python/                       # Python integration
│   ├── ace_calculator.py         # ASE-compatible calculator (uses matscipy)
│   └── examples/
│
├── scripts/                      # Convenience scripts
│   └── build_deployment.jl       # Export + compile + package
│
└── examples/                     # Complete example workflows
    └── silicon/
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

The Python calculator uses [matscipy](https://github.com/libAtoms/matscipy) for efficient O(N) neighbor list construction and calls the site-level C API for energy/force evaluation.

```python
from ace_calculator import ACECalculator
from ase.build import bulk

calc = ACECalculator("/path/to/libace_mymodel.so")
atoms = bulk('Si', 'diamond', a=5.43)
atoms.calc = calc

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()
```

**Dependencies**: `pip install numpy ase matscipy`

See [python/](python/) for examples and the full API.

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

The Python `ACECalculator` is single-threaded. For parallel Python calculations, we recommend:

1. **LAMMPS + LAMMPSlib**: Use ASE's LAMMPS interface for production MD with MPI parallelization
2. **JuliaCall/PythonCall**: Call Julia directly for Julia-level threading
3. **IPICalculator.jl**: For i-PI socket interface integration

## Requirements

### For Export (Julia side)
- Julia 1.12+ with juliac support
- ACEpotentials.jl and dependencies

### For Deployment (End users)
- **No Julia required** - runtime libraries are bundled
- LAMMPS: Any version with plugin support
- Python: 3.9+, numpy, ASE, matscipy

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
Install dependencies:
```bash
pip install numpy ase matscipy
```
