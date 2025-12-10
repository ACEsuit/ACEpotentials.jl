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
│   ├── ace_calculator.py         # ASE-compatible calculator
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

```python
from ace_calculator import ACECalculator
from ase.build import bulk

calc = ACECalculator("/path/to/libace_mymodel.so")
atoms = bulk('Si', 'diamond', a=5.43)
atoms.calc = calc

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

See [python/](python/) for examples and the full API.

## OpenMP Parallelization

Both the LAMMPS plugin and Python calculator support OpenMP parallelization for multi-core speedup.

### LAMMPS with OpenMP

The LAMMPS plugin automatically parallelizes the atom loop when built with OpenMP:

```bash
# Build with OpenMP (enabled by default)
# From the export/ directory:
cd lammps/plugin
mkdir build && cd build
cmake ../cmake -DLAMMPS_HEADER_DIR=/path/to/lammps/src
make

# Run with multiple threads
OMP_NUM_THREADS=4 lmp -in input.lmp
```

To disable OpenMP at build time:
```bash
cmake ../cmake -DBUILD_OMP=OFF -DLAMMPS_HEADER_DIR=/path/to/lammps/src
```

**Typical speedup** (216-atom Si system):
| Threads | Speedup |
|---------|---------|
| 1       | 1.0×    |
| 2       | 1.8×    |
| 4       | 3.2×    |
| 8       | 4.9×    |

### Python with OpenMP

For Python, use the OpenMP-accelerated calculator wrapper:

```python
from ace_calculator_omp import ACECalculatorOMP

# Use 4 OpenMP threads
calc = ACECalculatorOMP("/path/to/libace.so", num_threads=4)
atoms.calc = calc

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

Build the wrapper library first:
```bash
cd python
gcc -shared -fPIC -O3 -fopenmp -o libace_omp.so ace_omp_wrapper.c -ldl -lm
```

The wrapper parallelizes the neighbor-list construction and atom loop, providing similar speedups to the LAMMPS implementation.

### Threading Notes

- **Julia runtime is NOT multithreaded**: The compiled ACE library uses a single-threaded Julia runtime. Parallelization is done at the C/C++ level via OpenMP.
- **Thread safety**: The Julia library is thread-safe for concurrent calls after initialization.
- **Initialization**: Model loading (`ace_get_cutoff()`) must happen on a single thread before parallel evaluation.
- **MPI + OpenMP**: For large systems, combine MPI domain decomposition with OpenMP threading:
  ```bash
  OMP_NUM_THREADS=4 mpirun -np 4 lmp -in input.lmp
  ```

## Requirements

### For Export (Julia side)
- Julia 1.12+ with juliac support
- ACEpotentials.jl and dependencies

### For Deployment (End users)
- **No Julia required** - runtime libraries are bundled
- LAMMPS: Any version with plugin support
- Python: 3.8+, numpy, ASE

## C API Reference

The compiled library exports these C functions:

### Model Information Functions

```c
double ace_get_cutoff(void);      // Returns maximum cutoff radius (Angstroms)
int ace_get_n_species(void);       // Returns number of supported species
int ace_get_species(int idx);      // Returns atomic number for species index (1-indexed)
```

### Site-Level Evaluation (for LAMMPS)

These functions compute the contribution from a single site (atom i) given its neighbors.

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

### System-Level Evaluation (for Python)

These functions compute total energy, forces, and virial for an entire system.

```c
// Energy only
double ace_energy(
    int natoms,       // Number of atoms
    int* species,     // Array[natoms]: atomic numbers
    double* positions,// Array[natoms*3]: positions, row-major (x1,y1,z1,x2,y2,z2,...)
    double* cell,     // Array[9]: cell vectors, row-major (a1x,a1y,a1z,a2x,...) or NULL
    int* pbc          // Array[3]: periodic boundary conditions (0/1) or NULL
);

// Energy and forces
double ace_energy_forces(
    int natoms, int* species, double* positions, double* cell, int* pbc,
    double* forces    // Array[natoms*3]: output forces (row-major)
);

// Energy, forces, and virial
double ace_energy_forces_virial(
    int natoms, int* species, double* positions, double* cell, int* pbc,
    double* forces,   // Array[natoms*3]: output forces (row-major)
    double* virial    // Array[9]: output virial tensor (3x3 row-major: Vxx,Vxy,Vxz,Vyx,...)
);
```

### Important Conventions

**Cell Layout (row-major):**
```
cell[0..2] = a1 (first lattice vector)
cell[3..5] = a2 (second lattice vector)
cell[6..8] = a3 (third lattice vector)
```

**Virial Format:**
- **Site-level**: 6 elements in Voigt notation: (xx, yy, zz, yz, xz, xy)
- **System-level**: 9 elements as 3x3 matrix (row-major): Vxx, Vxy, Vxz, Vyx, Vyy, Vyz, Vzx, Vzy, Vzz

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

### Force Convention

The site-level API returns forces **on neighbors**:
- `forces[j]` = force on neighbor j due to center atom
- For total forces: `F[j] += forces[j]`, `F[i] -= sum(forces)`

### Virial Convention

Virial tensor in Voigt notation: `[xx, yy, zz, yz, xz, xy]`

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
pip install numpy ase
```
