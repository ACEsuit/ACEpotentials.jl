# ase-ace

ASE calculators for ACE (Atomic Cluster Expansion) potentials.

This package provides three ASE-compatible calculators for ACE potentials:

| Calculator | Backend | Threading | Startup | Julia Required |
|------------|---------|-----------|---------|----------------|
| `ACECalculator` | Socket/IPICalculator | Multi-threaded | 5-10s (JIT) | Yes (runtime) |
| `ACEJuliaCalculator` | JuliaCall | Multi-threaded | 10-30s (JIT) | Yes (managed) |
| `ACELibraryCalculator` | Compiled .so | Single-threaded | Instant | No (at runtime) |

## Features

- Full ASE Calculator interface (energy, forces, stress)
- ACE descriptor computation via `get_descriptors()` method
- Multi-threaded evaluation via `JULIA_NUM_THREADS`
- Three backends:
  - **Socket-based** (`ACECalculator`): Full Julia, multi-threaded, requires Julia installation
  - **JuliaCall** (`ACEJuliaCalculator`): Direct Julia integration, multi-threaded, auto-manages Julia
  - **Compiled library** (`ACELibraryCalculator`): Instant startup, single-threaded, no Julia at runtime
- Automatic Julia subprocess/environment management
- Context manager support for clean resource cleanup

## Installation

### 1. Install Julia

Install Julia 1.11+ using [juliaup](https://github.com/JuliaLang/juliaup) (recommended):

```bash
# Linux/macOS
curl -fsSL https://install.julialang.org | sh

# Windows
winget install julia -s msstore
```

Or download from [julialang.org](https://julialang.org/downloads/).

Verify installation:
```bash
julia --version
# Should show: julia version 1.11.x or higher
```

### 2. Install Julia Packages

Install the required Julia packages in the ase-ace Julia environment:

```bash
# Navigate to the ase-ace directory
cd path/to/ACEpotentials.jl/export/ase-ace

# Install Julia dependencies
julia --project=julia -e '
    using Pkg
    println("Adding ACE registry...")
    Pkg.Registry.add(Pkg.RegistrySpec(url="https://github.com/ACEsuit/ACEregistry"))
    println("Installing packages...")
    Pkg.instantiate()
    println("Precompiling (this may take a few minutes)...")
    Pkg.precompile()
    println("Done!")
'
```

### 3. Install ase-ace

```bash
# Base installation (no optional dependencies)
pip install .

# With JuliaCall support (ACEJuliaCalculator)
pip install ".[julia]"

# With library support (ACELibraryCalculator)
pip install ".[lib]"

# With all backends
pip install ".[all]"

# Development mode with test dependencies
pip install -e ".[dev]"
```

**Installation options:**
- `ase-ace` - Base package only (includes `ACECalculator`)
- `ase-ace[julia]` - Adds `juliacall` and `juliapkg` for `ACEJuliaCalculator`
- `ase-ace[lib]` - Adds `matscipy` for `ACELibraryCalculator`
- `ase-ace[all]` - All optional dependencies

## Quick Start

### Socket-based Calculator (ACECalculator)

Uses Julia runtime via sockets. Requires Julia installation.

```python
from ase.build import bulk
from ase_ace import ACECalculator

# Create a silicon structure
atoms = bulk('Si', 'diamond', a=5.43)

# Use ACECalculator with context manager (recommended)
with ACECalculator('path/to/model.json', num_threads=4) as calc:
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()

    print(f"Energy: {energy:.4f} eV")
    print(f"Max force: {abs(forces).max():.4f} eV/A")
```

### JuliaCall-based Calculator (ACEJuliaCalculator)

Uses JuliaCall for direct Julia integration. Julia environment is automatically
managed by `juliapkg`. Multi-threaded, no manual Julia setup required.

```python
from ase.build import bulk
from ase_ace import ACEJuliaCalculator

atoms = bulk('Si', 'diamond', a=5.43)

# JuliaCall manages Julia installation and dependencies automatically
calc = ACEJuliaCalculator('path/to/model.json', num_threads=4)
atoms.calc = calc

energy = atoms.get_potential_energy()
forces = atoms.get_forces()

# ACEJuliaCalculator also supports descriptor computation
descriptors = calc.get_descriptors(atoms)
print(f"Energy: {energy:.4f} eV")
print(f"Descriptors shape: {descriptors.shape}")
```

**Note**: Install with JuliaCall support: `pip install ase-ace[julia]`

The first call will download and install Julia and required packages automatically
(this may take a few minutes). Subsequent calls use the cached installation.

### Library-based Calculator (ACELibraryCalculator)

Uses pre-compiled shared library. Instant startup, no Julia needed at runtime.
Single-threaded (Julia's `--trim=safe` limitation).

```python
from ase.build import bulk
from ase_ace import ACELibraryCalculator

atoms = bulk('Si', 'diamond', a=5.43)

# Point to compiled library from ACEpotentials.jl deployment
calc = ACELibraryCalculator('deployment/lib/libace_model.so')
atoms.calc = calc

energy = atoms.get_potential_energy()

# ACELibraryCalculator also supports descriptor computation
descriptors = calc.get_descriptors(atoms)
print(f"Energy: {energy:.4f} eV")
```

**Note**: Install with library support: `pip install ase-ace[lib]`

## Computing ACE Descriptors

The `get_descriptors()` method returns the raw ACE basis vectors for each atom,
useful for fitting, analysis, and transfer learning.

**Availability:** `ACEJuliaCalculator` and `ACELibraryCalculator` only.
`ACECalculator` does not support descriptors (socket protocol limitation).

### Example

```python
from ase.build import bulk
from ase_ace import ACELibraryCalculator

atoms = bulk('Si', 'diamond', a=5.43) * (2, 2, 2)
calc = ACELibraryCalculator("deployment/lib/libace_model.so")

# Get descriptors for all atoms
descriptors = calc.get_descriptors(atoms)
print(f"Shape: {descriptors.shape}")  # (natoms, n_basis)

# Access model properties
print(f"Cutoff: {calc.cutoff} Å")
print(f"Species: {calc.species}")  # Atomic numbers
print(f"Basis size: {calc.n_basis}")
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `cutoff` | float | Cutoff radius in Angstroms |
| `species` | List[int] | Supported atomic numbers |
| `n_basis` | int | Number of basis functions per atom |

### Use Cases

- **Linear model verification**: For linear ACE, `E = sum(descriptors @ weights)`
- **Transfer learning**: Use descriptors as features for other ML models
- **Analysis**: Examine local atomic environments

## Configuration

### ACECalculator Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | required | Path to ACE model JSON file |
| `num_threads` | int/str | 'auto' | Julia threads |
| `port` | int | 0 | TCP port (0 = auto) |
| `unixsocket` | str | None | Unix socket name |
| `timeout` | float | 60.0 | Connection timeout (seconds) |
| `julia_executable` | str | 'julia' | Path to Julia |
| `julia_project` | str | None | Julia project path |
| `log_level` | str | 'WARNING' | Logging level |

### ACEJuliaCalculator Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | required | Path to ACE model JSON file |
| `num_threads` | int/str | 'auto' | Julia threads |

### ACELibraryCalculator Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `library_path` | str | required | Path to compiled .so file |

### Threading

The calculator uses Julia's multi-threading for parallel ACE evaluation:

```python
# Explicit thread count
calc = ACECalculator('model.json', num_threads=8)

# Auto-detect available cores
calc = ACECalculator('model.json', num_threads='auto')

# Single-threaded (deterministic)
calc = ACECalculator('model.json', num_threads=1)
```

### Unix Sockets (Faster for Local)

For local connections, Unix sockets have lower latency than TCP:

```python
calc = ACECalculator('model.json', unixsocket='ace_socket')
```

## Examples

### Geometry Optimization

```python
import numpy as np
from ase.build import bulk
from ase.optimize import BFGS
from ase_ace import ACECalculator

# Create perturbed structure
atoms = bulk('Si', 'diamond', a=5.43) * (2, 2, 2)
atoms.positions += np.random.randn(*atoms.positions.shape) * 0.1

with ACECalculator('model.json', num_threads='auto') as calc:
    atoms.calc = calc

    opt = BFGS(atoms, logfile='opt.log')
    opt.run(fmax=0.01)

    print(f"Optimized energy: {atoms.get_potential_energy():.4f} eV")
```

### Molecular Dynamics

```python
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
from ase_ace import ACECalculator

atoms = bulk('Si', 'diamond', a=5.43) * (3, 3, 3)

with ACECalculator('model.json', num_threads=8) as calc:
    atoms.calc = calc

    # Initialize velocities at 300 K
    MaxwellBoltzmannDistribution(atoms, temperature_K=300)

    # Run NVE dynamics
    dyn = VelocityVerlet(atoms, timestep=1.0 * units.fs)

    def print_energy():
        e_kin = atoms.get_kinetic_energy()
        e_pot = atoms.get_potential_energy()
        print(f"E_kin={e_kin:.3f} E_pot={e_pot:.3f} E_tot={e_kin+e_pot:.3f}")

    dyn.attach(print_energy, interval=10)
    dyn.run(100)
```

## Creating ACE Models

ACE models are created and fitted using ACEpotentials.jl in Julia:

```julia
using ACEpotentials

# Define model
model = ACEpotentials.ace1_model(
    elements = [:Si, :O],
    order = 3,
    totaldegree = 12,
    rcut = 5.5,
)

# Load training data
data = ACEpotentials.load_data("training_data.xyz")

# Fit model
ACEpotentials.acefit!(data, model;
    energy_key = "energy",
    force_key = "forces",
)

# Save for Python use
ACEpotentials.save_model(model, "model.json")
```

See the [ACEpotentials.jl documentation](https://acesuit.github.io/ACEpotentials.jl) for details.

## Performance Notes

### Startup Time

The first calculation takes 5-10 seconds due to Julia's JIT compilation.
Subsequent calculations are fast. For interactive use, consider:

```python
# Keep calculator alive between calculations
calc = ACECalculator('model.json', num_threads=4)

for atoms in structures:
    atoms.calc = calc
    energies.append(atoms.get_potential_energy())

calc.close()  # Clean up when done
```

### Comparison of Calculators

| Calculator | Julia Required | Threading | Startup | Best For |
|------------|---------------|-----------|---------|----------|
| `ACECalculator` | Yes (runtime) | Multi-threaded | 5-10s | Development, interactive use |
| `ACEJuliaCalculator` | Yes (managed) | Multi-threaded | 10-30s | Self-contained scripts, descriptors |
| `ACELibraryCalculator` | No (at runtime) | Single-threaded | Instant | Quick calculations, deployment |
| LAMMPS plugin | No | MPI + OpenMP | N/A | Large-scale MD, HPC |

- **Development/prototyping**: Use `ACECalculator` for convenience and threading
- **Self-contained scripts**: Use `ACEJuliaCalculator` - no manual Julia setup needed
- **Quick calculations**: Use `ACELibraryCalculator` for instant startup
- **Descriptor computation**: Use `ACEJuliaCalculator` or `ACELibraryCalculator`
- **Large-scale parallel MD**: Use LAMMPS plugin for MPI parallelization

## Troubleshooting

### Julia not found

```
RuntimeError: Julia executable not found
```

Ensure Julia is in your PATH:
```bash
export PATH="$HOME/.juliaup/bin:$PATH"
```

### Julia package errors

```
ERROR: LoadError: ArgumentError: Package ACEpotentials not found
```

Install Julia dependencies:
```bash
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
```

### Timeout during first calculation

The first calculation may take longer due to JIT compilation:
```python
calc = ACECalculator('model.json', timeout=120.0)  # 2 minutes
```

### Connection refused

If using a specific port that's in use:
```python
calc = ACECalculator('model.json', port=0)  # Auto-assign port
```

## Utility Functions

The `ase_ace.utils` module provides helper functions for Julia setup:

```python
from ase_ace.utils import find_julia, check_julia_version, setup_julia_environment

# Find Julia executable
julia_path = find_julia()

# Check Julia version
major, minor, patch = check_julia_version()

# Set up Julia environment with required packages
setup_julia_environment(verbose=True)
```

**Available functions:**
- `find_julia()` - Locate Julia executable in PATH
- `check_julia_version(julia_executable)` - Get Julia version as (major, minor, patch) tuple
- `check_julia_packages(julia_executable, julia_project)` - Check if required packages are installed
- `setup_julia_environment(julia_executable, julia_project, verbose)` - Install and configure Julia dependencies

## Running Tests

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run tests (requires test model and Julia)
pytest -v tests/

# Skip slow tests
pytest -v tests/ -m "not slow"
```

To create the test model fixture:
```bash
python tests/conftest.py
```

## License

MIT License - see the main ACEpotentials.jl repository.

## References

- [ACEpotentials.jl](https://github.com/ACEsuit/ACEpotentials.jl) - Julia ACE potentials package
- [IPICalculator.jl](https://github.com/JuliaMolSim/IPICalculator.jl) - i-PI socket protocol for Julia
- [ASE](https://wiki.fysik.dtu.dk/ase/) - Atomic Simulation Environment
- [i-PI](https://ipi-code.org/) - Universal force engine protocol
