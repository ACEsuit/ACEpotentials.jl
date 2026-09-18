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

### 1. Install Julia (optional)

This step is optional.  juliapkg will install a compatible Julia itself if it cannot find
one, so `pip install ase-ace` followed by section 2 is enough.  Installing
[juliaup](https://github.com/JuliaLang/juliaup) first is still recommended: juliapkg then
picks the newest channel matching `~1.11, ~1.12` rather than downloading a second copy of
Julia into your Python environment.

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
# Should show: julia version 1.11.x or 1.12.x
```

Note that a Julia on `PATH` is not necessarily the one the calculators run --
`ase_ace.server.julia_env()` reports the one juliapkg chose.

### 2. Install ase-ace's Julia Packages

`ase-ace` declares what it needs from Julia in **one** file, `src/ase_ace/juliapkg.json`
(installed as `<site-packages>/ase_ace/juliapkg.json`): a supported Julia range
(`~1.11, ~1.12`) and eight packages.  [juliapkg](https://github.com/JuliaPy/pyjuliapkg),
which is a base dependency, reads that file and builds the environment; **both** Julia-backed
calculators then use it -- `ACEJuliaCalculator` through juliacall, and `ACECalculator` by
spawning `julia --project=<that environment>`.  There is no Julia project inside the Python
package any more, and nothing is installed into `site-packages`.

You do not have to do anything: the environment is created on first use.  To do it once,
deliberately -- before a batch job, or at container-build time -- run

```bash
python -c "from ase_ace.utils import setup_julia_environment; setup_julia_environment(verbose=True)"
```

This is the same command in a source checkout and in a wheel install.  It installs a
compatible Julia if there is not one already, adds the packages, resolves and precompiles,
under a cross-process file lock.  First run takes minutes; afterwards startup is a
content-hash check.  You never have to re-run it after upgrading `ase-ace`: juliapkg
re-resolves automatically whenever `juliapkg.json`, the Julia version, or the set of
`juliapkg.json` files on `sys.path` changes.

To see what it chose:

```python
from ase_ace.server import julia_env
executable, project = julia_env()
```

#### Where the Julia environment lives

By default juliapkg puts it **inside the Python prefix**: `<sys.prefix>/julia_env` in a
virtualenv or conda environment, and `~/.julia/environments/pyjuliapkg` for a system Python.

**If the prefix is not writable** -- a system-wide install, a read-only container image, a
shared install serving several users -- creating it fails, and `ase-ace` reports:

```
RuntimeError: ase-ace could not create its Julia environment: [Errno 13] Permission denied: '<prefix>/julia_env'
```

followed by the fix.  Set juliapkg's own environment variable to somewhere writable, before
Python starts:

```bash
export PYTHON_JULIAPKG_PROJECT=$HOME/.julia/environments/ase_ace
python -c "from ase_ace.utils import setup_julia_environment; setup_julia_environment(verbose=True)"
```

`ase-ace` deliberately does **not** set this for you: it is process-global and shared with
juliacall and every other juliapkg consumer in the interpreter, and silently relocating
another package's Julia environment is not ours to do.

Three juliapkg variables are worth knowing:

| variable | effect |
|---|---|
| `PYTHON_JULIAPKG_PROJECT` | Put the environment at this absolute path.  Also marks it *shared*, which changes one thing: juliapkg then adds to the existing `Project.toml` instead of rebuilding it, and does not delete `Manifest.toml`, so a dependency that `ase-ace` **removes** in a later release lingers.  Version changes still apply. |
| `PYTHON_JULIAPKG_OFFLINE=yes` | Never touch the network; use the environment as it stands.  The run-time half of a build-time resolve, for read-only images. |
| `PYTHON_JULIAPKG_EXE` | Use this Julia.  Read once, at import, so it must be set before Python starts -- which is why `ACECalculator(julia_executable=...)` cannot steer juliapkg and instead bypasses it (below). |

For a container: build as root with `PYTHON_JULIAPKG_PROJECT` pointing somewhere
world-readable *outside* the Python prefix, run `setup_julia_environment()` at build time,
and set `PYTHON_JULIAPKG_OFFLINE=yes` at run time.

#### Bypassing juliapkg

Passing `julia_project=` or `julia_executable=` to `ACECalculator` means "I built this
environment myself, use exactly it".  It bypasses juliapkg entirely; it is not an override of
one of juliapkg's two choices.  That is also why both default to `None` rather than to
`'julia'` -- "not specified" has to be distinguishable from "specified".

#### A note on the Julia version specifier

`juliapkg.json` says `"julia": "~1.11, ~1.12"`, and the tildes matter.  In Julia compat
syntax a comma is a union of **caret** ranges, so the bare `"1.11, 1.12"` means `[1.11, 2.0)`
and admits 1.13, which ACEpotentials does not support.  juliapkg resolves with
`upgrade=True`, i.e. the newest compatible Julia available, so this distinction decides which
Julia you actually run.

One constraint this puts on the declaration: juliapkg merges every `juliapkg.json` on
`sys.path`, and on a Python linked against OpenSSL older than 3.5 its `openssl_compat()` rule
injects `julia = "1 - 1.11"` into that merge.  Intersected with `~1.11, ~1.12` that is
benign -- it pins 1.11.x.  But a future declaration of 1.12 *alone* would intersect to empty
and raise `'julia' compat entries have empty intersection` on those users' machines and not
on others.  Widen this specifier when ACEpotentials supports a newer Julia; do not narrow it
past 1.11.

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
- `ase-ace` - Base package only (includes `ACECalculator`; pulls in `juliapkg`)
- `ase-ace[julia]` - Adds `juliacall` for `ACEJuliaCalculator`
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
| `julia_executable` | str | None | Path to Julia; `None` = the one juliapkg resolved |
| `julia_project` | str | None | Julia project path; `None` = juliapkg's environment. Setting either this or `julia_executable` bypasses juliapkg |
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

Install the Julia dependencies:
```bash
python -c "from ase_ace.utils import setup_julia_environment; setup_julia_environment(verbose=True)"
```

If that reports `could not create its Julia environment: [Errno 13] Permission denied`, the
Python prefix is read-only -- see
[Where the Julia environment lives](#where-the-julia-environment-lives).

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
from ase_ace.server import julia_env

# Find a Julia executable on PATH (not necessarily the one the calculators use)
julia_path = find_julia()

# Check Julia version
major, minor, patch = check_julia_version()

# Install the Julia side (juliapkg resolves ase_ace/juliapkg.json)
setup_julia_environment(verbose=True)

# What the calculators actually run
executable, project = julia_env()
```

**Available functions:**
- `find_julia()` - Locate a Julia executable on `PATH`.  Note this is *not* what the
  calculators run -- they use the Julia juliapkg resolved; see `ase_ace.server.julia_env()`
- `check_julia_version(julia_executable)` - Get Julia version as (major, minor, patch) tuple
- `check_julia_packages(julia_executable, julia_project)` - Check that the declared packages
  load; the list is read from `juliapkg.json`, not hardcoded
- `setup_julia_environment(julia_executable, julia_project, verbose, update)` - Resolve the
  Julia environment via juliapkg.  Passing `julia_project` instantiates that project instead,
  bypassing juliapkg
- `declared_julia_packages()` - The Julia packages named in the shipped `juliapkg.json`

## Running Tests

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run tests (requires test model and Julia)
pytest -v tests/

# Skip slow tests
pytest -v tests/ -m "not slow"
```

`tests/test_packaging.py` checks that the bundled Julia assets resolve from the *installed*
package.  Its cheap tier runs in the command above; its expensive tier builds a wheel and
installs it into a throwaway venv, which is the only way to reproduce a non-editable install:

```bash
ACE_TEST_PACKAGING=1 pytest -v tests/test_packaging.py
```

CI runs it with that variable set in the `ase-ace (imports and utils)` job.  The reason it
exists is worth knowing before adding another asset: every other job installs this package
with `pip install -e`, and an editable install cannot see a file that the wheel does not ship.

To create the test model fixture:
```bash
python tests/conftest.py
```

## License

MIT License.  The full text ships with the package, as `LICENSE` in the source tree and in `ase_ace-<version>.dist-info/licenses/` in an installed wheel; it is a copy of the LICENSE at the root of the ACEpotentials.jl repository.

## References

- [ACEpotentials.jl](https://github.com/ACEsuit/ACEpotentials.jl) - Julia ACE potentials package
- [IPICalculator.jl](https://github.com/JuliaMolSim/IPICalculator.jl) - i-PI socket protocol for Julia
- [ASE](https://wiki.fysik.dtu.dk/ase/) - Atomic Simulation Environment
- [i-PI](https://ipi-code.org/) - Universal force engine protocol
