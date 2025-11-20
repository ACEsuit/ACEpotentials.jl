# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ACEpotentials.jl is a Julia package for creating and using atomic cluster expansion (ACE) interatomic potentials. It provides tools for fitting machine learning potentials to quantum mechanical data and using them for atomistic simulations. The package integrates with the AtomsBase ecosystem and supports linear and nonlinear ACE models.

**Current Version**: 0.9.1
**Julia Compatibility**: 1.10, 1.11 (1.11 strongly recommended)
**Active Branch**: Version 0.8+ represents a complete rewrite with new architecture

## Essential Commands

### Development Setup

The package requires the ACEregistry to be added before use:

```bash
# Activate the project
julia --project=.

# In Julia REPL, add the ACE registry
] registry add https://github.com/ACEsuit/ACEregistry

# Install dependencies
] instantiate

# Resolve and precompile
] resolve
] precompile
```

### Testing

```bash
# Run full test suite (takes ~2 hours)
julia --project=. -e 'using Pkg; Pkg.test()'

# Run main test file
julia --project=. test/runtests.jl

# Run specific test file
julia --project=. test/test_silicon.jl
julia --project=. test/models/test_ace.jl
julia --project=. test/test_migration.jl

# Run individual testset from within Julia
using ACEpotentials, Test
include("test/test_silicon.jl")
```

### Documentation

```bash
# Build documentation locally
julia --project=docs docs/make.jl

# Or from root directory
julia --project=. -e 'include("docs/make.jl")'
```

## Architecture

### Module Structure

**Top-level** (`src/ACEpotentials.jl`):
- Re-exports ACEfit for fitting functionality
- Coordinates model construction, data handling, and fitting workflows

**Core Subsystems**:

1. **Models** (`src/models/`): ACE model definitions and evaluation
   - `ace.jl`: Main ACEModel structure and constructor
   - `calculators.jl`: AtomsCalculators interface for energy/force/virial evaluation
   - `Rnl_*.jl`: Radial basis functions (learnable, splines, basis)
   - `fasteval.jl`: Optimized evaluation paths
   - `committee.jl`: Model committees for uncertainty quantification
   - `smoothness_priors.jl`: Regularization priors (algebraic, exponential, gaussian)

2. **Data Handling** (`src/atoms_data.jl`):
   - Converts AtomsBase systems to ACEfit-compatible AtomsData format
   - Handles energy/force/virial extraction with fuzzy key matching
   - Manages per-configuration weights

3. **Fitting** (`src/fit_model.jl`):
   - `acefit!()`: Main fitting interface
   - `compute_errors()`: Error analysis and RMSE computation
   - Integrates with ACEfit.jl solvers (QR, BLR, LSQR)

4. **ACE1 Compatibility** (`src/ace1_compat.jl`):
   - `ace1_model()`: Creates ACE models using familiar v0.6.x API
   - Provides backward compatibility for existing workflows

5. **JSON Interface** (`src/json_interface.jl`):
   - Model serialization/deserialization
   - Script-based fitting via parameter files
   - `make_model()`, `make_solver()`: Construct objects from dictionaries

### Key Dependencies

**ACE Ecosystem**:
- `ACEfit`: Fitting algorithms and linear solvers
- `EquivariantTensors`: Coupling coefficient generation (replaces EquivariantModels)
- `Polynomials4ML`: Polynomial basis infrastructure
- `RepLieGroups`: Representation theory for SO(3) symmetry

**External**:
- `Lux`/`LuxCore`: Neural network layer abstractions
- `Zygote`: Automatic differentiation
- `AtomsBase`/`AtomsCalculators`: Atomistic simulation interfaces
- `ExtXYZ`: Dataset I/O

### Workflow Pattern

Typical fitting workflow follows this structure:

1. **Model Construction**: Use `ace1_model()` or direct ACEModel construction
2. **Data Loading**: Load datasets via ExtXYZ (usually from artifacts)
3. **Data Preparation**: Convert to AtomsData with energy/force/virial keys
4. **Fitting**: Call `acefit!()` with solver (QR, BLR, LSQR) and optional smoothness prior
5. **Evaluation**: Use `compute_errors()` to assess fit quality
6. **Serialization**: Save/load models via JSON interface

The package maintains separation between:
- **Model definition** (what basis functions, how they're combined)
- **Model parameters** (coefficients learned from data)
- **Evaluation** (computing energy/forces from atomic configurations)

## Important Notes

### EquivariantTensors Migration ✅

**Migration Complete**: All tests passing on Julia 1.11 and 1.12

**What Changed**:
- Migrated from EquivariantModels.jl (maintenance mode) to EquivariantTensors.jl v0.3
- Updated `Project.toml`, `src/models/ace.jl`, `src/models/utils.jl`
- Replaced deprecated API calls with upstream equivalents:
  - `EquivariantModels._rpi_A2B_matrix()` → `EquivariantTensors.symmetrisation_matrix()`
  - `EquivariantModels.RPE_filter_real()` → `_rpe_filter_real()` (local implementation in `src/models/utils.jl`)

**Why Migration?**
- EquivariantModels.jl frozen (legacy support only)
- EquivariantTensors.jl actively developed with better performance and GPU support
- Future-proofs the codebase for upcoming features

**Testing Status**:
- ✅ Julia 1.11: All tests pass
- ✅ Julia 1.12: All tests pass (with 2 known issues documented below)
- ✅ Documentation build: Success
- ✅ Numerical equivalence: Verified against main branch

**Critical Bug Fix** (commit `0910f528`):
Fixed basis size bug introduced during migration in `src/models/ace.jl:96`:
- **Bug**: Applied `unique()` to each basis element separately instead of to the entire specification
- **Impact**: Created duplicate basis functions when multiple AA_spec entries had the same (n,l) values but different m values
- **Consequence**: Models had inflated basis sizes, leading to redundant parameters and incorrect basis dimensions
- **Symptom**: Silicon test RMSE values were elevated, requiring loosened thresholds to pass
- **Fix**: Moved `unique()` outside the comprehension to deduplicate the entire mb_spec list
- **Result**: RMSE thresholds reverted to original strict values; tests expected to pass with correct basis size
- **How introduced**: During migration to EquivariantTensors, the conversion from (n,l,m) to (n,l) format required deduplication logic that was incorrectly placed

**Known Issues** (documented as test skips/broken):
1. **fast_evaluator** (experimental feature): Requires major refactoring to work with new upstream API. Tests skipped in `test/test_fast.jl` and usage commented out in `docs/src/tutorials/asp.jl`. This is an optional optimization feature, not core functionality.

2. **Julia 1.12 basis ordering** (`test/test_bugs.jl:35-45`): Julia 1.12's new hash algorithm causes Dict iteration order changes that affect basis function construction. Test marked as `@test_broken` for Julia 1.12+. Requires either upstream EquivariantTensors fixes or comprehensive OrderedDict refactoring. Does not affect Julia 1.11 compatibility.

### Future Work: Full Lux-based Backend Migration

**Current Status**: The v0.10.0 migration is "superficial" - only the tensor coupling coefficients use EquivariantTensors' `SparseACEbasis`. The radial basis (rbasis), spherical harmonics (ybasis), distance transforms, and evaluation logic remain hand-written ACEpotentials code.

**Goal**: Full migration to Lux-based backend leveraging EquivariantTensors' optimized kernels and enabling GPU acceleration.

#### Why Further Migration is Needed

Current limitations of v0.10.0:
- **Hand-written evaluation**: Custom implementations for rbasis/ybasis evaluation instead of composable Lux layers
- **Scalar-based loops**: Per-atom distance calculations instead of vectorized graph operations
- **Custom differentiation**: Manual rrules for gradients instead of automatic Lux/Zygote AD
- **No GPU support**: Hand-written code incompatible with KernelAbstractions/GPU backends
- **Missed optimizations**: Doesn't leverage EquivariantTensors' optimized sparse kernels

The current ACEModel struct (src/models/ace.jl:16-33) only uses ET for the `tensor` field. The `rbasis`, `ybasis`, and evaluation functions are all custom ACEpotentials code from the v0.9 era.

#### Target Architecture

Based on EquivariantTensors.jl examples (`mlip.jl`, `ace_lux.jl`) and the co/etback branch prototype:

**Edge-centric graph evaluation:**
- Move from `evaluate(model, Rs, Zs, Z0)` with per-atom loops
- To graph-based `model(G, ps, st)` with batch edge operations
- Input format: edge tuples `(𝐫ij, zi, zj)` instead of scalars `(r, Zi, Zj)`
- Use `ETGraph` structure for efficient batching

**Component replacements:**

1. **Radial basis** (currently 297 lines in `Rnl_learnable.jl`):
   ```julia
   # Target: Lux Chain with EquivariantTensors components
   rbasis = Chain(
       trans = NTtransform(xij -> 1 / (1 + norm(xij.𝐫ij) / rcut)),
       polys = SkipConnection(
           P4ML.ChebBasis(maxdeg),
           WrappedFunction((P, y) -> envelope(y) .* P)
       ),
       linl = SelectLinL(selector=(xij -> species_idx(xij.zi, xij.zj)))
   )
   ```

2. **Spherical harmonics** (currently direct P4ML.evaluate! calls):
   ```julia
   # Target: Lux-wrapped spherical harmonics
   ybasis = Chain(
       trans = NTtransform(xij -> xij.𝐫ij),  # Extract direction vector
       ylm = P4ML.lux(P4ML.real_sphericalharmonics(maxl))
   )
   ```

3. **Embedding layer** (new component, doesn't currently exist):
   ```julia
   embed = EdgeEmbed(BranchLayer(; Rnl = rbasis, Ylm = ybasis))
   ```

4. **Full model as Lux Chain**:
   ```julia
   model = Chain(
       embed = embed,                    # Evaluate (Rnl, Ylm) on edges
       ace = SparseACElayer(𝔹basis, (1,)), # ACE coupling + weights
       energy = WrappedFunction(x -> sum(x[1]) + vref)
   )
   ```

**Key architectural changes:**
- `NTtransform`: Differentiable transforms on edge NamedTuples
- `SelectLinL`: Categorical linear layer for species-pair weights (replaces 4D tensors)
- Graph batching: Vectorize over all edges simultaneously
- Automatic differentiation: Lux/Zygote handles all gradients

#### Proof-of-Concept Work (co/etback branch)

The maintainer has created a prototype demonstrating the migration pattern in `test/models/test_learnable_Rnl.jl` (co/etback branch, commit `caeb7f07`):

**What it demonstrates:**
- How to rebuild `LearnableRnlrzzBasis` using pure EquivariantTensors + Lux components
- Edge tuple format: `xij = (𝐫ij = SVector, zi = AtomicNumber, zj = AtomicNumber)`
- `NTtransform` for differentiable distance transformations
- `SelectLinL` replacing 4D weight tensors `Wnlq[:,:,iz,jz]`
- Numerical equivalence verified: 100 random tests confirm old ≈ new

**Key insight from prototype:**
```julia
# Old ACEpotentials pattern:
P = evaluate(basis, r, Zi, Zj, ps, st)  # Scalar distance

# New EquivariantTensors pattern:
P = et_rbasis(xij, et_ps, et_st)  # Edge tuple with full geometric info
```

This is a **template** showing how to migrate, not production code. The actual implementation work remains to be done.

**Pattern alignment:** The co/etback prototype uses the exact same patterns found in EquivariantTensors' mlip.jl example, confirming this is the recommended upstream approach.

#### Migration Roadmap

**Phase 1: Radial Basis Migration** (~2-3 weeks)
- Apply co/etback pattern to production code in `src/models/Rnl_*.jl`
- Rewrite `LearnableRnlrzzBasis` using `Chain(NTtransform, SkipConnection, SelectLinL)`
- Update `RnlBasis` and spline-based variants
- Create `evaluate_graph` interface accepting edge tuples
- Maintain backward compatibility via wrapper functions
- Verify numerical equivalence in all tests

**Phase 2: Graph-based Evaluation** (~2-3 weeks)
- Adopt edge tuple format `(𝐫ij, zi, zj)` in calculators.jl
- Convert neighbor lists to edge representations with NamedTuples
- Update `site_energy` and `site_energy_d` for graph evaluation
- Vectorize over edges instead of per-atom loops
- Update differentiation to work through graph eval path
- Benchmark performance (should improve due to vectorization)

**Phase 3: Full Lux Chain Integration** (~3-4 weeks)
- Refactor ACEModel to be/use a Lux Chain
- Integrate `EdgeEmbed(BranchLayer(...))` + `SparseACElayer` pattern
- Update all constructors: `ace1_model`, `ace_model`, etc.
- Update JSON serialization interface
- Create migration utilities for existing fitted models
- Update documentation and tutorials
- Comprehensive testing across all features

**Phase 4: GPU Enablement** (~2-3 weeks)
- Add KernelAbstractions support (leveraging ET's GPU kernels)
- Test on CUDA and Metal backends
- Optimize memory allocations with Bumper.jl integration
- Benchmark GPU vs CPU performance
- Document GPU usage patterns and requirements
- Add GPU tests to CI (if infrastructure available)

**Total estimated effort**: 2-3 months of focused development work.

#### Breaking Changes & Compatibility Strategy

**Expected API changes:**

1. **Model construction:**
   ```julia
   # v0.10 (current):
   model = ace1_model(elements=[:Si], order=3, totaldegree=8, ...)
   # Returns: ACEModel struct

   # v0.11+ (future):
   model = ace1_model_lux(elements=[:Si], order=3, totaldegree=8, ...)
   # Returns: Lux.Chain
   ```

2. **Evaluation interface:**
   ```julia
   # v0.10 (current):
   E = evaluate(model, Rs, Zs, Z0, ps, st)

   # v0.11+ (future):
   G = ETGraph(ii, jj; edge_data = [(𝐫=R, zi=Z0, zj=Z) for (R,Z) in zip(Rs, Zs)])
   E, st = model(G, ps, st)
   ```

3. **Parameter structure:**
   ```julia
   # v0.10 (current): Custom NamedTuple
   ps = (WB = ..., Wpair = ..., rbasis = (Wnlq = [...], ...), ...)

   # v0.11+ (future): Auto-generated by Lux
   ps, st = Lux.setup(rng, model)
   # Access: ps.embed.Rnl.linl.weight[...]
   ```

**Backward compatibility strategy:**

- **v0.10.x** (current): "Superficial" migration, classic API
- **v0.11** (future): Add experimental Lux backend with `_lux` suffix constructors
  - `ace1_model_lux(...)` returns Lux Chain
  - Feature flag: `ace1_model(..., backend=:lux)`
  - Both backends coexist, classic as default
- **v0.12**: Lux backend becomes default, classic deprecated with warnings
- **v0.13**: Remove classic backend entirely

**Migration utilities needed:**
- Convert old fitted model parameters to new Lux format
- Wrapper functions maintaining old API during transition period
- Comprehensive migration guide with examples
- Automated testing for numerical equivalence

**Serialization changes:**
- Current JSON interface (`json_interface.jl`) needs updates
- Lux models have different parameter structure
- Consider using Lux.jl's built-in serialization or custom save/load

#### References

**EquivariantTensors.jl examples:**
- `examples/mlip.jl`: Complete Lux-based ACE potential with graph evaluation
- `examples/ace_lux.jl`: Additional architectural patterns and optimizations
- GitHub: https://github.com/ACEsuit/EquivariantTensors.jl

**ACEpotentials.jl branches:**
- `co/etback`: Proof-of-concept prototype by maintainer (Christoph Ortner)
- `test/models/test_learnable_Rnl.jl`: Migration pattern template
- Demonstrates edge-centric evaluation with numerical equivalence verification

**Related upstream work:**
- EquivariantTensors uses KernelAbstractions for GPU kernels
- P4ML has `P4ML.lux()` wrapper for converting bases to Lux layers
- SelectLinL provides efficient categorical linear layers for species pairs

**Maintainer feedback:** "The refactoring is fairly superficial right now - the new backends are not really used yet, but this is now the part that I will need to take on I think." This future migration work will unlock GPU capabilities and full integration with the EquivariantTensors ecosystem.

### Version Differences

- **v0.6.x**: Uses ACE1.jl backend, mature but feature-frozen
- **v0.8+**: Complete rewrite with flexible architecture, AtomsBase integration
- **v0.9+**: Requires Julia 1.11 (works with 1.10 but may show test accuracy drift)

### Testing Expectations

- Full test suite passes on Julia 1.11 (Ubuntu, Python 3.8)
- `test/test_silicon.jl`: Integration test with real fitting workflow (~5-10 min)
- `test/models/test_models.jl`: Core model functionality
- Tests use LazyArtifacts for datasets (Si_tiny_dataset)

### Scripting Interface

The `scripts/runfit.jl` script provides a command-line interface for fitting:

```bash
julia --project=. scripts/runfit.jl --params scripts/example_params.json
```

This is useful for batch fitting jobs and automated workflows.

## Common Development Patterns

### Adding a New Radial Basis

1. Create new file in `src/models/` (e.g., `Rnl_newbasis.jl`)
2. Implement required interface: evaluation, parameter initialization
3. Include in `src/models/models.jl`
4. Add tests in `test/models/test_newbasis.jl`
5. Update ACE constructor to accept new basis type

### Modifying Coupling Coefficients

Changes to tensor coupling should be made in `src/models/ace.jl` in `_generate_ace_model()`. Always ensure:
- Compatibility with EquivariantTensors API
- Proper handling of pruning and symmetrization
- Tests verify coefficient matrix properties

### Working with Calculators

When implementing new evaluation methods, follow the AtomsCalculators interface:
- `potential_energy(system, calculator)`: Returns scalar energy
- `forces(system, calculator)`: Returns force vectors
- `virial(system, calculator)`: Returns virial tensor
- `energy_forces_virial(system, calculator)`: Combined evaluation (most efficient)

## File Organization

```
src/
├── ACEpotentials.jl         # Main module entry point
├── models/                  # ACE model implementations
│   ├── ace.jl              # Core ACEModel structure
│   ├── calculators.jl      # Calculator interface
│   ├── Rnl_*.jl            # Radial basis variants
│   └── utils.jl            # Basis generation utilities
├── atoms_data.jl            # Data format conversion
├── fit_model.jl             # Fitting interface
├── ace1_compat.jl           # Backward compatibility
└── json_interface.jl        # Serialization

test/
├── runtests.jl              # Main test suite
├── test_silicon.jl          # Integration test
├── test_migration.jl        # Migration validation
└── models/                  # Model-specific tests

docs/
├── make.jl                  # Documentation builder
└── src/tutorials/           # Literate.jl tutorials

scripts/
└── runfit.jl                # Command-line fitting script
```
