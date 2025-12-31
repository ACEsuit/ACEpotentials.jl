# Plan: ETACE Calculator Interface and Training Support

## Overview

Create calculator wrappers and training assembly for the new ETACE backend, integrating with EquivariantTensors.jl.

**Status**: ✅ Core implementation complete. GPU acceleration working.

**Branch**: `jrk/etcalculators` (rebased on `acesuit/co/etback` including `co/etpair` merge)

---

## Progress Summary

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | ETACEPotential with AtomsCalculators interface | ✅ Complete |
| Phase 2 | WrappedSiteCalculator + StackedCalculator | ✅ Complete |
| Phase 3 | E0Model + PairModel | ✅ Complete (upstream ETOneBody, ETPairModel) |
| Phase 5 | Training assembly functions | ✅ Complete (many-body only) |
| Phase 6 | Full model integration | ✅ Complete |
| Benchmarks | CPU + GPU performance comparison | ✅ Complete |

### Key Design Decision: Unified Architecture

**All upstream ETACE-pattern models share the same interface:**

| Method | ETACE | ETPairModel | ETOneBody |
|--------|-------|-------------|-----------|
| `model(G, ps, st)` | site energies | site energies | site energies |
| `site_grads(model, G, ps, st)` | edge gradients | edge gradients | zero gradients |
| `site_basis(model, G, ps, st)` | basis matrix | basis matrix | empty |
| `site_basis_jacobian(model, G, ps, st)` | (basis, jac) | (basis, jac) | (empty, empty) |

This enables a **unified `WrappedSiteCalculator`** that works with all three model types directly.

---

## Benchmark Results

### GPU Benchmarks (Many-Body Only - ETACE)

**Energy:**
| Atoms | Edges | CPU (ms) | GPU (ms) | GPU Speedup |
|-------|-------|----------|----------|-------------|
| 64 | 2146 | 3.38 | 0.54 | **6.3x** |
| 512 | 17176 | 27.77 | 0.66 | **41.9x** |
| 800 | 26868 | 37.12 | 0.78 | **47.6x** |

**Forces:**
| Atoms | Edges | CPU (ms) | GPU (ms) | GPU Speedup |
|-------|-------|----------|----------|-------------|
| 64 | 2146 | 46.46 | 14.42 | **3.2x** |
| 512 | 17178 | 104.39 | 15.12 | **6.9x** |
| 800 | 26860 | 289.32 | 16.33 | **17.7x** |

### GPU Benchmarks (Full Model - E0 + Pair + Many-Body)

**Energy:**
| Atoms | Edges | CPU (ms) | GPU (ms) | GPU Speedup |
|-------|-------|----------|----------|-------------|
| 64 | 2140 | 3.40 | 0.94 | **3.6x** |
| 512 | 17166 | 31.18 | 0.95 | **32.9x** |
| 800 | 26858 | 45.16 | 1.24 | **36.4x** |

**Forces:**
| Atoms | Edges | CPU (ms) | GPU (ms) | GPU Speedup |
|-------|-------|----------|----------|-------------|
| 64 | 2134 | 24.05 | 19.34 | **1.2x** |
| 512 | 17178 | ~110 | ~20 | **~5x** |
| 800 | 26860 | ~300 | ~22 | **~14x** |

### CPU Benchmarks (ETACE vs Classic ACE)

**Forces (Full Model):**
| Atoms | Edges | ACE CPU (ms) | ETACE CPU (ms) | ETACE Speedup |
|-------|-------|--------------|----------------|---------------|
| 64 | 2146 | 73.6 | 30.5 | **2.4x** |
| 256 | 8596 | 307.7 | 74.4 | **4.1x** |
| 800 | 26886 | 975.0 | 225.6 | **4.3x** |

**Notes:**
- GPU forces require Polynomials4ML v0.5.8+ (bug fix Dec 29, 2024)
- GPU shows excellent scaling: larger systems see better speedups
- Full model GPU speedups are lower than many-body only due to graph construction overhead
- CPU forces are 2-4x faster with ETACE due to Zygote AD through ET graph

---

## Architecture

### Current Implementation (Complete)

```
StackedCalculator
├── WrappedSiteCalculator{ETOneBody}         # One-body reference energies
├── WrappedSiteCalculator{ETPairModel}       # Pair potential
└── WrappedSiteCalculator{ETACE}             # Many-body ACE
```

### Core Components

**WrappedSiteCalculator{M, PS, ST}** (`et_calculators.jl`)
- Unified wrapper for any ETACE-pattern model
- Provides AtomsCalculators interface (energy, forces, virial)
- Mutable to allow parameter updates during training

**ETACEPotential** - Type alias for `WrappedSiteCalculator{ETACE, PS, ST}`

**StackedCalculator{N, C}** (`stackedcalc.jl`)
- Combines multiple calculators by summing contributions
- Uses @generated functions for type-stable loop unrolling

### Conversion Functions

```julia
convert2et(model)                    # Many-body ACE → ETACE
convertpair(model)                   # Pair potential → ETPairModel
convert2et_full(model, ps, st)       # Full model → StackedCalculator
```

### Training Assembly (Many-Body Only)

```julia
length_basis(calc)                           # Total linear parameters
get_linear_parameters(calc)                  # Extract θ vector
set_linear_parameters!(calc, θ)              # Set θ vector
potential_energy_basis(sys, calc)            # Energy design row
energy_forces_virial_basis(sys, calc)        # Full EFV design row
```

---

## Files

### Source Files
- `src/et_models/et_ace.jl` - ETACE model implementation
- `src/et_models/et_pair.jl` - ETPairModel implementation
- `src/et_models/onebody.jl` - ETOneBody implementation
- `src/et_models/et_calculators.jl` - WrappedSiteCalculator, ETACEPotential, training assembly
- `src/et_models/stackedcalc.jl` - StackedCalculator with @generated
- `src/et_models/convert.jl` - Model conversion utilities
- `src/et_models/et_envbranch.jl` - EnvRBranchL for envelope × radial basis
- `src/et_models/et_models.jl` - Module includes and exports

### Test Files
- `test/etmodels/test_etbackend.jl` - ETACE tests
- `test/etmodels/test_etpair.jl` - ETPairModel tests
- `test/etmodels/test_etonebody.jl` - ETOneBody tests

### Benchmark Files
- `benchmark/gpu_benchmark.jl` - GPU energy/forces benchmarks
- `benchmark/benchmark_full_model.jl` - CPU comparison benchmarks

---

## Outstanding Work

### ~~1. Training Assembly for Pair Model~~ ✅ Complete
**Status**: Implemented in `et_calculators.jl` and `stackedcalc.jl`

**What was done**:
- Added `ETPairPotential` type alias with full training assembly support
- Added `ETOneBodyPotential` type alias (returns empty arrays - no learnable params)
- Implemented `length_basis`, `energy_forces_virial_basis`, `potential_energy_basis`, `get_linear_parameters`, `set_linear_parameters!` for all calculator types
- Extended `StackedCalculator` to concatenate basis functions from all components
- Added `ACEfit.basis_size` dispatch for all calculator types

### ~~2. ACEfit.assemble Dispatch Integration~~ ✅ Complete
**Status**: Works out-of-the-box after extending `length_basis` and `energy_forces_virial_basis`

**What was done**:
- Added empty function declarations in `models/models.jl` for `length_basis`, `energy_forces_virial_basis`, `potential_energy_basis`
- ETModels now imports and extends these functions
- `ACEfit.feature_matrix(d::AtomsData, calc)` works with ETACE calculators
- `ACEfit.assemble(data, calc)` works with `StackedCalculator`

### 3. Committee Support
**Priority**: Low
**Description**: Extend committee/uncertainty quantification to work with StackedCalculator.

### 4. Basis Index Design Discussion
**Priority**: Needs Discussion
**Description**: Moderator raised concern about basis indices:

> "I realized I made a mistake in the design of the basis interface. I'm returning the site energy basis but for each center-atom species, the basis occupies the same indices. We need to perform a transformation so that bases for different species occupy separate indices."

**Current Implementation**: Species separation is handled at the **calculator level** in `energy_forces_virial_basis` using `p = (s-1) * nbasis + k`. Each species gets separate parameter indices.

**Options**:
1. Keep current approach (calculator-level separation)
2. Move to site potential model level
3. Handle at WrappedSiteCalculator level

Moderator wants discussion before making changes.

---

## Dependencies

- EquivariantTensors.jl >= 0.4.2
- Polynomials4ML.jl >= 0.5.8 (for GPU forces)
- LuxCUDA (for GPU support, test dependency)

---

## Test Status

All tests pass: **945 passed, 1 broken** (known Julia 1.12 hash ordering issue)

```bash
# Run ET model tests
julia --project=test -e 'using Pkg; Pkg.test("ACEpotentials"; test_args=["etmodels"])'

# Run GPU benchmark
julia --project=test benchmark/gpu_benchmark.jl
```

---

## Notes

- Virial formula: `V = -∑ ∂E/∂𝐫ij ⊗ 𝐫ij`
- GPU time scales sub-linearly with system size
- Forces speedup (CPU) larger than energy speedup due to Zygote AD efficiency
- StackedCalculator uses @generated functions for zero-overhead composition
- Upstream `ETOneBody` stores E0s in state (`st.E0s`) for float type flexibility
- All models use `VState` for edge gradients in `site_grads()` return
