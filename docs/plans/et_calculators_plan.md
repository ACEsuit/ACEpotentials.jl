# Plan: ETACE Calculator Interface and Training Support

## Overview

Create calculator wrappers and training assembly for the new ETACE backend, integrating with EquivariantTensors.jl.

**Status**: ✅ Core implementation complete. Awaiting maintainer for E0/PairModel.

**Branch**: `jrk/etcalculators` (based on `acesuit/co/etback`)

---

## Progress Summary

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | ETACEPotential with AtomsCalculators interface | ✅ Complete |
| Phase 2 | WrappedSiteCalculator + StackedCalculator | ✅ Complete |
| Phase 3 | E0Model + PairModel | 🔄 Maintainer will implement |
| Phase 5 | Training assembly functions | ✅ Complete |
| Benchmarks | Performance comparison scripts | ✅ Complete |

### Benchmark Results

**Energy (test/benchmark_comparison.jl)**:
| Atoms | ACE CPU (ms) | ETACE CPU (ms) | ETACE GPU (ms) | CPU Speedup | GPU Speedup |
|-------|--------------|----------------|----------------|-------------|-------------|
| 8 | 0.87 | 0.43 | 0.39 | 2.0x | 2.2x |
| 64 | 5.88 | 2.79 | 0.45 | 2.1x | 13.0x |
| 256 | 17.77 | 11.81 | 0.48 | 1.5x | 37.1x |
| 800 | 53.03 | 30.32 | 0.61 | 1.7x | **87.6x** |

**Forces (test/benchmark_forces.jl)**:
| Atoms | ACE CPU (ms) | ETACE CPU (ms) | CPU Speedup |
|-------|--------------|----------------|-------------|
| 8 | 9.27 | 0.88 | 10.6x |
| 64 | 73.58 | 9.62 | 7.7x |
| 256 | 297.36 | 27.09 | 11.0x |
| 800 | 926.90 | 109.49 | **8.5x** |

---

## Files Created/Modified

### New Files
- `src/et_models/et_calculators.jl` - ETACEPotential, WrappedSiteCalculator, WrappedETACE, training assembly
- `src/et_models/stackedcalc.jl` - StackedCalculator with @generated loop unrolling
- `test/et_models/test_et_calculators.jl` - Comprehensive tests
- `test/benchmark_comparison.jl` - Energy benchmarks (CPU + GPU)
- `test/benchmark_forces.jl` - Forces benchmarks (CPU)

### Modified Files
- `src/et_models/et_models.jl` - Added includes for new files
- `test/Project.toml` - Updated EquivariantTensors compat to 0.4

---

## Implementation Details

### ETACEPotential (`et_calculators.jl`)

Standalone calculator wrapping ETACE with full AtomsCalculators interface:

```julia
mutable struct ETACEPotential{MOD<:ETACE, T} <: SitePotential
   model::MOD
   ps::T
   st::NamedTuple
   rcut::Float64
   co_ps::Any  # optional committee parameters
end
```

Implements:
- `potential_energy(sys, calc)`
- `forces(sys, calc)`
- `virial(sys, calc)`
- `energy_forces_virial(sys, calc)`

### WrappedSiteCalculator (`et_calculators.jl`)

Generic wrapper for models implementing site energy interface:

```julia
struct WrappedSiteCalculator{M}
   model::M
end
```

Site energy interface:
- `site_energies(model, G, ps, st) -> Vector`
- `site_energy_grads(model, G, ps, st) -> (edge_data = [...],)`
- `cutoff_radius(model) -> Unitful.Length`

### StackedCalculator (`stackedcalc.jl`)

Combines multiple AtomsCalculators using @generated functions for type-stable loop unrolling:

```julia
struct StackedCalculator{N, C<:Tuple}
   calcs::C
end

@generated function _stacked_energy(sys::AbstractSystem, calc::StackedCalculator{N}) where {N}
   # Generates: E_1 + E_2 + ... + E_N at compile time
end
```

### Training Assembly (`et_calculators.jl`)

Functions for linear least squares fitting:

- `length_basis(calc)` - Total number of linear parameters
- `get_linear_parameters(calc)` - Extract parameter vector
- `set_linear_parameters!(calc, θ)` - Set parameters from vector
- `potential_energy_basis(sys, calc)` - Energy design matrix row
- `energy_forces_virial_basis(sys, calc)` - Full design matrix row

---

## Maintainer Decisions (Phase 3)

**Q2: Parameter ownership** → **Option A**: PairModel owns its own `ps`/`st`

**Q3: Implementation approach** → **Option B**: Create new ET-native pair implementation
- Native GPU support
- Consistent with ETACE architecture

Maintainer will implement E0Model and PairModel given their ACE experience.

---

## Current State (Already Implemented)

### In ACEpotentials (`src/et_models/`)

**ETACE struct** (`et_ace.jl:11-16`):
```julia
@concrete struct ETACE <: AbstractLuxContainerLayer{(:rembed, :yembed, :basis, :readout)}
   rembed     # radial embedding layer
   yembed     # angular embedding layer
   basis      # many-body basis layer
   readout    # selectlinl readout layer
end
```

**Core functions** (`et_ace.jl`):
- ✅ `(l::ETACE)(X::ETGraph, ps, st)` - forward evaluation, returns site energies
- ✅ `site_grads(l::ETACE, X::ETGraph, ps, st)` - Zygote gradient for forces
- ✅ `site_basis(l::ETACE, X::ETGraph, ps, st)` - basis values per site
- ✅ `site_basis_jacobian(l::ETACE, X::ETGraph, ps, st)` - basis + jacobians

**Model conversion** (`convert.jl`):
- ✅ `convert2et(model::ACEModel)` - full conversion from ACEModel to ETACE

### In EquivariantTensors.jl (v0.4.0)

**Atoms extension** (`ext/NeighbourListsExt.jl`):
- ✅ `ET.Atoms.interaction_graph(sys, rcut)` - ETGraph from AtomsBase system
- ✅ `ET.Atoms.forces_from_edge_grads(sys, G, ∇E_edges)` - edge gradients to atomic forces
- ✅ `ET.rev_reshape_embedding` - neighbor-indexed to edge-indexed conversion

---

## Test Coverage

Tests in `test/et_models/test_et_calculators.jl`:

1. ✅ WrappedETACE site energies consistency
2. ✅ WrappedETACE site energy gradients (finite difference)
3. ✅ WrappedSiteCalculator AtomsCalculators interface
4. ✅ Forces finite difference validation
5. ✅ Virial finite difference validation
6. ✅ ETACEPotential consistency with WrappedSiteCalculator
7. ✅ StackedCalculator composition (E0 + ACE)
8. ✅ Training assembly: length_basis, get/set_linear_parameters
9. ✅ Training assembly: potential_energy_basis
10. ✅ Training assembly: energy_forces_virial_basis

---

## Remaining Work

### For Maintainer (Phase 3)

1. **E0Model**: One-body reference energies
   - Store E0s in state for float type conversion
   - Implement site energy interface (zero gradients)

2. **PairModel**: ET-native pair potential
   - New implementation using `ET.Atoms` patterns
   - GPU-compatible
   - Implement site energy interface

### Future Enhancements

- GPU forces benchmark (requires GPU gradient support)
- ACEfit.assemble dispatch integration
- Committee support for ETACEPotential

---

## Notes

- Virial formula: `V = -∑ ∂E/∂𝐫ij ⊗ 𝐫ij`
- GPU time nearly constant regardless of system size (~0.5ms)
- Forces speedup (8-11x) larger than energy speedup (1.5-2.5x) on CPU
- StackedCalculator uses @generated functions for zero-overhead composition
