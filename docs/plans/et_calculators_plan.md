# Plan: ETACE Calculator Interface and Training Support

## Overview

Create calculator wrappers and training assembly for the new ETACE backend, integrating with EquivariantTensors.jl.

**Status**: 🔄 Refactoring to unified architecture - remove duplicate E0Model, use upstream models directly.

**Branch**: `jrk/etcalculators` (rebased on `acesuit/co/etback` including `co/etpair` merge)

---

## Progress Summary

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | ETACEPotential with AtomsCalculators interface | ✅ Complete |
| Phase 2 | WrappedSiteCalculator + StackedCalculator | 🔄 Refactoring |
| Phase 3 | E0Model + PairModel | ✅ Upstream (ETOneBody, ETPairModel, convertpair) |
| Phase 5 | Training assembly functions | ✅ Complete (many-body only) |
| Phase 6 | Full model integration | 🔄 In Progress |
| Benchmarks | Performance comparison scripts | ✅ Complete |

### Key Design Decision: Unified Architecture

**All upstream ETACE-pattern models share the same interface:**

| Method | ETACE | ETPairModel | ETOneBody |
|--------|-------|-------------|-----------|
| `model(G, ps, st)` | site energies | site energies | site energies |
| `site_grads(model, G, ps, st)` | edge gradients | edge gradients | zero gradients |
| `site_basis(model, G, ps, st)` | basis matrix | basis matrix | empty |
| `site_basis_jacobian(model, G, ps, st)` | (basis, jac) | (basis, jac) | (empty, empty) |

This enables a **unified `WrappedSiteCalculator`** that works with all three model types directly, eliminating the need for multiple wrapper types.

### Current Limitations

**ETACE currently only implements the many-body basis, not pair potential or reference energies.**

In the integration test (`test/et_models/test_et_silicon.jl`), we compare ETACE against ACE with `Wpair=0` (pair disabled) because:
- `convert2et(model)` converts only the many-body basis
- `convertpair(model)` converts the pair potential separately (not yet integrated)
- Reference energies (E0/Vref) need separate handling via `ETOneBody`

Full model conversion will require combining all three components via `StackedCalculator`.

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

## Phase 3: Upstream Implementation (Now Complete)

The maintainer has implemented E0/PairModel in the `co/etback` branch (merged via PR #316):

### New Files from Upstream

1. **`src/et_models/onebody.jl`** - `ETOneBody` one-body energy model
2. **`src/et_models/et_pair.jl`** - `ETPairModel` pair potential
3. **`src/et_models/et_envbranch.jl`** - Environment branch layer utilities
4. **`test/etmodels/test_etonebody.jl`** - OneBody tests
5. **`test/etmodels/test_etpair.jl`** - Pair potential tests

### Upstream Interface Pattern

The upstream models implement the **ETACE interface** (different from our SiteEnergyModel):

```julia
# Upstream interface (ETACE pattern):
model(G, ps, st)                    # Returns (site_energies, st)
site_grads(model, G, ps, st)        # Returns edge gradient array
site_basis(model, G, ps, st)        # Returns basis matrix
site_basis_jacobian(model, G, ps, st)  # Returns (basis, jacobian)
```

```julia
# Our interface (SiteEnergyModel pattern):
site_energies(model, G, ps, st)     # Returns site energies vector
site_energy_grads(model, G, ps, st) # Returns (edge_data = [...],) named tuple
cutoff_radius(model)                # Returns Float64 in Ångström
```

### `ETOneBody` Details (`onebody.jl`)

```julia
struct ETOneBody{NZ, T, CAT, TSEL} <: AbstractLuxLayer
   E0s::SVector{NZ, T}        # Reference energies per species
   categories::SVector{NZ, CAT}
   selector::TSEL             # Maps atom state to species index
end

# Constructor from Dict
one_body(D::Dict, catfun) -> ETOneBody

# Interface implementation
(l::ETOneBody)(X::ETGraph, ps, st)              # Returns site energies
site_grads(l::ETOneBody, X, ps, st)             # Returns zeros (constant energy)
site_basis(l::ETOneBody, X, ps, st)             # Returns empty (0 basis functions)
site_basis_jacobian(l::ETOneBody, X, ps, st)    # Returns empty
```

Key design decisions:
- E0s stored in **state** (`st.E0s`) for float type conversion (Float32/Float64)
- Uses `SVector` for GPU compatibility
- Returns `fill(VState(), ...)` for zero gradients (maintains edge structure)
- Returns `(nnodes, 0)` sized arrays for basis (no learnable parameters)

### `ETPairModel` Details (`et_pair.jl`)

```julia
@concrete struct ETPairModel <: AbstractLuxContainerLayer{(:rembed, :readout)}
   rembed     # Radial embedding layer (basis)
   readout    # SelectLinL readout layer
end

# Interface implementation
(l::ETPairModel)(X::ETGraph, ps, st)            # Returns site energies
site_grads(l::ETPairModel, X, ps, st)           # Zygote gradient
site_basis(l::ETPairModel, X, ps, st)           # Sum over neighbor radial basis
site_basis_jacobian(l::ETPairModel, X, ps, st)  # Uses ET.evaluate_ed
```

Key design decisions:
- **Owns its own `ps`/`st`** (Option A from original plan)
- Uses ET-native implementation (Option B from original plan)
- Radial basis: `𝔹 = sum(Rnl, dims=1)` - sums radial embeddings over neighbors
- GPU-compatible via ET's existing kernels

### Model Conversion (`convert.jl`)

```julia
convertpair(model::ACEModel) -> ETPairModel
```

Converts ACEModel's pair potential component to ETPairModel:
- Extracts radial basis parameters
- Creates `EnvRBranchL` envelope layer
- Sets up species-pair `SelectLinL` readout

---

## Refactoring Plan: Unified Architecture

### Motivation

The current implementation has **duplicate functionality**:
- Our `E0Model` duplicates upstream `ETOneBody`
- Multiple wrapper types (`WrappedETACE`, planned `WrappedETPairModel`, `WrappedETOneBody`) all do the same thing

Since all upstream models share the same interface, we can **unify to a single `WrappedSiteCalculator`**.

### Changes Required

#### 1. Remove `E0Model` (BREAKING)

Delete the `E0Model` struct and related functions. Users should migrate to:

```julia
# Old (our E0Model):
E0 = E0Model(Dict(:Si => -0.846, :O => -2.15))
calc = WrappedSiteCalculator(E0, 5.5)

# New (upstream ETOneBody):
et_onebody = ETM.one_body(Dict(:Si => -0.846, :O => -2.15), x -> x.z)
_, st = Lux.setup(rng, et_onebody)
calc = WrappedSiteCalculator(et_onebody, nothing, st, 3.0)  # rcut=3.0 minimum for graph
```

#### 2. Unify `WrappedSiteCalculator`

Refactor to store `ps` and `st` and work with ETACE-pattern models directly:

```julia
"""
    WrappedSiteCalculator{M, PS, ST}

Wraps any ETACE-pattern model (ETACE, ETPairModel, ETOneBody) and provides
the AtomsCalculators interface.

All wrapped models must implement:
- `model(G, ps, st)` → `(site_energies, st)`
- `site_grads(model, G, ps, st)` → edge gradients

# Fields
- `model` - ETACE-pattern model (ETACE, ETPairModel, or ETOneBody)
- `ps` - Model parameters (can be `nothing` for ETOneBody)
- `st` - Model state
- `rcut::Float64` - Cutoff radius for graph construction (Å)
"""
mutable struct WrappedSiteCalculator{M, PS, ST}
   model::M
   ps::PS
   st::ST
   rcut::Float64
end

# Convenience constructor with automatic cutoff
function WrappedSiteCalculator(model, ps, st)
   rcut = _model_cutoff(model, ps, st)
   return WrappedSiteCalculator(model, ps, st, max(rcut, 3.0))
end

# Cutoff extraction (type-specific)
_model_cutoff(::ETOneBody, ps, st) = 0.0
_model_cutoff(model::ETPairModel, ps, st) = _extract_rcut_from_rembed(model.rembed)
_model_cutoff(model::ETACE, ps, st) = _extract_rcut_from_rembed(model.rembed)
# Fallback: require explicit rcut
```

#### 3. Remove `WrappedETACE`

The functionality moves into `WrappedSiteCalculator`:

```julia
# Old (with WrappedETACE):
wrapped = WrappedETACE(et_model, ps, st, rcut)
calc = WrappedSiteCalculator(wrapped, rcut)

# New (direct):
calc = WrappedSiteCalculator(et_model, ps, st, rcut)
```

#### 4. Update `ETACEPotential` Type Alias

```julia
# Old:
const ETACEPotential{MOD, PS, ST} = WrappedSiteCalculator{WrappedETACE{MOD, PS, ST}}

# New:
const ETACEPotential{MOD<:ETACE, PS, ST} = WrappedSiteCalculator{MOD, PS, ST}
```

#### 5. Unified Energy/Force/Virial Implementation

```julia
function _wrapped_energy(calc::WrappedSiteCalculator, sys::AbstractSystem)
   G = ET.Atoms.interaction_graph(sys, calc.rcut * u"Å")
   Ei, _ = calc.model(G, calc.ps, calc.st)
   return sum(Ei)
end

function _wrapped_forces(calc::WrappedSiteCalculator, sys::AbstractSystem)
   G = ET.Atoms.interaction_graph(sys, calc.rcut * u"Å")
   ∂G = site_grads(calc.model, G, calc.ps, calc.st)
   if isempty(∂G.edge_data)
      return zeros(SVector{3, Float64}, length(sys))
   end
   return -ET.Atoms.forces_from_edge_grads(sys, G, ∂G.edge_data)
end
```

### Benefits of Unified Architecture

1. **No code duplication** - Single wrapper handles all model types
2. **Use upstream directly** - `ETOneBody`, `ETPairModel` work out-of-the-box
3. **GPU-compatible** - Upstream models use `SVector` for efficient GPU ops
4. **Simpler mental model** - One wrapper type, one interface
5. **Easier testing** - Test interface once, works for all models

### Migration Path

| Old | New |
|-----|-----|
| `E0Model(Dict(:Si => -0.846))` | `ETM.one_body(Dict(:Si => -0.846), x -> x.z)` |
| `WrappedETACE(model, ps, st, rcut)` | `WrappedSiteCalculator(model, ps, st, rcut)` |
| `WrappedSiteCalculator(E0Model(...))` | `WrappedSiteCalculator(ETOneBody(...), nothing, st)` |

### Backward Compatibility

For a transition period, we could keep `E0Model` as a deprecated alias:

```julia
@deprecate E0Model(d::Dict) begin
   et = one_body(d, x -> x.z)
   _, st = Lux.setup(Random.default_rng(), et)
   (model=et, ps=nothing, st=st)
end
```

However, since this is internal API on a feature branch, clean removal is preferred.

---

## Files Created/Modified

### Our Branch (jrk/etcalculators)
- `src/et_models/et_calculators.jl` - WrappedSiteCalculator (unified), ETACEPotential, training assembly
  - **To Remove**: `E0Model`, `WrappedETACE`, old `SiteEnergyModel` interface
- `src/et_models/stackedcalc.jl` - StackedCalculator with @generated loop unrolling
- `test/et_models/test_et_calculators.jl` - Comprehensive unit tests
  - **To Update**: Remove E0Model tests, update WrappedSiteCalculator signature
- `test/et_models/test_et_silicon.jl` - Integration test (compares many-body only)
- `benchmark/benchmark_comparison.jl` - Energy benchmarks (CPU + GPU)
- `benchmark/benchmark_forces.jl` - Forces benchmarks (CPU)

### Upstream (now merged via co/etpair)
- `src/et_models/onebody.jl` - `ETOneBody` Lux layer with `one_body()` constructor (**replaces our E0Model**)
- `src/et_models/et_pair.jl` - `ETPairModel` Lux layer with site_basis/jacobian
- `src/et_models/et_envbranch.jl` - `EnvRBranchL` for envelope × radial basis
- `src/et_models/convert.jl` - Added `convertpair()`, envelope conversion utilities
- `test/etmodels/test_etonebody.jl` - OneBody tests
- `test/etmodels/test_etpair.jl` - Pair model tests (shows parameter copying pattern)
- `test/etmodels/test_etbackend.jl` - General ET backend tests

### Modified Files
- `src/et_models/et_models.jl` - Includes for all new files
- `docs/src/all_exported.md` - Added ETModels to autodocs

---

## Implementation Details

### Current Architecture (to be refactored)

The current implementation uses nested wrappers:
```
StackedCalculator
├── WrappedSiteCalculator{E0Model}           # Our duplicate (TO REMOVE)
├── WrappedSiteCalculator{WrappedETACE}      # Extra indirection (TO REMOVE)
```

### Target Architecture (unified)

After refactoring, use upstream models directly:
```
StackedCalculator
├── WrappedSiteCalculator{ETOneBody}         # Upstream one-body
├── WrappedSiteCalculator{ETPairModel}       # Upstream pair
└── WrappedSiteCalculator{ETACE}             # Upstream many-body
```

### WrappedSiteCalculator (`et_calculators.jl`) - TARGET

Unified wrapper for any ETACE-pattern model:

```julia
mutable struct WrappedSiteCalculator{M, PS, ST}
   model::M      # ETACE, ETPairModel, or ETOneBody
   ps::PS        # Parameters (nothing for ETOneBody)
   st::ST        # State
   rcut::Float64 # Cutoff for graph construction
end

# All ETACE-pattern models have identical interface:
function _wrapped_energy(calc::WrappedSiteCalculator, sys)
   G = ET.Atoms.interaction_graph(sys, calc.rcut * u"Å")
   Ei, _ = calc.model(G, calc.ps, calc.st)  # Works for all model types!
   return sum(Ei)
end

function _wrapped_forces(calc::WrappedSiteCalculator, sys)
   G = ET.Atoms.interaction_graph(sys, calc.rcut * u"Å")
   ∂G = site_grads(calc.model, G, calc.ps, calc.st)  # Works for all model types!
   return -ET.Atoms.forces_from_edge_grads(sys, G, ∂G.edge_data)
end
```

### ETACEPotential Type Alias - TARGET

```julia
const ETACEPotential{MOD<:ETACE, PS, ST} = WrappedSiteCalculator{MOD, PS, ST}

function ETACEPotential(model::ETACE, ps, st, rcut::Real)
   return WrappedSiteCalculator(model, ps, st, Float64(rcut))
end
```

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

**Note**: Training assembly currently only works with `ETACE` (many-body).
Extension to `ETPairModel` will use the same `site_basis_jacobian` interface.
`ETOneBody` has no learnable parameters (empty basis).

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

Upstream tests in `test/etmodels/`:
- ✅ `test_etonebody.jl` - ETOneBody evaluation and gradients
- ✅ `test_etpair.jl` - ETPairModel evaluation, gradients, basis, jacobian

---

## Remaining Work

### Phase 6: Unified Architecture Refactoring

**Goal**: Simplify codebase by using upstream models directly with unified `WrappedSiteCalculator`.

#### 6.1 Refactor `WrappedSiteCalculator` (et_calculators.jl)

1. Change struct to store `ps` and `st`:
   ```julia
   mutable struct WrappedSiteCalculator{M, PS, ST}
      model::M
      ps::PS
      st::ST
      rcut::Float64
   end
   ```

2. Update `_wrapped_energy`, `_wrapped_forces`, `_wrapped_virial` to call ETACE interface directly

3. Add cutoff extraction helpers:
   ```julia
   _model_cutoff(::ETOneBody, ps, st) = 0.0
   _model_cutoff(model::ETPairModel, ps, st) = ...  # extract from rembed
   _model_cutoff(model::ETACE, ps, st) = ...  # extract from rembed
   ```

#### 6.2 Remove Redundant Code

1. **Delete `E0Model`** - replaced by upstream `ETOneBody`
2. **Delete `WrappedETACE`** - functionality merged into `WrappedSiteCalculator`
3. **Remove old SiteEnergyModel interface** - use ETACE interface directly

#### 6.3 Update `ETACEPotential` Type Alias

```julia
const ETACEPotential{MOD<:ETACE, PS, ST} = WrappedSiteCalculator{MOD, PS, ST}

function ETACEPotential(model::ETACE, ps, st, rcut::Real)
   return WrappedSiteCalculator(model, ps, st, Float64(rcut))
end
```

#### 6.4 Full Model Conversion Function

```julia
"""
    convert2et_full(model::ACEModel, ps, st; rng=Random.default_rng()) -> StackedCalculator

Convert a complete ACE model (E0 + Pair + Many-body) to an ETACE calculator.
Returns a StackedCalculator combining ETOneBody, ETPairModel, and ETACE.
"""
function convert2et_full(model, ps, st; rng=Random.default_rng())
   rcut = maximum(a.rcut for a in model.pairbasis.rin0cuts)

   # 1. Convert E0/Vref to ETOneBody
   E0s = model.Vref.E0  # Dict{Int, Float64}
   zlist = ChemicalSpecies.(model.rbasis._i2z)
   E0_dict = Dict(z => E0s[z.number] for z in zlist)
   et_onebody = one_body(E0_dict, x -> x.z)
   _, onebody_st = Lux.setup(rng, et_onebody)
   onebody_calc = WrappedSiteCalculator(et_onebody, nothing, onebody_st, 3.0)

   # 2. Convert pair potential to ETPairModel
   et_pair = convertpair(model)
   et_pair_ps, et_pair_st = Lux.setup(rng, et_pair)
   _copy_pair_params!(et_pair_ps, ps, model)
   pair_calc = WrappedSiteCalculator(et_pair, et_pair_ps, et_pair_st, rcut)

   # 3. Convert many-body to ETACE
   et_ace = convert2et(model)
   et_ace_ps, et_ace_st = Lux.setup(rng, et_ace)
   _copy_ace_params!(et_ace_ps, ps, model)
   ace_calc = WrappedSiteCalculator(et_ace, et_ace_ps, et_ace_st, rcut)

   # 4. Stack all components
   return StackedCalculator((onebody_calc, pair_calc, ace_calc))
end
```

#### 6.5 Parameter Copying Utilities

From `test/etmodels/test_etpair.jl`, pair parameter copying for multi-species:
```julia
function _copy_pair_params!(et_ps, ps, model)
   NZ = length(model.rbasis._i2z)
   for i in 1:NZ, j in 1:NZ
      idx = (i-1)*NZ + j
      et_ps.rembed.rbasis.post.W[:, :, idx] = ps.pairbasis.Wnlq[:, :, i, j]
   end
   for s in 1:NZ
      et_ps.readout.W[1, :, s] .= ps.Wpair[:, s]
   end
end
```

#### 6.6 Update Tests

1. Update `test/et_models/test_et_calculators.jl`:
   - Remove `E0Model` tests
   - Add `ETOneBody` integration tests
   - Update `WrappedSiteCalculator` tests for new signature

2. Update `test/et_models/test_et_silicon.jl`:
   - Use `ETOneBody` instead of `E0Model` if testing E0

#### 6.7 Training Assembly Updates

1. Extend `energy_forces_virial_basis` to work with unified `WrappedSiteCalculator`:
   - Detect model type and call appropriate `site_basis_jacobian`
   - Works with `ETACE`, `ETPairModel` (both have `site_basis_jacobian`)
   - `ETOneBody` returns empty basis (no learnable params)

2. Update `length_basis`, `get_linear_parameters`, `set_linear_parameters!`

### Future Enhancements

- GPU forces benchmark (requires GPU gradient support in ET)
- ACEfit.assemble dispatch integration for full models
- Committee support for combined calculators
- Training assembly for pair model (similar structure to many-body)

---

## Notes

- Virial formula: `V = -∑ ∂E/∂𝐫ij ⊗ 𝐫ij`
- GPU time nearly constant regardless of system size (~0.5ms)
- Forces speedup (8-11x) larger than energy speedup (1.5-2.5x) on CPU
- StackedCalculator uses @generated functions for zero-overhead composition
- Upstream `ETOneBody` stores E0s in state (`st.E0s`) for float type flexibility (Float32/Float64)
- All upstream models use `VState` for gradients in `site_grads()` return value
- `site_grads` returns edge gradients as `∂G` with `.edge_data` field containing `VState` objects
