# Migration Status: EquivariantTensors v0.3

**Date**: 2025-11-12
**Branch**: `claude/explore-repo-structure-011CV4PfXPf4MHS4ceHdoMQe`
**Status**: 🚧 **BLOCKED - Custom _pfwd Functions Need Rewrite**

## Summary

Migration from EquivariantModels to EquivariantTensors v0.3 has made significant progress. Package now compiles successfully, but experimental pushforward (_pfwd) functions need rewriting for new API.

## Progress Overview

### Completed ✅

1. **Fixed symmetrisation_matrix API mismatch** - Converted `(n,l,m)` → `(n,l)` format
2. **Upgraded all dependencies** - Resolved cascading version conflicts
3. **Fixed LuxCore 1.x API changes** - Updated AbstractLux* types
4. **Fixed Polynomials4ML 0.5 changes** - Moved types to EquivariantTensors
5. **Renamed SparseSymmProdDAG → SparseSymmProd**
6. **Fixed metadata storage** - Moved from basis objects to SparseEquivTensor
7. **Removed projection field usage** - New API doesn't need intermediate projection
8. **Package compiles successfully** ✅

### Current Blocker ❌

**Experimental `_pfwd` functions incompatible with new SparseSymmProd API**

## Critical Findings

### 1. Dependency Upgrades ✅ COMPLETE

| Package | Old Version | New Version | Status |
|---------|-------------|-------------|--------|
| Polynomials4ML | 0.3 | 0.5 | ✅ Updated |
| Lux | 0.5 | 1.x | ✅ Updated |
| LuxCore | 0.1 | 1.x | ✅ Updated |
| SpheriCart | 0.1 | 0.2 | ✅ Updated |
| Bumper | 0.6 | 0.7 | ✅ Updated |
| EquivariantTensors | - | 0.3 | ✅ Added |

### 2. API Breaking Changes ✅ MOSTLY FIXED

#### LuxCore 1.x ✅ FIXED

| Old API | New API | Status |
|---------|---------|--------|
| `AbstractExplicitLayer` | `AbstractLuxLayer` | ✅ Fixed |
| `AbstractExplicitContainerLayer` | `AbstractLuxContainerLayer` | ✅ Fixed |

**Files Updated**: `src/models/models.jl`, `src/models/Rnl_basis.jl`, `src/models/ace.jl`

#### Polynomials4ML 0.5 ✅ FIXED

| Old Location | New Location | Status |
|--------------|--------------|--------|
| `Polynomials4ML.PooledSparseProduct` | `EquivariantTensors.PooledSparseProduct` | ✅ Fixed |
| `Polynomials4ML.SparseSymmProdDAG` | `EquivariantTensors.SparseSymmProd` | ✅ Fixed |

**Files Updated**: `src/models/ace.jl`, `src/models/sparse.jl`, `src/models/fasteval.jl`

#### SparseSymmProdDAG → SparseSymmProd API Changes

**Old Structure (SparseSymmProdDAG)**:
```julia
struct SparseSymmProdDAG
   num1::Int
   nodes::Vector{Tuple{Int,Int}}
   projection::Vector{Int}
   ...
end
```

**New Structure (SparseSymmProd)**:
```julia
struct SparseSymmProd{ORD, TS}
   specs::TS
   ranges::NTuple{ORD, UnitRange{Int}}
   hasconst::Bool
end
```

**Key Changes**:
- ❌ Removed `.projection` field - no longer needs intermediate projection
- ❌ Removed `.num1` and `.nodes` fields - completely different internal structure
- ❌ No `.meta` field - metadata stored in wrapper (SparseEquivTensor)

### 3. Fixes Applied

#### A. Metadata Storage ✅ FIXED

**Old**: Store metadata on basis objects
```julia
a_basis.meta["A_spec"] = A_spec
aa_basis.meta["AA_spec"] = AA_spec
```

**New**: Store in SparseEquivTensor wrapper
```julia
tensor_meta = Dict{String, Any}("A_spec" => A_spec, "AA_spec" => AA_spec)
tensor = SparseEquivTensor(a_basis, aa_basis, AA2BB_map, tensor_meta)
```

**Changed**:
- `src/models/ace.jl:142-151`
- `src/models/sparse.jl:114` - Changed `tensor.aabasis.meta` → `tensor.meta`
- `src/models/fasteval.jl:52` - Changed `aabasis.meta` → `model.model.tensor.meta`

#### B. Projection Field Removal ✅ FIXED

The old API used an intermediate "full" representation with projection to a "pruned" representation. The new API evaluates directly to the correct size without projection.

**Forward Pass** (`src/models/sparse.jl:16-32`):
```julia
# OLD:
P4ML.evaluate!(_AA, tensor.aabasis, A)
proj = tensor.aabasis.projection
AA = _AA[proj]
mul!(B, tensor.A2Bmap, AA)

# NEW:
P4ML.evaluate!(_AA, tensor.aabasis, A)
mul!(B, tensor.A2Bmap, _AA)  # Use _AA directly
```

**Backward Pass** (`src/models/sparse.jl:51-75`):
```julia
# OLD:
∂AA = @alloc(T_∂AA, size(tensor.A2Bmap, 2))
mul!(∂AA, tensor.A2Bmap', ∂B)
_∂AA = @alloc(T_∂AA, length(_AA))
fill!(_∂AA, zero(T_∂AA))
_∂AA[proj] = ∂AA
P4ML.unsafe_pullback!(∂A, _∂AA, tensor.aabasis, _AA)

# NEW:
_∂AA = @alloc(T_∂AA, size(tensor.A2Bmap, 2))
mul!(_∂AA, tensor.A2Bmap', ∂B)
P4ML.unsafe_pullback!(∂A, _∂AA, tensor.aabasis, _AA)
```

**Files Updated**:
- `src/models/sparse.jl` - evaluate!, pullback!, _pfwd
- `src/models/fasteval.jl:72` - weights mapping

### 4. Current Blocker: _pfwd Functions ❌

**Location**: `src/models/sparse.jl:166-199`, called from `src/models/ace.jl:698`

**Problem**: Custom pushforward implementations access fields that don't exist in new API:

```julia
function _pfwd(aabasis::EquivariantTensors.SparseSymmProd, A, ∂A)
   num1 = aabasis.num1  # ❌ ERROR: no field num1
   nodes = aabasis.nodes  # ❌ ERROR: no field nodes
   ...
end
```

**Impact**: `evaluate_basis_ed` function fails when computing gradients for forces/virials.

**Error Message**:
```
ERROR: type SparseSymmProd has no field num1
Stacktrace:
 [2] _pfwd(aabasis::EquivariantTensors.SparseSymmProd, A, ∂A)
   @ ACEpotentials.Models ~/ACEpotentials.jl/src/models/sparse.jl:168
```

## Solutions

### Option 1: Reimplement _pfwd for New API (Recommended for Production)

**Effort**: Medium (2-3 hours)
**Risk**: Medium

**Steps**:
1. Study new `SparseSymmProd` internal structure (`specs`, `ranges`, `hasconst`)
2. Rewrite `_pfwd(::SparseSymmProd, ...)` using new fields
3. Ensure equivalent performance to old DAG-based approach
4. Test thoroughly with baseline comparisons

**Pros**:
- Maintains performance optimizations
- Complete migration
- Preserves all functionality

**Cons**:
- Requires deep understanding of new API internals
- Testing burden to ensure correctness
- May need performance tuning

### Option 2: Use Standard evaluate_ed! Path (Quick Workaround)

**Effort**: Low (1 hour)
**Risk**: Low-Medium

**Steps**:
1. Check if Polynomials4ML/EquivariantTensors provide standard `evaluate_ed!` methods
2. Rewrite `evaluate_basis_ed` to use standard methods instead of custom `_pfwd`
3. Accept potential performance regression (experimental features)
4. Test for correctness

**Pros**:
- Simpler, less code to maintain
- Relies on upstream implementations
- Lower risk of bugs

**Cons**:
- Potential performance degradation
- May not have equivalent functionality
- Might need fallback to Zygote/automatic differentiation

### Option 3: Disable _pfwd-Dependent Code Paths ⚠️ IMPLEMENTED - VERY LIMITED

**Effort**: Very Low (30min) ✅ DONE
**Risk**: High
**Status**: ✅ Implemented, ⚠️ **Cannot fit models**

**Steps Completed**:
1. ✅ Disabled `evaluate_basis_ed` with clear error message
2. ✅ Created test script (`test_energy_only.jl`)
3. ✅ Tested package loading and model construction

**Key Finding**: **Fitting is not possible**, even for energy-only data.

**Root Cause**: `energy_forces_virial_basis` (calculators.jl:309) **always** calls
`evaluate_basis_ed` during assembly to build the full feature matrix, regardless of
whether force/virial observations exist. The basis matrix needs derivatives for all
basis functions to construct the linear system.

**What Works**:
- ✅ Package loading and compilation
- ✅ Model construction (unfitted parameters)
- ✅ Energy prediction (returns reference energies)

**What Doesn't Work**:
- ❌ Model fitting (all modes: energy-only, energy+forces, etc.)
- ❌ Any workflow requiring `evaluate_basis_ed`

**Conclusion**: Option 3 enables testing package compilation but **cannot validate
migration correctness** through fitting/comparison with baseline.

**See**: `OPTION3_FINDINGS.md` for detailed analysis

## Recommendations

1. **Immediate**: Proceed with **Option 2** to enable fitting and full testing
2. **Medium-term**: Option 2 sufficient for migration validation
3. **Long-term (for optimization)**: Option 1 for production performance

## Testing Status

### Package Compilation ✅
```bash
julia +1.11 --project=. -e 'using ACEpotentials'
# SUCCESS: Package precompiled successfully
```

### Unit Tests ❌ BLOCKED
```bash
julia +1.11 --project=. test/test_fast.jl
# ERROR: type SparseSymmProd has no field num1
```

**Test Progress**:
- ✅ Package loads
- ✅ Model construction starts
- ✅ Basis assembly begins
- ❌ Force/virial computation fails (_pfwd error)

## Files Modified

### Code Changes
- `src/models/models.jl` - Added EquivariantTensors import, fixed LuxCore API
- `src/models/Rnl_basis.jl` - Fixed LuxCore API
- `src/models/ace.jl` - Fixed symmetrisation_matrix, LuxCore, metadata, type renames
- `src/models/sparse.jl` - Fixed type renames, removed projection, metadata
- `src/models/fasteval.jl` - Fixed type renames, metadata, removed projection
- `Project.toml` - Updated all dependency versions

### Documentation
- `CLAUDE.md` - Added migration context
- `BASELINE_TEST_RESULTS.md` - Documented `main` branch baseline
- `MIGRATION_STATUS.md` - This document
- `MIGRATION_TESTING.md` - Testing guide

## Next Steps

1. **Immediate**: Choose solution strategy (Option 1, 2, or 3)
2. **Short-term**: Implement chosen solution
3. **Medium-term**: Test with full test suite
4. **Long-term**: Compare with baseline and validate numerical equivalence

## References

- EquivariantTensors.jl v0.3: `/tmp/EquivariantTensors.jl/`
- SparseSymmProd source: `/tmp/EquivariantTensors.jl/src/ace/sparsesymmprod.jl`
- Baseline tests: `BASELINE_TEST_RESULTS.md`
- Migration testing: `MIGRATION_TESTING.md`

---

**Last Updated**: 2025-11-12
**Current Blocker**: Custom _pfwd functions need rewrite for SparseSymmProd API
**Recommended Action**: Choose solution strategy and implement
