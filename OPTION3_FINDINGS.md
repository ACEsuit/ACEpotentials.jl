# Option 3 Findings: Disabling evaluate_basis_ed

**Date**: 2025-11-12
**Status**: ✅ Implemented, ⚠️ Limited Functionality

## Summary

Option 3 (temporarily disabling `evaluate_basis_ed`) has been successfully implemented. However, testing reveals significant limitations: **fitting is not possible**, only inference with pre-fitted models.

## Implementation

### Changes Made

**File**: `src/models/ace.jl:659-683`

Replaced the `evaluate_basis_ed` function body with:
```julia
function evaluate_basis_ed(model::ACEModel, ...)
   error("""
   evaluate_basis_ed is temporarily disabled during EquivariantTensors v0.3 migration.

   This function requires custom pushforward (_pfwd) implementations that need
   rewriting for the new SparseSymmProd API.

   Impact: Cannot compute forces or virials
   Available: Energy-only predictions still work

   For solutions, see MIGRATION_STATUS.md
   """)
end
```

## Testing Results

### ✅ What Works

1. **Package Loading**
   ```bash
   julia +1.11 --project=. -e 'using ACEpotentials'
   # SUCCESS: Package compiles and loads
   ```

2. **Model Construction**
   ```julia
   model = ace1_model(elements = [:Si], ...)
   # SUCCESS: Model created with unfitted parameters
   ```

3. **Energy Prediction (Unfitted Model)**
   ```julia
   E = potential_energy(atoms, model)
   # SUCCESS: Returns reference energy (-158.54 eV for Si)
   ```

4. **Force Prediction Returns Zeros**
   ```julia
   F = forces(atoms, model)
   # SUCCESS: Returns zero forces (unfitted model)
   # Does NOT throw error as expected (uses fallback path)
   ```

### ❌ What Doesn't Work

1. **Model Fitting**
   ```julia
   acefit!(data, model; ...)
   # FAILS: evaluate_basis_ed is temporarily disabled
   ```

   **Root Cause**: The fitting process calls `energy_forces_virial_basis`
   (src/models/calculators.jl:309) to assemble the feature matrix. This function
   **always** calls `evaluate_basis_ed`, even for energy-only data.

2. **Why Energy-Only Fitting Fails**

   Even with:
   - ✅ Non-existent force/virial keys → 0 force/virial observations
   - ✅ `smoothness = 0` → No smoothness regularization
   - ✅ Zero weights on forces/virials

   The assembly still fails because:
   ```julia
   # In energy_forces_virial_basis (calculators.jl:309)
   for i in domain
      v, dv = evaluate_basis_ed(calc.model, Rs, Zs, z0, ps, st)
      # ^^^ Always called to build the full basis matrix
      #     (energies AND force/virial derivatives for each basis function)
   end
   ```

   The basis matrix includes:
   - Energy basis: E[k] for each basis function k
   - Force basis: F[atom, k] for each atom and basis function
   - Virial basis: V[k] for each basis function

   Even if we have zero force/virial **observations**, the assembly needs
   force/virial **basis derivatives** to build the feature matrix.

## Implications

### What Option 3 Enables

✅ **Testing migration without fitting**:
- Verify package compiles
- Test model construction
- Test inference with pre-fitted models (if we had them)

### What Option 3 Does NOT Enable

❌ **Anything requiring derivatives**:
- Model fitting (even energy-only)
- Force predictions (returns zeros, but should error)
- Virial predictions
- Any workflow requiring `evaluate_basis_ed`

## Recommendations

### For Testing the Migration

**Option 3 is insufficient** for comprehensive testing because:
1. Cannot fit models to validate numerical correctness
2. Cannot compare energy predictions (unfitted vs baseline)
3. Cannot test the complete migration path

**Better Approach**: Implement Option 2 or Option 1

### For Production Use

Option 3 is **not suitable** for production. It blocks:
- Training new models
- Fine-tuning existing models
- Any force/virial calculations

**Required**: Implement Option 1 or Option 2

## Next Steps

### Option 2: Use Standard evaluate_ed! (Recommended Next)

**Effort**: 1-2 hours
**Benefit**: Enables fitting and full functionality

**Approach**: Replace custom `_pfwd` with standard differentiation:
1. Check if `Polynomials4ML.evaluate_ed!` works with new `SparseSymmProd`
2. Rewrite `evaluate_basis_ed` to use standard AD path
3. Test fitting and compare with baseline

**Code Location**: `src/models/ace.jl:659-683` (replace error with implementation)

### Option 1: Reimplement _pfwd (Production Solution)

**Effort**: 2-3 hours
**Benefit**: Full functionality with optimal performance

**Approach**: Rewrite custom pushforward for new API:
1. Study `SparseSymmProd` internals (`specs`, `ranges`, `hasconst`)
2. Implement `_pfwd(::SparseSymmProd, A, ∂A)` using new structure
3. Update `_pfwd(::PooledSparseProduct, ...)` if needed
4. Test and benchmark performance

**Code Location**: `src/models/sparse.jl:166-199`

## Test Script

A test script `test_energy_only.jl` has been created to verify Option 3:

```bash
julia +1.11 --project=. test_energy_only.jl
```

**Expected Results**:
- ✅ Package loads
- ✅ Model construction
- ❌ Fitting fails (evaluate_basis_ed disabled)
- ✅ Energy prediction (returns reference energy)
- ✅ Force prediction (returns zeros, no error)

## Conclusion

**Option 3 Status**: ✅ Successfully implemented, ⚠️ Very limited functionality

**Key Finding**: Disabling `evaluate_basis_ed` prevents ALL fitting workflows, not just force/virial fitting. The assembly process requires basis derivatives even for energy-only observations.

**Recommendation**: Proceed with **Option 2** (standard evaluate_ed!) as the next step to enable full testing of the migration.

---

**Last Updated**: 2025-11-12
**Blocking Issue**: Cannot fit models with Option 3
**Next Action**: Implement Option 2 or Option 1
