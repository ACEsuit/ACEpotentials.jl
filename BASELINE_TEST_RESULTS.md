# Baseline Test Results - Main Branch

**Date**: 2025-11-12
**Branch**: `main` (commit `08ca97f`)
**Julia Version**: 1.11.7
**Dependency**: `EquivariantModels = "0.0.6"`

## Environment Setup

```bash
julia +1.11 --project=. -e 'using Pkg; Pkg.instantiate()'
```

**Status**: ✅ SUCCESS
**Time**: ~7 minutes (including package downloads)

### Precompilation Notes

- **Warning**: Method overwriting between ChainRules and Lux
  - `rrule(typeof(Base.merge), ...)` definition conflict
  - Non-fatal, existing issue in upstream dependencies
  - Package still loads and functions correctly

## Test Results

### test/test_fast.jl

**Status**: ✅ PASSED
**Time**: ~2-3 minutes

#### Test Components:

1. **Silicon Model Construction and Fitting**
   - Model: Si, order=3, totaldegree=12
   - Dataset: Si_tiny_dataset (53 configs, 1052 data points)
   - Solver: BLR with L-BFGS optimization
   - **Result**: Converged successfully in 20 iterations
   - Final objective value: -194.1988

2. **Fast Evaluator Validation**
   - **Result**: Predictions identical between standard and fast evaluator
   - ✅ Site-level predictions match
   - ✅ System-level predictions match

3. **TiAl Model Test**
   - **Result**: UF_ACE format conversion successful
   - ✅ Site predictions identical
   - ✅ System predictions identical

## Summary

**Overall Status**: ✅ BASELINE ESTABLISHED

The main branch (using EquivariantModels v0.0.6) works correctly with Julia 1.11:
- Package installation: ✅ Success
- Package loading: ✅ Success (with benign warnings)
- Model construction: ✅ Success
- Model fitting: ✅ Success
- Fast evaluator: ✅ Success
- Format conversion: ✅ Success

## Next Steps

1. Switch to migration branch (`claude/explore-repo-structure-011CV4PfXPf4MHS4ceHdoMQe`)
2. Test with EquivariantTensors v0.3
3. Compare results against this baseline
4. Verify no regressions introduced by migration

---

**Testing Environment**:
- Machine: HPC cluster node
- Julia: 1.11.7 (via juliaup)
- ACEregistry: Installed and configured
- Artifacts: Downloaded successfully (Si_tiny_dataset)
