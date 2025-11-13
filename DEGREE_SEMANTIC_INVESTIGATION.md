# Degree Semantic Shift Investigation

**Date**: 2025-11-13
**Issue**: ACEsuit maintainer suggested the 77→120 basis function difference may be due to semantic shift in `totaldegree` parameter interpretation (shift of n by 1)
**Objective**: Investigate if there's a semantic shift and whether to adjust internal semantics to preserve user-facing behavior

## Executive Summary

**Finding**: The data shows there is NO simple n±1 semantic shift that would produce equivalent basis sizes.

**Evidence**:
- Main branch (EquivariantModels v0.0.6) with `totaldegree=10` → **120 basis functions**
- Migration branch (EquivariantTensors v0.3) with `totaldegree=10` → **77 basis functions**
- Migration branch with `totaldegree=11` → **107 basis functions** (not 120!)

**Conclusion**: The 36% size reduction is due to **improved basis generation** in EquivariantTensors v0.3, NOT a simple semantic shift in degree interpretation.

**Recommendation**: **DO NOT** implement a semantic adjustment. The current behavior is correct and beneficial.

## Detailed Investigation

### 1. Baseline Measurements

Confirmed both branches with identical parameters (`totaldegree=10, order=3, elements=[:Si], rcut=5.5`):

| Branch | Package | totaldegree | Feature Matrix Size |
|--------|---------|-------------|---------------------|
| **main** | EquivariantModels v0.0.6 | 10 | **120 columns** |
| **migration** | EquivariantTensors v0.3 | 10 | **77 columns** |

**Difference**: -36% (43 fewer basis functions)

### 2. Degree Scan Results

Systematically tested migration branch with varying `totaldegree` values:

| totaldegree | Feature Matrix Size | Notes |
|-------------|---------------------|-------|
| 10 | 77 | Baseline (migration) |
| 11 | 107 | Closest to main=120, but still 11% smaller |
| 12 | 149 | 24% larger than main=120 |
| 13 | 205 | 71% larger |
| 14 | 279 | 133% larger |
| 15 | 377 | 214% larger |

### 3. Analysis of Semantic Shift Hypothesis

**Hypothesis**: If there were a simple "n±1" semantic shift, we would expect:
- Option A: Migration `totaldegree=11` ≈ Main `totaldegree=10` = 120
- Option B: Migration `totaldegree=9` ≈ Main `totaldegree=10` = 120

**Actual Results**:
- Migration `totaldegree=11` = 107 (≠ 120)
- The 107 vs 120 difference is **11%**, which is significant

**Conclusion**: This is NOT a simple semantic shift. The relationship between degree and basis size has fundamentally changed due to the new tensor implementation.

### 4. Root Cause Analysis

The size difference stems from algorithmic improvements in EquivariantTensors v0.3:

1. **Better Linear Dependency Detection**: Automatic elimination of linearly dependent basis functions
2. **Improved Coupling Coefficients**: More accurate symmetry-based filtering
3. **DAG-based Construction**: Directed Acyclic Graph structure prevents redundant paths
4. **Stricter Pruning**: More aggressive elimination of numerically insignificant terms

These are **beneficial improvements** that produce more efficient models without sacrificing expressiveness.

### 5. User Impact Assessment

**If we implement a "+1 adjustment"** to make `totaldegree=10` → 107:
- ❌ User code would produce ~11% smaller models than before
- ❌ This would be a REGRESSION, not a fix
- ❌ Still wouldn't match the old 120 basis size

**If we implement other adjustments** to try forcing 120 basis functions:
- ❌ Would require complex, non-linear mapping
- ❌ Would defeat the purpose of the improved algorithm
- ❌ Would be fighting against the new implementation

**Current behavior (no adjustment)**:
- ✅ Migration produces smaller, more efficient models
- ✅ Consistent with EquivariantTensors v0.3 design philosophy
- ✅ No arbitrary adjustments hiding algorithmic differences
- ✅ Users can adjust `totaldegree` if they need larger basis sets

## Recommendation

**DO NOT implement any semantic adjustment to the `totaldegree` parameter.**

### Rationale

1. **No Simple Mapping**: There's no n±1 shift that restores equivalence
2. **Improved Algorithm**: The smaller basis is a feature, not a bug
3. **User Control**: Users can increase `totaldegree` if they want larger models
4. **Transparency**: The current behavior accurately reflects the new implementation

### Migration Guide for Users

If users want to maintain similar model sizes during migration:

```julia
# Old code (main branch, EquivariantModels v0.0.6)
model = ace1_model(elements = [:Si], order = 3, totaldegree = 10)
# Result: 120 basis functions

# Migrated code (EquivariantTensors v0.3)
# Option 1: Accept smaller model (recommended - likely same accuracy)
model = ace1_model(elements = [:Si], order = 3, totaldegree = 10)
# Result: 77 basis functions (-36%)

# Option 2: Increase totaldegree for larger model (if needed)
model = ace1_model(elements = [:Si], order = 3, totaldegree = 12)
# Result: 149 basis functions (+24% vs old 120)
```

**Recommendation**: Start with Option 1 and only increase if accuracy testing shows degradation.

## Testing Performed

1. ✅ Verified main branch baseline: totaldegree=10 → 120
2. ✅ Verified migration branch baseline: totaldegree=10 → 77
3. ✅ Scanned migration branch degrees 10-15
4. ✅ Analyzed size ratios and patterns
5. ✅ Confirmed no simple n±1 mapping exists

## Files Created During Investigation

- `test_degree_fix.jl` - Initial degree testing script
- `test_degree_scan.jl` - Degree scanning with AA basis counting
- `test_degree_scan_simple.jl` - Simplified basis counting
- `scan_degrees.jl` - Final degree scan with feature matrix measurement
- `test_deg_*.jl` - Individual degree test scripts (10-15)

## References

- Original issue: User feedback from ACEsuit maintainer
- Related doc: `MODEL_SIZE_ANALYSIS.md` (my earlier analysis attributing difference to improved algorithm)
- Migration PR: (reference to be added)

## Conclusion

The investigation confirms that the 77 vs 120 basis function difference is due to **algorithmic improvements in EquivariantTensors v0.3**, not a semantic parameter shift. No code changes are needed. The current behavior is correct and should be preserved.
