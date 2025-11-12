# CI Failure Diagnosis and Fix Plan

## Problem Statement

CI is running on `main` but failing on the PR branch (`claude/explore-repo-structure-011CV4PfXPf4MHS4ceHdoMQe`).

**Branch**: `claude/explore-repo-structure-011CV4PfXPf4MHS4ceHdoMQe`
**PR**: https://github.com/jameskermode/ACEpotentials.jl/pull/1
**Last Commit**: `5b486b5`

---

## Initial Investigation

### CI Status from Actions Page
- **Run #2** (PR branch): Completed in 41 seconds - Status unclear
- **Run #1** (main branch): In progress

**Note**: 41 seconds is suspiciously fast for a Julia test suite, suggesting early failure (likely during package instantiation or precompilation).

---

## Likely Root Causes

Based on the migration changes, here are the most probable failure points:

### 1. EquivariantTensors.jl Dependency Issues ⭐ **MOST LIKELY**

**Problem**: EquivariantTensors v0.3 might not be registered properly or has compatibility issues.

**Evidence**:
- Added `EquivariantTensors = "0.3"` to `Project.toml`
- EquivariantTensors is marked as "work in progress"
- Custom ACEregistry is used for ACEsuit packages

**Symptoms**:
- Package instantiation failure
- "Package not found" error
- Dependency resolution conflicts

**Fix Strategy**:
```julia
# Check if EquivariantTensors is in ACEregistry
# May need to adjust version or use different source
```

### 2. Missing Imports/Exports

**Problem**: `symmetrisation_matrix` might not be exported by EquivariantTensors.

**Current Code** (`src/models/ace.jl:109-112`):
```julia
AA2BB_map, _ = EquivariantTensors.symmetrisation_matrix(0, AA_spec;
                                                         prune = true,
                                                         PI = true,
                                                         basis = real)
```

**Potential Issues**:
- Function not exported (need `import EquivariantTensors: symmetrisation_matrix`)
- Function in submodule (need `EquivariantTensors.Utils.symmetrisation_matrix`)
- Different function name or signature

**Fix Strategy**:
- Check EquivariantTensors exports
- Use explicit import
- Verify function location in module hierarchy

### 3. `real` and `complex` Basis Constants

**Problem**: `real` used as basis parameter might not be defined/available.

**Current Code**:
```julia
basis = real  # Line 112 in ace.jl
```

**Investigation Needed**:
- In Julia, `real` is a built-in function
- EquivariantTensors uses `typeof(real)` for type signatures
- Should work but verify in context

### 4. API Signature Mismatch

**Problem**: `symmetrisation_matrix` might have different parameters than expected.

**Our Call**:
```julia
symmetrisation_matrix(0, AA_spec; prune=true, PI=true, basis=real)
```

**Need to Verify**:
- Parameter names match
- Parameter types are correct
- Return type is tuple `(matrix, spec)` as assumed

### 5. Test File Issues

**Problem**: Our new `test/test_migration.jl` might have errors.

**Potential Issues**:
- Import errors
- Missing dependencies in test environment
- Incorrect function calls
- Type mismatches

---

## Diagnostic Steps

### Step 1: Check Dependency Resolution

```bash
# Locally test package instantiation
julia --project=. -e '
using Pkg
Pkg.instantiate()
'
```

**Expected Outcomes**:
- ✅ Success: Dependencies resolve correctly
- ❌ Failure: EquivariantTensors not found or incompatible

### Step 2: Test Import

```bash
julia --project=. -e '
using ACEpotentials
import EquivariantTensors
println("Import successful")
println("EquivariantTensors version: ", pkgversion(EquivariantTensors))
'
```

**Expected Outcomes**:
- ✅ Success: Both packages import
- ❌ Failure: Import error with message

### Step 3: Test Function Availability

```bash
julia --project=. -e '
import EquivariantTensors
# Check if function exists
if isdefined(EquivariantTensors, :symmetrisation_matrix)
    println("✅ symmetrisation_matrix is available")
else
    println("❌ symmetrisation_matrix not found")
    println("Available: ", names(EquivariantTensors))
end
'
```

### Step 4: Test Function Call

```bash
julia --project=. -e '
import EquivariantTensors

# Simple test case
AA_spec = [
    [(n=1, l=0, m=0)],
]

try
    result = EquivariantTensors.symmetrisation_matrix(0, AA_spec;
                                                      prune=true,
                                                      PI=true,
                                                      basis=real)
    println("✅ Function call successful")
    println("Result type: ", typeof(result))
catch e
    println("❌ Function call failed")
    println("Error: ", e)
end
'
```

### Step 5: Run Migration Tests

```bash
julia --project=. test/test_migration.jl
```

### Step 6: Run Full Test Suite

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

---

## Potential Fixes

### Fix 1: Adjust EquivariantTensors Version

If v0.3 is not available or incompatible:

```toml
# Project.toml
[deps]
EquivariantTensors = "5e107534-7145-4f8f-b06f-47a52840c895"

[compat]
EquivariantTensors = "0.1, 0.2, 0.3"  # Allow broader range
```

Or pin to specific commit:
```toml
[deps]
EquivariantTensors = "5e107534-7145-4f8f-b06f-47a52840c895"

# In Manifest.toml (after Pkg.add with specific rev)
```

### Fix 2: Explicit Import

If function is not exported:

```julia
# src/models/ace.jl
import EquivariantTensors
import EquivariantTensors: symmetrisation_matrix  # Explicit import

# Or if in submodule:
import EquivariantTensors.Utils: symmetrisation_matrix
```

### Fix 3: Check ACEregistry

EquivariantTensors might need to be registered in ACEregistry:

```bash
# In CI workflow (.github/workflows/CI.yml line 47-49)
julia -e '
using Pkg
Pkg.Registry.add("https://github.com/ACEsuit/ACEregistry")
Pkg.Registry.add(General)  # Ensure General registry too
'
```

### Fix 4: Alternative Import Pattern

If module structure is different:

```julia
# Try different access patterns
using EquivariantTensors
# vs
import EquivariantTensors
# vs
using EquivariantTensors: symmetrisation_matrix
```

### Fix 5: Fallback to EquivariantModels

Temporary workaround while debugging:

```julia
# src/models/ace.jl
const USE_EQUIVARIANT_TENSORS = false  # Toggle for testing

if USE_EQUIVARIANT_TENSORS
    import EquivariantTensors
    AA2BB_map, _ = EquivariantTensors.symmetrisation_matrix(...)
else
    import EquivariantModels
    AA2BB_map = EquivariantModels._rpi_A2B_matrix(...)
end
```

---

## Action Plan

### Phase 1: Gather Information (If CI logs available)

1. **Access CI logs** from failed run
2. **Identify exact error message**
3. **Determine failure point** (instantiate, precompile, test)
4. **Note any stack traces**

### Phase 2: Reproduce Locally

1. **Clone fork** and checkout PR branch
2. **Run diagnostic steps** 1-6 above
3. **Document errors** and outcomes
4. **Identify root cause**

### Phase 3: Implement Fix

Based on diagnostic results:

**If dependency issue**:
- Adjust version constraints
- Check registry availability
- Consider using dev/add with URL

**If import issue**:
- Add explicit imports
- Fix module paths
- Update function calls

**If API issue**:
- Check EquivariantTensors source
- Adjust function signatures
- Update parameter names

**If test issue**:
- Fix test file errors
- Add missing test dependencies
- Update test assertions

### Phase 4: Verify Fix

1. **Run tests locally** until passing
2. **Commit fixes** with descriptive message
3. **Push to PR branch**
4. **Wait for CI** to re-run
5. **Monitor results**

### Phase 5: Document

1. **Update MIGRATION_README.md** with any gotchas
2. **Add to CI_INVESTIGATION.md** if CI-specific
3. **Note in commit message** what was fixed

---

## Quick Fix Attempts (Without Full Diagnosis)

If unable to access detailed CI logs, try these common fixes:

### Quick Fix 1: Broaden Version Constraints

```julia
# Project.toml - line 60
EquivariantTensors = "0.1, 0.2, 0.3"  # Instead of just "0.3"
```

### Quick Fix 2: Add Development Dependency

```julia
# If not in registry, use direct URL
julia --project=. -e '
using Pkg
Pkg.develop(url="https://github.com/ACEsuit/EquivariantTensors.jl")
'
```

### Quick Fix 3: Ensure Test Dependencies

```toml
# Project.toml
[extras]
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
EquivariantModels = "73ee3e68-46fd-466f-9c56-451dc0291ebc"  # For comparison tests

[targets]
test = ["Test", "EquivariantModels"]
```

### Quick Fix 4: Update CI Workflow

```yaml
# .github/workflows/CI.yml - ensure ACEregistry is added
- run: |
    using Pkg
    Pkg.Registry.add(General)  # Add this line
    Pkg.pkg"registry add https://github.com/ACEsuit/ACEregistry"
  shell: bash -c "julia --color=yes {0}"
```

---

## Expected CI Failure Patterns

### Pattern 1: Dependency Resolution Failure

```
ERROR: Unsatisfiable requirements detected for package EquivariantTensors
```

**Fix**: Adjust version constraints or add to registry

### Pattern 2: Package Not Found

```
ERROR: The following package names could not be resolved:
 * EquivariantTensors (5e107534-7145-4f8f-b06f-47a52840c895)
```

**Fix**: Ensure package is registered in ACEregistry or use URL

### Pattern 3: Import Error

```
ERROR: LoadError: UndefVarError: symmetrisation_matrix not defined
```

**Fix**: Add explicit import or fix module path

### Pattern 4: Type/Signature Error

```
ERROR: MethodError: no method matching symmetrisation_matrix(::Int64, ::Vector{...}; prune=true, PI=true, basis=typeof(real))
```

**Fix**: Adjust function call parameters or signature

### Pattern 5: Test Failure

```
Test Failed at /home/runner/work/.../test/test_migration.jl:XX
  Expression: ...
```

**Fix**: Update test assertions or fix test logic

---

## Monitoring and Next Steps

### After Implementing Fixes

1. **Commit changes** with clear message
2. **Push to branch** (triggers CI if Actions enabled)
3. **Monitor CI progress** (should take 10-30 minutes)
4. **Check logs** for any remaining errors
5. **Iterate** if needed

### Success Criteria

- ✅ Package instantiation succeeds
- ✅ All imports work
- ✅ Migration tests pass
- ✅ Full test suite passes
- ✅ No errors or warnings

### If Still Failing

1. **Get CI logs** from repository owner
2. **Share logs** for analysis
3. **Consider alternative approaches**:
   - Staged migration (keep both backends temporarily)
   - Feature flag to toggle backends
   - Upstream coordination

---

## Summary

**Most Likely Issue**: EquivariantTensors v0.3 dependency resolution or availability in ACEregistry

**Primary Diagnostic**: Run `julia --project=. -e 'using Pkg; Pkg.instantiate()'` locally

**Quick Fix**: Broaden version constraints or use direct URL for EquivariantTensors

**Full Fix**: Requires access to CI logs to see exact error message

**Fallback**: Keep EquivariantModels as backup while debugging

---

**Document Created**: 2025-11-12
**Status**: Diagnostic plan ready, awaiting CI logs or local testing
**Next Action**: Access CI logs or test locally to identify exact failure
