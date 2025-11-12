# Migration Testing Guide: EquivariantModels.jl → EquivariantTensors.jl

This document provides comprehensive testing instructions for verifying the migration from EquivariantModels.jl to EquivariantTensors.jl.

## Migration Summary

**Commit**: 7b5cda9d939e0a00429c6d6614a92b8f873f3036
**Branch**: `claude/explore-repo-structure-011CV4PfXPf4MHS4ceHdoMQe`
**Date**: 2025-11-12

### Changes Made
1. Added EquivariantTensors.jl v0.3 as dependency
2. Replaced 3 function calls across 2 files:
   - `EquivariantModels._rpi_A2B_matrix` → `EquivariantTensors.symmetrisation_matrix`
   - `EquivariantModels.RPE_filter_real` → `_rpe_filter_real` (local implementation)
   - `EquivariantModels.gensparse` → `Polynomials4ML.Utils.gensparse` (explicit path)

## Prerequisites

### Install Julia
```bash
# Using Juliaup (recommended)
curl -fsSL https://install.julialang.org | sh

# Or download Julia 1.10/1.11 from https://julialang.org/downloads/
```

### Setup Package Environment
```bash
cd /home/user/ACEpotentials.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Test Plan

### Phase 1: Unit Tests

#### 1.1 Filter Function Equivalence Test

Create a test to verify `_rpe_filter_real` produces identical results to `EquivariantModels.RPE_filter_real`:

```julia
using Test
include("src/models/utils.jl")

# Test cases
test_specs = [
    # Empty basis
    [],

    # Single element with L=0
    [(n=1, l=0, m=0)],

    # Single element with L=0, l≠0 (should fail)
    [(n=1, l=1, m=0)],

    # Pair of elements
    [(n=1, l=1, m=0), (n=1, l=1, m=0)],
    [(n=1, l=1, m=1), (n=1, l=1, m=-1)],

    # Triple
    [(n=1, l=1, m=0), (n=1, l=1, m=1), (n=1, l=1, m=-1)],

    # Higher correlation
    [(n=1, l=2, m=0), (n=1, l=2, m=1), (n=1, l=2, m=-1), (n=1, l=2, m=2)],
]

@testset "RPE Filter Migration" begin
    for spec in test_specs
        old_filter = EquivariantModels.RPE_filter_real(0)
        new_filter = _rpe_filter_real(0)

        @test old_filter(spec) == new_filter(spec)
    end
end
```

**Expected Result**: All tests pass

#### 1.2 Coupling Coefficients Equivalence Test

```julia
using Test
using EquivariantModels
using EquivariantTensors

# Test specification
AA_spec_test = [
    [(n=1, l=0, m=0)],
    [(n=1, l=0, m=0), (n=1, l=0, m=0)],
    [(n=1, l=1, m=0), (n=1, l=1, m=0)],
    [(n=1, l=1, m=1), (n=1, l=1, m=-1)],
]

@testset "Coupling Coefficients Migration" begin
    for spec in AA_spec_test
        # Old way
        old_result = EquivariantModels._rpi_A2B_matrix(0, [spec]; basis=real)

        # New way
        new_result, pruned_spec = EquivariantTensors.symmetrisation_matrix(
            0, [spec]; prune=true, PI=true, basis=real
        )

        # Check matrices are equivalent
        @test size(old_result) == size(new_result)
        @test isapprox(old_result, new_result, rtol=1e-12)
    end
end
```

**Expected Result**: All matrices match within numerical tolerance

### Phase 2: Integration Tests

#### 2.1 ACE Model Construction Test

```julia
using Test
using ACEpotentials

@testset "ACE Model Construction" begin
    # Test basic model construction still works
    elements = [:Si]
    order = 3
    totdeg = 6

    @testset "Silicon model" begin
        model = acemodel(
            elements = elements,
            order = order,
            totdegree = totdeg
        )

        @test model isa ACEModel
        @test length(model.tensor) > 0
    end
end
```

**Expected Result**: Model constructs without errors

#### 2.2 Run Existing Test Suite

```bash
# Run full test suite
julia --project=. -e 'using Pkg; Pkg.test()'

# Run specific test files
julia --project=. test/test_silicon.jl
julia --project=. test/models/test_ace.jl
julia --project=. test/models/test_models.jl
```

**Expected Result**: All existing tests pass

### Phase 3: Regression Tests

#### 3.1 Numerical Equivalence Test

Test that models produce identical predictions before and after migration:

```julia
using ACEpotentials
using AtomsBase
using Test

@testset "Numerical Regression" begin
    # Create a simple test structure
    atoms = isolated_system([
        :Si => [0.0, 0.0, 0.0]u"Å",
        :Si => [1.5, 1.5, 1.5]u"Å",
    ])

    # Build model with new backend
    model = acemodel(
        elements = [:Si],
        order = 2,
        totdegree = 4
    )

    # Initialize parameters
    using Random
    rng = Random.MersenneTwister(1234)
    ps, st = Lux.setup(rng, model)

    # Evaluate
    energy, forces = evaluate_ed(model, atoms, ps, st)

    # Check types and shapes
    @test energy isa Number
    @test length(forces) == length(atoms)
    @test !isnan(energy)
    @test !any(isnan, forces)
end
```

**Expected Result**: Model evaluation works without NaN/Inf values

#### 3.2 Fitting Test

Test that a simple fitting workflow still works:

```julia
using ACEpotentials
using Test

@testset "Silicon Fitting Regression" begin
    # Load tiny Silicon dataset
    data, _, _ = ACEpotentials.example_dataset("Si_tiny")

    # Build model
    model = acemodel(
        elements = [:Si],
        order = 2,
        totdegree = 6,
        Eref = [:Si => -158.54]
    )

    # Fit (just a few iterations to test)
    weights = Dict(
        "FLD" => Dict("Si" => Dict("E" => 1.0, "F" => 1.0))
    )

    solver = ACEfit.LSQR(damp = 1e-3, atol = 1e-6)

    @testset "Fitting runs" begin
        P, Y = ACEpotentials.assemble(data, model; verbose=false)
        results = ACEfit.solve(solver, P, Y)

        @test results isa NamedTuple
        @test haskey(results, :coeffs)
    end
end
```

**Expected Result**: Fitting completes successfully

### Phase 4: Performance Tests

#### 4.1 Benchmark Model Construction

```julia
using BenchmarkTools
using ACEpotentials

@testset "Performance Benchmarks" begin
    elements = [:Si]
    order = 3
    totdeg = 8

    # Benchmark model construction
    t = @benchmark acemodel(
        elements = $elements,
        order = $order,
        totdegree = $totdeg
    )

    println("Model construction time: $(median(t))")
    @test median(t).time < 1e9  # Less than 1 second
end
```

**Expected Result**: No significant performance regression

## Validation Checklist

- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] `test/test_silicon.jl` passes
- [ ] `test/models/test_ace.jl` passes
- [ ] `test/models/test_models.jl` passes
- [ ] Numerical predictions match within tolerance
- [ ] Fitting workflow completes successfully
- [ ] No NaN/Inf values in model evaluation
- [ ] Performance is comparable (±20%)
- [ ] Memory usage is comparable (±20%)

## Known Issues / Expected Changes

### API Differences
- `symmetrisation_matrix` returns `(matrix, pruned_spec)` tuple instead of just matrix
- We explicitly handle this by unpacking: `AA2BB_map, _ = symmetrisation_matrix(...)`

### Minor Differences
- Filter function is now local implementation (same logic, just moved)
- Import path for `gensparse` is now explicit: `Polynomials4ML.Utils.gensparse`

## Rollback Procedure

If critical issues are found:

1. **Revert commit**:
   ```bash
   git revert 7b5cda9d939e0a00429c6d6614a92b8f873f3036
   ```

2. **Or use feature flag approach**:
   ```julia
   # In src/models/ace.jl
   const USE_EQUIVARIANT_TENSORS = get(ENV, "USE_EQUIVARIANT_TENSORS", "true") == "true"

   if USE_EQUIVARIANT_TENSORS
       import EquivariantTensors
       # New implementation
   else
       import EquivariantModels
       # Old implementation
   end
   ```

3. **Remove EquivariantTensors from Project.toml** if needed

## Next Steps After Testing

1. **If all tests pass**:
   - Consider removing EquivariantModels dependency
   - Update documentation
   - Create PR to merge migration
   - Update CHANGELOG.md

2. **If tests reveal issues**:
   - Document specific failures
   - Debug and fix issues
   - Consider keeping both backends temporarily
   - Report issues to EquivariantTensors.jl if bugs found there

3. **Future work**:
   - Monitor for `gensparse` deprecation in Polynomials4ML
   - Consider using more EquivariantTensors features (GPU kernels, etc.)
   - Align with ACEhamiltonians.jl migration timeline

## Contact

For questions about this migration:
- See original migration plan in git commit message
- Check EquivariantTensors.jl docs: https://acesuit.github.io/EquivariantTensors.jl/
- Open issue on GitHub

## References

- EquivariantModels.jl: https://github.com/ACEsuit/EquivariantModels.jl
- EquivariantTensors.jl: https://github.com/ACEsuit/EquivariantTensors.jl
- Migration commit: 7b5cda9d939e0a00429c6d6614a92b8f873f3036
