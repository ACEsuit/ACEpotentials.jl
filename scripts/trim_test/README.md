# Julia 1.12 `juliac --trim` Compilation Tests for ACE Potentials

This directory contains tests for native code compilation using Julia 1.12's `juliac --trim` feature, with the goal of creating a LAMMPS-compatible shared library for ACE potential evaluation.

## Environment

- **Julia version**: 1.12.2
- **JuliaC.jl version**: 0.2.2
- **Date**: 2024-12-09

## Test Results Summary

| Test | Package | Result | Notes |
|------|---------|--------|-------|
| hello.jl | (none) | ✅ PASS | Basic executable, ~1.6MB |
| staticarrays_test.jl | StaticArrays, LinearAlgebra | ✅ PASS | SVector operations work |
| p4ml_test.jl | Polynomials4ML | ✅ PASS | Spherical harmonics evaluation works |
| et_test.jl | EquivariantTensors | ❌ FAIL | Dynamic dispatch in model construction |
| et_hardcoded_eval.jl | EquivariantTensors | ✅ PASS | Evaluation works when structures are pre-built |
| ace_eval_forces.jl | EquivariantTensors | ✅ PASS | Energy + forces (finite diff) |
| **test_silicon_export.jl** | ACEpotentials | ✅ PASS | **Full fitted model with analytic forces** |

## Key Achievement: Analytic Forces with `--trim=safe`

We successfully implemented **analytic forces and virial stress** that compile with `--trim=safe`:

```bash
./silicon_bundle/bin/silicon
# Output:
# === ACE Potential Evaluation ===
# Number of species: 1
# Tensor length: 17
# Radial basis size: 14
# Spherical harmonics: L=2 (9 functions)
#
# Test evaluation:
#   Center species: Z=14
#   Number of neighbors: 3
#   Site energy: -237.4746776432051 eV
#
# Analytic forces:
#   F[1] = [-1034.36..., 1.09e-5, 2.25e-5]
#   F[2] = [325.94..., -927.67..., 2.00e-5]
#   F[3] = [338.01..., 481.02..., -832.03...]
#
# With virial stress:
#   Energy: -237.4746776432051 eV
#   Virial tensor: [3x3 matrix]
#
# Force verification (analytic vs finite difference):
#   Max force error: 1.46e-5
#
# Evaluation successful!
```

**Binary size**: 3.1 MB

## The Challenge: ET.pullback Returns `Any` Types

The standard `ET.pullback` function from EquivariantTensors uses dynamic allocation patterns (`zeros(...)`) that the trim compiler cannot statically analyze, resulting in 80+ verifier errors.

## The Solution: Manual Pullback Implementation

We implemented a **manual backward pass** that is fully type-stable:

1. **Typed constants** - `AABASIS_SPECS_1`, `AABASIS_SPECS_2` etc. as `Tuple` of `NTuple{N,Int}` instead of `Vector{Any}`

2. **Manual pullback functions**:
   - `pullback_abasis!`: Gradients through PooledSparseProduct (A = Rnl × Ylm)
   - `pullback_aabasis!`: Gradients through SparseSymmProd (AA = products of A)
   - `tensor_pullback!`: Full backward pass A2Bmap' → aabasis → abasis

3. **Analytic derivatives** for radial basis:
   - `transform_d_*`: Distance transform derivatives
   - `envelope_d_*`: Envelope derivatives
   - `spline_d_*`: Spline derivatives
   - `evaluate_Rnl_d`: Combined radial basis with derivatives

## Workflow

### 1. Build and Fit Model (Full Julia)

```julia
using ACEpotentials

# Create model
model = ACEpotentials.ACE1compat.ace1_model(
    elements = [:Si],
    order = 2,
    totaldegree = 6,
    rcut = 5.5
)

# Fit to data
ACEpotentials.acefit!(data, model; ...)
```

### 2. Export to Trim-Compatible Code

```julia
include("export_ace_model.jl")
export_ace_model(model, "silicon_model.jl"; splinify_first=true)
```

### 3. Compile with `--trim=safe`

```bash
mkdir -p silicon_bundle/bin
julia --project=. ~/.julia/juliaup/julia-1.12.2+0.x64.linux.gnu/share/julia/juliac/juliac.jl \
    --output-exe silicon_bundle/bin/silicon \
    --experimental --trim=safe \
    silicon_model.jl
./silicon_bundle/bin/silicon
```

## What the Export Script Generates

The `export_ace_model.jl` script extracts and hardcodes:

- **Tensor structures**: PooledSparseProduct, SparseSymmProd specs as typed tuples
- **A2B coupling matrices**: Sparse matrix data (I, J, V arrays)
- **Radial basis**: Spline coefficients as `Tuple{SVector...}` (type-stable)
- **Transforms & envelopes**: All parameters for distance transforms
- **Model weights**: WB and Wpair weights per species
- **Reference energies**: E0 values per species
- **Manual pullback**: Type-stable backward pass for analytic forces
- **Virial stress**: Computed alongside forces

## Key Type-Stability Fixes

1. **Spline coefficients** as `Tuple{SVector...}` not `Vector{Vector}`
2. **AABASIS specs** as `AABASIS_SPECS_1`, `AABASIS_SPECS_2` typed tuples per order
3. **Explicit return type annotations** on all evaluation functions
4. **Use `zero(SVector{N,T})`** not `zeros(...)`
5. **Manual pullback** instead of `ET.pullback`

## Remaining Work for LAMMPS

1. **C Interface**: Add `@ccallable` functions:
   ```julia
   @ccallable function ace_compute_energy(...)::Cdouble
   @ccallable function ace_compute_forces(...)::Cint
   ```

2. **Shared Library**: Use `--output-lib` instead of `--output-exe`

3. **Multi-element**: Export script already supports multiple species

## Important Notes on `--trim` Mode

1. **No dynamic dispatch** - All code paths must be statically resolvable
2. **Use `Core.stdout`** instead of `stdout` for printing
3. **Avoid printing complex types** - Use element access instead of `show`
4. **Construction at module load is OK** - Just not at runtime in `main()`
5. **Pre-build structures as const** - Tensor structures created once at compile time

## References

- [JuliaC.jl](https://github.com/JuliaLang/JuliaC.jl)
- [Julia 1.12 Highlights - Trim Feature](https://julialang.org/blog/2025/10/julia-1.12-highlights/)
- [PR #55047 - add --trim option](https://github.com/JuliaLang/julia/pull/55047)
