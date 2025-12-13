#!/usr/bin/env julia
#
# Benchmark script for evaluate_basis_ed
# Compares current Zygote pullback approach vs ForwardDiff.jacobian
#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ACEpotentials, AtomsBase, Unitful, BenchmarkTools, Random
using Lux, StaticArrays
using AtomsBuilder

# Access functions from internal Models module
const M = ACEpotentials.Models
const build_et_calculator = M.build_et_calculator
const evaluate_basis = M.evaluate_basis
const evaluate_basis_ed = M.evaluate_basis_ed
const length_basis = M.length_basis

# Build a small test system (Si crystal)
function make_si_system(supercell=(2,1,1))
    sys = AtomsBuilder.bulk(:Si) * supercell
    AtomsBuilder.rattle!(sys, 0.1u"Å")
    return sys
end

# Build model and calculator (following test_et_calculator.jl pattern)
println("Building ACE model...")

elements = (:Si,)
level = M.TotalDegree()
max_level = 8
order = 2
maxl = 4

rin0cuts = M._default_rin0cuts(elements)
rin0cuts = (x -> (rin = x.rin, r0 = x.r0, rcut = 5.5)).(rin0cuts)

model = M.ace_model(; elements = elements, order = order,
            Ytype = :solid, level = level, max_level = max_level,
            maxl = maxl, pair_maxn = max_level,
            rin0cuts = rin0cuts,
            init_WB = :glorot_normal, init_Wpair = :glorot_normal)

rng = Random.MersenneTwister(1234)
ps, st = Lux.setup(rng, model)

# Zero out pair basis (not implemented in ET backend)
for s in model.pairbasis.splines
   s.itp.itp.coefs[:] *= 0
end

println("Building ET calculator...")
calc = build_et_calculator(model, ps, st)

# Test with different system sizes
for supercell in [(2,1,1)]  # Can add larger systems
    sys = make_si_system(supercell)
    n_atoms = length(sys)

    println("\n" * "="^60)
    println("System: $(n_atoms) atoms (supercell $(supercell))")
    println("Basis length per species: $(calc.len_Bi)")
    println("Total basis length: $(length_basis(calc))")
    println("="^60)

    # Warmup
    println("\nWarmup...")
    B = evaluate_basis(calc, sys)
    println("B shape: $(size(B))")

    # Benchmark evaluate_basis (forward only)
    println("\n--- evaluate_basis (forward only) ---")
    b1 = @benchmark evaluate_basis($calc, $sys) samples=10 evals=1
    display(b1)

    # Benchmark evaluate_basis_ed (forward + Jacobian)
    println("\n--- evaluate_basis_ed (current: Zygote pullback) ---")
    println("This may take a while for larger systems...")

    # First just time it once to see how long it takes
    t0 = time()
    B_ed, dB = evaluate_basis_ed(calc, sys)
    t1 = time()
    println("Single call time: $(round(t1-t0, digits=2)) seconds")
    println("B_ed shape: $(size(B_ed)), dB shape: $(size(dB))")

    # Only do proper benchmark if it's not too slow
    if t1 - t0 < 30.0
        b2 = @benchmark evaluate_basis_ed($calc, $sys) samples=5 evals=1
        display(b2)
    else
        println("Skipping full benchmark - single call took > 30 seconds")
    end

    # Calculate theoretical efficiency
    n_outputs = n_atoms * calc.len_Bi
    println("\n--- Analysis ---")
    println("Number of outputs (n_atoms × len_Bi): $n_outputs")
    println("With Zygote pullback: $n_outputs backward passes")
    println("With ForwardDiff: ~$(3 * n_atoms) forward passes (rough estimate)")
    println("Theoretical speedup: $(round(n_outputs / (3 * n_atoms), digits=1))×")
end

println("\n\nBenchmark complete.")
