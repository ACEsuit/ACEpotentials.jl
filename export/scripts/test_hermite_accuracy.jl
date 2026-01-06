#!/usr/bin/env julia
# Test Hermite spline accuracy vs original P4ML splines

using ACEpotentials
using ACEpotentials.Models
using ACEpotentials.ETModels
using ACEpotentials.ETModels: splinify, ETACE
using Lux
using LuxCore
using Random
using StaticArrays
using Statistics
using Printf

# Load the extraction and code generation functions
include("../src/splinify.jl")
include("../src/codegen.jl")

const M = ACEpotentials.Models
const ETM = ACEpotentials.ETModels

println("="^80)
println("Hermite Spline Accuracy Test")
println("="^80)

# Create and splinify test model
println("\n[1] Creating and splinifying ETACE model...")
elements = (:Ti, :Al)
order = 2
totaldegree = 6
rcut = 5.5
maxl = 1

rin0cuts = M._default_rin0cuts(elements)
rin0cuts = (x -> (rin = x.rin, r0 = x.r0, rcut = rcut)).(rin0cuts)

rng = Random.MersenneTwister(1234)

ace_model = M.ace_model(;
    elements = elements,
    order = order,
    Ytype = :solid,
    level = M.TotalDegree(),
    max_level = totaldegree,
    maxl = maxl,
    pair_maxn = totaldegree,
    rin0cuts = rin0cuts,
    init_WB = :glorot_normal,
    init_Wpair = :glorot_normal
)

ps, st = Lux.setup(rng, ace_model)

# Convert to ETACE
etace = ETM.convert2et(ace_model)
et_ps, et_st = LuxCore.setup(MersenneTwister(1234), etace)

# Copy parameters
n_species = length(elements)
for iz in 1:n_species
    for jz in 1:n_species
        et_ps.rembed.post.W[:, :, (iz-1)*n_species + jz] .= ps.rbasis.Wnlq[:, :, iz, jz]
    end
end

for iz in 1:n_species
    et_ps.readout.W[1, :, iz] .= ps.WB[:, iz]
end

# Splinify
etace_splined = splinify(etace, et_ps, et_st; Nspl=50)
println("   ✓ Model splinified with 50 knots")

# Extract Hermite data and generate code
println("\n[2] Extracting Hermite spline data and generating code...")
hermite_data = extract_hermite_spline_data(etace_splined, et_ps, et_st, rcut)
code = generate_hermite_spline_code(hermite_data, n_species)

# Write generated code to temp file
codefile = "/tmp/hermite_test_generated.jl"
open(codefile, "w") do f
    write(f, code)
end
println("   ✓ Generated code written to $codefile")

# Define helper to load generated code in isolated module
println("\n[3] Loading generated code...")
# We need zz2pair_sym function for the generated code
module GeneratedHermite
    using StaticArrays

    # Species pair index mapping (symmetric)
    function zz2pair_sym(iz::Int, jz::Int)
        # For 2 species: (1,1)→1, (1,2)→2, (2,1)→2, (2,2)→3
        # But our code uses asymmetric indexing: (iz-1)*NZ + jz
        # Actually, looking at the generated code, it expects pair_idx from selector
        # Let's use asymmetric indexing to match extraction
        NZ = 2
        return (iz - 1) * NZ + jz
    end

    include("/tmp/hermite_test_generated.jl")
end
println("   ✓ Generated code loaded successfully")

# Test accuracy for each species pair
println("\n[4] Testing accuracy...")
n_test_points = 1000
Random.seed!(42)

# Import EquivariantTensors at top level
using EquivariantTensors
using AtomsBase: ChemicalSpecies
using DecoratedParticles
const ET = EquivariantTensors

global total_max_err = 0.0
global total_rmse = 0.0
global n_total_tests = 0

for pair_idx in 1:4
    # Map pair_idx to (iz, jz)
    iz = (pair_idx - 1) ÷ n_species + 1
    jz = (pair_idx - 1) % n_species + 1

    species_names = [(1,1)=>"Ti-Ti", (1,2)=>"Ti-Al", (2,1)=>"Al-Ti", (2,2)=>"Al-Al"]
    pair_name = species_names[iz, jz]

    println("\n   Testing $pair_name (pair $pair_idx):")

    # Get parameter range for this pair
    data = hermite_data[pair_idx]
    p = data.agnesi_params
    r_min = Float64(p.rin) + 0.001
    r_max = rcut - 0.001

    # Generate random test points
    r_test = r_min .+ (r_max - r_min) .* rand(n_test_points)

    # Evaluate with original splined model
    # We need to construct edge data and call the model
    zlist = etace_splined.rembed.layer.trans.refstate.zlist
    zi_chem = ChemicalSpecies(zlist[iz])
    zj_chem = ChemicalSpecies(zlist[jz])

    original_values = zeros(data.n_rnl, n_test_points)

    for (k, r) in enumerate(r_test)
        # Create edge state - XState is just a NamedTuple with specific fields
        Rij = SVector(r, 0.0, 0.0)
        # The actual signature from DecoratedParticles needs specific fields
        # Let me try wrapping in a StructArray or similar
        # Actually, looking at DecoratedParticles, X = (𝐫=..., ...) should work
        edge_state = DecoratedParticles.state(; 𝐫 = Rij, zi = zi_chem, zj = zj_chem)

        # Evaluate radial basis - TransSelSplines expects a vector of XStates
        rbasis_out, _ = etace_splined.rembed.layer([edge_state], et_ps.rembed, et_st.rembed)

        # Output is (n_edges, n_rnl), we want the single edge
        original_values[:, k] = rbasis_out[1, :]
    end

    # Evaluate with generated Hermite code
    generated_values = zeros(data.n_rnl, n_test_points)

    for (k, r) in enumerate(r_test)
        generated_values[:, k] = GeneratedHermite.evaluate_Rnl(r, iz, jz)
    end

    # Compute errors
    abs_errors = abs.(generated_values .- original_values)
    max_err = maximum(abs_errors)
    rmse = sqrt(mean(abs_errors.^2))
    mean_err = mean(abs_errors)

    println("     Max absolute error:  $(@sprintf("%.3e", max_err))")
    println("     RMSE:                $(@sprintf("%.3e", rmse))")
    println("     Mean absolute error: $(@sprintf("%.3e", mean_err))")

    # Check relative errors for non-zero values
    nonzero_mask = abs.(original_values) .> 1e-10
    if any(nonzero_mask)
        rel_errors = abs_errors[nonzero_mask] ./ abs.(original_values[nonzero_mask])
        max_rel_err = maximum(rel_errors)
        mean_rel_err = mean(rel_errors)
        println("     Max relative error:  $(@sprintf("%.3e", max_rel_err))")
        println("     Mean relative error: $(@sprintf("%.3e", mean_rel_err))")
    end

    global total_max_err = max(total_max_err, max_err)
    global total_rmse += rmse
    global n_total_tests += 1

    # Check if errors are acceptably small
    if max_err < 1e-6
        println("     ✓ PASS: Accuracy excellent (< 1e-6)")
    elseif max_err < 1e-4
        println("     ✓ PASS: Accuracy good (< 1e-4)")
    else
        println("     ⚠ WARNING: Larger errors than expected")
    end
end

avg_rmse = total_rmse / n_total_tests

println("\n" * "="^80)
println("Accuracy Test Summary")
println("="^80)
println("Test points per pair: $n_test_points")
println("Total species pairs tested: 4")
println("Overall max error: $(@sprintf("%.3e", total_max_err))")
println("Average RMSE: $(@sprintf("%.3e", avg_rmse))")

if total_max_err < 1e-6
    println("\n✓ EXCELLENT: Hermite splines reproduce P4ML with machine precision!")
elseif total_max_err < 1e-4
    println("\n✓ GOOD: Hermite splines match P4ML to high accuracy")
else
    println("\n⚠ WARNING: Accuracy lower than expected - investigate")
end

println("\n" * "="^80)
