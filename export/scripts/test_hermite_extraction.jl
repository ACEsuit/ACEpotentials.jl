#!/usr/bin/env julia
# Test script for Hermite spline data extraction

using ACEpotentials
using ACEpotentials.Models
using ACEpotentials.ETModels
using ACEpotentials.ETModels: splinify, ETACE
using Lux
using LuxCore
using Random

# Load the extraction function
include("../src/splinify.jl")

const M = ACEpotentials.Models
const ETM = ACEpotentials.ETModels

println("="^80)
println("Testing Hermite Spline Extraction")
println("="^80)

# Create test model (same as investigation script)
println("\n[1] Creating minimal ETACE model...")
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
println("   ✓ ACE model created")

# Convert to ETACE
println("\n[2] Converting to ETACE...")
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

println("   ✓ ETACE model created")

# Splinify
println("\n[3] Splinifying with Nspl=50...")
etace_splined = splinify(etace, et_ps, et_st; Nspl=50)
println("   ✓ Splinified")

# Extract Hermite data
println("\n[4] Extracting Hermite spline data...")
try
    hermite_data = extract_hermite_spline_data(etace_splined, et_ps, et_st, rcut)

    println("   ✓ Extraction successful!")
    println("\n   Number of species pairs: $(length(hermite_data))")

    # Examine first pair
    pair1 = hermite_data[1]
    println("\n[5] First species pair (Ti-Ti):")
    println("   pair_idx: $(pair1.pair_idx)")
    println("   iz, jz: $(pair1.iz), $(pair1.jz)")
    println("   n_rnl: $(pair1.n_rnl)")
    println("   n_knots: $(pair1.n_knots)")
    println("   y_min, y_max: $(pair1.y_min), $(pair1.y_max)")
    println("   F size: $(size(pair1.F))")
    println("   G size: $(size(pair1.G))")
    println("   Agnesi params: $(pair1.agnesi_params)")

    println("\n   Sample F values (first 3 knots, first basis function):")
    for i in 1:min(3, pair1.n_knots)
        println("     F[$i, 1] = $(pair1.F[i, 1])")
    end

    println("\n   Sample G values (first 3 knots, first basis function):")
    for i in 1:min(3, pair1.n_knots)
        println("     G[$i, 1] = $(pair1.G[i, 1])")
    end

    println("\n[6] ✓ All data extracted successfully!")

catch e
    println("   ✗ Extraction failed!")
    println("   Error: $e")
    rethrow(e)
end

println("\n" * "="^80)
println("Test complete!")
println("="^80)
