#!/usr/bin/env julia
# Test script for Hermite spline code generation

using ACEpotentials
using ACEpotentials.Models
using ACEpotentials.ETModels
using ACEpotentials.ETModels: splinify, ETACE
using Lux
using LuxCore
using Random

# Load the extraction and code generation functions
include("../src/splinify.jl")
include("../src/codegen.jl")

const M = ACEpotentials.Models
const ETM = ACEpotentials.ETModels

println("="^80)
println("Testing Hermite Spline Code Generation")
println("="^80)

# Create test model
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
println("   ✓ Model splinified")

# Extract Hermite data
println("\n[2] Extracting Hermite spline data...")
hermite_data = extract_hermite_spline_data(etace_splined, et_ps, et_st, rcut)
println("   ✓ Extracted data for $(length(hermite_data)) species pairs")

# Generate code
println("\n[3] Generating Julia code...")
code = generate_hermite_spline_code(hermite_data, n_species)
println("   ✓ Generated $(length(code)) characters of code")

# Write to file
output_file = "/tmp/hermite_radial_basis_generated.jl"
println("\n[4] Writing to file: $output_file")
open(output_file, "w") do f
    write(f, code)
end
println("   ✓ Code written")

# Show first few lines
println("\n[5] First 30 lines of generated code:")
lines = split(code, '\n')
for (i, line) in enumerate(lines[1:min(30, end)])
    println("   ", line)
end
if length(lines) > 30
    println("   ...")
    println("   ($(length(lines) - 30) more lines)")
end

println("\n" * "="^80)
println("Code generation complete!")
println("="^80)
println("\nGenerated file: $output_file")
println("Code size: $(length(code)) characters, $(length(lines)) lines")
println("\nNext steps:")
println("  1. Review the generated code")
println("  2. Test it can be included and evaluated")
println("  3. Compare accuracy vs original splines")
