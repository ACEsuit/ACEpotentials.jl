#!/usr/bin/env julia
# Compare Rnl evaluation: calculator vs exported

using ACEpotentials
using ACEpotentials.Models
using ACEpotentials.ETModels
using ACEpotentials.ETModels: splinify
using StaticArrays
using Random
using Lux
using LuxCore
import EquivariantTensors as ET

const M = ACEpotentials.Models
const ETM = ACEpotentials.ETModels
const EXPORT_DIR = dirname(@__DIR__)

include(joinpath(EXPORT_DIR, "src", "export_ace_model.jl"))

println("\nComparing Rnl Evaluation: Calculator vs Exported")
println("="^80)

# Create and splinify model
elements = (:Si,)
rcut = 5.5
rng = Random.MersenneTwister(1234)

ace_model = M.ace_model(;
    elements = elements,
    order = 2,
    Ytype = :solid,
    level = M.TotalDegree(),
    max_level = 8,
    maxl = 2,
    pair_maxn = 8,
    rin0cuts = (x -> (rin = x.rin, r0 = x.r0, rcut = rcut)).(M._default_rin0cuts(elements)),
    init_WB = :glorot_normal,
    init_Wpair = :glorot_normal
)

ps, st = Lux.setup(rng, ace_model)

et_model = ETM.convert2et(ace_model)
et_ps, et_st = LuxCore.setup(rng, et_model)

# Copy parameters
for iz in 1:1, jz in 1:1
    et_ps.rembed.post.W[:, :, (iz-1)*1 + jz] .= ps.rbasis.Wnlq[:, :, iz, jz]
end
for iz in 1:1
    et_ps.readout.W[1, :, iz] .= ps.WB[:, iz]
end

et_model_splined = splinify(et_model, et_ps, et_st; Nspl=50)
et_ps_splined, et_st_splined = LuxCore.setup(rng, et_model_splined)

for iz in 1:1
    et_ps_splined.readout.W[1, :, iz] .= et_ps.readout.W[1, :, iz]
end

# Export
etace_calc = ETM.ETACEPotential(et_model_splined, et_ps_splined, et_st_splined, rcut)
model_file = joinpath(@__DIR__, "build", "compare_rnl.jl")
export_ace_model(etace_calc, model_file; for_library=false, radial_basis=:hermite_spline)

exported = Module(:Exported)
Base.include(exported, model_file)

# Test at r = 3.0 Å
r_test = 3.0
println("\nTest distance: r = $r_test Å")

# Evaluate with exported code
Rnl_exported = exported.evaluate_Rnl_1(r_test)
println("\nExported Rnl:")
println("  Values: $(Rnl_exported)")
println("  Norm: $(norm(Rnl_exported))")

# Now evaluate with calculator's spline layer
# Access the rembed layer (TransSelSplines)
rembed_layer = et_model_splined.rembed.layer

# Create edge input for evaluation
using AtomsBase: ChemicalSpecies
zi = ChemicalSpecies(:Si)
zj = ChemicalSpecies(:Si)
r_vec = SVector(r_test, 0.0, 0.0)

# Create edge tuple
edge = (; 𝐫 = r_vec, z0 = zi, z1 = zj)

# Evaluate the rembed layer
Rnl_calc, _ = rembed_layer(edge, et_ps_splined.rembed, et_st_splined.rembed)

println("\nCalculator Rnl:")
println("  Values: $(Rnl_calc)")
println("  Norm: $(norm(Rnl_calc))")

println("\nDifference:")
println("  Abs diff: $(norm(Rnl_calc - Rnl_exported))")
println("  Rel diff: $(norm(Rnl_calc - Rnl_exported) / norm(Rnl_calc))")

# Compare element by element
println("\nElement-by-element comparison:")
for i in 1:min(length(Rnl_calc), length(Rnl_exported))
    diff = Rnl_calc[i] - Rnl_exported[i]
    println("  [$i] calc=$(Rnl_calc[i]), exported=$(Rnl_exported[i]), diff=$diff")
end
