#!/usr/bin/env julia
# Compare basis evaluation between calculator and exported code

using ACEpotentials
using ACEpotentials.Models
using ACEpotentials.ETModels
using ACEpotentials.ETModels: splinify
using StaticArrays
using LinearAlgebra
using Random
using Lux
using LuxCore
using AtomsCalculators
import AtomsBase
using Unitful
import EquivariantTensors as ET

const M = ACEpotentials.Models
const ETM = ACEpotentials.ETModels
const EXPORT_DIR = dirname(@__DIR__)

include(joinpath(EXPORT_DIR, "src", "export_ace_model.jl"))

println("\nComparing Basis Evaluation")
println("="^80)

# Create model (same as debug script)
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

# Copy readout weights
for iz in 1:1
    et_ps_splined.readout.W[1, :, iz] .= et_ps.readout.W[1, :, iz]
end

etace_calc = ETM.ETACEPotential(et_model_splined, et_ps_splined, et_st_splined, rcut)

# Create system
positions = [SVector(0.0, 0.0, 0.0), SVector(3.0, 0.0, 0.0)]
box = [SVector(10.0, 0.0, 0.0), SVector(0.0, 10.0, 0.0), SVector(0.0, 0.0, 10.0)]
sys = AtomsBase.periodic_system(
    [:Si => pos * u"Å" for pos in positions],
    [b * u"Å" for b in box]
)

# Compute with calculator
E_calc = AtomsCalculators.potential_energy(sys, etace_calc)
E_calc_val = ustrip(u"eV", E_calc)
println("\nCalculator energy: $E_calc_val eV")

# Get graph
G = ET.Atoms.interaction_graph(sys, rcut * u"Å")

# Evaluate model to get basis
y_calc, st_out = et_model_splined(G, et_ps_splined, et_st_splined)
println("Model output (site energies): $(y_calc)")

# Now try to get intermediate basis values
# The model structure is: rembed -> yembed -> basis -> readout
println("\n Checking model evaluation step by step...")

# Export and load
model_file = joinpath(@__DIR__, "build", "compare_basis.jl")
export_ace_model(etace_calc, model_file; for_library=false, radial_basis=:hermite_spline)

exported = Module(:Exported)
Base.include(exported, model_file)

# Evaluate with exported code
neighbor_Rs = [SVector(3.0, 0.0, 0.0)]
neighbor_Zs = [14]
Z0 = 14

B_exported = exported.site_basis(neighbor_Rs, neighbor_Zs, Z0)
println("\nExported basis:")
println("  Size: $(length(B_exported))")
println("  Norm: $(norm(B_exported))")
println("  Max: $(maximum(abs, B_exported))")
println("  First 5: $(B_exported[1:5])")
println("  Last 5: $(B_exported[end-4:end])")

E_exported = exported.site_energy(neighbor_Rs, neighbor_Zs, Z0)
println("\nExported energy: $E_exported eV")
println("Calculator energy (per atom): $(E_calc_val / 2) eV")

# Check weights
WB = exported.WB_1
println("\nWeights:")
println("  Size: $(length(WB))")
println("  Norm: $(norm(WB))")
println("  First 5: $(WB[1:5])")
println("  Last 5: $(WB[end-4:end])")

# Manual energy
E_manual = dot(B_exported, WB)
println("\nManual dot(B, WB): $E_manual eV")

println("\n" * "="^80)
