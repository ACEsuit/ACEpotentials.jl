#!/usr/bin/env julia
# Debug splinified ETACE export

using Test
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
const TEST_DIR = @__DIR__

# Include export function
include(joinpath(EXPORT_DIR, "src", "export_ace_model.jl"))

println("\n" * "="^80)
println("Debugging Splinified ETACE Export")
println("="^80)

# Create minimal test model
println("\n[1] Creating ETACE model...")
elements = (:Si,)
order = 2
max_level = 8
maxl = 2
rcut = 5.5

rin0cuts = M._default_rin0cuts(elements)
rin0cuts = (x -> (rin = x.rin, r0 = x.r0, rcut = rcut)).(rin0cuts)

rng = Random.MersenneTwister(1234)

ace_model = M.ace_model(;
    elements = elements,
    order = order,
    Ytype = :solid,
    level = M.TotalDegree(),
    max_level = max_level,
    maxl = maxl,
    pair_maxn = max_level,
    rin0cuts = rin0cuts,
    init_WB = :glorot_normal,
    init_Wpair = :glorot_normal
)

ps, st = Lux.setup(rng, ace_model)

# Convert to ETACE
println("[2] Converting to ETACE...")
et_model = ETM.convert2et(ace_model)
et_ps, et_st = LuxCore.setup(MersenneTwister(1234), et_model)

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
println("[3] Splinifying model (Nspl=50)...")
et_model_splined = splinify(et_model, et_ps, et_st; Nspl=50)

et_ps_splined, et_st_splined = LuxCore.setup(MersenneTwister(1234), et_model_splined)

# Copy readout weights
for iz in 1:n_species
    et_ps_splined.readout.W[1, :, iz] .= et_ps.readout.W[1, :, iz]
end

# Create calculator
etace_calc = ETM.ETACEPotential(et_model_splined, et_ps_splined, et_st_splined, rcut)

# Create test system - single Si atom with one neighbor
println("[4] Creating minimal test system...")
# Two atoms at distance r = 3.0 Å
positions = [
    SVector(0.0, 0.0, 0.0),
    SVector(3.0, 0.0, 0.0),
]
box = [SVector(10.0, 0.0, 0.0), SVector(0.0, 10.0, 0.0), SVector(0.0, 0.0, 10.0)]

sys = AtomsBase.periodic_system(
    [:Si => pos * u"Å" for pos in positions],
    [b * u"Å" for b in box]
)

# Compute reference energy
println("[5] Computing reference energy...")
E_calc = AtomsCalculators.potential_energy(sys, etace_calc)
E_calc_val = ustrip(u"eV", E_calc)
println("   Calculator energy: $E_calc_val eV")

# Now let's manually evaluate the calculator to see intermediate values
println("\n[6] Manual evaluation with calculator...")
G = ET.Atoms.interaction_graph(sys, rcut * u"Å")

# Get the model output
y_calc, st_out = et_model_splined(G, et_ps_splined, et_st_splined)
println("   Raw model output (per atom): $(y_calc)")
println("   Sum: $(sum(y_calc)) eV")

# Get site energies
site_energies_calc = sum(y_calc)
println("   Site energies sum: $site_energies_calc eV")

# Now let's trace through the model step by step
println("\n[7] Detailed model evaluation...")

# Get edge data
println("   Number of edges: $(length(G.edge_data))")
for (i, edge) in enumerate(G.edge_data)
    println("   Edge $i: r = $(edge.𝐫), |r| = $(norm(edge.𝐫))")
end

# Export the model
println("\n[8] Exporting model...")
build_dir = joinpath(TEST_DIR, "build")
mkpath(build_dir)
model_file = joinpath(build_dir, "debug_spline.jl")

export_ace_model(etace_calc, model_file;
                 for_library=false,
                 radial_basis=:hermite_spline)

println("   ✓ Exported to $model_file")

# Load and test exported model
println("\n[9] Testing exported model...")
exported = Module(:ExportedDebug)
Base.include(exported, model_file)

# Evaluate for first atom (at origin) with one neighbor at (3,0,0)
neighbor_Rs = [SVector(3.0, 0.0, 0.0)]
neighbor_Zs = [14]  # Si
Z0 = 14

E_exported_atom1 = exported.site_energy(neighbor_Rs, neighbor_Zs, Z0)
println("   Exported energy (atom 1): $E_exported_atom1 eV")

# Evaluate for second atom with one neighbor at (-3,0,0)
neighbor_Rs2 = [SVector(-3.0, 0.0, 0.0)]
E_exported_atom2 = exported.site_energy(neighbor_Rs2, neighbor_Zs, Z0)
println("   Exported energy (atom 2): $E_exported_atom2 eV")

E_exported_total = E_exported_atom1 + E_exported_atom2
println("   Exported total: $E_exported_total eV")
println("   Calculator total: $E_calc_val eV")
println("   Difference: $(abs(E_exported_total - E_calc_val)) eV")

# Now let's manually evaluate the radial basis at r=3.0
println("\n[10] Detailed radial basis evaluation at r=3.0...")

# Call the exported evaluate_Rnl function
r_test = 3.0
Rnl_exported = exported.evaluate_Rnl_1(r_test)
println("   Exported Rnl(r=$r_test): $Rnl_exported")
println("   Rnl norm: $(norm(Rnl_exported))")
println("   Rnl values: $(Rnl_exported)")

# Now let's evaluate the full basis
println("\n[11] Full basis evaluation...")
B_exported = exported.site_basis(neighbor_Rs, neighbor_Zs, Z0)
println("   Exported basis B: $(B_exported)")
println("   Basis norm: $(norm(B_exported))")
println("   Basis size: $(length(B_exported))")

# Check weights
println("\n[12] Weights check...")
WB = exported.WB_1
println("   WB size: $(length(WB))")
println("   WB norm: $(norm(WB))")
println("   WB[1:5]: $(WB[1:5])")

# Compute energy manually
E_manual = dot(B_exported, WB)
println("   Manual energy (dot(B, WB)): $E_manual eV")
println("   Should match exported: $E_exported_atom1 eV")

println("\n" * "="^80)
println("Debug Complete")
println("="^80)
