#!/usr/bin/env julia
# Test minimal export approach

using Test
using ACEpotentials
using ACEpotentials.Models
using ACEpotentials.ETModels
using ACEpotentials.ETModels: splinify, ETOneBody, StackedCalculator, one_body
using StaticArrays
using Random
using Lux
using LuxCore
using AtomsCalculators
import AtomsBase
using Unitful
using AtomsBase: ChemicalSpecies

const M = ACEpotentials.Models
const ETM = ACEpotentials.ETModels
const EXPORT_DIR = dirname(@__DIR__)

# Load minimal export
include(joinpath(EXPORT_DIR, "src", "export_ace_model_minimal.jl"))

@testset "Minimal Export" begin
    println("\n" * "="^80)
    println("Testing Minimal Export (No Code Generation!)")
    println("="^80)

    # Create test model
    println("\n[1] Creating ETACE model...")
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

    # Convert to ETACE
    println("[2] Converting to ETACE...")
    et_model = ETM.convert2et(ace_model)
    et_ps, et_st = LuxCore.setup(rng, et_model)

    # Copy parameters
    for iz in 1:1, jz in 1:1
        et_ps.rembed.post.W[:, :, (iz-1)*1 + jz] .= ps.rbasis.Wnlq[:, :, iz, jz]
    end
    for iz in 1:1
        et_ps.readout.W[1, :, iz] .= ps.WB[:, iz]
    end

    # Splinify
    println("[3] Splinifying model...")
    et_model_splined = splinify(et_model, et_ps, et_st; Nspl=50)
    et_ps_splined, et_st_splined = LuxCore.setup(rng, et_model_splined)

    for iz in 1:1
        et_ps_splined.readout.W[1, :, iz] .= et_ps.readout.W[1, :, iz]
    end

    # Create ETACE calculator
    etace_calc = ETM.ETACEPotential(et_model_splined, et_ps_splined, et_st_splined, rcut)

    # Create ETOneBody
    println("[4] Creating ETOneBody...")
    E0_dict_species = Dict(ChemicalSpecies(:Si) => -5.0)
    e0_model = one_body(E0_dict_species, x -> x.z)
    e0_ps, e0_st = LuxCore.setup(rng, e0_model)
    e0_calc = ETM.ETOneBodyPotential(e0_model, e0_ps, e0_st, rcut)

    # Create StackedCalculator
    println("[5] Creating StackedCalculator...")
    calc = StackedCalculator((e0_calc, etace_calc))

    # Create test system
    println("[6] Creating test system...")
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
    println("[7] Computing reference energy...")
    E_etace_only = AtomsCalculators.potential_energy(sys, etace_calc)
    E_e0_only = AtomsCalculators.potential_energy(sys, e0_calc)
    E_calc = AtomsCalculators.potential_energy(sys, calc)

    E_etace_val = ustrip(u"eV", E_etace_only)
    E_e0_val = ustrip(u"eV", E_e0_only)
    E_calc_val = ustrip(u"eV", E_calc)

    println("   ETACE only: $E_etace_val eV")
    println("   E0 only: $E_e0_val eV")
    println("   Total: $E_calc_val eV")

    # Export with minimal approach
    println("\n[8] Exporting with minimal approach...")
    output_dir = joinpath(@__DIR__, "build", "minimal_export")
    export_ace_model_minimal(calc, output_dir; model_name="test_ace")

    # Load exported module
    println("\n[9] Loading exported module...")
    wrapper_file = joinpath(output_dir, "test_ace.jl")
    include(wrapper_file)
    exported_mod = Test_ace

    # Test evaluation
    println("\n[10] Testing exported model...")
    neighbor_Rs = [SVector(3.0, 0.0, 0.0)]
    neighbor_Zs = [14]
    Z0 = 14

    E_exported_1 = exported_mod.site_energy(neighbor_Rs, neighbor_Zs, Z0)
    E_exported_2 = exported_mod.site_energy([SVector(-3.0, 0.0, 0.0)], neighbor_Zs, Z0)
    E_exported_total = E_exported_1 + E_exported_2

    println("   Exported energy (atom 1): $E_exported_1 eV")
    println("   Exported energy (atom 2): $E_exported_2 eV")
    println("   Exported total: $E_exported_total eV")
    println("   Calculator total: $E_calc_val eV")
    println("   Difference: $(abs(E_exported_total - E_calc_val)) eV")

    @testset "Energy Accuracy" begin
        @test E_exported_total ≈ E_calc_val atol=1e-10 rtol=1e-8
    end

    # Test forces
    println("\n[11] Testing forces...")
    E_exp, F_exp = exported_mod.site_energy_forces(neighbor_Rs, neighbor_Zs, Z0)
    println("   Exported energy (with forces): $E_exp eV")
    println("   Exported forces: $F_exp")
    @test E_exp ≈ E_exported_1 atol=1e-10

    # Test basis
    println("\n[12] Testing basis...")
    B_exp = exported_mod.site_basis(neighbor_Rs, neighbor_Zs, Z0)
    println("   Basis size: $(length(B_exp))")
    println("   Basis norm: $(norm(B_exp))")

    println("\n" * "="^80)
    println("✓ Minimal Export Test Complete")
    println("="^80)
end
