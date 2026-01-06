#!/usr/bin/env julia
# Test Hermite spline export with a properly splinified ETACE model
# This validates that the Hermite cubic spline code generation works correctly

using Test
using ACEpotentials
using ACEpotentials.Models
using ACEpotentials.ETModels
using ACEpotentials.ETModels: splinify, ETOneBody, StackedCalculator, one_body
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

@testset "Hermite Spline Export" begin
    println("\n" * "="^80)
    println("Testing Hermite Spline Export")
    println("="^80)

    # Create test model
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

    # CRITICAL: Splinify BEFORE creating calculator
    println("[3] Splinifying model (Nspl=50)...")
    et_model_splined = splinify(et_model, et_ps, et_st; Nspl=50)
    println("   ✓ Model splinified")

    # Setup parameters and state for splinified model
    et_ps_splined, et_st_splined = LuxCore.setup(MersenneTwister(1234), et_model_splined)

    # Copy readout weights (splinify doesn't preserve these)
    for iz in 1:n_species
        et_ps_splined.readout.W[1, :, iz] .= et_ps.readout.W[1, :, iz]
    end

    # Create ETACE calculator with splinified model
    etace_calc = ETM.ETACEPotential(et_model_splined, et_ps_splined, et_st_splined, rcut)

    # Create ETOneBody model for E0 values
    println("[3b] Creating ETOneBody model...")
    using AtomsBase: ChemicalSpecies
    # Define E0 values: map ChemicalSpecies to energy
    E0_dict_species = Dict(ChemicalSpecies(:Si) => -5.0)  # -5.0 eV per Si atom
    e0_model = one_body(E0_dict_species, x -> x.z)
    e0_ps, e0_st = LuxCore.setup(MersenneTwister(1234), e0_model)
    e0_calc = ETM.ETOneBodyPotential(e0_model, e0_ps, e0_st, rcut)

    # Create StackedCalculator combining ETOneBody + ETACE
    println("[3c] Creating StackedCalculator...")
    et_calc = StackedCalculator((e0_calc, etace_calc))
    println("   ✓ StackedCalculator created (ETOneBody + ETACE)")

    # Create test system (4-atom Si diamond structure)
    println("[4] Creating test system...")
    a0 = 5.43
    positions = [
        SVector(0.0, 0.0, 0.0),
        SVector(a0/4, a0/4, a0/4),
        SVector(a0/2, a0/2, 0.0),
        SVector(a0/2, 0.0, a0/2),
    ]
    box = [SVector(a0, 0.0, 0.0), SVector(0.0, a0, 0.0), SVector(0.0, 0.0, a0)]

    sys = AtomsBase.periodic_system(
        [:Si => pos * u"Å" for pos in positions],
        [b * u"Å" for b in box]
    )

    # Compute reference energy with calculator
    println("[5] Computing reference energy with calculator...")

    # Test ETACE alone first
    E_etace_only = AtomsCalculators.potential_energy(sys, etace_calc)
    E_etace_only_val = ustrip(u"eV", E_etace_only)
    println("   ETACE only energy: $E_etace_only_val eV")

    # Test E0 alone
    E_e0_only = AtomsCalculators.potential_energy(sys, e0_calc)
    E_e0_only_val = ustrip(u"eV", E_e0_only)
    println("   E0 only energy: $E_e0_only_val eV")

    # Test stacked
    E_calc = AtomsCalculators.potential_energy(sys, et_calc)
    E_calc_val = ustrip(u"eV", E_calc)
    println("   StackedCalculator energy: $E_calc_val eV")
    println("   Expected (E0 + ETACE): $(E_e0_only_val + E_etace_only_val) eV")

    # Export with Hermite splines (default)
    println("[6] Exporting model with Hermite splines...")
    build_dir = joinpath(TEST_DIR, "build")
    mkpath(build_dir)

    # Test: Export ETACE only first (without StackedCalculator)
    # Try polynomial first to see if export works at all
    println("[6a] Testing ETACE-only export with polynomial radial basis...")
    model_file_poly = joinpath(build_dir, "test_poly_etace_only.jl")
    export_ace_model(etace_calc, model_file_poly;
                     for_library=false,
                     radial_basis=:polynomial)
    println("   ✓ Polynomial export complete")

    println("[6b] Testing ETACE-only export with Hermite splines...")
    model_file_etace = joinpath(build_dir, "test_hermite_etace_only.jl")
    export_ace_model(etace_calc, model_file_etace;
                     for_library=false,
                     radial_basis=:hermite_spline)
    println("   ✓ Hermite spline export complete")

    # Now export with StackedCalculator
    println("[6b] Exporting with StackedCalculator...")
    model_file = joinpath(build_dir, "test_hermite_spline.jl")
    export_ace_model(et_calc, model_file;
                     for_library=false,
                     radial_basis=:hermite_spline)

    @test isfile(model_file)
    println("   ✓ Model exported to $model_file")

    # Verify Hermite spline code is present
    content = read(model_file, String)
    @test occursin("HERMITE", content) || occursin("hermite", content) || occursin("PAIR_", content)
    println("   ✓ Hermite spline code detected")

    # Test polynomial export first
    println("[7a] Testing polynomial exported model...")
    exported_poly = Module(:ExportedPoly)
    Base.include(exported_poly, model_file_poly)

    G = ET.Atoms.interaction_graph(sys, rcut * u"Å")
    n_atoms = length(sys)
    Z0 = 14

    E_poly_exported = 0.0
    for i in 1:n_atoms
        neighbor_Rs = SVector{3, Float64}[]
        neighbor_Zs = Int[]
        for (edge_idx, edge) in enumerate(G.edge_data)
            if G.ii[edge_idx] == i
                push!(neighbor_Rs, SVector{3, Float64}(edge.𝐫))
                push!(neighbor_Zs, Int(edge.z1.atomic_number))
            end
        end
        E_i = exported_poly.site_energy(neighbor_Rs, neighbor_Zs, Z0)
        E_poly_exported += E_i
    end
    println("   Polynomial exported energy: $E_poly_exported eV")
    println("   ETACE calculator energy: $E_etace_only_val eV")
    println("   Polynomial difference: $(abs(E_poly_exported - E_etace_only_val)) eV")

    # Load ETACE-only Hermite spline exported model
    println("[7b] Testing Hermite spline exported model...")
    exported_etace = Module(:ExportedETACE)
    Base.include(exported_etace, model_file_etace)

    E_etace_exported = 0.0
    for i in 1:n_atoms
        neighbor_Rs = SVector{3, Float64}[]
        neighbor_Zs = Int[]
        for (edge_idx, edge) in enumerate(G.edge_data)
            if G.ii[edge_idx] == i
                push!(neighbor_Rs, SVector{3, Float64}(edge.𝐫))
                push!(neighbor_Zs, Int(edge.z1.atomic_number))
            end
        end
        E_i = exported_etace.site_energy(neighbor_Rs, neighbor_Zs, Z0)
        E_etace_exported += E_i
    end
    println("   Hermite spline exported energy: $E_etace_exported eV")
    println("   ETACE calculator energy: $E_etace_only_val eV")
    println("   Hermite difference: $(abs(E_etace_exported - E_etace_only_val)) eV")

    # Load exported model in isolated module
    println("[7b] Loading StackedCalculator exported model...")
    exported = Module(:ExportedHermite)
    Base.include(exported, model_file)
    println("   ✓ Exported model loaded")

    # Compute energy with exported model
    println("[8] Computing energy with StackedCalculator exported model...")
    G = ET.Atoms.interaction_graph(sys, rcut * u"Å")
    n_atoms = length(sys)
    Z0 = 14  # Silicon

    E_exported = 0.0
    for i in 1:n_atoms
        neighbor_Rs = SVector{3, Float64}[]
        neighbor_Zs = Int[]
        for (edge_idx, edge) in enumerate(G.edge_data)
            if G.ii[edge_idx] == i
                push!(neighbor_Rs, SVector{3, Float64}(edge.𝐫))
                push!(neighbor_Zs, Int(edge.z1.atomic_number))
            end
        end
        # site_energy handles empty neighbor lists (returns E0)
        E_i = exported.site_energy(neighbor_Rs, neighbor_Zs, Z0)
        E_exported += E_i
    end

    println("   Exported energy:   $E_exported eV")
    println("   Difference vs StackedCalculator: $(abs(E_exported - E_calc_val)) eV")
    println("   Difference vs E0+ETACE expected: $(abs(E_exported - (E_e0_only_val + E_etace_only_val))) eV")

    # Test accuracy - exported model should match E0 + ETACE
    @testset "Energy Accuracy" begin
        @test E_exported ≈ (E_e0_only_val + E_etace_only_val) atol=1e-10 rtol=1e-8
    end

    # Also test forces
    println("\n[9] Testing forces...")
    F_calc = AtomsCalculators.forces(sys, et_calc)
    F_calc_vals = [ustrip.(u"eV/Å", f) for f in F_calc]

    # Compute forces with exported model using full Newton's 3rd law assembly
    # For each site energy E_i:
    #   - dE_i/dR_j is the force on neighbor j due to site i
    #   - The reaction force on center i is -sum_j(dE_i/dR_j)
    F_exported = [zero(SVector{3, Float64}) for _ in 1:n_atoms]

    for i in 1:n_atoms
        neighbor_Rs = SVector{3, Float64}[]
        neighbor_Zs = Int[]
        neighbor_indices = Int[]

        for (edge_idx, edge) in enumerate(G.edge_data)
            if G.ii[edge_idx] == i
                push!(neighbor_Rs, SVector{3, Float64}(edge.𝐫))
                push!(neighbor_Zs, Int(edge.z1.atomic_number))
                push!(neighbor_indices, G.jj[edge_idx])
            end
        end

        if isempty(neighbor_Rs)
            continue
        end

        _, dE_dR = exported.site_energy_forces(neighbor_Rs, neighbor_Zs, Z0)

        # Full Newton's 3rd law assembly:
        # dE_dR[k] is the force on neighbor j_k (= -dE_i/dR_{j_k})
        for (k, F_on_j) in enumerate(dE_dR)
            j = neighbor_indices[k]

            # Force on neighbor j
            F_exported[j] += F_on_j

            # Reaction force on center i (Newton's 3rd law)
            F_exported[i] -= F_on_j
        end
    end

    @testset "Force Accuracy" begin
        for i in 1:n_atoms
            force_error = norm(F_exported[i] - F_calc_vals[i])
            println("   Atom $i force error: $(force_error) eV/Å")
            # Expect machine precision with exact Hermite splines
            @test force_error < 1e-10
        end
    end

    println("\n" * "="^80)
    println("✓ Hermite Spline Export Test Complete")
    println("="^80)
end
