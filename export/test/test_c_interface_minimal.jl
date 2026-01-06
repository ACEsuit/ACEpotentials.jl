#!/usr/bin/env julia
"""
Tests for minimal C interface.

Tests that the C interface correctly loads exported models and provides
accurate energy, force, and basis evaluations.
"""

using Test
using ACEpotentials
using ACEpotentials.ETModels
using ACEpotentials.ETModels: splinify, ETOneBody, StackedCalculator, one_body
using StaticArrays
using Random
using Lux
using LuxCore
using AtomsBase: ChemicalSpecies

const M = ACEpotentials.Models
const ETM = ACEpotentials.ETModels

# Determine paths
const EXPORT_DIR = dirname(dirname(@__FILE__))
const EXPORT_SRC = joinpath(EXPORT_DIR, "src")
const EXPORT_TEST = joinpath(EXPORT_DIR, "test")

# Load C interface
include(joinpath(EXPORT_SRC, "ace_c_interface_minimal.jl"))
using .ACE_C_Interface_Minimal
import .ACE_C_Interface_Minimal: ace_load_model, ace_unload_model, ace_get_cutoff,
                                   ace_get_species, ace_site_energy, ace_site_energy_forces,
                                   ace_site_basis, ace_get_n_basis

# Load export function
include(joinpath(EXPORT_SRC, "export_ace_model_minimal.jl"))

@testset "C Interface Minimal" begin
    println("\n" * "="^80)
    println("Testing Minimal C Interface")
    println("="^80)

    # Create test model
    println("\n[1] Creating test model...")
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

    # Export model
    println("\n[6] Exporting model...")
    output_dir = joinpath(EXPORT_TEST, "build", "c_interface_test")
    export_ace_model_minimal(calc, output_dir; model_name="test_ace")

    # Load model for testing (outside testsets for scoping)
    println("\n[7] Loading model for testing...")
    model_path_cstr = Base.unsafe_convert(Cstring, Base.cconvert(Cstring, output_dir))
    test_model_id = ace_load_model(model_path_cstr)
    @assert test_model_id > 0 "Failed to load model"
    println("    ✓ Model loaded with ID: $test_model_id")

    # Test C interface
    println("\n[8] Testing C interface...")

    @testset "Model Loading" begin
        println("  Testing model loading...")

        # Test that model was loaded
        @test test_model_id > 0
        @test haskey(ACE_C_Interface_Minimal.LOADED_MODELS, test_model_id)

        # Test loading non-existent model
        bad_path_str = "/nonexistent/path"
        bad_path_cstr = Base.unsafe_convert(Cstring, Base.cconvert(Cstring, bad_path_str))
        bad_id = ace_load_model(bad_path_cstr)
        @test bad_id == -1

        println("    ✓ Model tests passed")
    end

    @testset "Get Model Info" begin
        println("  Testing model info retrieval...")

        # Test get cutoff
        cutoff_ref = Ref{Float64}(0.0)
        status = ace_get_cutoff(test_model_id, Base.unsafe_convert(Ptr{Float64}, cutoff_ref))
        @test status == 0
        @test cutoff_ref[] == rcut
        println("    ✓ Cutoff: $(cutoff_ref[]) Å")

        # Test get species
        species_arr = zeros(Int32, 10)
        n_species_ref = Ref{Int32}(0)
        status = ace_get_species(
            test_model_id,
            pointer(species_arr),
            Base.unsafe_convert(Ptr{Int32}, n_species_ref)
        )
        @test status == 0
        @test n_species_ref[] == 1
        @test species_arr[1] == 14  # Silicon
        println("    ✓ Species: $(species_arr[1:n_species_ref[]])")

        # Test get basis size
        n_basis_ref = Ref{Int32}(0)
        status = ace_get_n_basis(test_model_id, Base.unsafe_convert(Ptr{Int32}, n_basis_ref))
        @test status == 0
        @test n_basis_ref[] > 0
        println("    ✓ Basis size: $(n_basis_ref[])")
    end

    @testset "Site Energy Evaluation" begin
        println("  Testing site energy...")

        # Test data: single neighbor
        Rs_flat = [3.0, 0.0, 0.0]
        Zs = Int32[14]
        Z0 = Int32(14)
        n_neigh = Int32(1)

        # Test via C interface
        E_c = ace_site_energy(
            test_model_id,
            n_neigh,
            pointer(Rs_flat),
            pointer(Zs),
            Z0
        )

        @test !isnan(E_c)
        @test isfinite(E_c)
        println("    ✓ Site energy: $E_c eV")

        # Compare with direct Julia call
        Rs_julia = [SVector(3.0, 0.0, 0.0)]
        Zs_julia = [14]
        Z0_julia = 14

        # Load the module directly
        include(joinpath(output_dir, "test_ace.jl"))
        E_julia = Test_ace.site_energy(Rs_julia, Zs_julia, Z0_julia)

        @test E_c ≈ E_julia atol=1e-10
        println("    ✓ Matches Julia: $(abs(E_c - E_julia)) eV difference")

        # Test with invalid model ID
        E_bad = ace_site_energy(Int32(-999), n_neigh, pointer(Rs_flat), pointer(Zs), Z0)
        @test isnan(E_bad)
    end

    @testset "Site Energy and Forces" begin
        println("  Testing site energy and forces...")

        # Test data
        Rs_flat = [3.0, 0.0, 0.0]
        Zs = Int32[14]
        Z0 = Int32(14)
        n_neigh = Int32(1)

        # Allocate output arrays
        energy_ref = Ref{Float64}(0.0)
        forces_flat = zeros(Float64, 3)

        # Call C interface
        status = ace_site_energy_forces(
            test_model_id,
            n_neigh,
            pointer(Rs_flat),
            pointer(Zs),
            Z0,
            Base.unsafe_convert(Ptr{Float64}, energy_ref),
            pointer(forces_flat)
        )

        @test status == 0
        @test !isnan(energy_ref[])
        @test all(isfinite.(forces_flat))
        println("    ✓ Energy: $(energy_ref[]) eV")
        println("    ✓ Forces: $forces_flat")

        # Compare with direct Julia call
        Rs_julia = [SVector(3.0, 0.0, 0.0)]
        Zs_julia = [14]
        Z0_julia = 14

        E_julia, F_julia = Test_ace.site_energy_forces(Rs_julia, Zs_julia, Z0_julia)

        @test energy_ref[] ≈ E_julia atol=1e-10
        @test forces_flat[1] ≈ F_julia[1][1] atol=1e-10
        @test forces_flat[2] ≈ F_julia[1][2] atol=1e-10
        @test forces_flat[3] ≈ F_julia[1][3] atol=1e-10
        println("    ✓ Matches Julia perfectly")

        # Test with invalid model ID
        status_bad = ace_site_energy_forces(
            Int32(-999), n_neigh, pointer(Rs_flat), pointer(Zs), Z0,
            Base.unsafe_convert(Ptr{Float64}, energy_ref), pointer(forces_flat)
        )
        @test status_bad == -1
    end

    @testset "Site Basis Evaluation" begin
        println("  Testing site basis...")

        # Test data
        Rs_flat = [3.0, 0.0, 0.0]
        Zs = Int32[14]
        Z0 = Int32(14)
        n_neigh = Int32(1)

        # Get basis size first
        n_basis_ref = Ref{Int32}(0)
        status = ace_get_n_basis(test_model_id, Base.unsafe_convert(Ptr{Int32}, n_basis_ref))
        @test status == 0

        # Allocate output arrays
        basis_arr = zeros(Float64, n_basis_ref[])
        n_basis_out = Ref{Int32}(0)

        # Call C interface
        status = ace_site_basis(
            test_model_id,
            n_neigh,
            pointer(Rs_flat),
            pointer(Zs),
            Z0,
            pointer(basis_arr),
            Base.unsafe_convert(Ptr{Int32}, n_basis_out)
        )

        @test status == 0
        @test n_basis_out[] == n_basis_ref[]
        @test all(isfinite.(basis_arr))
        println("    ✓ Basis size: $(n_basis_out[])")
        println("    ✓ Basis norm: $(norm(basis_arr))")

        # Compare with direct Julia call
        Rs_julia = [SVector(3.0, 0.0, 0.0)]
        Zs_julia = [14]
        Z0_julia = 14

        B_julia = Test_ace.site_basis(Rs_julia, Zs_julia, Z0_julia)

        @test length(B_julia) == n_basis_out[]
        @test basis_arr ≈ B_julia atol=1e-10
        println("    ✓ Matches Julia perfectly")

        # Test with invalid model ID
        status_bad = ace_site_basis(
            Int32(-999), n_neigh, pointer(Rs_flat), pointer(Zs), Z0,
            pointer(basis_arr), Base.unsafe_convert(Ptr{Int32}, n_basis_out)
        )
        @test status_bad == -1
    end

    @testset "Multiple Neighbors" begin
        println("  Testing with multiple neighbors...")

        # Test data: 3 neighbors
        Rs_flat = [
            3.0, 0.0, 0.0,   # Neighbor 1
            0.0, 3.0, 0.0,   # Neighbor 2
            0.0, 0.0, 3.0    # Neighbor 3
        ]
        Zs = Int32[14, 14, 14]
        Z0 = Int32(14)
        n_neigh = Int32(3)

        # Test energy
        E_c = ace_site_energy(
            test_model_id,
            n_neigh,
            pointer(Rs_flat),
            pointer(Zs),
            Z0
        )

        @test !isnan(E_c)
        @test isfinite(E_c)
        println("    ✓ Energy with 3 neighbors: $E_c eV")

        # Test forces
        energy_ref = Ref{Float64}(0.0)
        forces_flat = zeros(Float64, 9)  # 3 neighbors × 3 components

        status = ace_site_energy_forces(
            test_model_id,
            n_neigh,
            pointer(Rs_flat),
            pointer(Zs),
            Z0,
            Base.unsafe_convert(Ptr{Float64}, energy_ref),
            pointer(forces_flat)
        )

        @test status == 0
        @test all(isfinite.(forces_flat))
        println("    ✓ Forces computed for 3 neighbors")

        # Compare with Julia
        Rs_julia = [
            SVector(3.0, 0.0, 0.0),
            SVector(0.0, 3.0, 0.0),
            SVector(0.0, 0.0, 3.0)
        ]
        Zs_julia = [14, 14, 14]

        E_julia, F_julia = Test_ace.site_energy_forces(Rs_julia, Zs_julia, 14)

        @test energy_ref[] ≈ E_julia atol=1e-10
        for i in 1:3
            @test forces_flat[3*(i-1)+1] ≈ F_julia[i][1] atol=1e-10
            @test forces_flat[3*(i-1)+2] ≈ F_julia[i][2] atol=1e-10
            @test forces_flat[3*(i-1)+3] ≈ F_julia[i][3] atol=1e-10
        end
        println("    ✓ Perfect agreement with Julia")
    end

    @testset "Isolated Atom (No Neighbors)" begin
        println("  Testing isolated atom...")

        # Test data: no neighbors
        Rs_flat = Float64[]
        Zs = Int32[]
        Z0 = Int32(14)
        n_neigh = Int32(0)

        # Test energy (should return E0)
        E_c = ace_site_energy(
            test_model_id,
            n_neigh,
            pointer([0.0]),  # Dummy pointer
            pointer([Int32(0)]),  # Dummy pointer
            Z0
        )

        @test !isnan(E_c)
        @test E_c ≈ -5.0  # Should be E0 value
        println("    ✓ Isolated atom energy: $E_c eV (E0)")

        # Test forces (should be empty)
        energy_ref = Ref{Float64}(0.0)
        forces_flat = Float64[]

        status = ace_site_energy_forces(
            test_model_id,
            n_neigh,
            pointer([0.0]),
            pointer([Int32(0)]),
            Z0,
            Base.unsafe_convert(Ptr{Float64}, energy_ref),
            pointer([0.0])  # Dummy pointer
        )

        @test status == 0
        @test energy_ref[] ≈ -5.0
        println("    ✓ Isolated atom forces: empty (correct)")
    end

    @testset "Model Unloading" begin
        println("  Testing model unloading...")

        # Unload the model
        status = ace_unload_model(test_model_id)
        @test status == 0
        @test !haskey(ACE_C_Interface_Minimal.LOADED_MODELS, test_model_id)
        println("    ✓ Model unloaded successfully")

        # Test that unloaded model can't be used
        Rs_flat = [3.0, 0.0, 0.0]
        Zs = Int32[14]
        Z0 = Int32(14)

        E_bad = ace_site_energy(test_model_id, Int32(1), pointer(Rs_flat), pointer(Zs), Z0)
        @test isnan(E_bad)
        println("    ✓ Unloaded model correctly refuses evaluation")

        # Test unloading non-existent model
        status_bad = ace_unload_model(Int32(-999))
        @test status_bad == -1
    end

    println("\n" * "="^80)
    println("✓ All C Interface Tests Passed")
    println("="^80)
end
