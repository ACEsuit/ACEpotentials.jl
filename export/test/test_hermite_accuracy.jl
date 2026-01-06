#!/usr/bin/env julia
#=
Test Hermite Spline Export Accuracy

This test verifies that the Hermite cubic spline export produces results
that accurately match the reference splinified model.

Key tests:
1. Hermite export energy matches reference calculator (splinified model)
2. Hermite export forces match reference calculator
3. Forces are consistent with finite differences of energy
4. Both polynomial (unsplinified) and hermite (splinified) exports work correctly
=#

using Test
using ACEpotentials
using ACEpotentials.Models
using ACEpotentials.ETModels
using ACEpotentials.ETModels: splinify, ETACEPotential
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

"""
Create a splinified ETACE model for testing.
Returns (calc_splined, calc_unsplined, rcut) where calc_splined is the
splinified model and calc_unsplined is the original polynomial model.
"""
function setup_splinified_model(; elements=(:Si,), order=2, max_level=8, maxl=2, rcut=5.5)
    # Build ACE model
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

    # Convert to ETACE (unsplinified)
    et_model = ETM.convert2et(ace_model)
    et_ps, et_st = LuxCore.setup(MersenneTwister(1234), et_model)

    # Copy parameters from ACE to ETACE
    n_species = length(elements)
    for iz in 1:n_species
        for jz in 1:n_species
            et_ps.rembed.post.W[:, :, (iz-1)*n_species + jz] .= ps.rbasis.Wnlq[:, :, iz, jz]
        end
    end
    for iz in 1:n_species
        et_ps.readout.W[1, :, iz] .= ps.WB[:, iz]
    end

    # Create unsplinified calculator
    calc_unsplined = ETM.ETACEPotential(et_model, et_ps, et_st, rcut)

    # Splinify the model
    et_model_splined = splinify(et_model, et_ps, et_st; Nspl=50)
    et_ps_splined, et_st_splined = LuxCore.setup(MersenneTwister(1234), et_model_splined)

    # Copy readout weights (splinify doesn't preserve these)
    for iz in 1:n_species
        et_ps_splined.readout.W[1, :, iz] .= et_ps.readout.W[1, :, iz]
    end

    # Create splinified calculator
    calc_splined = ETM.ETACEPotential(et_model_splined, et_ps_splined, et_st_splined, rcut)

    return calc_splined, calc_unsplined, rcut
end

"""
Create a test atomic system.
"""
function create_test_system(a0=5.43)
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
    return sys
end

"""
Compute energy from exported model.
"""
function compute_exported_energy(sys, exported_module, rcut)
    G = ET.Atoms.interaction_graph(sys, rcut * u"Å")
    n_atoms = length(sys)
    Z0 = 14  # Silicon

    E_total = 0.0
    for i in 1:n_atoms
        neighbor_Rs = SVector{3, Float64}[]
        neighbor_Zs = Int[]
        for (edge_idx, edge) in enumerate(G.edge_data)
            if G.ii[edge_idx] == i
                push!(neighbor_Rs, SVector{3, Float64}(edge.𝐫))
                push!(neighbor_Zs, Int(edge.z1.atomic_number))
            end
        end
        E_i = isempty(neighbor_Rs) ? 0.0 : exported_module.site_energy(neighbor_Rs, neighbor_Zs, Z0)
        E_total += E_i
    end
    return E_total
end

"""
Compute forces from exported model with proper accumulation.
"""
function compute_exported_forces(sys, exported_module, rcut)
    G = ET.Atoms.interaction_graph(sys, rcut * u"Å")
    n_atoms = length(sys)
    Z0 = 14

    F_total = [zeros(SVector{3, Float64}) for _ in 1:n_atoms]

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

        if !isempty(neighbor_Rs)
            _, F_site = exported_module.site_energy_forces(neighbor_Rs, neighbor_Zs, Z0)
            for (j_local, f) in enumerate(F_site)
                j_global = neighbor_indices[j_local]
                F_total[i] += -f
                F_total[j_global] += f
            end
        end
    end

    return F_total
end

@testset "Hermite Spline Export Accuracy" verbose=true begin

    println("\n" * "="^80)
    println("Testing Hermite Spline Export Accuracy")
    println("="^80)

    # Setup
    build_dir = joinpath(TEST_DIR, "build")
    mkpath(build_dir)

    @testset "Hermite Export vs Reference (Si)" begin
        println("\n[1] Testing Hermite export accuracy against reference (Si)...")

        calc_splined, calc_unsplined, rcut = setup_splinified_model()
        sys = create_test_system()

        # Get reference energy and forces from splinified calculator
        E_ref = ustrip(u"eV", AtomsCalculators.potential_energy(sys, calc_splined))
        F_ref = AtomsCalculators.forces(sys, calc_splined)
        F_ref_val = [SVector{3}(ustrip.(u"eV/Å", f)) for f in F_ref]

        println("   Reference energy (splinified calc): $E_ref eV")

        # Export with Hermite spline method
        hermite_file = joinpath(build_dir, "hermite_accuracy_test.jl")
        export_ace_model(calc_splined, hermite_file; for_library=false, radial_basis=:hermite_spline)
        @test isfile(hermite_file)

        # Load export
        hermite_mod = Module(:HermiteExport)
        Base.include(hermite_mod, hermite_file)

        # Test energy accuracy
        E_hermite = compute_exported_energy(sys, hermite_mod, rcut)
        energy_diff = abs(E_hermite - E_ref)
        energy_rel_diff = energy_diff / abs(E_ref)

        println("   Hermite export energy:  $E_hermite eV")
        println("   Energy difference:      $energy_diff eV")
        println("   Relative difference:    $energy_rel_diff")

        # Hermite export should match reference very closely
        # (differences only from floating point and spline representation)
        @test E_hermite ≈ E_ref atol=1e-10 rtol=1e-8

        # Test force accuracy
        F_hermite = compute_exported_forces(sys, hermite_mod, rcut)
        max_force_diff = 0.0
        for i in 1:length(sys)
            diff = norm(F_hermite[i] - F_ref_val[i])
            max_force_diff = max(max_force_diff, diff)
        end
        println("   Max force difference:   $max_force_diff eV/Å")

        for i in 1:length(sys)
            @test F_hermite[i] ≈ F_ref_val[i] atol=1e-8 rtol=1e-6
        end

        println("   Energy and force accuracy tests passed")
    end

    @testset "Polynomial Export vs Reference (Si)" begin
        println("\n[2] Testing Polynomial export accuracy against reference (Si)...")

        _, calc_unsplined, rcut = setup_splinified_model()
        sys = create_test_system()

        # Get reference from unsplinified calculator
        E_ref = ustrip(u"eV", AtomsCalculators.potential_energy(sys, calc_unsplined))
        F_ref = AtomsCalculators.forces(sys, calc_unsplined)
        F_ref_val = [SVector{3}(ustrip.(u"eV/Å", f)) for f in F_ref]

        println("   Reference energy (polynomial calc): $E_ref eV")

        # Export with polynomial method (unsplinified model)
        poly_file = joinpath(build_dir, "poly_accuracy_test.jl")
        export_ace_model(calc_unsplined, poly_file; for_library=false, radial_basis=:polynomial)
        @test isfile(poly_file)

        # Load export
        poly_mod = Module(:PolyExport)
        Base.include(poly_mod, poly_file)

        # Test energy accuracy
        E_poly = compute_exported_energy(sys, poly_mod, rcut)
        energy_diff = abs(E_poly - E_ref)
        energy_rel_diff = energy_diff / abs(E_ref)

        println("   Polynomial export energy: $E_poly eV")
        println("   Energy difference:        $energy_diff eV")
        println("   Relative difference:      $energy_rel_diff")

        @test E_poly ≈ E_ref atol=1e-10 rtol=1e-8

        # Test force accuracy
        F_poly = compute_exported_forces(sys, poly_mod, rcut)
        max_force_diff = 0.0
        for i in 1:length(sys)
            diff = norm(F_poly[i] - F_ref_val[i])
            max_force_diff = max(max_force_diff, diff)
        end
        println("   Max force difference:     $max_force_diff eV/Å")

        for i in 1:length(sys)
            @test F_poly[i] ≈ F_ref_val[i] atol=1e-8 rtol=1e-6
        end

        println("   Polynomial export accuracy tests passed")
    end

    @testset "Finite Difference Verification (Hermite)" begin
        println("\n[3] Verifying Hermite forces with finite differences...")

        calc_splined, _, rcut = setup_splinified_model()

        hermite_file = joinpath(build_dir, "hermite_fd_test.jl")
        export_ace_model(calc_splined, hermite_file; for_library=false, radial_basis=:hermite_spline)

        hermite_mod = Module(:HermiteFD)
        Base.include(hermite_mod, hermite_file)

        # Test configuration
        Z0 = 14
        Rs = [
            SVector(2.35, 0.0, 0.0),
            SVector(0.0, 2.35, 0.0),
            SVector(0.0, 0.0, 2.35),
            SVector(1.5, 1.5, 1.5),
        ]
        Zs = [14, 14, 14, 14]

        # Analytic forces
        _, F_analytic = hermite_mod.site_energy_forces(Rs, Zs, Z0)

        # Finite difference
        h = 1e-6
        max_fd_error = 0.0
        for j in 1:length(Rs)
            F_fd = zeros(3)
            for α in 1:3
                e_α = zeros(3)
                e_α[α] = h

                Rs_p = copy(Rs)
                Rs_m = copy(Rs)
                Rs_p[j] = Rs[j] + SVector{3}(e_α)
                Rs_m[j] = Rs[j] - SVector{3}(e_α)

                Ep = hermite_mod.site_energy(Rs_p, Zs, Z0)
                Em = hermite_mod.site_energy(Rs_m, Zs, Z0)

                F_fd[α] = -(Ep - Em) / (2h)
            end

            fd_error = norm(F_fd - F_analytic[j])
            max_fd_error = max(max_fd_error, fd_error)

            # With analytical derivatives, FD error should be O(h^2) ≈ 1e-12
            # Allow some margin for numerical effects
            @test fd_error < 1e-5
        end
        println("   Max FD error: $max_fd_error eV/Å")
        println("   Finite difference verification passed")
    end

    @testset "Finite Difference Verification (Polynomial)" begin
        println("\n[4] Verifying Polynomial forces with finite differences...")

        _, calc_unsplined, rcut = setup_splinified_model()

        poly_file = joinpath(build_dir, "poly_fd_test.jl")
        export_ace_model(calc_unsplined, poly_file; for_library=false, radial_basis=:polynomial)

        poly_mod = Module(:PolyFD)
        Base.include(poly_mod, poly_file)

        # Test configuration
        Z0 = 14
        Rs = [
            SVector(2.35, 0.0, 0.0),
            SVector(0.0, 2.35, 0.0),
            SVector(0.0, 0.0, 2.35),
            SVector(1.5, 1.5, 1.5),
        ]
        Zs = [14, 14, 14, 14]

        # Analytic forces
        _, F_analytic = poly_mod.site_energy_forces(Rs, Zs, Z0)

        # Finite difference
        h = 1e-6
        max_fd_error = 0.0
        for j in 1:length(Rs)
            F_fd = zeros(3)
            for α in 1:3
                e_α = zeros(3)
                e_α[α] = h

                Rs_p = copy(Rs)
                Rs_m = copy(Rs)
                Rs_p[j] = Rs[j] + SVector{3}(e_α)
                Rs_m[j] = Rs[j] - SVector{3}(e_α)

                Ep = poly_mod.site_energy(Rs_p, Zs, Z0)
                Em = poly_mod.site_energy(Rs_m, Zs, Z0)

                F_fd[α] = -(Ep - Em) / (2h)
            end

            fd_error = norm(F_fd - F_analytic[j])
            max_fd_error = max(max_fd_error, fd_error)

            @test fd_error < 1e-5
        end
        println("   Max FD error: $max_fd_error eV/Å")
        println("   Polynomial FD verification passed")
    end

    @testset "Random Perturbation Tests (Hermite)" begin
        println("\n[5] Testing Hermite export with random perturbations...")

        calc_splined, _, rcut = setup_splinified_model()

        hermite_file = joinpath(build_dir, "hermite_random_test.jl")
        export_ace_model(calc_splined, hermite_file; for_library=false, radial_basis=:hermite_spline)

        hermite_mod = Module(:HermiteRandom)
        Base.include(hermite_mod, hermite_file)

        rng = MersenneTwister(42)

        # Test 5 random configurations
        for trial in 1:5
            # Create perturbed system
            a0 = 5.43
            perturb = 0.1  # Angstrom
            positions = [
                SVector(0.0, 0.0, 0.0) + perturb * (rand(rng, 3) .- 0.5),
                SVector(a0/4, a0/4, a0/4) + perturb * (rand(rng, 3) .- 0.5),
                SVector(a0/2, a0/2, 0.0) + perturb * (rand(rng, 3) .- 0.5),
                SVector(a0/2, 0.0, a0/2) + perturb * (rand(rng, 3) .- 0.5),
            ]
            box = [SVector(a0, 0.0, 0.0), SVector(0.0, a0, 0.0), SVector(0.0, 0.0, a0)]

            sys_perturbed = AtomsBase.periodic_system(
                [:Si => pos * u"Å" for pos in positions],
                [b * u"Å" for b in box]
            )

            # Compare exported energy to reference
            E_ref = ustrip(u"eV", AtomsCalculators.potential_energy(sys_perturbed, calc_splined))
            E_hermite = compute_exported_energy(sys_perturbed, hermite_mod, rcut)

            @test E_hermite ≈ E_ref atol=1e-10 rtol=1e-8
        end
        println("   5 random perturbation tests passed")
    end

    println("\n" * "="^80)
    println("All Hermite Spline Accuracy Tests Passed!")
    println("="^80 * "\n")
end
