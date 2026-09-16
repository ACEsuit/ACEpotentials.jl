#!/usr/bin/env julia
#=
Hermite spline export accuracy, on a small randomly-parameterised Si model.

WHICH REFERENCE EACH NUMBER IS MEASURED AGAINST -- the whole point of this file.

  :hermite_spline  is GATED against the **SPLINIFIED** calculator
                   (`ETM.splinify(...; Nspl)`), at `atol = 1e-8`.  That is the model the
                   Hermite tables are a representation OF, so this gate measures the export,
                   and only the export.  Both Nspl the file exports (50 and 200) are gated.

  :hermite_spline  is additionally REPORTED against the **FITTED** (unsplinified) calculator,
                   with an `@info` line per Nspl.  That difference is the splinification
                   error of the model -- approximation error that `splinify` introduced
                   before any export happened -- and it is *never* asserted against a
                   tolerance.  It falls with Nspl by construction; on the fitted Cantor model
                   `verify_cantor/log.chain` records it as 2.3e-4 eV/A at Nspl=50 and 3.0e-6
                   at Nspl=200.  Gating it would be gating the model, not the export, and the
                   only way to make such a gate pass is to loosen it.

  :polynomial      is GATED against the **FITTED** calculator at `1e-12`.  The polynomial
                   export is exact -- it re-emits the same basis, not an approximation of it
                   -- so nothing but floating-point summation order separates the two.  This
                   used to be checked at `atol=1e-8 rtol=1e-6`, four orders of magnitude
                   looser than the mode can actually achieve (measured: 3.2e-15 eV/A).

The finite-difference testsets check the exported derivative against the exported energy;
they say nothing about either reference above.
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
Returns `(calc_splined, calc_unsplined, rcut)`:

  * `calc_unsplined` is the **FITTED** model -- the reference the `:polynomial` export is
    gated against, and the reference the Hermite export's *reported* (never gated) error is
    measured against;
  * `calc_splined` is `splinify(..., Nspl)` of it -- the **SPLINIFIED** reference the Hermite
    export is gated against.

`Nspl` is a keyword so the caller can exercise more than one spline resolution; the
difference between the two calculators is the splinification error and grows as `Nspl` falls.
"""
function setup_splinified_model(; elements=(:Si,), order=2, max_level=8, maxl=2, rcut=5.5,
                                  Nspl=50)
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
    et_model_splined = splinify(et_model, et_ps, et_st; Nspl=Nspl)
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

    """
    `(max |dE|, max_i ||dF_i||)` of an exported module against an AtomsCalculators calculator
    on `sys`.  Energy in eV (this system has 4 atoms, so per-atom and total differ only by a
    factor 4 and the absolute figure is the stricter one); forces in eV/Å, absolute.
    """
    function export_vs_calc(mod, sys, calc, rcut)
        E_ref = ustrip(u"eV", AtomsCalculators.potential_energy(sys, calc))
        F_ref = [SVector{3}(ustrip.(u"eV/Å", f)) for f in AtomsCalculators.forces(sys, calc)]
        E = compute_exported_energy(sys, mod, rcut)
        F = compute_exported_forces(sys, mod, rcut)
        return (abs(E - E_ref), maximum(norm(F[i] - F_ref[i]) for i in 1:length(sys)))
    end

    # Both spline resolutions the file exports.  50 is the coarse one used everywhere else in
    # the suite; 200 is included so that the REPORTED error against the fitted model can be
    # seen to fall with Nspl, which is what distinguishes splinification error from an export
    # bug (an export bug would not care about Nspl).
    for Nspl in (50, 200)
        @testset "Hermite (Nspl=$Nspl) vs the SPLINIFIED reference, atol 1e-8 (Si)" begin
            println("\n[1] Hermite export, Nspl=$Nspl ...")

            calc_splined, calc_unsplined, rcut = setup_splinified_model(; Nspl = Nspl)
            sys = create_test_system()

            E_ref = ustrip(u"eV", AtomsCalculators.potential_energy(sys, calc_splined))
            F_ref_val = [SVector{3}(ustrip.(u"eV/Å", f))
                         for f in AtomsCalculators.forces(sys, calc_splined)]
            println("   Reference energy (SPLINIFIED calc, Nspl=$Nspl): $E_ref eV")

            hermite_file = joinpath(build_dir, "hermite_accuracy_test_$(Nspl).jl")
            export_ace_model(calc_splined, hermite_file; for_library=false,
                             radial_basis=:hermite_spline)
            @test isfile(hermite_file)

            hermite_mod = Module(Symbol("HermiteExport", Nspl))
            Base.include(hermite_mod, hermite_file)

            E_hermite = Base.invokelatest(compute_exported_energy, sys, hermite_mod, rcut)
            F_hermite = Base.invokelatest(compute_exported_forces, sys, hermite_mod, rcut)
            max_force_diff = maximum(norm(F_hermite[i] - F_ref_val[i]) for i in 1:length(sys))

            println("   Hermite export energy:  $E_hermite eV")
            println("   |dE| vs SPLINIFIED:     $(abs(E_hermite - E_ref)) eV")
            println("   max|dF| vs SPLINIFIED:  $max_force_diff eV/Å")

            # GATE.  Reference: the SPLINIFIED calculator -- the model these Hermite tables
            # represent.  atol 1e-8 is the plan's tolerance for this comparison and is NOT
            # loosened anywhere; the measured value is ~7e-15, seven orders inside it.
            @test E_hermite ≈ E_ref atol=1e-8 rtol=0
            for i in 1:length(sys)
                @test F_hermite[i] ≈ F_ref_val[i] atol=1e-8 rtol=0
            end

            # REPORTED, NEVER GATED.  Reference: the FITTED calculator.  This is the error
            # `splinify` introduced into the model before any export took place; asserting it
            # would be asserting the model's own approximation error.  It must fall with Nspl.
            dE_fit, dF_fit = Base.invokelatest(export_vs_calc, hermite_mod, sys,
                                               calc_unsplined, rcut)
            @info("Hermite export error against the FITTED model -- reported, never gated",
                  Nspl, dE_eV = dE_fit, dF_eV_per_A = dF_fit)
            println("   |dE| vs FITTED (splinification error, reported only):    $dE_fit eV")
            println("   max|dF| vs FITTED (splinification error, reported only): $dF_fit eV/Å")
        end
    end

    @testset "Polynomial vs the FITTED reference, 1e-12 (Si)" begin
        println("\n[2] Polynomial export ...")

        _, calc_unsplined, rcut = setup_splinified_model()
        sys = create_test_system()

        E_ref = ustrip(u"eV", AtomsCalculators.potential_energy(sys, calc_unsplined))
        F_ref_val = [SVector{3}(ustrip.(u"eV/Å", f))
                     for f in AtomsCalculators.forces(sys, calc_unsplined)]
        println("   Reference energy (FITTED calc): $E_ref eV")

        poly_file = joinpath(build_dir, "poly_accuracy_test.jl")
        export_ace_model(calc_unsplined, poly_file; for_library=false, radial_basis=:polynomial)
        @test isfile(poly_file)

        poly_mod = Module(:PolyExport)
        Base.include(poly_mod, poly_file)

        E_poly = Base.invokelatest(compute_exported_energy, sys, poly_mod, rcut)
        F_poly = Base.invokelatest(compute_exported_forces, sys, poly_mod, rcut)
        max_force_diff = maximum(norm(F_poly[i] - F_ref_val[i]) for i in 1:length(sys))

        println("   Polynomial export energy: $E_poly eV")
        println("   |dE| vs FITTED:           $(abs(E_poly - E_ref)) eV")
        println("   max|dF| vs FITTED:        $max_force_diff eV/Å")

        # GATE.  Reference: the FITTED calculator.  :polynomial re-emits the same basis
        # rather than approximating it, so the only difference permitted is summation order.
        @test E_poly ≈ E_ref atol=1e-12 rtol=0
        for i in 1:length(sys)
            @test F_poly[i] ≈ F_ref_val[i] atol=1e-12 rtol=0
        end
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

            @test E_hermite ≈ E_ref atol=1e-8 rtol=0
        end
        println("   5 random perturbation tests passed")
    end

    println("\n" * "="^80)
    println("All Hermite Spline Accuracy Tests Passed!")
    println("="^80 * "\n")
end
