#!/usr/bin/env julia
#=
Export accuracy, on a small randomly-parameterised Si model.

WAS `test_hermite_accuracy.jl`.  The `:hermite_spline` radial mode was removed (evidence and
the maintainer's answer: `export/bench/FINDINGS_parity.md` §7), so this file lost its three
Hermite testsets -- the two-Nspl gate against the splinified reference, the Hermite
finite-difference check and the Hermite random-perturbation check.  It was KEPT, and renamed,
because the rest of it was never about splines: the `:polynomial` gate against the fitted
model is the strictest small-model check in the suite, and nothing else runs it.

WHICH REFERENCE EACH NUMBER IS MEASURED AGAINST -- still the point of this file.

  :polynomial      is GATED against the **FITTED** calculator at `1e-12`.  The polynomial
                   export is exact -- it re-emits the same basis, not an approximation of it
                   -- so nothing but floating-point summation order separates the two.  This
                   used to be checked at `atol=1e-8 rtol=1e-6`, four orders of magnitude
                   looser than the mode can actually achieve (measured: 3.2e-15 eV/A).

  the REFUSALS     are checked against no model at all: a splinified model can no longer be
                   exported by any route, and `radial_basis=:hermite_spline` raises.  Both
                   messages are asserted to name the workflow that works, because a removal
                   message that only says "that is gone" costs the reader the same search
                   twice.  This is the only place the splinified-model refusal is checked
                   since `test_hermite_spline_export.jl` was deleted, and that file's step
                   [6a] is where the assertion comes from.

The finite-difference testset checks the exported derivative against the exported energy; it
says nothing about either reference above.
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
Create an ETACE model for testing, and a splinified copy of it.
Returns `(calc_splined, calc_unsplined, rcut)`:

  * `calc_unsplined` is the **FITTED** model -- the reference the `:polynomial` export is
    gated against, and the only one of the two that can be exported at all;
  * `calc_splined` is `splinify(..., Nspl)` of it.  It exists ONLY so that the refusal
    testset has a splinified model to be refused.  `splinify` itself is untouched by the
    removal of the spline export; it is still a callable, working function, which is exactly
    why the exporter has to refuse its output loudly rather than rely on nobody producing it.
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

@testset "Export Accuracy (:polynomial)" verbose=true begin

    println("\n" * "="^80)
    println("Testing export accuracy (:polynomial) and the removed spline path's refusals")
    println("="^80)

    # Setup
    build_dir = joinpath(TEST_DIR, "build")
    mkpath(build_dir)

    @testset "Polynomial vs the FITTED reference, 1e-12 (Si)" begin
        println("\n[1] Polynomial export ...")

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

    @testset "Finite Difference Verification (Polynomial)" begin
        println("\n[2] Verifying Polynomial forces with finite differences...")

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

    @testset "Random Perturbation Tests (Polynomial)" begin
        println("\n[3] Testing the :polynomial export with random perturbations...")

        # Was the Hermite random-perturbation testset, at atol 1e-8 against the SPLINIFIED
        # calculator.  Re-pointed at the FITTED calculator and TIGHTENED to the 1e-12 the
        # exact mode actually achieves -- the 1e-8 was the spline interpolation's tolerance,
        # and carrying it over to a mode that is four orders better would have been a gate
        # that could not fail.
        _, calc_unsplined, rcut = setup_splinified_model()

        poly_file = joinpath(build_dir, "poly_random_test.jl")
        export_ace_model(calc_unsplined, poly_file; for_library=false,
                         radial_basis=:polynomial)

        poly_mod = Module(:PolyRandom)
        Base.include(poly_mod, poly_file)

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
            E_ref = ustrip(u"eV", AtomsCalculators.potential_energy(sys_perturbed,
                                                                    calc_unsplined))
            E_poly = Base.invokelatest(compute_exported_energy, sys_perturbed, poly_mod, rcut)

            @test E_poly ≈ E_ref atol=1e-12 rtol=0
        end
        println("   5 random perturbation tests passed")
    end

    @testset "the removed spline export path refuses, and says what to do" begin
        println("\n[4] :hermite_spline and splinified models must be REFUSED...")

        calc_splined, calc_unsplined, _ = setup_splinified_model()

        # (a) the keyword.  It is kept, accepting :polynomial only, because every existing
        # call site passes it; asking for the removed mode must raise rather than silently
        # give the caller a different mode from the one they named.
        f = joinpath(build_dir, "refused_hermite_keyword.jl")
        isfile(f) && rm(f)
        e1 = try
            export_ace_model(calc_unsplined, f; for_library=false,
                             radial_basis=:hermite_spline)
            nothing
        catch e
            e
        end
        @test e1 isa ErrorException
        m1 = sprint(showerror, e1)
        @test occursin("REMOVED", m1)
        @test occursin("radial_basis=:polynomial", m1)   # the remedy, named
        @test isfile(f) == false                          # refused before any file was opened
        println("   ✓ radial_basis=:hermite_spline refused, no file written")

        # (b) an unknown mode still raises its own error, not the removal one.
        e2 = try
            export_ace_model(calc_unsplined, f; for_library=false, radial_basis=:banana)
            nothing
        catch e
            e
        end
        @test e2 isa ErrorException
        @test occursin("unknown radial_basis", sprint(showerror, e2))

        # (c) the splinified model.  Inherited from test_hermite_spline_export.jl step [6a],
        # which used to check that :polynomial was refused rather than PROMOTED to the spline
        # mode.  Now there is no mode to promote to and nothing exports such a model at all,
        # so what the message has to carry is the retirement of the fit-on-splines route and
        # the workflow that replaces it.
        g = joinpath(build_dir, "refused_splinified.jl")
        isfile(g) && rm(g)
        e3 = try
            export_ace_model(calc_splined, g; for_library=false)   # the DEFAULT mode
            nothing
        catch e
            e
        end
        @test e3 isa ErrorException
        m3 = sprint(showerror, e3)
        @test occursin("already been splinified", m3)
        @test occursin("BEFORE splinify()", m3)                  # the workflow that works
        @test occursin("FIT-ON-SPLINES DEPLOYMENT ROUTE IS THEREFORE RETIRED", m3)
        @test isfile(g) == false
        println("   ✓ splinified model refused in the default mode, no file written")
    end

    println("\n" * "="^80)
    println("All export accuracy tests passed!")
    println("="^80 * "\n")
end
