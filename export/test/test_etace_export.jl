#=
ETACE Export Tests

Tests for exporting ETACE models to standalone Julia code.
Verifies that exported models produce identical results to
the ETACE AtomsCalculator for:
- Energies
- Forces
- Virials
- Analytic vs finite difference force accuracy
=#

using Test
using ACEpotentials
using ACEpotentials.Models
using ACEpotentials.ETModels
using StaticArrays
using LinearAlgebra
using Random
using Lux
using LuxCore
using AtomsCalculators
import AtomsBase
using Unitful

import EquivariantTensors as ET

# Test configuration
const EXPORT_DIR = dirname(@__DIR__)
const TEST_DIR = @__DIR__
const M = ACEpotentials.Models
const ETM = ACEpotentials.ETModels

"""
    setup_etace_model()

Create an ETACE model for testing. Returns (et_calc, et_model, et_ps, et_st, rcut).
"""
function setup_etace_model(; elements=(:Si,), order=2, max_level=8, maxl=4, rcut=5.5)
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

    # Convert to ETACE
    et_model = ETM.convert2et(ace_model)
    et_ps, et_st = LuxCore.setup(MersenneTwister(1234), et_model)

    # Copy radial basis parameters
    n_species = length(elements)
    for iz in 1:n_species
        for jz in 1:n_species
            et_ps.rembed.post.W[:, :, (iz-1)*n_species + jz] .= ps.rbasis.Wnlq[:, :, iz, jz]
        end
    end

    # Copy readout parameters
    for iz in 1:n_species
        et_ps.readout.W[1, :, iz] .= ps.WB[:, iz]
    end

    # Create calculator
    et_calc = ETM.ETACEPotential(et_model, et_ps, et_st, rcut)

    return et_calc, et_model, et_ps, et_st, rcut
end

"""
    setup_stacked_model(; kwargs...) -> (stacked, rcut)

The model the **compiled library** (`build/test_etace_model.jl` -> `libace_test.so`) is
exported from: a full `StackedCalculator` of `ETOneBody + ETPairModel + ETACE`, built with
`ETModels.convert2et_full`.

WHY THIS EXISTS, AND WHY THE LIBRARY MODEL IS NOT `setup_etace_model`'s BARE ETACE.
Until Task 3's fix round the library was exported from a bare `ETACEPotential`, so the
generated file carried `# PAIR POTENTIAL: none in this model` and a `pair_energy_d` that
returned a hard `(0.0, 0.0)`.  Every downstream gate that runs against that library --
the 1e-10 LAMMPS gate, the 1e-12 Python-C-API gate, the two-rank gate, the whole
`test_python.jl` group and the ase-ace jobs -- therefore exercised a many-body-only model.
A sign error or a wrong per-pair cutoff in the generated pair code would have left all of
them green, and the only check that would have caught it (`test_pair_export.jl`) is
Julia-only AND guarded on host-local fixture data, so it never runs on a hosted CI runner.

Nothing about the pair term needs fitted data, so the fix costs nothing: random parameters
and a hard-coded `E0s` are enough to put the pair path under every one of those gates.

`E0s` is not decoration either -- `convert2et_full` reads `model.Vref.E0` to build the
`ETOneBody` term, and the exported site energy includes E0 of the central atom.
"""
function setup_stacked_model(; elements=(:Si,), order=2, max_level=8, maxl=4, rcut=5.5,
                               E0s=Dict(:Si => -5.0))
    rin0cuts = M._default_rin0cuts(elements)
    rin0cuts = (x -> (rin = x.rin, r0 = x.r0, rcut = rcut)).(rin0cuts)

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
        init_Wpair = :glorot_normal,
        pair_learnable = true,
        E0s = E0s,
    )
    ps, st = Lux.setup(Random.MersenneTwister(1234), ace_model)
    stacked = ETM.convert2et_full(ace_model, ps, st; rng = Random.MersenneTwister(1234))
    return stacked, rcut
end

"""
    create_test_system(a0=5.43)

Create a periodic silicon test system.
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
    compute_exported_forces(G, exported_module, n_atoms, Z0)

Compute forces from exported model with proper site→system accumulation.
"""
function compute_exported_forces(G, exported_module, n_atoms, Z0)
    F_exported = [zeros(SVector{3, Float64}) for _ in 1:n_atoms]

    for i in 1:n_atoms
        neighbor_Rs = SVector{3, Float64}[]
        neighbor_Zs = Int[]
        neighbor_global_indices = Int[]

        for (edge_idx, edge) in enumerate(G.edge_data)
            if G.ii[edge_idx] == i
                push!(neighbor_Rs, SVector{3, Float64}(edge.𝐫))
                push!(neighbor_Zs, Int(edge.z1.atomic_number))
                push!(neighbor_global_indices, G.jj[edge_idx])
            end
        end

        if !isempty(neighbor_Rs)
            _, F_site = exported_module.site_energy_forces(neighbor_Rs, neighbor_Zs, Z0)

            for (j_local, f) in enumerate(F_site)
                j_global = neighbor_global_indices[j_local]
                F_exported[i] += -f
                F_exported[j_global] += f
            end
        end
    end

    return F_exported
end

"""
    compute_exported_virial(G, exported_module, n_atoms, Z0)

Compute virial from exported model.
"""
function compute_exported_virial(G, exported_module, n_atoms, Z0)
    V_total = zeros(SMatrix{3, 3, Float64, 9})

    for i in 1:n_atoms
        neighbor_Rs = SVector{3, Float64}[]
        neighbor_Zs = Int[]

        for (edge_idx, edge) in enumerate(G.edge_data)
            if G.ii[edge_idx] == i
                push!(neighbor_Rs, SVector{3, Float64}(edge.𝐫))
                push!(neighbor_Zs, Int(edge.z1.atomic_number))
            end
        end

        if !isempty(neighbor_Rs)
            _, _, V_i = exported_module.site_energy_forces_virial(neighbor_Rs, neighbor_Zs, Z0)
            V_total += V_i
        end
    end

    return V_total
end

@testset "ETACE Export" verbose=true begin

    @testset "Model Export" begin
        # Create ETACE model
        et_calc, et_model, et_ps, et_st, rcut = setup_etace_model()

        # Export to file
        build_dir = joinpath(TEST_DIR, "build")
        mkpath(build_dir)

        include(joinpath(EXPORT_DIR, "src", "export_ace_model.jl"))
        model_file = joinpath(build_dir, "test_etace_model.jl")

        # Export with library interface.  This is the file juliac compiles into
        # libace_test.so, i.e. the model behind EVERY gate that goes through the compiled
        # library: the 1e-10 LAMMPS gate, the 1e-12 Python-C-API gate, the two-rank gate,
        # test_python.jl and the ase-ace jobs.  It is therefore exported from the FULL stack
        # (E0 + pair + many-body), not from the bare ETACE -- see setup_stacked_model.
        stacked, _ = setup_stacked_model()
        export_ace_model(stacked, model_file; for_library=true)
        @test isfile(model_file)

        # Verify file contents
        content = read(model_file, String)
        @test occursin("I2Z", content)
        @test occursin("ABASIS_SPEC", content)  # Tensor specification (replaces old "TENSOR")
        @test occursin("eval_ylm", content)  # Inline solid harmonics (replaces SpheriCart import)
        @test occursin("POLY_A", content)  # Orthonormalized poly coefficients
        @test occursin("ace_site_energy", content)

        # The pair term and E0 must really be in the library model.  Without these three the
        # gates listed above silently degrade to many-body-only checks, which is exactly what
        # they had been doing.
        @test occursin("const PAIR_C", content)
        @test occursin("PAIR POTENTIAL: none in this model", content) == false
        @test occursin("const E0_1", content)

        # Also test standalone export
        exe_file = joinpath(build_dir, "test_etace_exe.jl")
        export_ace_model(et_calc, exe_file; for_library=false)
        @test isfile(exe_file)
        @test occursin("@main", read(exe_file, String))
    end

    @testset "Energy Correctness" begin
        et_calc, et_model, et_ps, et_st, rcut = setup_etace_model()
        sys = create_test_system()

        # Export and load
        build_dir = joinpath(TEST_DIR, "build")
        include(joinpath(EXPORT_DIR, "src", "export_ace_model.jl"))
        model_file = joinpath(build_dir, "test_etace_energy.jl")
        export_ace_model(et_calc, model_file; for_library=false)

        # Create a module to isolate the exported code
        exported = Module(:ExportedModel)
        Base.include(exported, model_file)

        # Get ETACE energy
        E_ac = AtomsCalculators.potential_energy(sys, et_calc)
        E_ac_val = ustrip(u"eV", E_ac)

        # Compute exported energy
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
            E_i = isempty(neighbor_Rs) ? 0.0 : exported.site_energy(neighbor_Rs, neighbor_Zs, Z0)
            E_exported += E_i
        end

        @test E_exported ≈ E_ac_val atol=1e-10
    end

    @testset "Force Correctness" begin
        et_calc, et_model, et_ps, et_st, rcut = setup_etace_model()
        sys = create_test_system()

        # Export and load
        build_dir = joinpath(TEST_DIR, "build")
        include(joinpath(EXPORT_DIR, "src", "export_ace_model.jl"))
        model_file = joinpath(build_dir, "test_etace_forces.jl")
        export_ace_model(et_calc, model_file; for_library=false)

        exported = Module(:ExportedModel)
        Base.include(exported, model_file)

        # Get ETACE forces
        _, F_ac, _ = AtomsCalculators.energy_forces_virial(sys, et_calc)
        F_ac_val = [SVector{3}(ustrip.(u"eV/Å", f)) for f in F_ac]

        # Compute exported forces with proper accumulation
        G = ET.Atoms.interaction_graph(sys, rcut * u"Å")
        n_atoms = length(sys)
        Z0 = 14

        F_exported = compute_exported_forces(G, exported, n_atoms, Z0)

        # Compare
        for i in 1:n_atoms
            @test F_exported[i] ≈ F_ac_val[i] atol=1e-10
        end
    end

    @testset "Virial Correctness" begin
        et_calc, et_model, et_ps, et_st, rcut = setup_etace_model()
        sys = create_test_system()

        # Export and load
        build_dir = joinpath(TEST_DIR, "build")
        include(joinpath(EXPORT_DIR, "src", "export_ace_model.jl"))
        model_file = joinpath(build_dir, "test_etace_virial.jl")
        export_ace_model(et_calc, model_file; for_library=false)

        exported = Module(:ExportedModel)
        Base.include(exported, model_file)

        # Get ETACE virial
        _, _, V_ac = AtomsCalculators.energy_forces_virial(sys, et_calc)
        V_ac_val = ustrip.(u"eV", V_ac)

        # Compute exported virial
        G = ET.Atoms.interaction_graph(sys, rcut * u"Å")
        n_atoms = length(sys)
        Z0 = 14

        V_exported = compute_exported_virial(G, exported, n_atoms, Z0)

        # Compare
        @test maximum(abs.(collect(V_exported) - collect(V_ac_val))) < 1e-10
    end

    @testset "Finite Difference Verification" begin
        et_calc, et_model, et_ps, et_st, rcut = setup_etace_model()

        # Export and load
        build_dir = joinpath(TEST_DIR, "build")
        include(joinpath(EXPORT_DIR, "src", "export_ace_model.jl"))
        model_file = joinpath(build_dir, "test_etace_fd.jl")
        export_ace_model(et_calc, model_file; for_library=false)

        exported = Module(:ExportedModel)
        Base.include(exported, model_file)

        # Test with random neighbor configuration
        Z0 = 14
        Rs = [
            SVector(2.35, 0.0, 0.0),
            SVector(0.0, 2.35, 0.0),
            SVector(0.0, 0.0, 2.35),
        ]
        Zs = [14, 14, 14]

        # Get analytic forces
        _, F_analytic = exported.site_energy_forces(Rs, Zs, Z0)

        # Finite difference
        h = 1e-6
        for j in 1:length(Rs)
            F_fd = zeros(3)
            for α in 1:3
                e_α = zeros(3)
                e_α[α] = h

                Rs_p = copy(Rs)
                Rs_m = copy(Rs)
                Rs_p[j] = Rs[j] + SVector{3}(e_α)
                Rs_m[j] = Rs[j] - SVector{3}(e_α)

                Ep = exported.site_energy(Rs_p, Zs, Z0)
                Em = exported.site_energy(Rs_m, Zs, Z0)

                F_fd[α] = -(Ep - Em) / (2h)
            end

            err = norm(F_fd - F_analytic[j])
            @test err < 1e-7  # Should match to ~h^2 precision
        end
    end

    @testset "Multi-species Export" begin
        # Test with TiAl system
        et_calc, et_model, et_ps, et_st, rcut = setup_etace_model(
            elements = (:Ti, :Al),
            order = 2,
            max_level = 6,
            maxl = 3,
            rcut = 5.0
        )

        # Export
        build_dir = joinpath(TEST_DIR, "build")
        include(joinpath(EXPORT_DIR, "src", "export_ace_model.jl"))
        model_file = joinpath(build_dir, "test_etace_multispecies.jl")
        export_ace_model(et_calc, model_file; for_library=false)

        # Verify content
        content = read(model_file, String)
        @test occursin("I2Z = [22, 13]", content)  # Ti, Al atomic numbers
        @test occursin("NZ = 2", content)

        # Load and test basic evaluation
        exported = Module(:ExportedModel)
        Base.include(exported, model_file)

        # Test with mixed species neighbors
        Z0 = 22  # Ti center
        Rs = [SVector(2.5, 0.0, 0.0), SVector(0.0, 2.5, 0.0)]
        Zs = [22, 13]  # Ti and Al neighbors

        E = exported.site_energy(Rs, Zs, Z0)
        @test isfinite(E)

        E2, F = exported.site_energy_forces(Rs, Zs, Z0)
        @test E2 ≈ E
        @test length(F) == 2
        @test all(isfinite, norm.(F))
    end
end
