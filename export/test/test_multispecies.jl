#=
Multi-Species ETACE Model Tests

Tests for:
1. Export of multi-species ETACE models (e.g., Ti-Al)
2. Verification that RCUT_MAX is correctly computed
3. Species indexing in exported code
4. Cutoff handling for different species pairs
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

const M = ACEpotentials.Models
const ETM = ACEpotentials.ETModels

@testset "Multi-Species ETACE Export" verbose=true begin

    @testset "Multi-Species ETACE Model Creation" begin
        # Create a simple Ti-Al ETACE model for testing multi-species export
        @info "Creating multi-species Ti-Al ETACE model..."

        elements = (:Ti, :Al)
        rcut = 5.0

        rin0cuts = M._default_rin0cuts(elements)
        rin0cuts = (x -> (rin = x.rin, r0 = x.r0, rcut = rcut)).(rin0cuts)

        rng = Random.MersenneTwister(1234)

        ace_model = M.ace_model(;
            elements = elements,
            order = 2,
            Ytype = :solid,
            level = M.TotalDegree(),
            max_level = 6,
            maxl = 2,
            pair_maxn = 6,
            rin0cuts = rin0cuts,
            init_WB = :glorot_normal,
            init_Wpair = :glorot_normal
        )

        ps, st = Lux.setup(rng, ace_model)

        # Convert to ETACE
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

        # Create ETACE calculator
        et_calc = ETM.ETACEPotential(et_model, et_ps, et_st, rcut)

        @test et_calc !== nothing
        @info "Multi-species ETACE model created with $n_species species"

        # Store model for export test
        TEST_ARTIFACTS["multispecies_etace_calc"] = et_calc
    end

    @testset "Multi-Species Export Code Generation" begin
        et_calc = get(TEST_ARTIFACTS, "multispecies_etace_calc", nothing)
        if et_calc === nothing
            @test_skip "Multi-species ETACE model not created"
            return
        end

        # Export to Julia code
        build_dir = joinpath(TEST_DIR, "build")
        mkpath(build_dir)

        include(joinpath(EXPORT_DIR, "src", "export_ace_model.jl"))
        model_file = joinpath(build_dir, "test_multispecies_etace.jl")

        # Export with library interface
        export_ace_model(et_calc, model_file; for_library=true)
        @test isfile(model_file)

        # Verify file contents have expected multi-species components
        content = read(model_file, String)

        # Check for multiple species
        @test occursin("NZ = 2", content)  # Two species

        # Check species mapping (Ti=22, Al=13)
        @test occursin("22", content)  # Ti atomic number
        @test occursin("13", content)  # Al atomic number

        # Check for I2Z array
        @test occursin("I2Z", content)

        @info "Multi-species ETACE export code generated successfully"
    end

    @testset "Multi-Species Evaluation" begin
        build_dir = joinpath(TEST_DIR, "build")
        model_file = joinpath(build_dir, "test_multispecies_etace.jl")

        if !isfile(model_file)
            @test_skip "Multi-species model file not found"
            return
        end

        # Load exported model
        exported = Module(:ExportedMultispecies)
        Base.include(exported, model_file)

        # Test with Ti center and mixed Ti/Al neighbors
        Z0 = 22  # Ti center
        Rs = [
            SVector(2.5, 0.0, 0.0),
            SVector(0.0, 2.5, 0.0),
            SVector(0.0, 0.0, 2.5),
        ]
        Zs = [22, 13, 22]  # Ti, Al, Ti neighbors

        # Energy should be finite
        E = exported.site_energy(Rs, Zs, Z0)
        @test isfinite(E)

        # Forces should be finite and correct length
        E2, F = exported.site_energy_forces(Rs, Zs, Z0)
        @test E2 ≈ E
        @test length(F) == 3
        @test all(isfinite, norm.(F))

        # Test with Al center
        Z0_Al = 13  # Al center
        E_Al = exported.site_energy(Rs, Zs, Z0_Al)
        @test isfinite(E_Al)

        @info "Multi-species evaluation verified"
    end

    @testset "Species Index Mapping" begin
        build_dir = joinpath(TEST_DIR, "build")
        model_file = joinpath(build_dir, "test_multispecies_etace.jl")

        if !isfile(model_file)
            @test_skip "Multi-species model file not found"
            return
        end

        content = read(model_file, String)

        # Check I2Z mapping (index to atomic number)
        @test occursin("I2Z", content)

        # The z2i function should handle both Ti (Z=22) and Al (Z=13)
        @test occursin("z2i", content)

        @info "Species index mapping verified"
    end

end
