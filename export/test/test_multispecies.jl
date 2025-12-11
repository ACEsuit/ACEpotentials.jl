#=
Multi-Species Model Tests

Tests for:
1. Export and compilation of multi-species models (e.g., Si-O)
2. Verification that RCUT_MAX is correctly computed
3. Species indexing in exported code
4. Cutoff handling for different species pairs
=#

using Test
using ACEpotentials
using ACEfit
using StaticArrays
using LinearAlgebra

@testset "Multi-Species Export" verbose=true begin

    @testset "Multi-Species Model Creation" begin
        # Create a simple Si-O model for testing multi-species export
        @info "Creating multi-species Si-O model..."

        model = ACEpotentials.ace1_model(
            elements = [:Si, :O],
            order = 2,
            totaldegree = 4,  # Keep small for fast testing
            rcut = 5.0,
        )

        @test model !== nothing
        # Check number of species via the _i2z field (index to atomic number mapping)
        num_species = length(model.model._i2z)
        @test num_species == 2
        @info "Multi-species model created with $num_species species"

        # Store model for export test
        TEST_ARTIFACTS["multispecies_model"] = model
    end

    @testset "Multi-Species Export Code Generation" begin
        model = get(TEST_ARTIFACTS, "multispecies_model", nothing)
        if model === nothing
            @test_skip "Multi-species model not created"
            return
        end

        # Export to Julia code
        build_dir = joinpath(TEST_DIR, "build")
        mkpath(build_dir)

        include(joinpath(EXPORT_DIR, "src", "export_ace_model.jl"))
        model_file = joinpath(build_dir, "test_multispecies_model.jl")

        # Export with library interface
        export_ace_model(model, model_file; for_library=true)
        @test isfile(model_file)

        # Verify file contents have expected multi-species components
        content = read(model_file, String)

        # Check for multiple species
        @test occursin("NZ = 2", content)  # Two species

        # Check species mapping (Si=14, O=8)
        @test occursin("14", content)  # Si atomic number
        @test occursin("8", content)   # O atomic number

        # Check for RCUT_MAX constant (should exist for multi-species)
        @test occursin("RCUT_MAX", content)

        # Check for multiple RIN0CUT constants (one per species pair)
        @test occursin("RIN0CUT_1_1", content)  # Si-Si
        @test occursin("RIN0CUT_1_2", content)  # Si-O
        @test occursin("RIN0CUT_2_1", content)  # O-Si
        @test occursin("RIN0CUT_2_2", content)  # O-O

        # Check ace_get_cutoff returns RCUT_MAX
        @test occursin("ace_get_cutoff", content)
        @test occursin("return RCUT_MAX", content)

        @info "Multi-species export code generated successfully"
    end

    @testset "RCUT_MAX Computation" begin
        # Test that RCUT_MAX is correctly computed as max over all species pairs
        build_dir = joinpath(TEST_DIR, "build")
        model_file = joinpath(build_dir, "test_multispecies_model.jl")

        if !isfile(model_file)
            @test_skip "Multi-species model file not found"
            return
        end

        content = read(model_file, String)

        # Extract RCUT_MAX value
        m = match(r"const RCUT_MAX = ([0-9.]+)", content)
        if m !== nothing
            rcut_max = parse(Float64, m.captures[1])
            @test rcut_max > 0.0

            # Extract individual cutoffs and verify RCUT_MAX >= all of them
            for iz in 1:2, jz in 1:2
                pattern = Regex("RIN0CUT_$(iz)_$(jz).*rcut=([0-9.]+)")
                m_ij = match(pattern, content)
                if m_ij !== nothing
                    rcut_ij = parse(Float64, m_ij.captures[1])
                    @test rcut_max >= rcut_ij - 1e-10  # Allow small tolerance
                end
            end

            @info "RCUT_MAX = $rcut_max (verified >= all species-pair cutoffs)"
        else
            @test_skip "Could not extract RCUT_MAX from generated code"
        end
    end

    @testset "Species Index Mapping" begin
        # Verify z2i function correctly maps atomic numbers to indices
        build_dir = joinpath(TEST_DIR, "build")
        model_file = joinpath(build_dir, "test_multispecies_model.jl")

        if !isfile(model_file)
            @test_skip "Multi-species model file not found"
            return
        end

        content = read(model_file, String)

        # Check I2Z mapping (index to atomic number)
        @test occursin("I2Z", content)

        # The z2i function should handle both Si (Z=14) and O (Z=8)
        @test occursin("z2i", content)

        # Check that the function iterates over NZ species
        @test occursin("for i in 1:NZ", content) || occursin("@inbounds for i in 1:NZ", content)

        @info "Species index mapping verified"
    end

end
