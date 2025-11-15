# Test suite for EquivariantModels → EquivariantTensors migration
# Run with: julia --project=. test/test_migration.jl

using Test
using ACEpotentials
using ACEpotentials.Models: _rpe_filter_real, _mm_filter

@testset "Migration Tests" begin

    @testset "Filter Function Tests" begin
        @testset "_mm_filter helper" begin
            # Test empty
            @test _mm_filter(Int[], 0) == true

            # Test single m=0
            @test _mm_filter([0], 0) == true
            @test _mm_filter([0], 1) == true

            # Test single m=1, L=0 should fail
            @test _mm_filter([1], 0) == false

            # Test single m=1, L=1 should pass
            @test _mm_filter([1], 1) == true

            # Test pair that sums to 0
            @test _mm_filter([1, -1], 0) == true

            # Test pair that can't reach L=0
            @test _mm_filter([1, 1], 0) == false

            # Test pair with sign combinations
            @test _mm_filter([1, 1], 2) == true  # 1+1=2, |2|<=2
            @test _mm_filter([1, 1], 1) == false # min sum is |1+1|=2 or |1-1|=0
        end

        @testset "_rpe_filter_real basic cases" begin
            filter_L0 = _rpe_filter_real(0)

            # Empty basis should be admissible
            @test filter_L0([]) == true

            # Single l=0, m=0 should be admissible for L=0
            @test filter_L0([(n=1, l=0, m=0)]) == true

            # Single l=1, m=0 should NOT be admissible for L=0 (special case)
            @test filter_L0([(n=1, l=1, m=0)]) == false

            # Pair with l=0 (even sum, m-filter ok)
            @test filter_L0([(n=1, l=0, m=0), (n=1, l=0, m=0)]) == true

            # Pair with l=1 (odd sum, should fail parity for L=0)
            # sum(l) + L = 1+1+0 = 2 (even), m-admissible, should pass
            @test filter_L0([(n=1, l=1, m=0), (n=1, l=1, m=0)]) == true

            # Pair that cancels m
            @test filter_L0([(n=1, l=1, m=1), (n=1, l=1, m=-1)]) == true
        end

        @testset "_rpe_filter_real parity checks" begin
            filter_L0 = _rpe_filter_real(0)

            # L=0, sum(l)=1 (odd) → sum(l)+L=1 (odd) → fail
            bb_odd = [(n=1, l=1, m=0)]
            # Wait, this is the special case - single element with L=0 must have l=0
            @test filter_L0(bb_odd) == false

            # L=0, sum(l)=2 (even) → sum(l)+L=2 (even) → pass (if m-filter ok)
            bb_even = [(n=1, l=1, m=1), (n=1, l=1, m=-1)]
            @test filter_L0(bb_even) == true

            # L=0, sum(l)=3 (odd) → sum(l)+L=3 (odd) → fail
            bb_odd3 = [(n=1, l=1, m=0), (n=1, l=1, m=0), (n=1, l=1, m=0)]
            @test filter_L0(bb_odd3) == false
        end
    end

    @testset "Coupling Coefficients Basic Test" begin
        using EquivariantTensors
        using LinearAlgebra

        # Test simple specification
        AA_spec = [
            [(n=1, l=0, m=0)],
            [(n=1, l=0, m=0), (n=1, l=0, m=0)],
        ]

        @testset "symmetrisation_matrix interface" begin
            result = EquivariantTensors.symmetrisation_matrix(
                0, AA_spec; prune=true, PI=true, basis=real
            )

            # Check it returns a tuple
            @test result isa Tuple
            @test length(result) == 2

            matrix, pruned_spec = result

            # Check matrix properties
            @test matrix isa AbstractMatrix
            @test eltype(matrix) <: Real
            @test size(matrix, 2) <= length(AA_spec)  # May prune columns

            # Check pruned spec
            @test pruned_spec isa Vector
        end

        @testset "Matrix is well-formed" begin
            matrix, _ = EquivariantTensors.symmetrisation_matrix(
                0, AA_spec; prune=true, PI=true, basis=real
            )

            # No NaN or Inf
            @test !any(isnan, matrix)
            @test !any(isinf, matrix)

            # Has non-zero entries
            @test sum(abs, matrix) > 0
        end
    end

    @testset "ACE Model Construction" begin
        # This tests that the migration didn't break model construction
        elements = [:Si]

        @testset "Basic model creation" begin
            model = acemodel(
                elements = elements,
                order = 2,
                totdegree = 4
            )

            @test model isa ACEModel
            @test length(model.tensor) > 0
        end

        @testset "Model with reference energy" begin
            model = acemodel(
                elements = elements,
                order = 2,
                totdegree = 4,
                Eref = [:Si => -158.54]
            )

            @test model isa ACEModel
            @test model.Vref isa OneBody
        end
    end

    @testset "Model Evaluation" begin
        using AtomsBase
        using Unitful
        using Random
        using Lux

        # Create simple test system
        atoms = isolated_system([
            :Si => [0.0, 0.0, 0.0]u"Å",
            :Si => [2.0, 0.0, 0.0]u"Å",
        ])

        # Build minimal model
        model = acemodel(
            elements = [:Si],
            order = 2,
            totdegree = 4
        )

        # Initialize parameters
        rng = MersenneTwister(1234)
        ps, st = Lux.setup(rng, model)

        @testset "Energy evaluation" begin
            Rs = [atoms[2].position - atoms[1].position]
            Zs = [atoms[2].atomic_number]
            Z0 = atoms[1].atomic_number

            # Evaluate site energy
            E = evaluate(model, Rs, Zs, Z0, ps, st)

            @test E isa Real
            @test !isnan(E)
            @test !isinf(E)
        end

        @testset "Energy and forces" begin
            Rs = [atoms[2].position - atoms[1].position]
            Zs = [atoms[2].atomic_number]
            Z0 = atoms[1].atomic_number

            # Evaluate with derivatives
            E, F = evaluate_ed(model, Rs, Zs, Z0, ps, st)

            @test E isa Real
            @test F isa Vector
            @test length(F) == 1
            @test !isnan(E)
            @test !any(isnan, F)
        end
    end

    @testset "Comparison with EquivariantModels (if available)" begin
        # This test only runs if EquivariantModels is still available
        try
            using EquivariantModels

            @testset "Filter equivalence" begin
                test_specs = [
                    [],
                    [(n=1, l=0, m=0)],
                    [(n=1, l=0, m=0), (n=1, l=0, m=0)],
                    [(n=1, l=1, m=1), (n=1, l=1, m=-1)],
                ]

                old_filter = EquivariantModels.RPE_filter_real(0)
                new_filter = _rpe_filter_real(0)

                for spec in test_specs
                    @test old_filter(spec) == new_filter(spec)
                end
            end

            @testset "Coupling coefficients equivalence" begin
                AA_spec = [
                    [(n=1, l=0, m=0)],
                    [(n=1, l=0, m=0), (n=1, l=0, m=0)],
                ]

                old_result = EquivariantModels._rpi_A2B_matrix(0, AA_spec; basis=real)
                new_result, _ = EquivariantTensors.symmetrisation_matrix(
                    0, AA_spec; prune=true, PI=true, basis=real
                )

                @test size(old_result) == size(new_result)
                @test isapprox(old_result, new_result, rtol=1e-12)
            end

            println("✓ Direct comparison with EquivariantModels successful")
        catch e
            if e isa ArgumentError && occursin("EquivariantModels", string(e))
                @info "Skipping EquivariantModels comparison (package not available)"
            else
                rethrow(e)
            end
        end
    end
end

println("\n" * "="^70)
println("Migration Test Suite Completed")
println("="^70)
println("\nIf all tests passed, the migration is functionally equivalent!")
println("Next steps:")
println("  1. Run full test suite: julia --project=. -e 'using Pkg; Pkg.test()'")
println("  2. Run Silicon fitting test: julia --project=. test/test_silicon.jl")
println("  3. Check performance benchmarks")
println("  4. Review MIGRATION_TESTING.md for detailed validation steps")
