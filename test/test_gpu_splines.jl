# Test GPU splines against Interpolations.jl reference

using Pkg; Pkg.activate(joinpath(@__DIR__(), ".."))

##

using ACEpotentials
M = ACEpotentials.Models

using Interpolations
using StaticArrays
using LinearAlgebra
using Lux
using Test

using Random
Random.seed!(1234)

##

println("Testing GPUCubicSpline...")

@testset "GPUCubicSpline" begin

    @testset "scalar spline conversion" begin
        println("\n  Testing scalar spline conversion...")

        # Create a test function and spline it
        f(x) = sin(2π * x) + 0.5 * cos(4π * x)
        x_nodes = range(-1.0, 1.0, length=50)
        y_nodes = f.(x_nodes)

        # Create Interpolations.jl spline
        itp = cubic_spline_interpolation(x_nodes, y_nodes)

        # Convert to GPUCubicSpline
        gpu_spl = M.GPUCubicSpline(itp)

        println("    GPU spline: ", gpu_spl)

        # Test at random points
        x_test = -1.0 .+ 2.0 * rand(100)
        max_err = 0.0

        for x in x_test
            y_ref = itp(x)
            y_gpu = gpu_spl(x)
            err = abs(y_ref - y_gpu)
            max_err = max(max_err, err)
        end

        println("    Max error (scalar): ", max_err)
        @test max_err < 1e-9  # Hermite reconstruction has ~1e-10 error
    end

    @testset "vector spline conversion" begin
        println("\n  Testing vector (SVector) spline conversion...")

        # Create multi-output spline
        N = 5
        x_nodes = range(-1.0, 1.0, length=50)
        y_nodes = [SVector{N}(sin(2π*x + k*π/N) for k in 1:N) for x in x_nodes]

        # Create Interpolations.jl spline
        itp = cubic_spline_interpolation(x_nodes, y_nodes)

        # Convert to GPUCubicSpline
        gpu_spl = M.GPUCubicSpline(itp)

        println("    GPU spline: ", gpu_spl)
        @test M.output_dim(gpu_spl) == N

        # Test at random points
        x_test = -1.0 .+ 2.0 * rand(100)
        max_err = 0.0

        for x in x_test
            y_ref = itp(x)
            y_gpu = gpu_spl(x)
            err = maximum(abs.(y_ref - y_gpu))
            max_err = max(max_err, err)
        end

        println("    Max error (vector): ", max_err)
        @test max_err < 1e-9  # Hermite reconstruction has ~1e-10 error
    end

    @testset "batched evaluation" begin
        println("\n  Testing batched evaluation...")

        # Create spline
        N = 8
        x_nodes = range(-1.0, 1.0, length=100)
        y_nodes = [SVector{N}(sin(2π*x + k*π/N) * exp(-x^2) for k in 1:N) for x in x_nodes]
        itp = cubic_spline_interpolation(x_nodes, y_nodes)
        gpu_spl = M.GPUCubicSpline(itp)

        # Batched evaluation
        x_batch = collect(range(-0.9, 0.9, length=50))
        y_batch = M.evaluate(gpu_spl, x_batch)

        @test size(y_batch) == (50, N)

        # Compare with single evaluations
        max_err = 0.0
        for (i, x) in enumerate(x_batch)
            y_ref = itp(x)
            y_gpu = y_batch[i, :]
            err = maximum(abs.(y_ref - y_gpu))
            max_err = max(max_err, err)
        end

        println("    Max error (batched): ", max_err)
        @test max_err < 1e-9  # Hermite reconstruction has ~1e-10 error
    end

    @testset "derivative evaluation" begin
        println("\n  Testing derivative evaluation...")

        # Create spline for a function with known derivative
        f(x) = x^3 - x
        df(x) = 3x^2 - 1

        x_nodes = range(-1.0, 1.0, length=100)
        y_nodes = f.(x_nodes)
        itp = cubic_spline_interpolation(x_nodes, y_nodes)
        gpu_spl = M.GPUCubicSpline(itp)

        # Evaluate with derivative
        x_test = collect(range(-0.8, 0.8, length=30))
        y_out, dy_out = M.evaluate_ed(gpu_spl, x_test)

        # Check values
        max_val_err = 0.0
        max_deriv_err = 0.0

        for (i, x) in enumerate(x_test)
            val_err = abs(y_out[i, 1] - f(x))
            deriv_err = abs(dy_out[i, 1] - df(x))
            max_val_err = max(max_val_err, val_err)
            max_deriv_err = max(max_deriv_err, deriv_err)
        end

        println("    Max value error: ", max_val_err)
        println("    Max derivative error: ", max_deriv_err)

        @test max_val_err < 1e-9   # Hermite reconstruction has ~1e-10 error
        @test max_deriv_err < 1e-6  # Derivative accuracy limited by spline interpolation
    end

    @testset "boundary behavior" begin
        println("\n  Testing boundary clamping...")

        # Simple spline
        x_nodes = range(-1.0, 1.0, length=20)
        y_nodes = x_nodes .^ 2
        itp = cubic_spline_interpolation(x_nodes, y_nodes)
        gpu_spl = M.GPUCubicSpline(itp)

        # Test at boundaries
        @test gpu_spl(-1.0) ≈ 1.0 atol=1e-10
        @test gpu_spl(1.0) ≈ 1.0 atol=1e-10

        # Test clamping outside range
        @test gpu_spl(-2.0) ≈ gpu_spl(-1.0) atol=1e-10
        @test gpu_spl(2.0) ≈ gpu_spl(1.0) atol=1e-10
    end

    @testset "ACEpotentials spline compatibility" begin
        println("\n  Testing ACEpotentials spline compatibility...")

        # Build an ACE model and splinify it
        elements = (:Si,)
        level = M.TotalDegree()
        max_level = 6
        order = 2
        maxl = 2

        model = M.ace_model(; elements = elements, order = order,
                    Ytype = :solid, level = level, max_level = max_level,
                    maxl = maxl, pair_maxn = max_level)

        rng = Random.MersenneTwister(1234)
        ps, st = Lux.setup(rng, model)

        # Splinify the model
        model_spl = M.splinify(model, ps)

        # Get one of the splines
        spl_ref = model_spl.pairbasis.splines[1, 1]

        println("    Reference spline type: ", typeof(spl_ref))

        # Convert to GPU spline
        gpu_spl = M.GPUCubicSpline(spl_ref)

        println("    GPU spline: ", gpu_spl)

        # Test evaluation
        x_test = range(-0.9, 0.9, length=30)
        max_err = 0.0

        for x in x_test
            y_ref = spl_ref(x)
            y_gpu = gpu_spl(x)
            err = maximum(abs.(y_ref - y_gpu))
            max_err = max(max_err, err)
        end

        println("    Max error vs ACE spline: ", max_err)
        @test max_err < 1e-8
    end

end

println("\nAll GPU spline tests completed!")
