# Master script to create all four fair benchmark deployments
# Run this to build Old ACE, ML-PACE, ETACE Spline, and ETACE Polynomial

using Dates

println("="^70)
println("Fair Benchmark: Building All Four Deployments")
println("="^70)
println("Start time: $(Dates.now())")
println()

benchmark_dir = @__DIR__

# Build Old ACE deployment
println("\n" * "="^70)
println("1/4: Building Old ACE deployment...")
println("="^70)
try
    include(joinpath(benchmark_dir, "create_oldace.jl"))
    println("Old ACE deployment: SUCCESS")
catch e
    println("Old ACE deployment: FAILED")
    println("Error: $e")
end

# Build ML-PACE deployment
println("\n" * "="^70)
println("2/4: Building ML-PACE deployment...")
println("="^70)
try
    include(joinpath(benchmark_dir, "create_mlpace.jl"))
    println("ML-PACE deployment: SUCCESS")
catch e
    println("ML-PACE deployment: FAILED")
    println("Error: $e")
end

# Build ETACE deployments (both spline and polynomial)
println("\n" * "="^70)
println("3-4/4: Building ETACE deployments...")
println("="^70)
try
    include(joinpath(benchmark_dir, "create_etace.jl"))
    println("ETACE deployments: SUCCESS")
catch e
    println("ETACE deployments: FAILED")
    println("Error: $e")
end

# Summary
println("\n" * "="^70)
println("Deployment Summary")
println("="^70)
println("End time: $(Dates.now())")
println()

deployments_dir = joinpath(benchmark_dir, "deployments")
for method in ["oldace", "mlpace", "etace_spline", "etace_poly"]
    method_dir = joinpath(deployments_dir, method)
    if isdir(method_dir)
        lib_dir = joinpath(method_dir, "lib")
        if isdir(lib_dir)
            libs = filter(f -> endswith(f, ".so"), readdir(lib_dir))
            if !isempty(libs)
                lib_file = joinpath(lib_dir, libs[1])
                size_mb = round(filesize(lib_file) / 1024^2, digits=1)
                println("  [OK] $method: $(libs[1]) ($size_mb MB)")
            else
                println("  [MISSING] $method: no .so files in lib/")
            end
        else
            # Check for .yace files (ML-PACE)
            yace_files = filter(f -> endswith(f, ".yace"), readdir(method_dir))
            if !isempty(yace_files)
                yace_file = joinpath(method_dir, yace_files[1])
                size_kb = round(filesize(yace_file) / 1024, digits=1)
                println("  [OK] $method: $(yace_files[1]) ($size_kb KB)")
            else
                println("  [PARTIAL] $method: directory exists but no artifacts")
            end
        end
    else
        println("  [MISSING] $method: directory not found")
    end
end

println()
println("Next steps:")
println("  1. Run: ./run_benchmark.sh")
println("  2. Analyze: julia analyze_results.jl")
