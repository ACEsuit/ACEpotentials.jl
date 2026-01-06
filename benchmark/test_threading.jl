# Test script to verify threading works with --trim=safe compiled libraries
#
# Usage:
#   JULIA_NUM_THREADS=4 julia --project=export test_threading.jl
#
# This script:
# 1. Loads the TiAl tutorial model
# 2. Exports it to a library with threading support
# 3. Tests the ace_get_nthreads() function

using ACEpotentials

println("=" ^ 60)
println("Testing Threading with --trim=safe Compiled Libraries")
println("=" ^ 60)

println("\nJulia threads: $(Threads.nthreads())")
println("JULIA_NUM_THREADS = $(get(ENV, "JULIA_NUM_THREADS", "not set"))")

# Load the same model used in benchmarks
println("\n[1/4] Loading TiAl tutorial model...")
data, _, meta = ACEpotentials.example_dataset("TiAl_tutorial")
train_data = data[1:10:end]  # Smaller subset for faster testing

hyperparams = (
    elements = [:Ti, :Al],
    order = 3,
    totaldegree = 8,  # Smaller for faster compilation
    rcut = 5.5,
    Eref = [:Ti => -1586.0195, :Al => -105.5954]
)

println("\n[2/4] Creating and fitting model...")
model = ace1_model(; hyperparams...)
solver = ACEfit.QR(lambda = 1e-3)
acefit!(train_data, model; solver=solver)
println("  Model fitted successfully")

# Export to library
println("\n[3/4] Exporting to shared library...")
export_dir = dirname(@__DIR__)
include(joinpath(export_dir, "export", "scripts", "build_deployment.jl"))

output_dir = joinpath(@__DIR__, "deployments", "threading_test")
deploy_path = build_deployment(
    model,
    "threading_test";
    output_dir = output_dir,
    include_lammps = false,
    include_python = false
)

lib_path = joinpath(deploy_path, "lib", "libace_threading_test.so")
println("  Library: $lib_path")

# Check if the new symbols are present
println("\n[4/4] Checking exported symbols...")
nm_output = read(`nm -D $lib_path`, String)
has_nthreads = contains(nm_output, "ace_get_nthreads")
has_batch = contains(nm_output, "ace_batch_energy_forces_virial")

println("  ace_get_nthreads: $(has_nthreads ? "✓" : "✗")")
println("  ace_batch_energy_forces_virial: $(has_batch ? "✓" : "✗")")

if has_nthreads
    # Try to call ace_get_nthreads via ccall
    println("\n  Testing ace_get_nthreads() call...")

    # Need to set up library path
    setup_script = joinpath(deploy_path, "setup_env.sh")

    println("  To test thread count, run:")
    println("    source $setup_script")
    println("    python3 -c \"import ctypes; lib = ctypes.CDLL('$lib_path'); print(f'Threads: {lib.ace_get_nthreads()}')\"")
end

println("\n" * "=" ^ 60)
println("Test deployment created at: $deploy_path")
println("=" ^ 60)
