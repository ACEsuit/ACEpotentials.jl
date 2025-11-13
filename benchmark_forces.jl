"""
Simple force computation benchmark

Measures the performance of force computation using the current AD implementation.
"""

using ACEpotentials
using AtomsCalculators
using BenchmarkTools
using ExtXYZ
using LazyArtifacts
using Printf

println("=" ^ 80)
println("Force Computation Benchmark")
println("=" ^ 80)
println()

# Setup: Create and fit a small model
println("Setting up model...")
model = ace1_model(;
    elements = [:Si],
    order = 3,
    totaldegree = 12,
    rcut = 5.5,
    Eref = [:Si => -158.54496821]
)

# Load and fit
data = ExtXYZ.load(artifact"Si_tiny_dataset" * "/Si_tiny.xyz")
data_keys = (energy_key = "dft_energy", force_key = "dft_force", virial_key = "dft_virial")
solver = ACEfit.LSQR(; damp=1e-3, atol=1e-6)
acefit!(data, model; data_keys..., solver=solver)

println("Model fitted successfully")
println()

# Select a test system (liquid config has more atoms)
at_test = data[5]

println("Benchmark configuration:")
println("  Number of atoms: ", length(at_test))
println()

#Verify forces work
println("Testing force computation...")
f = AtomsCalculators.forces(at_test, model)
println("  ✓ Forces computed successfully")
println("  Example force: ", f[1])
println()

# Benchmark
println("=" ^ 80)
println("Running Benchmark")
println("=" ^ 80)
println()

b = @benchmark AtomsCalculators.forces($at_test, $model) samples=50 evals=3
display(b)
println()
println()

# Summary
println("=" ^ 80)
println("Summary")
println("=" ^ 80)
println()

median_time = median(b.times) / 1e6  # ms
mean_time = mean(b.times) / 1e6
allocs = b.allocs
memory = b.memory / 1024  # KB

println(@sprintf("Median time:     %10.3f ms", median_time))
println(@sprintf("Mean time:       %10.3f ms", mean_time))
println(@sprintf("Allocations:     %10d", allocs))
println(@sprintf("Memory:          %10.1f KB", memory))
println()
