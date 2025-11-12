#!/usr/bin/env julia
# Test energy-only predictions during EquivariantTensors v0.3 migration
# This tests that the migration works for energy calculations

using ACEpotentials
using ACEpotentials.ACE1compat
using LazyArtifacts, ExtXYZ
using LinearAlgebra
using AtomsCalculators: forces, potential_energy, energy_forces_virial

println("="^60)
println("Testing Energy-Only Predictions")
println("="^60)

# Create a simple Si model
println("\n1. Creating Si model...")
model = ace1_model(; elements = [:Si],
                     Eref = [:Si => -158.54496821],
                     rcut = 5.0,
                     order = 3,
                     totaldegree = 8)
println("✓ Model created successfully")

# Load test data
println("\n2. Loading test data...")
data = ExtXYZ.load(artifact"Si_tiny_dataset" * "/Si_tiny.xyz")
println("✓ Loaded $(length(data)) configurations")

# Fit model with energy-only (use non-existent keys to avoid force/virial data)
println("\n3. Attempting to fit model (energy-only)...")
println("   Using non-existent force/virial keys to ensure only energy data is used")
try
    # Use non-existent key names so no force/virial observations are found
    data_keys = [:energy_key => "dft_energy",
                 :force_key => "nonexistent_force_key",
                 :virial_key => "nonexistent_virial_key"]

    # Fit with only energy data (no smoothness to avoid evaluate_basis_ed)
    acefit!(data, model;
            data_keys...,
            weights = Dict("default" => Dict("E" => 30.0, "F" => 1.0, "V" => 1.0)),
            solver = ACEfit.BLR(),
            smoothness = 0)  # No smoothness regularization

    println("✓ Model fitted successfully (energy-only)!")
catch e
    println("✗ Fitting failed:")
    println("  ", sprint(showerror, e, catch_backtrace())[1:min(500, end)])
    println("  ...")
    println("\n  If this still fails, fitting may require forces internally.")
end

# Test energy prediction directly
println("\n4. Testing direct energy prediction...")
try
    test_at = data[1]

    # Try computing just potential energy (should work with evaluate! only)
    E = potential_energy(test_at, model)

    println("✓ Energy prediction successful: E = $E")
    println("  (Note: Model may have random/zero parameters if fitting failed)")
catch e
    println("✗ Energy prediction failed:")
    println("  ", sprint(showerror, e))
end

# Try to compute forces (should fail with clear error)
println("\n5. Testing force prediction (should fail with clear error)...")
try
    test_at = data[1]
    F = forces(test_at, model)
    println("✗ Unexpected: Force calculation succeeded (should have failed)")
    println("  Forces: $(F[1:min(3, length(F))])")
catch e
    if occursin("evaluate_basis_ed is temporarily disabled", sprint(showerror, e))
        println("✓ Force calculation failed with expected error message")
        println("  Error: evaluate_basis_ed is temporarily disabled during migration")
    else
        println("⚠  Force calculation failed (expected), but with unexpected error:")
        errstr = sprint(showerror, e)
        println("  ", split(errstr, "\n")[1])
        if occursin("evaluate_basis_ed", errstr)
            println("  (But it does mention evaluate_basis_ed)")
        end
    end
end

# Try energy_forces_virial (more likely to hit the error)
println("\n6. Testing energy_forces_virial (should definitely fail)...")
try
    test_at = data[1]
    result = energy_forces_virial(test_at, model)
    println("✗ Unexpected: energy_forces_virial succeeded")
catch e
    if occursin("evaluate_basis_ed is temporarily disabled", sprint(showerror, e))
        println("✓ Correctly failed with expected error message")
    else
        println("⚠  Failed (expected), error:")
        println("  ", split(sprint(showerror, e), "\n")[1])
    end
end

println("\n" * "="^60)
println("Energy-Only Test Summary")
println("="^60)
println("✓ Package loads")
println("✓ Model construction works")
println("? Energy predictions may work (depends on fitting)")
println("✗ Force/virial predictions disabled (as expected)")
println("\nFor full functionality, see MIGRATION_STATUS.md")
println("="^60)
