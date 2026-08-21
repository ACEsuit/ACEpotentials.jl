#=
fit_and_export.jl - Complete example workflow for ACE model export

This script demonstrates the full workflow:
1. Load training data
2. Fit an ACE model
3. Export to deployment package

Usage:
    julia --project=../.. fit_and_export.jl
=#

using ACEpotentials
using ExtXYZ

println("=" ^ 60)
println("ACE Model Export Example: Silicon")
println("=" ^ 60)

# Step 1: Load training data
println("\n[1/4] Loading training data...")

# Use the Si_tiny_dataset artifact for this example
data_path = ACEpotentials.artifact("Si_tiny_dataset")
data = ExtXYZ.load(joinpath(data_path, "Si_tiny.xyz"))
println("  Loaded $(length(data)) configurations")

# Step 2: Create and fit ACE model
println("\n[2/4] Creating ACE model...")

model = ACEpotentials.ace1_model(
    elements = [:Si],
    order = 2,           # Body order (2 = up to 3-body)
    totaldegree = 6,     # Polynomial degree
    rcut = 5.5,          # Cutoff radius in Angstrom
)

println("  Model created with $(length(model)) parameters")

println("\n[3/4] Fitting model to data...")

# Convert data to ACEpotentials format
# Note: Si_tiny dataset uses dft_energy, dft_force, dft_virial keys
train_data = [
    ACEpotentials.AtomsData(at;
        energy_key = "dft_energy",
        force_key = "dft_force",
        virial_key = "dft_virial",
        weights = Dict("default" => Dict("E" => 1.0, "F" => 1.0, "V" => 1.0))
    )
    for at in data
]

# Fit the model
acefit!(train_data, model;
    solver = ACEpotentials.BLR(),  # Bayesian Linear Regression
    smoothness_prior = true,
)

# Compute errors
errors = ACEpotentials.compute_errors(train_data, model)
println("  Training RMSE:")
println("    Energy: $(round(errors[:rmse_E], digits=4)) eV/atom")
println("    Forces: $(round(errors[:rmse_F], digits=4)) eV/Å")

# Step 3: Export to deployment package
println("\n[4/4] Creating deployment package...")

# Include the build_deployment script
include(joinpath(@__DIR__, "..", "..", "scripts", "build_deployment.jl"))

# Create deployment
deploy_dir = build_deployment(
    model,
    "silicon_example";
    output_dir = joinpath(@__DIR__, "deployments"),
    include_lammps = true,
    include_python = true,
    verbose = true
)

println("\n" * "=" ^ 60)
println("Export complete!")
println("=" ^ 60)
println("\nDeployment package: $deploy_dir")
println("\nNext steps:")
println("  1. source $deploy_dir/setup_env.sh")
println("  2. Run LAMMPS: lmp -in $deploy_dir/lammps/example.lmp")
println("  3. Run Python: python $deploy_dir/python/example.py")
