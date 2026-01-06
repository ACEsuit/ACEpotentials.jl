# Fit TiAl ACE model and export to shared library for LAMMPS
# This script fits a model using the current ACEpotentials.jl (lammps-export branch)
# and exports it to a compiled shared library

using ACEpotentials

println("="^60)
println("PART 1: Fitting TiAl Model")
println("="^60)

println("\nLoading TiAl tutorial dataset...")
data, _, meta = ACEpotentials.example_dataset("TiAl_tutorial")
train_data = data[1:5:end]
println("Training set size: $(length(train_data)) configurations")

# Model hyperparameters (medium complexity)
hyperparams = (
    elements = [:Ti, :Al],
    order = 3,
    totaldegree = 10,
    rcut = 5.5,
    Eref = [:Ti => -1586.0195, :Al => -105.5954]
)

println("\nCreating ACE model with hyperparameters:")
println("  elements: $(hyperparams.elements)")
println("  order: $(hyperparams.order)")
println("  totaldegree: $(hyperparams.totaldegree)")
println("  rcut: $(hyperparams.rcut)")

model = ace1_model(; hyperparams...)
println("Basis size: $(length_basis(model))")

# Fitting configuration - use QR solver (BLR can have numerical issues)
solver = ACEfit.QR(lambda = 1e-3)
P = algebraic_smoothness_prior(model; p=4)
weights = Dict(
    "FLD_TiAl" => Dict("E" => 60.0, "F" => 1.0, "V" => 1.0),
    "TiAl_T5000" => Dict("E" => 5.0, "F" => 1.0, "V" => 1.0)
)

println("\nFitting model with QR solver...")
result = acefit!(train_data, model; solver=solver, prior=P, weights=weights)

# Compute and display errors
println("\n" * "="^60)
println("Training Errors:")
println("="^60)
err = ACEpotentials.compute_errors(train_data, model; weights=weights)

println("\n" * "="^60)
println("PART 2: Exporting to Shared Library")
println("="^60)

# Export model to trim-compatible library
include(joinpath(@__DIR__, "..", "export", "scripts", "build_deployment.jl"))

# Set LAMMPS header dir if available
lammps_header_dir = expanduser("~/lammps/lammps-22Jul2025/src")
if !isdir(lammps_header_dir)
    lammps_header_dir = nothing
    @warn "LAMMPS headers not found at $lammps_header_dir, plugin won't be built"
end

deploy_path = build_deployment(
    model,
    "tial_ace";
    output_dir = joinpath(@__DIR__, "deployments"),
    include_lammps = true,
    include_python = false,
    lammps_header_dir = lammps_header_dir
)

println("\n" * "="^60)
println("Deployment complete!")
println("Library location: $(joinpath(deploy_path, "lib", "libace_tial_ace.so"))")
println("="^60)
