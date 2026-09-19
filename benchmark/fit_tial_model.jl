# Fit TiAl ACE model for LAMMPS benchmark comparison
# This script fits a model using the current ACEpotentials.jl (lammps-export branch)

using ACEpotentials

println("Loading TiAl tutorial dataset...")
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

# Fitting configuration
# Use QR solver (BLR can have numerical issues)
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

# Save model for export
println("\nModel fitting complete. Ready for export.")

# Return model for use in export script
model
