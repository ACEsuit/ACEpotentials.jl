# Fit TiAl ACE model with v0.6.9 and export to .yace format
# Run with: julia +1.10 --project=benchmark/v069_env benchmark/fit_tial_v069.jl

using ACEpotentials
# JuLIP is re-exported through ACEpotentials
using ACEpotentials.JuLIP: read_extxyz, chemical_symbol
using Pkg.Artifacts

println("="^60)
println("Fitting TiAl Model with ACEpotentials v0.6.9")
println("="^60)

# Load the TiAl tutorial dataset
println("\nLoading TiAl tutorial dataset...")
# Get path to dataset artifact
artifact_toml = joinpath(dirname(dirname(pathof(ACEpotentials))), "Artifacts.toml")
data_path = artifact_path(artifact_hash("TiAl_tutorial", artifact_toml))
train = read_extxyz(joinpath(data_path, "TiAl_tutorial.xyz"))[1:5:end]
println("Training set size: $(length(train)) configurations")

# Reference energies for OneBody potential
e0 = Dict("Ti" => -1586.0195, "Al" => -105.5954)

# Create model with v0.6.9 API - include Eref for export compatibility
println("\nCreating ACE model...")
model = acemodel(
    elements = [:Ti, :Al],
    order = 3,
    totaldegree = 10,
    rcut = 5.5,
    Eref = [:Ti => e0["Ti"], :Al => e0["Al"]]
)
@show length(model.basis)

# Set up fitting
weights = Dict(
    "FLD_TiAl" => Dict("E" => 60.0, "F" => 1.0, "V" => 1.0),
    "TiAl_T5000" => Dict("E" => 5.0, "F" => 1.0, "V" => 1.0)
)

# Use QR solver
println("\nFitting model with QR solver...")
solver = ACEfit.QR(; lambda=1e-3)
data_keys = (energy_key = "energy", force_key = "force", virial_key = "virial")
acefit!(model, train; solver=solver, e0=e0, weights=weights, data_keys...)

# Compute errors
println("\n" * "="^60)
println("Training Errors:")
println("="^60)
ACEpotentials.linear_errors(train, model; data_keys...)

# Export to .yace format for LAMMPS ML-PACE
println("\n" * "="^60)
println("Exporting to LAMMPS .yace format")
println("="^60)
output_dir = joinpath(@__DIR__, "lammps")
yace_path = joinpath(output_dir, "tial_model.yace")
export2lammps(yace_path, model)
println("Exported to: $yace_path")

# List all exported files
println("\nExported files:")
for f in readdir(output_dir)
    if startswith(f, "tial_model")
        fpath = joinpath(output_dir, f)
        println("  - $f ($(round(filesize(fpath)/1024, digits=1)) KB)")
    end
end

println("\nDone!")
