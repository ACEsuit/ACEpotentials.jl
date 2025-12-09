# Test export of ACE model as a shared library with C interface
# This script:
# 1. Fits a small ACE model to Silicon data
# 2. Exports it with for_library=true to generate C interface
# 3. The exported file can then be compiled with --output-lib

using ACEpotentials
using ACEpotentials.Models: ACEPotential
using ExtXYZ, AtomsBase, Unitful, StaticArrays
using AtomsCalculators: potential_energy, forces
using Pkg
using Pkg.Artifacts
using LinearAlgebra

include("export_ace_model.jl")

# ============================================================================
# 1. Create and fit the model
# ============================================================================

@info "Creating ACE model for Silicon..."

params = (
    elements = [:Si],
    Eref = [:Si => -158.54496821],
    rcut = 5.5,
    order = 2,           # Smaller for faster test
    totaldegree = 6,     # Smaller for faster test
)

model = ACEpotentials.ACE1compat.ace1_model(; params...)

@info "Loading training data..."
# Load the artifact
pkg_dir = dirname(dirname(pathof(ACEpotentials)))
test_dir = joinpath(pkg_dir, "test")
push!(LOAD_PATH, test_dir)
artifact_toml = joinpath(test_dir, "Artifacts.toml")
artifact_hash = Pkg.Artifacts.artifact_hash("Si_tiny_dataset", artifact_toml)
artifact_path = Pkg.Artifacts.artifact_path(artifact_hash)
if !Pkg.Artifacts.artifact_exists(artifact_hash)
    @info "Downloading Si_tiny_dataset artifact..."
    Pkg.Artifacts.ensure_artifact_installed("Si_tiny_dataset", artifact_toml)
end
data = ExtXYZ.load(joinpath(artifact_path, "Si_tiny.xyz"))
pop!(LOAD_PATH)

data_keys = (
    :energy_key => "dft_energy",
    :force_key  => "dft_force",
    :virial_key => "dft_virial",
)

weights = Dict(
    "default" => Dict("E"=>30.0, "F"=>1.0, "V"=>1.0),
    "liq" => Dict("E"=>10.0, "F"=>0.66, "V"=>0.25),
)

@info "Fitting model with QR solver..."
ACEpotentials.acefit!(data, model;
    data_keys...,
    weights = weights,
    solver = ACEpotentials.ACEfit.QR(),
)

@info "Computing errors..."
err = ACEpotentials.compute_errors(data, model; data_keys..., weights=weights)
println("RMSE (dia): E=$(err["rmse"]["dia"]["E"]), F=$(err["rmse"]["dia"]["F"])")

# ============================================================================
# 2. Export the fitted model as a library
# ============================================================================

export_filename = joinpath(@__DIR__, "silicon_lib.jl")
@info "Exporting fitted model as library to $export_filename..."
export_ace_model(model, export_filename; splinify_first=true, for_library=true)

# ============================================================================
# 3. Summary
# ============================================================================

println("\n" * "="^60)
println("LIBRARY EXPORT SUMMARY")
println("="^60)
println("Model parameters:")
println("  Elements: $(params.elements)")
println("  Order: $(params.order)")
println("  Total degree: $(params.totaldegree)")
println("  Cutoff: $(params.rcut) Å")
println()
println("Exported library model:")
println("  File: $export_filename")
println()
println("Next steps:")
println("  1. Compile as shared library with --output-lib:")
println("     cd $(dirname(export_filename))")
println("     mkdir -p silicon_lib/lib")
println("     julia --project=. ~/.julia/juliaup/julia-1.12.2+0.x64.linux.gnu/share/julia/juliac/juliac.jl \\")
println("         --output-lib silicon_lib/lib/libace.so \\")
println("         --experimental --trim=safe \\")
println("         silicon_lib.jl")
println()
println("  2. Test from Python:")
println("     python test_ace_library.py")
println("="^60)
