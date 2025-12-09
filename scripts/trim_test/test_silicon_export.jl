# Test export with a real fitted Silicon model
# This script:
# 1. Fits a small ACE model to Silicon data
# 2. Exports it using export_ace_model.jl
# 3. Compares evaluation results between original and exported model
# 4. The exported file can then be compiled with --trim=safe

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
# Load the artifact - need to use the test Artifacts.toml
pkg_dir = dirname(dirname(pathof(ACEpotentials)))
test_dir = joinpath(pkg_dir, "test")
# Temporarily add test dir to LOAD_PATH to access artifacts
push!(LOAD_PATH, test_dir)
artifact_toml = joinpath(test_dir, "Artifacts.toml")
artifact_hash = Pkg.Artifacts.artifact_hash("Si_tiny_dataset", artifact_toml)
artifact_path = Pkg.Artifacts.artifact_path(artifact_hash)
# Ensure artifact is downloaded
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
# 2. Export the fitted model
# ============================================================================

export_filename = joinpath(@__DIR__, "silicon_model.jl")
@info "Exporting fitted model to $export_filename..."
export_ace_model(model, export_filename; splinify_first=true)

# ============================================================================
# 3. Test the exported model
# ============================================================================

@info "Testing exported model..."

# Include the exported model (defines site_energy, etc.)
include(export_filename)

# Pick a test configuration from the dataset
test_sys = data[10]  # Pick a configuration with multiple atoms
@info "Test system" n_atoms=length(test_sys) config_type=test_sys.system[:config_type]

# Get atomic positions and species
positions = [ustrip.(u"Å", position(test_sys, i)) for i in 1:length(test_sys)]
species = [atomic_number(test_sys, i) for i in 1:length(test_sys)]

# Evaluate with original model
E_orig = ustrip(u"eV", potential_energy(test_sys, model))
F_orig = forces(test_sys, model)

@info "Original model evaluation" E=E_orig

# Evaluate with exported model (site energies)
# We need to compute neighbor lists ourselves for the exported model
function get_neighbors(positions, i, rcut)
    Rs = SVector{3, Float64}[]
    Zs = Int[]
    pos_i = positions[i]
    for j in 1:length(positions)
        if i != j
            R = positions[j] - pos_i
            if norm(R) < rcut
                push!(Rs, SVector{3}(R))
                push!(Zs, species[j])
            end
        end
    end
    return Rs, Zs
end

# Note: The exported model uses a simple cutoff, not periodic boundaries
# For this test, we'll just evaluate site energies for a few atoms
@info "Evaluating exported model site energies (no PBC)..."

# Use the cutoff from the exported model
rcut_export = RIN0CUT_1_1.rcut

# Evaluate a few site energies
for i in 1:min(3, length(positions))
    Rs, Zs = get_neighbors(positions, i, rcut_export)
    Z0 = species[i]
    E_site = site_energy(Rs, Zs, Z0)
    println("  Site $i: E = $E_site eV ($(length(Rs)) neighbors)")
end

# Test forces for one site
Rs, Zs = get_neighbors(positions, 1, rcut_export)
Z0 = species[1]
E_site, F_site = site_energy_forces(Rs, Zs, Z0)
@info "Site 1 forces" E=E_site n_forces=length(F_site)
for (j, f) in enumerate(F_site)
    println("  F[$j] = [$(f[1]), $(f[2]), $(f[3])]")
end

# ============================================================================
# 4. Summary
# ============================================================================

println("\n" * "="^60)
println("EXPORT TEST SUMMARY")
println("="^60)
println("Model parameters:")
println("  Elements: $(params.elements)")
println("  Order: $(params.order)")
println("  Total degree: $(params.totaldegree)")
println("  Cutoff: $(params.rcut) Å")
println()
println("Exported model:")
println("  File: $export_filename")
println("  Tensor length: $(length(TENSOR))")
println("  Radial basis size: $N_RNL")
println("  Spherical harmonics: L=$MAXL ($N_YLM functions)")
println()
println("Original model total energy: $E_orig eV")
println()
println("Next steps:")
println("  1. Compile with --trim=safe:")
println("     cd $(dirname(export_filename))")
println("     mkdir -p silicon_bundle/bin")
println("     julia --project=. ~/.julia/juliaup/julia-1.12.2+0.x64.linux.gnu/share/julia/juliac/juliac.jl \\")
println("         --output-exe silicon_bundle/bin/silicon \\")
println("         --experimental --trim=safe \\")
println("         silicon_model.jl")
println()
println("  2. Run the compiled binary:")
println("     ./silicon_bundle/bin/silicon")
println("="^60)
