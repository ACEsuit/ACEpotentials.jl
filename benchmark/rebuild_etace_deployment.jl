# Rebuild the TiAl ETACE deployment with codegen-based export
# Run with: julia --project=. benchmark/rebuild_etace_deployment.jl
#
# This script creates an ETACE model with LearnableRnlrzzBasis (not splinified)
# because convert2et only works with LearnableRnlrzzBasis.
# We use Models.ace_model() directly to avoid the splinification in ace1_model.

using ACEpotentials
using ACEpotentials.Models
using ACEpotentials.ETModels
using ACEpotentials.ETModels: splinify
using Lux
using LuxCore
using Random
using ACEfit
using Unitful

const M = ACEpotentials.Models
const ETM = ACEpotentials.ETModels

# Include the updated export code
include(joinpath(@__DIR__, "..", "export/src/export_ace_model.jl"))

println("="^60)
println("Rebuilding TiAl ETACE Deployment with Codegen Export")
println("="^60)

# Load TiAl tutorial dataset
println("\nLoading TiAl tutorial dataset...")
data, _, meta = ACEpotentials.example_dataset("TiAl_tutorial")
train_data = data[1:5:end]
println("Training set size: $(length(train_data)) configurations")

# Model hyperparameters - matching ace1_model defaults for fair comparison
println("\nCreating ACE model with hyperparameters (matching ace1_model):")
elements = (:Ti, :Al)
order = 3
totaldegree = 8    # Match ace1_model default
rcut = 5.5
maxl = 6           # Match ace1_model default (not restricted)

# Use ace1_model's level specification for matching basis size
# ace1_model uses TotalDegree(1.0*NZ, 1/wL) with wL=1.5
NZ = length(elements)
wL = 1.5
level = M.TotalDegree(1.0*NZ, 1/wL)  # TotalDegree(2.0, 0.667)

println("  elements: $elements")
println("  order: $order")
println("  totaldegree: $totaldegree")
println("  rcut: $rcut")
println("  maxl: $maxl")
println("  level: TotalDegree($(1.0*NZ), $(1/wL)) - matching ace1_model")

# Create ACE model with LearnableRnlrzzBasis (not splinified)
# This allows convert2et to work properly
rin0cuts = M._default_rin0cuts(elements)
rin0cuts = (x -> (rin = x.rin, r0 = x.r0, rcut = rcut)).(rin0cuts)

rng = Random.MersenneTwister(1234)

ace_model = M.ace_model(;
    elements = elements,
    order = order,
    Ytype = :solid,
    level = level,           # Use ace1_model-compatible level
    max_level = totaldegree,
    maxl = maxl,
    pair_maxn = totaldegree,
    rin0cuts = rin0cuts,
    init_WB = :zeros,        # Start with zero readout weights (will be fitted)
    init_Wpair = :onehot,    # Standard polynomials for pair basis
    init_Wradial = :onehot,  # Fixed radial basis (not learnable) like ACE1
    E0s = Dict(:Ti => -1586.0195u"eV", :Al => -105.5954u"eV")
)

ps, st = Lux.setup(rng, ace_model)
println("Basis size: $(length(ps.WB[:, 1]))")

# Create ACEPotential wrapper for fitting
ace_pot = ACEpotentials.ACEPotential(ace_model, ps, st)

# Fit the model
println("\nFitting model with QR solver...")
solver = ACEfit.QR(lambda = 1e-3)
P = algebraic_smoothness_prior(ace_pot; p=4)
weights = Dict(
    "FLD_TiAl" => Dict("E" => 60.0, "F" => 1.0, "V" => 1.0),
    "TiAl_T5000" => Dict("E" => 5.0, "F" => 1.0, "V" => 1.0)
)

acefit!(train_data, ace_pot; solver=solver, prior=P, weights=weights)

# Compute and display errors
println("\n" * "="^60)
println("Training Errors:")
println("="^60)
ACEpotentials.compute_errors(train_data, ace_pot; weights=weights)

# Get fitted parameters
fitted_ps = ace_pot.ps
println("\nFitted WB shape: $(size(fitted_ps.WB))")

# Convert to ETACE
println("\nConverting ACE to ETACE...")
et_model = ETM.convert2et(ace_model)
et_ps, et_st = LuxCore.setup(MersenneTwister(1234), et_model)

# Copy fitted radial basis parameters
n_species = length(elements)
for iz in 1:n_species
    for jz in 1:n_species
        et_ps.rembed.post.W[:, :, (iz-1)*n_species + jz] .= fitted_ps.rbasis.Wnlq[:, :, iz, jz]
    end
end

# Copy fitted readout parameters
for iz in 1:n_species
    et_ps.readout.W[1, :, iz] .= fitted_ps.WB[:, iz]
end

println("ETACE model created with fitted weights")

# CRITICAL: Splinify the model BEFORE creating calculator for Hermite export
println("\nSplinifying model (Nspl=50)...")
et_model_splined = splinify(et_model, et_ps, et_st; Nspl=50)
et_ps_splined, et_st_splined = LuxCore.setup(MersenneTwister(1234), et_model_splined)

# Copy readout weights (splinify doesn't preserve these)
for iz in 1:n_species
    et_ps_splined.readout.W[1, :, iz] .= et_ps.readout.W[1, :, iz]
end

# Create calculator with splinified model
et_calc = ETM.ETACEPotential(et_model_splined, et_ps_splined, et_st_splined, rcut)

# Export with Hermite cubic splines (exact, machine-precision accuracy)
deploy_dir = joinpath(@__DIR__, "deployments/tial_etace_codegen")
mkpath(deploy_dir)
mkpath(joinpath(deploy_dir, "lib"))

export_file = joinpath(deploy_dir, "tial_etace_model.jl")
println("\nExporting ETACE model to $export_file...")
println("  Using: radial_basis=:hermite_spline (codegen-based, exact)")
export_ace_model(et_calc, export_file; for_library=true, radial_basis=:hermite_spline)

# Also export with polynomial for comparison (needs non-splinified model)
# Create a non-splinified calculator
et_calc_poly = ETM.ETACEPotential(et_model, et_ps, et_st, rcut)
export_file_poly = joinpath(deploy_dir, "tial_etace_model_poly.jl")
println("\nExporting polynomial version to $export_file_poly...")
export_ace_model(et_calc_poly, export_file_poly; for_library=true, radial_basis=:polynomial)

# Report file sizes
println("\n" * "="^60)
println("Export complete!")
println("="^60)

spline_size = filesize(export_file)
poly_size = filesize(export_file_poly)
println("\nFile sizes:")
println("  Spline: $(round(spline_size/1024/1024, digits=2)) MB")
println("  Polynomial: $(round(poly_size/1024, digits=1)) KB")

# Verify the exports are trim-safe
println("\nVerifying exports are self-contained:")
for (name, path) in [("Hermite Spline", export_file), ("Polynomial", export_file_poly)]
    content = read(path, String)
    code_lines = filter(line -> !startswith(strip(line), "#") && !isempty(strip(line)), split(content, '\n'))

    has_sphericart = any(line -> occursin("SpheriCart", line), code_lines)
    has_et = any(line -> occursin("EquivariantTensors", line), code_lines)
    has_p4ml = any(line -> occursin("Polynomials4ML", line), code_lines)

    status = !has_sphericart && !has_et && !has_p4ml ? "✓" : "✗"
    println("  $status $name: SpheriCart=$(!has_sphericart), ET=$(!has_et), P4ML=$(!has_p4ml)")
end

println("\n" * "="^60)
println("Next steps: Compile with JuliaC")
println("="^60)
println("""
cd $deploy_dir

# Create Project.toml
cat > Project.toml << 'EOF'
[deps]
JuliaC = "acedd4c2-ced6-4a15-accc-2607eb759ba2"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
EOF

# Compile spline version
julia --project=. -e '
using Pkg; Pkg.instantiate()
using JuliaC
recipe = ImageRecipe(;
    file="tial_etace_model.jl",
    output_type="sharedlib",
    trim_mode="safe",
    project=".",
    img_path="lib/libace_etace_spline.so",
    add_ccallables=true
)
compile_products(recipe)
'
""")
