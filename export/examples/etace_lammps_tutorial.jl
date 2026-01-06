# # ETACE Model: Fitting and Export to LAMMPS
#
# This tutorial demonstrates the complete workflow for creating an ETACE
# (EquivariantTensors ACE) model and exporting it to LAMMPS.
#
# ## Overview
#
# ETACE models offer approximately 2x performance improvement over standard ACE
# models while maintaining the same accuracy. The workflow is:
#
# 1. **Create** an ACE model with learnable radial basis
# 2. **Fit** the model to training data
# 3. **Convert** to ETACE format
# 4. **Splinify** for efficient evaluation (critical step!)
# 5. **Export** to trim-compatible Julia code
# 6. **Compile** to a shared library
# 7. **Deploy** to LAMMPS
#
# ## When to Use ETACE vs Standard ACE
#
# | Feature | Standard ACE | ETACE |
# |---------|-------------|-------|
# | Evaluation speed | Baseline | ~2x faster |
# | Export complexity | Simple | Requires conversion |
# | Radial basis | Pre-splinified | Learnable → Splinified |
# | Recommended for | Quick tests, small systems | Production MD |

# ## Step 1: Setup and Load Data

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

# Load the TiAl tutorial dataset (included with ACEpotentials)
data, _, meta = ACEpotentials.example_dataset("TiAl_tutorial")
train_data = data[1:5:end]  # Use subset for faster demo
println("Training set: $(length(train_data)) configurations")

# ## Step 2: Define Model Hyperparameters
#
# Key hyperparameters:
# - **elements**: Atomic species in the system
# - **order**: Correlation order (2-4 typical, higher = more accurate but slower)
# - **totaldegree**: Polynomial degree (6-12 typical)
# - **rcut**: Cutoff radius in Angstroms (material-dependent)
# - **maxl**: Maximum angular momentum (4-8 typical)

elements = (:Ti, :Al)
order = 3
totaldegree = 8
rcut = 5.5
maxl = 6

# Reference energies per atom (from DFT calculations)
E0s = Dict(:Ti => -1586.0195u"eV", :Al => -105.5954u"eV")

println("Model hyperparameters:")
println("  elements: $elements")
println("  order: $order")
println("  totaldegree: $totaldegree")
println("  rcut: $rcut Å")

# ## Step 3: Create ACE Model
#
# For ETACE export, we use `Models.ace_model()` directly instead of `ace1_model()`
# because we need a LearnableRnlrzzBasis (not pre-splinified).
#
# The key difference: `ace1_model()` automatically splinifies the radial basis,
# but `convert2et()` requires the learnable form.

NZ = length(elements)
wL = 1.5
level = M.TotalDegree(1.0*NZ, 1/wL)

# Setup cutoffs for each element pair
rin0cuts = M._default_rin0cuts(elements)
rin0cuts = (x -> (rin = x.rin, r0 = x.r0, rcut = rcut)).(rin0cuts)

rng = Random.MersenneTwister(1234)

ace_model = M.ace_model(;
    elements = elements,
    order = order,
    Ytype = :solid,
    level = level,
    max_level = totaldegree,
    maxl = maxl,
    pair_maxn = totaldegree,
    rin0cuts = rin0cuts,
    init_WB = :zeros,        # Zero readout weights (fitted later)
    init_Wpair = :onehot,    # Standard polynomials for pair basis
    init_Wradial = :onehot,  # Fixed radial basis (like ACE1)
    E0s = E0s
)

ps, st = Lux.setup(rng, ace_model)
println("Basis size: $(length(ps.WB[:, 1])) functions")

# ## Step 4: Fit the Model

ace_pot = ACEpotentials.ACEPotential(ace_model, ps, st)

# Use QR solver with regularization
solver = ACEfit.QR(lambda = 1e-3)

# Smoothness prior for better extrapolation
P = algebraic_smoothness_prior(ace_pot; p=4)

# Weights for different configuration types
weights = Dict(
    "FLD_TiAl" => Dict("E" => 60.0, "F" => 1.0, "V" => 1.0),
    "TiAl_T5000" => Dict("E" => 5.0, "F" => 1.0, "V" => 1.0)
)

println("\nFitting model...")
acefit!(train_data, ace_pot; solver=solver, prior=P, weights=weights)

# Show training errors
println("\nTraining Errors:")
ACEpotentials.compute_errors(train_data, ace_pot; weights=weights)

# ## Step 5: Convert to ETACE
#
# The ETACE format uses EquivariantTensors for faster evaluation.
# We need to:
# 1. Convert the model structure
# 2. Copy the fitted parameters

println("\nConverting to ETACE format...")
et_model = ETM.convert2et(ace_model)
et_ps, et_st = LuxCore.setup(MersenneTwister(1234), et_model)

# Copy fitted radial basis parameters
n_species = length(elements)
fitted_ps = ace_pot.ps
for iz in 1:n_species, jz in 1:n_species
    et_ps.rembed.post.W[:, :, (iz-1)*n_species + jz] .= fitted_ps.rbasis.Wnlq[:, :, iz, jz]
end

# Copy fitted readout weights
for iz in 1:n_species
    et_ps.readout.W[1, :, iz] .= fitted_ps.WB[:, iz]
end

println("ETACE model created")

# ## Step 6: Splinify the Radial Basis (CRITICAL!)
#
# This step converts the polynomial radial basis to Hermite cubic splines.
# **This must be done BEFORE export** for the Hermite spline radial basis mode.
#
# The splinified model evaluates much faster and produces machine-precision
# accurate results compared to the original polynomials.

println("\nSplinifying radial basis (Nspl=50 knots)...")
et_model_splined = splinify(et_model, et_ps, et_st; Nspl=50)
et_ps_splined, et_st_splined = LuxCore.setup(MersenneTwister(1234), et_model_splined)

# Copy readout weights (splinify doesn't preserve these)
for iz in 1:n_species
    et_ps_splined.readout.W[1, :, iz] .= et_ps.readout.W[1, :, iz]
end

# Create calculator for export
et_calc = ETM.ETACEPotential(et_model_splined, et_ps_splined, et_st_splined, rcut)
println("Splinified ETACE calculator ready for export")

# ## Step 7: Export to Trim-Compatible Code
#
# The export generates standalone Julia code that can be compiled with
# `juliac --trim=safe`. This code has no dependencies on ACEpotentials,
# EquivariantTensors, or Polynomials4ML.
#
# ### Radial Basis Options
#
# | Mode | Accuracy | File Size | Speed | Recommended |
# |------|----------|-----------|-------|-------------|
# | `:hermite_spline` | Machine precision | ~1 MB | Fast | Yes |
# | `:polynomial` | Exact | ~100 KB | Medium | For debugging |
#
# Use `:hermite_spline` for production - it's both faster and accurate.

include(joinpath(@__DIR__, "../../src/export_ace_model.jl"))

deploy_dir = joinpath(@__DIR__, "tial_etace_deployment")
mkpath(deploy_dir)
mkpath(joinpath(deploy_dir, "lib"))

export_file = joinpath(deploy_dir, "tial_etace_model.jl")
println("\nExporting ETACE model...")
export_ace_model(et_calc, export_file; for_library=true, radial_basis=:hermite_spline)

println("Exported to: $export_file")
println("File size: $(round(filesize(export_file)/1024, digits=1)) KB")

# ## Step 8: Compile with JuliaC
#
# Create Project.toml and compile:

project_toml = """
[deps]
JuliaC = "acedd4c2-ced6-4a15-accc-2607eb759ba2"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
"""
write(joinpath(deploy_dir, "Project.toml"), project_toml)

println("\n" * "="^60)
println("Export complete! Next steps:")
println("="^60)
println("""
1. Compile the shared library:
   cd $deploy_dir
   julia --project=. -e 'using Pkg; Pkg.instantiate(); using JuliaC; ...'

2. Use in LAMMPS (add to your input file):
   plugin load /path/to/aceplugin.so
   pair_style ace
   pair_coeff * * $deploy_dir/lib/libace_tial_etace.so Ti Al

3. Run LAMMPS:
   source $deploy_dir/setup_env.sh
   mpirun -np 4 lmp -in your_input.lmp
""")

# ## Summary
#
# The key steps for ETACE export are:
#
# 1. Use `Models.ace_model()` (not `ace1_model()`) for learnable radial basis
# 2. Fit with `acefit!()` as usual
# 3. Convert with `ETModels.convert2et()` and copy parameters
# 4. **Splinify BEFORE export** with `splinify()`
# 5. Export with `radial_basis=:hermite_spline`
# 6. Compile with `juliac --trim=safe`
#
# The resulting library is ~2x faster than standard ACE exports and requires
# no Julia installation at runtime.
