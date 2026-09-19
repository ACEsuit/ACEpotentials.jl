# # ETACE Model: Fitting and Export to LAMMPS
#
# This tutorial demonstrates the complete workflow for creating an ETACE
# (EquivariantTensors ACE) model and exporting it to LAMMPS.
#
# ## Overview
#
# ETACE models use the EquivariantTensors backend and export to standalone, trim-compatible
# Julia code. No speed comparison against standard ACE has been measured in this repository,
# so none is quoted here. The workflow is:
#
# 1. **Create** an ACE model with learnable radial basis
# 2. **Fit** the model to training data
# 3. **Convert** to ETACE format
# 4. **Export** to trim-compatible Julia code
# 5. **Compile** to a shared library
# 6. **Deploy** to LAMMPS
#
# An earlier version of this tutorial had a **Splinify** step here, called "critical", and
# exported with `radial_basis=:hermite_spline`. That mode has been REMOVED, and with it the
# export path for a splinified model: `splinify()` is now something you must NOT do before
# exporting. Step 6 says why, and `export/bench/FINDINGS_parity.md` §7 has the measurements
# the decision was made on.
#
# ## When to Use ETACE vs Standard ACE
#
# | Feature | Standard ACE | ETACE |
# |---------|-------------|-------|
# | Evaluation speed | not measured here | not measured here |
# | Export complexity | Simple | Requires conversion |
# | Radial basis | Pre-splinified | Learnable, exported as the polynomial recurrence |
# | Recommended for | Quick tests, small systems | Production MD |

# ## Step 1: Setup and Load Data

using ACEpotentials
using ACEpotentials.Models
using ACEpotentials.ETModels
# (splinify is deliberately NOT imported: a splinified model cannot be exported -- Step 6)
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
# The ETACE format uses the EquivariantTensors backend.
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

# ## Step 6: Build the Export Calculator (and do NOT splinify)
#
# **This step used to call `splinify()`, and called it critical. Do not do that.**
# `ETModels.splinify` replaces the radial polynomial recurrence with cubic-spline knot
# tables, and the export emits that recurrence — so a splinified model has nothing left to
# export and `export_ace_model` refuses it outright, by design.
#
# The `:hermite_spline` mode that once consumed those knot tables was removed. It was
# **slower** than the exact export on both reference models (62.5 vs 58.1 µs/site on Cantor,
# 152.5 vs 94.0 on TiAl), **approximate** (2.7e-4 eV/Å on Cantor and 1.6e-2 eV/Å on TiAl at
# `Nspl = 50`, against a mode gated at 1e-12), and unusable with per-pair cutoffs. The
# evidence is in `export/bench/FINDINGS_parity.md` §7; the answer was to remove it, which
# retires the whole splinify-then-deploy route.
#
# So the fitted, UNSPLINIFIED ETACE model of Step 5 is what gets exported, and what the
# library evaluates is the model that was fitted — to 1e-12 in energies, forces and virial.
# There is no approximation anywhere in this tutorial's deployment path any more, which is a
# straightforward improvement on the version that had a Step 6 warning about one.
#
# `splinify` itself still exists and still works; it is only useful for in-Julia experiments
# now, not for deployment.

# Create calculator for export -- the UNSPLINIFIED model from Step 5
et_calc = ETM.ETACEPotential(et_model, et_ps, et_st, rcut)
println("\nETACE calculator ready for export (not splinified -- see the note above)")

# ## Step 7: Export to Trim-Compatible Code
#
# The export generates standalone Julia code that can be compiled with
# `juliac --trim=safe`. This code has no dependencies on ACEpotentials,
# EquivariantTensors, or Polynomials4ML.
#
# ### Radial Basis Options
#
# There is one mode, and it is the default, so there is nothing to choose. The
# `radial_basis` keyword is vestigial and is omitted below.
#
# | Mode | Accuracy | Reproduces | Status |
# |------|----------|------------|--------|
# | `:polynomial` | exact (1e-12 in energy, forces and virial) | the fitted model | the only mode |
# | `:hermite_spline` | — | — | **REMOVED** (see Step 6); raises |

include(joinpath(@__DIR__, "../../src/export_ace_model.jl"))

deploy_dir = joinpath(@__DIR__, "tial_etace_deployment")
mkpath(deploy_dir)
mkpath(joinpath(deploy_dir, "lib"))

export_file = joinpath(deploy_dir, "tial_etace_model.jl")
println("\nExporting ETACE model...")
# No radial_basis keyword: :polynomial is the default and the only mode, and the model has
# deliberately not been splinified (Step 6).
export_ace_model(et_calc, export_file; for_library=true)

println("Exported to: $export_file")
println("File size: $(round(filesize(export_file)/1024, digits=1)) KB")

# ## Step 8: Compile with JuliaC
#
# Create Project.toml and compile. The following code writes a Project.toml
# and prints the next steps:

#jl project_toml = """
#jl [deps]
#jl JuliaC = "acedd4c2-ced6-4a15-accc-2607eb759ba2"
#jl LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
#jl StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
#jl """
#jl write(joinpath(deploy_dir, "Project.toml"), project_toml)

#jl println("\n" * "="^60)
#jl println("Export complete! Next steps:")
#jl println("="^60)
#jl println("""
#jl 1. Compile the shared library:
#jl    cd $deploy_dir
#jl    julia --project=. -e 'using Pkg; Pkg.instantiate(); using JuliaC; ...'
#jl
#jl 2. Use in LAMMPS (add to your input file):
#jl    plugin load /path/to/aceplugin.so
#jl    pair_style ace
#jl    pair_coeff * * $deploy_dir/lib/libace_tial_etace.so Ti Al
#jl
#jl 3. Run LAMMPS:
#jl    source $deploy_dir/setup_env.sh
#jl    mpirun -np 4 lmp -in your_input.lmp
#jl """)

# ```julia
# # Write Project.toml for compilation dependencies
# project_toml = """
# [deps]
# JuliaC = "acedd4c2-ced6-4a15-accc-2607eb759ba2"
# LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
# StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
# """
# write(joinpath(deploy_dir, "Project.toml"), project_toml)
# ```
#
# After export, compile and use in LAMMPS:
#
# ```bash
# # 1. Compile the shared library
# cd tial_etace_deployment
# julia --project=. -e '
#     using Pkg; Pkg.instantiate()
#     using JuliaC
#     recipe = ImageRecipe(;
#         file="tial_etace_model.jl",
#         output_type="sharedlib",
#         trim_mode="safe",
#         img_path="lib/libace_tial_etace.so",
#         add_ccallables=true
#     )
#     compile_products(recipe)
# '
#
# # 2. Use in LAMMPS input file
# plugin load /path/to/aceplugin.so
# pair_style ace
# pair_coeff * * /path/to/lib/libace_tial_etace.so Ti Al
#
# # 3. Run LAMMPS
# mpirun -np 4 lmp -in your_input.lmp
# ```

# ## Summary
#
# The key steps for ETACE export are:
#
# 1. Use `Models.ace_model()` (not `ace1_model()`) for learnable radial basis
# 2. Fit with `acefit!()` as usual
# 3. Convert with `ETModels.convert2et()` and copy parameters
# 4. **Do NOT splinify.** A splinified model cannot be exported at all; the export emits the
#    polynomial recurrence that `splinify()` removes (see Step 6)
# 5. Export -- there is one mode, it is the default, and it is exact against the fitted model
#    to 1e-12
# 6. Compile with `juliac --trim=safe`
#
# The resulting library requires no Julia installation at runtime. Its evaluation cost has
# not been benchmarked against a standard ACE export in this repository, so no speed ratio is
# quoted here; see `export/bench/README.md`.
