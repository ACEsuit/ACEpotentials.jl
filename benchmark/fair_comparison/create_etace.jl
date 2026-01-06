# Create ETACE deployments (both Spline and Polynomial)
# Uses modern EquivariantTensors backend with Models.ace_model()

using Pkg
Pkg.activate(@__DIR__)

using ACEpotentials
using ACEpotentials.Models
const M = ACEpotentials.Models
using ACEpotentials.ETModels
const ETM = ACEpotentials.ETModels
using ACEfit
using ExtXYZ
using Unitful
using LinearAlgebra
using Random
using Lux
using Dates

# Load unified parameters
include("params.jl")
using .FairParams

println("="^60)
println("Creating ETACE Deployments (Spline + Polynomial)")
println("="^60)
FairParams.print_summary()

# ============================================================================
# 1. Load training data
# ============================================================================
println("\n--- Loading training data ---")
raw_data, _, _ = ACEpotentials.example_dataset("TiAl_tutorial")
train_data = raw_data[1:TRAIN_STRIDE:end]
println("Using $(length(train_data)) configurations for training")

# ============================================================================
# 2. Create ETACE model using Models.ace_model()
# ============================================================================
println("\n--- Creating ETACE model ---")

# Models.ace_model() uses LearnableRnlrzzBasis which can be converted to ETACE
# Use TotalDegree level spec to match ace1_model behavior
NZ = length(ELEMENTS)
level = M.TotalDegree(1.0 * NZ, 1/1.5)  # Match ace1_model level spec

# Create custom rin0cuts matrix for the specific cutoff (5.5 Å)
# Format: SMatrix of named tuples (rin, r0, rcut)
using StaticArrays
r0_TiTi = 2.95  # Ti-Ti bond length estimate
r0_AlAl = 2.86  # Al-Al bond length estimate
r0_TiAl = 2.90  # Ti-Al bond length estimate
# Create 2x2 matrix: [Ti-Ti Ti-Al; Ti-Al Al-Al]
rin0cuts = SMatrix{NZ, NZ}([
    (rin=0.0, r0=r0_TiTi, rcut=RCUT) (rin=0.0, r0=r0_TiAl, rcut=RCUT)
    (rin=0.0, r0=r0_TiAl, rcut=RCUT) (rin=0.0, r0=r0_AlAl, rcut=RCUT)
])

model = M.ace_model(
    elements = ELEMENTS,
    order = ORDER,
    level = level,
    max_level = TOTALDEGREE,
    maxl = MAXL,
    rin0cuts = rin0cuts,
    pair_maxn = 8,  # Pair potential polynomial degree
    init_Wradial = :onehot,
    init_Wpair = :zero,
    E0s = E0S,  # Reference energies (creates built-in OneBody)
)

# Initialize parameters
rng = Random.MersenneTwister(12345)
ps, st = Lux.setup(rng, model)

println("Model type: $(typeof(model))")
println("Radial basis type: $(typeof(model.rbasis))")
nbasis = size(ps.WB, 1)
println("Basis size: $nbasis")

# Create calculator for fitting (model has built-in OneBody from E0s)
pot = M.ACEPotential(model, ps, st)

# ============================================================================
# 3. Fit the ETACE model
# ============================================================================
println("\n--- Fitting model ---")

# Use acefit! which handles all the assembly/solving internally
acefit!(train_data, pot;
    solver = ACEfit.QR(lambda=SOLVER_LAMBDA),
    weights = WEIGHTS,
    smoothness = PRIOR_P,
)

println("Fitting complete")

# ============================================================================
# 4. Convert to ETACE and create both exports
# ============================================================================
println("\n--- Converting to ETACE ---")

using LuxCore
using AtomsBase: ChemicalSpecies

# Convert ACEModel to ETACE
et_model = ETM.convert2et(pot.model)

# Setup new parameters for ETACE model and copy from fitted ACE model
et_ps, et_st = LuxCore.setup(Random.MersenneTwister(1234), et_model)

# Copy radial basis parameters from ps.rbasis.Wnlq to et_ps.rembed.post.W
n_species = NZ
for iz in 1:n_species
    for jz in 1:n_species
        et_ps.rembed.post.W[:, :, (iz-1)*n_species + jz] .= pot.ps.rbasis.Wnlq[:, :, iz, jz]
    end
end

# Copy readout (WB) parameters
for iz in 1:n_species
    et_ps.readout.W[1, :, iz] .= pot.ps.WB[:, iz]
end

println("ETACE model created")
println("ETACE type: $(typeof(et_model))")

# Create ETACE calculator
et_calc = ETM.ETACEPotential(et_model, et_ps, et_st, RCUT)

# Include the export function
include(joinpath(pkgdir(ACEpotentials), "export", "src", "export_ace_model.jl"))

# ============================================================================
# 4a. Create ETACE Polynomial deployment
# ============================================================================
println("\n--- Creating ETACE Polynomial deployment ---")

output_dir_poly = joinpath(DEPLOYMENTS_DIR, "etace_poly")
lib_dir_poly = joinpath(output_dir_poly, "lib")
mkpath(lib_dir_poly)

# Add OneBody calculator using one_body helper with ChemicalSpecies keys
E0_dict_species = Dict(ChemicalSpecies(e) => E0S[e] for e in ELEMENTS)
e0_model = ETM.one_body(E0_dict_species, x -> x.z)
e0_ps, e0_st = LuxCore.setup(Random.MersenneTwister(1234), e0_model)
e0_calc = ETM.ETOneBodyPotential(e0_model, e0_ps, e0_st, RCUT)
stacked_calc = ETM.StackedCalculator((e0_calc, et_calc))

model_file_poly = joinpath(output_dir_poly, "fair_etace_poly_model.jl")
export_ace_model(stacked_calc, model_file_poly; for_library=true, radial_basis=:polynomial)
println("Exported polynomial model to: $model_file_poly")

# Compile with juliac
julia_bindir = Sys.BINDIR
juliac_path = joinpath(julia_bindir, "..", "share", "julia", "juliac.jl")
if !isfile(juliac_path)
    juliac_path = joinpath(julia_bindir, "..", "share", "julia", "juliac", "juliac.jl")
end

if isfile(juliac_path)
    output_lib_poly = joinpath(lib_dir_poly, "libace_fair_etace_poly.so")
    cmd = `julia --project=$(pkgdir(ACEpotentials))/export $(juliac_path) --output-lib $(output_lib_poly) --experimental --trim=safe $(model_file_poly)`
    println("Compiling polynomial model...")
    run(cmd)
    if isfile(output_lib_poly)
        filesize_mb = round(filesize(output_lib_poly) / 1024^2, digits=1)
        println("Success! Created: $output_lib_poly ($filesize_mb MB)")
    end
end

# ============================================================================
# 4b. Create ETACE Spline deployment
# ============================================================================
println("\n--- Creating ETACE Spline deployment ---")

# Splinify the model for faster evaluation
# The splinify function embeds the radial basis parameters into splines
println("Splinifying model...")
et_model_splined = ETM.splinify(et_model, et_ps, et_st; Nspl=50)

# Setup new parameters for splined model (radial params are embedded in splines)
et_ps_splined, et_st_splined = LuxCore.setup(Random.MersenneTwister(1234), et_model_splined)

# Copy readout (WB) parameters to the splined model
for iz in 1:n_species
    et_ps_splined.readout.W[1, :, iz] .= pot.ps.WB[:, iz]
end

# Create splinified calculator
et_calc_splined = ETM.ETACEPotential(et_model_splined, et_ps_splined, et_st_splined, RCUT)
stacked_calc_splined = ETM.StackedCalculator((e0_calc, et_calc_splined))

output_dir_spline = joinpath(DEPLOYMENTS_DIR, "etace_spline")
lib_dir_spline = joinpath(output_dir_spline, "lib")
mkpath(lib_dir_spline)

model_file_spline = joinpath(output_dir_spline, "fair_etace_spline_model.jl")
export_ace_model(stacked_calc_splined, model_file_spline; for_library=true, radial_basis=:hermite_spline)
println("Exported spline model to: $model_file_spline")

# Compile with juliac
if isfile(juliac_path)
    output_lib_spline = joinpath(lib_dir_spline, "libace_fair_etace_spline.so")
    cmd = `julia --project=$(pkgdir(ACEpotentials))/export $(juliac_path) --output-lib $(output_lib_spline) --experimental --trim=safe $(model_file_spline)`
    println("Compiling spline model...")
    run(cmd)
    if isfile(output_lib_spline)
        filesize_mb = round(filesize(output_lib_spline) / 1024^2, digits=1)
        println("Success! Created: $output_lib_spline ($filesize_mb MB)")
    end
end

# ============================================================================
# 5. Save metadata
# ============================================================================
println("\n--- Saving metadata ---")

for (output_dir, variant) in [(output_dir_poly, "Polynomial"), (output_dir_spline, "Spline")]
    metadata = """
# ETACE $(variant) Fair Benchmark Deployment
# Generated: $(Dates.now())

## Model Parameters
- Elements: $(ELEMENTS)
- Order: $(ORDER)
- Total Degree: $(TOTALDEGREE)
- Max L: $(MAXL)
- Cutoff: $(RCUT) A
- E0s: Ti=$(E0S[:Ti]) eV, Al=$(E0S[:Al]) eV

## Training
- Configurations: $(length(train_data))
- Solver: QR(lambda=$(SOLVER_LAMBDA))
- Prior: algebraic_smoothness_prior(p=$(PRIOR_P))

## Results
- Basis size: $(nbasis)

## Files
- Model source: fair_etace_$(lowercase(variant))_model.jl
- Compiled lib: lib/libace_fair_etace_$(lowercase(variant)).so

## Notes
- Uses EquivariantTensors backend with LearnableRnlrzzBasis
- Converted to ETACE format for export
$(variant == "Spline" ? "- Splinified with Nspl=50 for Hermite cubic spline evaluation" : "- Direct polynomial evaluation (no splines)")
"""

    open(joinpath(output_dir, "README.md"), "w") do io
        write(io, metadata)
    end
end

println("\n" * "="^60)
println("ETACE deployments complete!")
println("="^60)
