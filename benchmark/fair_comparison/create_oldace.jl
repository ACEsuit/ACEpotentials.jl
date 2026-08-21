# Create Old ACE deployment using the pre-ETACE code generation approach
# This uses ACEPotential with SplineRnlrzzBasis, exported via the old code path

using Pkg
Pkg.activate(@__DIR__)

using ACEpotentials
using ACEpotentials.Models
const M = ACEpotentials.Models
using ACEfit
using ExtXYZ
using Unitful
using LinearAlgebra

# Load unified parameters
include("params.jl")
using .FairParams

println("="^60)
println("Creating Old ACE Deployment (pre-ETACE code generation)")
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
# 2. Create ACE model using ace1_model (creates SplineRnlrzzBasis)
# ============================================================================
println("\n--- Creating ACE model ---")

# Note: ace1_model uses wL=1.5 internally, doesn't expose maxl directly
# We use totaldegree and order to control complexity
model = ace1_model(
    elements = ELEMENTS,
    order = ORDER,
    totaldegree = TOTALDEGREE,
    rcut = RCUT,
    E0s = E0S_UNITFUL,
)

println("Model type: $(typeof(model))")
println("Radial basis type: $(typeof(model.model.rbasis))")
nbasis = length(model.ps.WB[:, 1])
println("Basis size: $nbasis")

# ============================================================================
# 3. Fit the model
# ============================================================================
println("\n--- Fitting model ---")

# Fit with QR solver and smoothness prior
acefit!(
    train_data, model;
    solver = ACEfit.QR(lambda=SOLVER_LAMBDA),
    weights = WEIGHTS,
    smoothness = PRIOR_P,
)

# Training errors are already printed by acefit!

# ============================================================================
# 4. Export using old code-generation approach
# ============================================================================
println("\n--- Exporting model ---")

# Create output directory
output_dir = joinpath(DEPLOYMENTS_DIR, "oldace")
lib_dir = joinpath(output_dir, "lib")
mkpath(lib_dir)

# Include the old export code
include("export_ace_model_oldace.jl")

# Export to Julia file
model_file = joinpath(output_dir, "fair_oldace_model.jl")
export_ace_model(model, model_file; splinify_first=true, for_library=true)
println("Exported to: $model_file")

# ============================================================================
# 5. Compile with juliac
# ============================================================================
println("\n--- Compiling with juliac ---")

# Find juliac.jl in the Julia installation
julia_bindir = Sys.BINDIR
juliac_path = joinpath(julia_bindir, "..", "share", "julia", "juliac.jl")

if !isfile(juliac_path)
    # Try alternate location
    juliac_path = joinpath(julia_bindir, "..", "share", "julia", "juliac", "juliac.jl")
end

if isfile(juliac_path)
    output_lib = joinpath(lib_dir, "libace_fair_oldace.so")

    # Build the compile command (note: -C generic no longer supported in juliac, --experimental needed for --trim)
    cmd = `julia --project=$(pkgdir(ACEpotentials))/export $(juliac_path) --output-lib $(output_lib) --experimental --trim=safe $(model_file)`

    println("Running: $cmd")
    println("This may take a few minutes...")

    run(cmd)

    if isfile(output_lib)
        filesize_mb = round(filesize(output_lib) / 1024^2, digits=1)
        println("\nSuccess! Created: $output_lib ($filesize_mb MB)")
    else
        println("\nWarning: Library file not found after compilation")
    end
else
    println("Warning: juliac.jl not found at $juliac_path")
    println("Please compile manually:")
    println("  julia --project=\$ACEPOT/export \$JULIAC --output-lib $lib_dir/libace_fair_oldace.so --trim=safe $model_file")
end

# ============================================================================
# 6. Save metadata
# ============================================================================
println("\n--- Saving metadata ---")
metadata = """
# Old ACE Fair Benchmark Deployment
# Generated: $(Dates.now())

## Model Parameters
- Elements: $(ELEMENTS)
- Order: $(ORDER)
- Total Degree: $(TOTALDEGREE)
- Cutoff: $(RCUT) A
- E0s: Ti=$(E0S[:Ti]) eV, Al=$(E0S[:Al]) eV

## Training
- Configurations: $(length(train_data))
- Solver: QR(lambda=$(SOLVER_LAMBDA))
- Prior: algebraic_smoothness_prior(p=$(PRIOR_P))

## Results
- Basis size: $(nbasis)
- Energy RMSE: $(round(train_errors["rmse"]["E"] * 1000, digits=2)) meV/atom
- Force RMSE: $(round(train_errors["rmse"]["F"], digits=3)) eV/A

## Files
- Model source: fair_oldace_model.jl
- Compiled lib: lib/libace_fair_oldace.so
"""

open(joinpath(output_dir, "README.md"), "w") do io
    write(io, metadata)
end

println("\n" * "="^60)
println("Old ACE deployment complete!")
println("="^60)
