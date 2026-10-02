# Create ML-PACE deployment using LAMMPS native PACE format
# This uses ACE1.jl backend and exports to .yace format

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
println("Creating ML-PACE Deployment (LAMMPS native .yace format)")
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
# 2. Create ACE model using ace1_model
# ============================================================================
println("\n--- Creating ACE model ---")

model = ace1_model(
    elements = ELEMENTS,
    order = ORDER,
    totaldegree = TOTALDEGREE,
    rcut = RCUT,
    E0s = E0S_UNITFUL,
)

println("Model type: $(typeof(model))")
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
# 4. Export to LAMMPS PACE format
# ============================================================================
println("\n--- Exporting to LAMMPS PACE format ---")

# Create output directory
output_dir = joinpath(DEPLOYMENTS_DIR, "mlpace")
mkpath(output_dir)

# Export to .yace format
output_base = joinpath(output_dir, "fair_mlpace")

# Include the export2lammps function from outdated code
include(joinpath(pkgdir(ACEpotentials), "src", "outdated", "export.jl"))

# Export using the legacy function
export2lammps(output_base, model)

println("Created files:")
for f in readdir(output_dir)
    println("  - $f ($(round(filesize(joinpath(output_dir, f)) / 1024, digits=1)) KB)")
end

# ============================================================================
# 5. Save metadata
# ============================================================================
println("\n--- Saving metadata ---")
metadata = """
# ML-PACE Fair Benchmark Deployment
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
- PACE model: fair_mlpace.yace
- Pair table: fair_mlpace_pairpot.table

## LAMMPS Usage
```lammps
pair_style      hybrid/overlay pace table spline 5500
pair_coeff      * * pace fair_mlpace.yace Ti Al
pair_coeff      1 1 table fair_mlpace_pairpot.table Ti_Ti
pair_coeff      1 2 table fair_mlpace_pairpot.table Al_Ti
pair_coeff      2 2 table fair_mlpace_pairpot.table Al_Al
```
"""

open(joinpath(output_dir, "README.md"), "w") do io
    write(io, metadata)
end

println("\n" * "="^60)
println("ML-PACE deployment complete!")
println("="^60)
