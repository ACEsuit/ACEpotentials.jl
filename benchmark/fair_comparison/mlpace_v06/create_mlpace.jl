# Create ML-PACE deployment using ACEpotentials v0.6.x
# This uses the old ACE1.jl backend and exports to .yace format

using Pkg
Pkg.activate(@__DIR__)

# Check ACEpotentials version
using ACEpotentials
println("ACEpotentials version: ", pkgversion(ACEpotentials))

using ACE1
using ACEfit
using ExtXYZ
using Unitful
using LinearAlgebra
using JuLIP

# Model parameters (matching fair comparison params)
const ELEMENTS = [:Ti, :Al]
const ORDER = 3
const TOTALDEGREE = 8
const RCUT = 5.5
const E0S = Dict(:Ti => -1586.0195u"eV", :Al => -105.5954u"eV")
const SOLVER_LAMBDA = 1e-3
const PRIOR_P = 4
const TRAIN_STRIDE = 5
const WEIGHTS = Dict(
    "FLD_TiAl" => Dict("E" => 60.0, "F" => 1.0, "V" => 1.0),
    "TiAl_T5000" => Dict("E" => 5.0, "F" => 1.0, "V" => 1.0)
)

println("=" ^ 60)
println("Creating ML-PACE Deployment (ACEpotentials v0.6.x)")
println("=" ^ 60)
println("Elements:     $(join(ELEMENTS, ", "))")
println("Order:        $ORDER")
println("Totaldegree:  $TOTALDEGREE")
println("Cutoff:       $RCUT Å")
println("=" ^ 60)

# Output directory
output_dir = joinpath(dirname(@__DIR__), "deployments", "mlpace")
mkpath(output_dir)

# ============================================================================
# 1. Load training data
# ============================================================================
println("\n--- Loading training data ---")

# Try to use example_dataset if available, otherwise load from artifact
train_data = try
    raw_data, _, _ = ACEpotentials.example_dataset("TiAl_tutorial")
    raw_data[1:TRAIN_STRIDE:end]
catch
    # Fallback: load from artifact path
    artifact_path = joinpath(homedir(), ".julia", "artifacts")
    tial_dirs = filter(d -> isdir(joinpath(artifact_path, d)) &&
                       isfile(joinpath(artifact_path, d, "TiAl_tutorial.xyz")),
                       readdir(artifact_path))
    if !isempty(tial_dirs)
        xyz_file = joinpath(artifact_path, first(tial_dirs), "TiAl_tutorial.xyz")
        raw = read_extxyz(xyz_file)
        raw[1:TRAIN_STRIDE:end]
    else
        error("Could not find TiAl_tutorial.xyz data")
    end
end

println("Using $(length(train_data)) configurations for training")

# ============================================================================
# 2. Create ACE model using v0.6 API
# ============================================================================
println("\n--- Creating ACE model ---")

# In v0.6, use acemodel() function
model = acemodel(
    elements = ELEMENTS,
    order = ORDER,
    totaldegree = TOTALDEGREE,
    rcut = RCUT,
    E0s = E0S,
)

println("Model type: $(typeof(model))")
println("Basis size: $(length(model.basis))")

# ============================================================================
# 3. Fit the model
# ============================================================================
println("\n--- Fitting model ---")

# ACEpotentials v0.6 fitting API
acefit!(model, train_data;
    solver = ACEfit.QR(lambda=SOLVER_LAMBDA),
    weights = WEIGHTS,
    prior = smoothness_prior(model; p=PRIOR_P),
)

# ============================================================================
# 4. Export to LAMMPS PACE format
# ============================================================================
println("\n--- Exporting to LAMMPS PACE format ---")

# The export2lammps function requires a SumIP with exactly 3 components:
# OneBody, PolyPairPot, and PIPotential
# The model.potential only has 2 components (PolyPairPot and PIPotential)
# We need to construct a full potential with OneBody included

# Create the OneBody reference potential (without units for export)
E0s_no_units = Dict(:Ti => -1586.0195, :Al => -105.5954)
vref = OneBody(E0s_no_units)

# Construct the full 3-component potential
full_potential = JuLIP.MLIPs.SumIP(vref, model.potential.components...)

println("Full potential components: $(length(full_potential.components))")
for (i, c) in enumerate(full_potential.components)
    println("  Component $i: $(typeof(c).name.name)")
end

output_yace = joinpath(output_dir, "fair_mlpace.yace")
export2lammps(output_yace, full_potential)

println("Created files:")
for f in readdir(output_dir)
    fpath = joinpath(output_dir, f)
    if isfile(fpath)
        println("  - $f ($(round(filesize(fpath) / 1024, digits=1)) KB)")
    end
end

# ============================================================================
# 5. Save metadata
# ============================================================================
println("\n--- Saving metadata ---")
using Dates

metadata = """
# ML-PACE Fair Benchmark Deployment
# Generated: $(Dates.now())
# ACEpotentials version: $(pkgversion(ACEpotentials))

## Model Parameters
- Elements: $(ELEMENTS)
- Order: $(ORDER)
- Total Degree: $(TOTALDEGREE)
- Cutoff: $(RCUT) Å
- E0s: Ti=$(-1586.0195) eV, Al=$(-105.5954) eV

## Training
- Configurations: $(length(train_data))
- Solver: QR(lambda=$(SOLVER_LAMBDA))
- Prior: smoothness_prior(p=$(PRIOR_P))

## Results
- Basis size: $(length(model.basis))

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

println("\n" * "=" ^ 60)
println("ML-PACE deployment complete!")
println("=" ^ 60)
