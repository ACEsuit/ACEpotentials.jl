# # ETACE Models Tutorial
#
# This tutorial demonstrates how to use the EquivariantTensors (ET) backend
# for ACE models in ACEpotentials.jl. The ET backend provides:
# - Graph-based evaluation (edge-centric computation)
# - Automatic differentiation via Zygote
# - GPU-ready architecture via KernelAbstractions
# - Lux.jl layer integration
#
# We cover two approaches:
# 1. **Converting from an existing ACE model** - The recommended approach
# 2. **Creating an ETACE model from scratch** - For advanced users
#

## Load required packages
using ACEpotentials, StaticArrays, Lux, AtomsBase, AtomsBuilder, Unitful
using AtomsCalculators, Random, LinearAlgebra

M = ACEpotentials.Models
ETM = ACEpotentials.ETModels
import EquivariantTensors as ET
import Polynomials4ML as P4ML

rng = Random.MersenneTwister(1234)

# =============================================================================
# Part 1: Converting from an Existing ACE Model (Recommended)
# =============================================================================
#
# The simplest way to get an ETACE model is to convert from a standard ACE model.
# This approach ensures consistency with the familiar ACE model construction API.

## Define model hyperparameters
elements = (:Si, :O)
order = 3          # correlation order (body-order = order + 1)
max_level = 10     # total polynomial degree
maxl = 6           # maximum angular momentum
rcut = 5.5         # cutoff radius in Angstrom

## Create the standard ACE model
rin0cuts = M._default_rin0cuts(elements)
rin0cuts = (x -> (rin = x.rin, r0 = x.r0, rcut = rcut)).(rin0cuts)

## Note: pair_learnable=true is required for ET conversion
## (default uses splines which aren't yet supported by convert2et)
model = M.ace_model(;
   elements = elements,
   order = order,
   Ytype = :solid,
   level = M.TotalDegree(),
   max_level = max_level,
   maxl = maxl,
   pair_maxn = max_level,
   rin0cuts = rin0cuts,
   E0s = Dict(:Si => -0.846, :O => -1.023),  # reference energies
   pair_learnable = true   # required for ET conversion
)

## Initialize parameters with Lux
ps, st = Lux.setup(rng, model)

@info "Standard ACE model created"
@info "  Number of basis functions: $(M.length_basis(model))"

# -----------------------------------------------------------------------------
# Method A: Convert full model (E0 + Pair + Many-body) to StackedCalculator
# -----------------------------------------------------------------------------

## convert2et_full creates a StackedCalculator combining:
##   - ETOneBody (reference energies per species)
##   - ETPairModel (pair potential)
##   - ETACE (many-body ACE potential)

et_calc_full = ETM.convert2et_full(model, ps, st; rng=rng)

@info "Full conversion to StackedCalculator"
@info "  Contains: ETOneBody + ETPairPotential + ETACEPotential"
@info "  Total linear parameters: $(ETM.length_basis(et_calc_full))"

# -----------------------------------------------------------------------------
# Method B: Convert only the many-body ACE component
# -----------------------------------------------------------------------------

## convert2et creates just the ETACE model (many-body only, no E0 or pair)
et_ace = ETM.convert2et(model)
et_ace_ps, et_ace_st = Lux.setup(rng, et_ace)

## Copy parameters from the original model
ETM.copy_ace_params!(et_ace_ps, ps, model)

## Wrap in calculator for AtomsCalculators interface
et_ace_calc = ETM.ETACEPotential(et_ace, et_ace_ps, et_ace_st, rcut)

@info "Many-body only conversion"
@info "  ETACE basis size: $(ETM.length_basis(et_ace_calc))"

# -----------------------------------------------------------------------------
# Method C: Convert only the pair potential
# -----------------------------------------------------------------------------

## convertpair creates an ETPairModel
et_pair = ETM.convertpair(model)
et_pair_ps, et_pair_st = Lux.setup(rng, et_pair)

## Copy parameters from the original model
ETM.copy_pair_params!(et_pair_ps, ps, model)

## Wrap in calculator
et_pair_calc = ETM.ETPairPotential(et_pair, et_pair_ps, et_pair_st, rcut)

@info "Pair potential only conversion"
@info "  ETPairModel basis size: $(ETM.length_basis(et_pair_calc))"


# =============================================================================
# Part 2: Using ETACE Calculators
# =============================================================================

## Create a test system
sys = AtomsBuilder.bulk(:Si) * (2, 2, 1)
rattle!(sys, 0.1u"Å")
AtomsBuilder.randz!(sys, [:Si => 0.5, :O => 0.5])

@info "Test system: $(length(sys)) atoms"

## Evaluate energy, forces, virial using AtomsCalculators interface
E = AtomsCalculators.potential_energy(sys, et_calc_full)
F = AtomsCalculators.forces(sys, et_calc_full)
V = AtomsCalculators.virial(sys, et_calc_full)

@info "Energy evaluation with full ETACE calculator"
@info "  Energy: $E"
@info "  Max force magnitude: $(maximum(norm.(F)))"

## Combined evaluation (more efficient)
efv = AtomsCalculators.energy_forces_virial(sys, et_calc_full)
@info "  Combined EFV evaluation successful"


# =============================================================================
# Part 3: Training Assembly (for Linear Fitting)
# =============================================================================
#
# The ETACE calculators support training assembly functions for ACEfit integration.
# These compute the design matrix rows for linear least squares fitting.

## Energy-only basis evaluation (fastest)
E_basis = ETM.potential_energy_basis(sys, et_ace_calc)
@info "Energy basis: $(length(E_basis)) components"

## Full energy, forces, virial basis
efv_basis = ETM.energy_forces_virial_basis(sys, et_ace_calc)
@info "EFV basis shapes:"
@info "  Energy:  $(size(efv_basis.energy))"
@info "  Forces:  $(size(efv_basis.forces))"
@info "  Virial:  $(size(efv_basis.virial))"

## Get/set linear parameters
params = ETM.get_linear_parameters(et_ace_calc)
@info "Linear parameters: $(length(params)) values"

## Parameters can be updated for fitting:
## ETM.set_linear_parameters!(et_ace_calc, new_params)


# =============================================================================
# Part 4: Creating an ETACE Model from Scratch (Advanced)
# =============================================================================
#
# For advanced users who want direct control over the model architecture.
# This requires understanding the EquivariantTensors.jl API.

## Define model parameters
scratch_elements = [:Si, :O]
scratch_maxn = 6      # number of radial basis functions
scratch_maxl = 4      # maximum angular momentum
scratch_order = 2     # correlation order
scratch_rcut = 5.5    # cutoff radius

## Species information
zlist = ChemicalSpecies.(scratch_elements)
NZ = length(zlist)

# -----------------------------------------------------------------------------
# Build the radial embedding (Rnl)
# -----------------------------------------------------------------------------

## Radial specification (n, l pairs)
Rnl_spec = [(n=n, l=l) for n in 1:scratch_maxn for l in 0:scratch_maxl]

## Distance transform: r -> transformed coordinate y
## Using standard Agnesi transform parameters
f_trans = let rcut = scratch_rcut
   (x, st) -> begin
      r = norm(x.𝐫)
      # Simple polynomial transform (normalized to [-1, 1])
      y = 1 - 2 * r / rcut
      return y
   end
end
trans = ET.NTtransformST(f_trans, NamedTuple())

## Envelope function: smooth cutoff
f_env = y -> (1 - y^2)^2  # quartic envelope

## Polynomial basis (Chebyshev)
polys = P4ML.ChebBasis(scratch_maxn)
Penv = P4ML.wrapped_basis(Lux.BranchLayer(
   polys,
   Lux.WrappedFunction(y -> f_env.(y)),
   fusion = Lux.WrappedFunction(Pe -> Pe[2] .* Pe[1])
))

## Species-pair selector for radial weights
selector_ij = let zlist = tuple(zlist...)
   xij -> ET.catcat2idx(zlist, xij.z0, xij.z1)
end

## Linear layer: P(yij) -> W[(Zi, Zj)] * P(yij)
linl = ET.SelectLinL(scratch_maxn, length(Rnl_spec), NZ^2, selector_ij)

## Complete radial embedding
rbasis = ET.EmbedDP(trans, Penv, linl)
rembed = ET.EdgeEmbed(rbasis)

# -----------------------------------------------------------------------------
# Build the angular embedding (Ylm)
# -----------------------------------------------------------------------------

## Spherical harmonics basis
ylm_basis = P4ML.real_sphericalharmonics(scratch_maxl)
Ylm_spec = P4ML.natural_indices(ylm_basis)

## Angular embedding: edge direction -> spherical harmonics
ybasis = ET.EmbedDP(
   ET.NTtransformST((x, st) -> x.𝐫, NamedTuple()),
   ylm_basis
)
yembed = ET.EdgeEmbed(ybasis)

# -----------------------------------------------------------------------------
# Build the many-body basis (sparse ACE)
# -----------------------------------------------------------------------------

## Define the many-body specification
## This specifies which (n,l) combinations appear in each correlation
## For simplicity, use all 1-correlations up to given degree
mb_spec = [[(n=n, l=l)] for n in 1:scratch_maxn for l in 0:scratch_maxl]

## Create sparse equivariant tensor (ACE basis)
mb_basis = ET.sparse_equivariant_tensor(
   L = 0,                # scalar (invariant) output
   mb_spec = mb_spec,
   Rnl_spec = Rnl_spec,
   Ylm_spec = Ylm_spec,
   basis = real          # real-valued basis
)

# -----------------------------------------------------------------------------
# Build the readout layer
# -----------------------------------------------------------------------------

## Species selector for readout
selector_i = let zlist = zlist
   x -> ET.cat2idx(zlist, x.z)
end

## Readout: basis values -> site energies
readout = ET.SelectLinL(
   mb_basis.lens[1],     # input dimension (basis length)
   1,                    # output dimension (site energy)
   NZ,                   # number of species categories
   selector_i
)

# -----------------------------------------------------------------------------
# Assemble the ETACE model
# -----------------------------------------------------------------------------

scratch_etace = ETM.ETACE(rembed, yembed, mb_basis, readout)

## Initialize with Lux
scratch_ps, scratch_st = Lux.setup(rng, scratch_etace)

@info "ETACE model created from scratch"
@info "  Radial basis size: $(length(Rnl_spec))"
@info "  Angular basis size: $(length(Ylm_spec))"
@info "  Many-body basis size: $(mb_basis.lens[1])"

## Wrap in calculator
scratch_calc = ETM.ETACEPotential(scratch_etace, scratch_ps, scratch_st, scratch_rcut)

## Test evaluation
E_scratch = AtomsCalculators.potential_energy(sys, scratch_calc)
@info "Scratch model energy: $E_scratch"


# =============================================================================
# Part 5: Creating One-Body and Pair Models from Scratch
# =============================================================================

# -----------------------------------------------------------------------------
# ETOneBody: Reference energies
# -----------------------------------------------------------------------------

## Define reference energies per species
E0_dict = Dict(ChemicalSpecies(:Si) => -0.846,
               ChemicalSpecies(:O) => -1.023)

## Category function extracts species from atom state
catfun = x -> x.z  # x.z is the ChemicalSpecies

## Create one-body model
et_onebody = ETM.one_body(E0_dict, catfun)
_, onebody_st = Lux.setup(rng, et_onebody)

## Wrap in calculator (uses small cutoff since no neighbors needed)
onebody_calc = ETM.ETOneBodyPotential(et_onebody, nothing, onebody_st, 3.0)

@info "ETOneBody model created"
@info "  Reference energies: $E0_dict"

E_onebody = AtomsCalculators.potential_energy(sys, onebody_calc)
@info "  One-body energy for test system: $E_onebody"


# =============================================================================
# Part 6: Combining Models with StackedCalculator
# =============================================================================
#
# StackedCalculator combines multiple calculators by summing their contributions.

## Stack our from-scratch models
combined_calc = ETM.StackedCalculator((onebody_calc, scratch_calc))

@info "StackedCalculator created"
@info "  Components: ETOneBody + ETACE"
@info "  Total basis size: $(ETM.length_basis(combined_calc))"

## Evaluate combined model
E_combined = AtomsCalculators.potential_energy(sys, combined_calc)
@info "  Combined energy: $E_combined"

## Training assembly works on StackedCalculator too
efv_combined = ETM.energy_forces_virial_basis(sys, combined_calc)
@info "  Combined EFV basis shapes: E=$(size(efv_combined.energy)), F=$(size(efv_combined.forces))"

@info "Tutorial complete!"
