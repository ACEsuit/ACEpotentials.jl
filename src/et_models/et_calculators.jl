
# Calculator interfaces for ETACE models
# Provides AtomsCalculators-compatible energy/forces/virial evaluation
#
# Architecture:
# - WrappedSiteCalculator: Unified wrapper for ETACE-pattern models (ETACE, ETPairModel, ETOneBody)
# - ETACEPotential: Type alias for WrappedSiteCalculator with ETACE model
# - StackedCalculator: Combines multiple calculators (see stackedcalc.jl)
#
# All wrapped models must implement the ETACE interface:
#   model(G, ps, st) -> (site_energies, st)
#   site_grads(model, G, ps, st) -> edge gradients
#
# See also: stackedcalc.jl for StackedCalculator (combines multiple calculators)

import AtomsCalculators
import AtomsBase: AbstractSystem, ChemicalSpecies
import EquivariantTensors as ET
using DecoratedParticles: PState
using StaticArrays
using Unitful
using LinearAlgebra: norm


# ============================================================================
#  WrappedSiteCalculator - Unified wrapper for ETACE-pattern models
# ============================================================================

"""
    WrappedSiteCalculator{M, PS, ST}

Wraps any ETACE-pattern model (ETACE, ETPairModel, ETOneBody) and provides
the AtomsCalculators interface.

All wrapped models must implement the ETACE interface:
- `model(G, ps, st)` → `(site_energies, st)`
- `site_grads(model, G, ps, st)` → edge gradients

Mutable to allow parameter updates during training.

# Example
```julia
# With ETACE model
calc = WrappedSiteCalculator(et_model, ps, st, 5.5)

# With ETOneBody (upstream)
et_onebody = ETM.one_body(Dict(:Si => -0.846), x -> x.z)
_, onebody_st = Lux.setup(rng, et_onebody)
calc = WrappedSiteCalculator(et_onebody, nothing, onebody_st, 3.0)

E = potential_energy(sys, calc)
F = forces(sys, calc)
```

# Fields
- `model` - ETACE-pattern model (ETACE, ETPairModel, or ETOneBody)
- `ps` - Model parameters (can be `nothing` for ETOneBody)
- `st` - Model state
- `rcut::Float64` - Cutoff radius for graph construction (Å)
- `co_ps` - Optional committee parameters for uncertainty quantification
"""
mutable struct WrappedSiteCalculator{M, PS, ST}
   model::M
   ps::PS
   st::ST
   rcut::Float64
   co_ps::Any
end

# Constructor without committee parameters
function WrappedSiteCalculator(model, ps, st, rcut::Real)
   return WrappedSiteCalculator(model, ps, st, Float64(rcut), nothing)
end

cutoff_radius(calc::WrappedSiteCalculator) = calc.rcut * u"Å"

function _wrapped_energy(calc::WrappedSiteCalculator, sys::AbstractSystem)
   G = ET.Atoms.interaction_graph(sys, calc.rcut * u"Å")
   Ei, _ = calc.model(G, calc.ps, calc.st)
   return sum(Ei)
end

function _wrapped_forces(calc::WrappedSiteCalculator, sys::AbstractSystem)
   G = ET.Atoms.interaction_graph(sys, calc.rcut * u"Å")
   ∂G = site_grads(calc.model, G, calc.ps, calc.st)
   # Handle empty edge case (e.g., ETOneBody with small cutoff)
   if isempty(∂G.edge_data)
      return zeros(SVector{3, Float64}, length(sys))
   end
   # forces_from_edge_grads returns +∇E, negate for forces
   return -ET.Atoms.forces_from_edge_grads(sys, G, ∂G.edge_data)
end

# Compute virial tensor from edge gradients
function _compute_virial(G::ET.ETGraph, ∂G)
   # V = -∑ (∂E/∂𝐫ij) ⊗ 𝐫ij
   V = zeros(SMatrix{3,3,Float64,9})
   for (edge, ∂edge) in zip(G.edge_data, ∂G.edge_data)
      V -= ∂edge.𝐫 * edge.𝐫'
   end
   return V
end

function _wrapped_virial(calc::WrappedSiteCalculator, sys::AbstractSystem)
   G = ET.Atoms.interaction_graph(sys, calc.rcut * u"Å")
   ∂G = site_grads(calc.model, G, calc.ps, calc.st)
   # Handle empty edge case
   if isempty(∂G.edge_data)
      return zeros(SMatrix{3,3,Float64,9})
   end
   return _compute_virial(G, ∂G)
end

function _wrapped_energy_forces_virial(calc::WrappedSiteCalculator, sys::AbstractSystem)
   G = ET.Atoms.interaction_graph(sys, calc.rcut * u"Å")

   # Energy from site energies (call model directly - ETACE interface)
   Ei, _ = calc.model(G, calc.ps, calc.st)
   E = sum(Ei)

   # Forces and virial from edge gradients
   ∂G = site_grads(calc.model, G, calc.ps, calc.st)

   # Handle empty edge case (e.g., ETOneBody with small cutoff)
   if isempty(∂G.edge_data)
      F = zeros(SVector{3, Float64}, length(sys))
      V = zeros(SMatrix{3,3,Float64,9})
   else
      F = -ET.Atoms.forces_from_edge_grads(sys, G, ∂G.edge_data)
      V = _compute_virial(G, ∂G)
   end

   return (energy=E, forces=F, virial=V)
end

# AtomsCalculators interface for WrappedSiteCalculator
AtomsCalculators.@generate_interface function AtomsCalculators.potential_energy(
      sys::AbstractSystem, calc::WrappedSiteCalculator; kwargs...)
   return _wrapped_energy(calc, sys) * u"eV"
end

AtomsCalculators.@generate_interface function AtomsCalculators.forces(
      sys::AbstractSystem, calc::WrappedSiteCalculator; kwargs...)
   return _wrapped_forces(calc, sys) .* u"eV/Å"
end

AtomsCalculators.@generate_interface function AtomsCalculators.virial(
      sys::AbstractSystem, calc::WrappedSiteCalculator; kwargs...)
   return _wrapped_virial(calc, sys) * u"eV"
end

function AtomsCalculators.energy_forces_virial(
      sys::AbstractSystem, calc::WrappedSiteCalculator; kwargs...)
   efv = _wrapped_energy_forces_virial(calc, sys)
   return (
      energy = efv.energy * u"eV",
      forces = efv.forces .* u"eV/Å",
      virial = efv.virial * u"eV"
   )
end


# ============================================================================
#  ETACEPotential - Type alias for WrappedSiteCalculator{ETACE}
# ============================================================================

"""
    ETACEPotential

AtomsCalculators-compatible calculator wrapping an ETACE model.
This is a type alias for `WrappedSiteCalculator{<:ETACE, PS, ST}`.

Access underlying components via:
- `calc.model` - The ETACE model
- `calc.ps` - Model parameters
- `calc.st` - Model state
- `calc.rcut` - Cutoff radius in Ångström
- `calc.co_ps` - Committee parameters (optional)

# Example
```julia
calc = ETACEPotential(et_model, ps, st, 5.5)
E = potential_energy(sys, calc)
```
"""
const ETACEPotential{MOD<:ETACE, PS, ST} = WrappedSiteCalculator{MOD, PS, ST}

# Constructor: creates WrappedSiteCalculator with ETACE model directly
function ETACEPotential(model::ETACE, ps, st, rcut::Real)
   return WrappedSiteCalculator(model, ps, st, Float64(rcut))
end

# ============================================================================
#  Training Assembly Interface
# ============================================================================
#
# These functions compute the basis values for linear least squares fitting.
# The linear parameters are the readout weights W[1, k, s] where:
#   k = basis function index (1:nbasis)
#   s = species index (1:nspecies)
#
# Total parameters: nbasis * nspecies
#
# Energy basis: E = ∑_i ∑_k W[k, species[i]] * 𝔹[i, k]
# Force basis:  F_atom = -∑ edges ∂E/∂r_edge, computed per basis function
# Virial basis: V = -∑ edges (∂E/∂r_edge) ⊗ r_edge, computed per basis function

# Accessor helpers for ETACEPotential (which is WrappedSiteCalculator{ETACE})
_etace(calc::ETACEPotential) = calc.model      # Underlying ETACE model (direct)
_ps(calc::ETACEPotential) = calc.ps            # Model parameters
_st(calc::ETACEPotential) = calc.st            # Model state

"""
    length_basis(calc::ETACEPotential)

Return the number of linear parameters in the model (nbasis * nspecies).
"""
function length_basis(calc::ETACEPotential)
   etace = _etace(calc)
   nbasis = etace.readout.in_dim
   nspecies = etace.readout.ncat
   return nbasis * nspecies
end

# ACEfit integration
import ACEfit
ACEfit.basis_size(calc::ETACEPotential) = length_basis(calc)

"""
    energy_forces_virial_basis(sys::AbstractSystem, calc::ETACEPotential)

Compute the basis functions for energy, forces, and virial.
Returns a named tuple with:
- `energy::Vector{Float64}` - length = length_basis(calc)
- `forces::Matrix{SVector{3,Float64}}` - size = (natoms, length_basis)
- `virial::Vector{SMatrix{3,3,Float64}}` - length = length_basis(calc)

The linear combination of basis values with parameters gives:
  E = dot(energy, params)
  F = forces * params
  V = sum(params .* virial)
"""
function energy_forces_virial_basis(sys::AbstractSystem, calc::ETACEPotential)
   G = ET.Atoms.interaction_graph(sys, calc.rcut * u"Å")
   etace = _etace(calc)

   # Get basis and jacobian
   # 𝔹: (nnodes, nbasis) - basis values per site (Float64)
   # ∂𝔹: (maxneigs, nnodes, nbasis) - directional derivatives (VState objects)
   𝔹, ∂𝔹 = site_basis_jacobian(etace, G, _ps(calc), _st(calc))

   natoms = length(sys)
   nnodes = size(𝔹, 1)
   nbasis = etace.readout.in_dim
   nspecies = etace.readout.ncat
   nparams = nbasis * nspecies
   maxneigs = size(∂𝔹, 1)

   # Species indices for each node
   iZ = etace.readout.selector.(G.node_data)

   # Initialize outputs
   E_basis = zeros(nparams)
   F_basis = zeros(SVector{3, Float64}, natoms, nparams)
   V_basis = zeros(SMatrix{3, 3, Float64, 9}, nparams)

   # Pre-allocate work buffer for gradient (same element type as ∂𝔹)
   # This avoids allocating a new matrix in each iteration
   ∇Ei_buf = similar(∂𝔹, maxneigs, nnodes)

   # Pre-compute a zero element for masking (same type as ∂𝔹 elements)
   zero_grad = zero(∂𝔹[1, 1, 1])

   # Pre-compute edge vectors for virial (avoid repeated access)
   edge_𝐫 = [edge.𝐫 for edge in G.edge_data]

   # Compute basis values for each parameter (k, s) pair
   # Parameter index: p = (s-1) * nbasis + k
   for s in 1:nspecies
      for k in 1:nbasis
         p = (s - 1) * nbasis + k

         # Energy basis: sum of 𝔹[i, k] for atoms of species s
         for i in 1:nnodes
            if iZ[i] == s
               E_basis[p] += 𝔹[i, k]
            end
         end

         # Fill gradient buffer: ∇Ei[:, i] = ∂𝔹[:, i, k] if iZ[i] == s, else zeros
         # This avoids allocating W_unit and doing matrix-vector multiply
         for i in 1:nnodes
            if iZ[i] == s
               @views ∇Ei_buf[:, i] .= ∂𝔹[:, i, k]
            else
               @views ∇Ei_buf[:, i] .= Ref(zero_grad)
            end
         end

         # Reshape for rev_reshape_embedding (needs 3D array) - this is a view, no allocation
         ∇Ei_3d = reshape(∇Ei_buf, maxneigs, nnodes, 1)

         # Convert to edge-indexed format with 3D vectors
         ∇E_edges = ET.rev_reshape_embedding(∇Ei_3d, G)[:]

         # Convert edge gradients to atomic forces (negate for forces)
         F_basis[:, p] = -ET.Atoms.forces_from_edge_grads(sys, G, ∇E_edges)

         # Compute virial: V = -∑ (∂E/∂𝐫ij) ⊗ 𝐫ij
         V = zero(SMatrix{3, 3, Float64, 9})
         for (e, ∂edge) in enumerate(∇E_edges)
            V -= ∂edge.𝐫 * edge_𝐫[e]'
         end
         V_basis[p] = V
      end
   end

   return (
      energy = E_basis * u"eV",
      forces = F_basis .* u"eV/Å",
      virial = V_basis * u"eV"
   )
end

"""
    potential_energy_basis(sys::AbstractSystem, calc::ETACEPotential)

Compute only the energy basis (faster when forces/virial not needed).
"""
function potential_energy_basis(sys::AbstractSystem, calc::ETACEPotential)
   G = ET.Atoms.interaction_graph(sys, calc.rcut * u"Å")
   etace = _etace(calc)

   # Get basis values
   𝔹 = site_basis(etace, G, _ps(calc), _st(calc))

   nbasis = etace.readout.in_dim
   nspecies = etace.readout.ncat
   nparams = nbasis * nspecies

   # Species indices for each node
   iZ = etace.readout.selector.(G.node_data)

   # Compute energy basis
   E_basis = zeros(nparams)
   for s in 1:nspecies
      for k in 1:nbasis
         p = (s - 1) * nbasis + k
         for i in 1:length(G.node_data)
            if iZ[i] == s
               E_basis[p] += 𝔹[i, k]
            end
         end
      end
   end

   return E_basis * u"eV"
end

"""
    get_linear_parameters(calc::ETACEPotential)

Extract the linear parameters (readout weights) as a flat vector.
Parameters are ordered as: [W[1,:,1]; W[1,:,2]; ... ; W[1,:,nspecies]]
"""
function get_linear_parameters(calc::ETACEPotential)
   return vec(_ps(calc).readout.W)
end

"""
    set_linear_parameters!(calc::ETACEPotential, θ::AbstractVector)

Set the linear parameters (readout weights) from a flat vector.
"""
function set_linear_parameters!(calc::ETACEPotential, θ::AbstractVector)
   etace = _etace(calc)
   nbasis = etace.readout.in_dim
   nspecies = etace.readout.ncat
   @assert length(θ) == nbasis * nspecies

   # Reshape and copy into ps (WrappedSiteCalculator is mutable)
   ps = _ps(calc)
   new_W = reshape(θ, 1, nbasis, nspecies)
   calc.ps = merge(ps, (readout = merge(ps.readout, (W = new_W,)),))
   return calc
end


# ============================================================================
#  Full Model Conversion
# ============================================================================

using Random: AbstractRNG, default_rng
using Lux: setup

"""
    convert2et_full(model, ps, st; rng=default_rng()) -> StackedCalculator

Convert a complete ACE model (E0 + Pair + Many-body) to an ETACE-based
StackedCalculator. This creates a calculator that combines:
1. ETOneBody - reference energies per species
2. ETPairModel - pair potential
3. ETACE - many-body ACE potential

The returned StackedCalculator is fully compatible with AtomsCalculators
and can be used for energy, forces, and virial evaluation.

# Arguments
- `model`: ACE model (from ACEpotentials.Models)
- `ps`: Model parameters (from Lux.setup)
- `st`: Model state (from Lux.setup)
- `rng`: Random number generator (default: `default_rng()`)

# Returns
- `StackedCalculator` combining ETOneBody, ETPairModel, and ETACE

# Example
```julia
model = ace_model(elements=[:Si], order=3, totaldegree=8)
ps, st = Lux.setup(rng, model)
# ... fit model ...
calc = convert2et_full(model, ps, st)
E = potential_energy(sys, calc)
```
"""
function convert2et_full(model, ps, st; rng::AbstractRNG=default_rng())
   # Extract cutoff radius from pair basis
   rcut = maximum(a.rcut for a in model.pairbasis.rin0cuts)

   # 1. Convert E0/Vref to ETOneBody
   E0s = model.Vref.E0  # Dict{Int, Float64}
   zlist = ChemicalSpecies.(model.rbasis._i2z)
   E0_dict = Dict(z => E0s[z.number] for z in zlist)
   et_onebody = one_body(E0_dict, x -> x.z)
   _, onebody_st = setup(rng, et_onebody)
   # Use minimum cutoff for graph construction (ETOneBody needs no neighbors)
   onebody_calc = WrappedSiteCalculator(et_onebody, nothing, onebody_st, 3.0)

   # 2. Convert pair potential to ETPairModel
   et_pair = convertpair(model)
   et_pair_ps, et_pair_st = setup(rng, et_pair)
   _copy_pair_params!(et_pair_ps, ps, model)
   pair_calc = WrappedSiteCalculator(et_pair, et_pair_ps, et_pair_st, rcut)

   # 3. Convert many-body to ETACE
   et_ace = convert2et(model)
   et_ace_ps, et_ace_st = setup(rng, et_ace)
   _copy_ace_params!(et_ace_ps, ps, model)
   ace_calc = WrappedSiteCalculator(et_ace, et_ace_ps, et_ace_st, rcut)

   # 4. Stack all components
   return StackedCalculator((onebody_calc, pair_calc, ace_calc))
end


# ============================================================================
#  Parameter Copying Utilities
# ============================================================================

"""
    _copy_ace_params!(et_ps, ps, model)

Copy many-body (ACE) parameters from ACE model format to ETACE format.
"""
function _copy_ace_params!(et_ps, ps, model)
   NZ = length(model.rbasis._i2z)

   # Copy radial basis parameters (Wnlq)
   # ACE format: Wnlq[:, :, iz, jz] for species pair (iz, jz)
   # ETACE format: W[:, :, idx] where idx = (i-1)*NZ + j (or symmetric idx)
   for i in 1:NZ, j in 1:NZ
      idx = (i-1)*NZ + j
      et_ps.rembed.basis.linl.W[:, :, idx] .= ps.rbasis.Wnlq[:, :, i, j]
   end

   # Copy readout (many-body) parameters
   # ACE format: WB[:, s] for species s
   # ETACE format: W[1, :, s]
   for s in 1:NZ
      et_ps.readout.W[1, :, s] .= ps.WB[:, s]
   end
end


"""
    _copy_pair_params!(et_ps, ps, model)

Copy pair potential parameters from ACE model format to ETPairModel format.
Based on parameter mapping from test/etmodels/test_etpair.jl.
"""
function _copy_pair_params!(et_ps, ps, model)
   NZ = length(model.pairbasis._i2z)

   # Copy pair radial basis parameters
   # ACE format: pairbasis.Wnlq[:, :, i, j] for species pair (i, j)
   # ETACE format: rembed.basis.rbasis.linl.W[:, :, idx] where idx = (i-1)*NZ + j
   for i in 1:NZ, j in 1:NZ
      idx = (i-1)*NZ + j
      et_ps.rembed.basis.rbasis.linl.W[:, :, idx] .= ps.pairbasis.Wnlq[:, :, i, j]
   end

   # Copy pair readout parameters
   # ACE format: Wpair[:, s] for species s
   # ETACE format: readout.W[1, :, s]
   for s in 1:NZ
      et_ps.readout.W[1, :, s] .= ps.Wpair[:, s]
   end
end

