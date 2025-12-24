
# Calculator interfaces for ETACE models
# Provides AtomsCalculators-compatible energy/forces/virial evaluation
#
# Architecture:
# - SiteEnergyModel interface: Any model producing per-site energies can implement this
# - E0Model: One-body reference energies (constant per species)
# - WrappedETACE: Wraps ETACE model with the SiteEnergyModel interface
# - WrappedSiteCalculator: Converts SiteEnergyModel to AtomsCalculators interface
# - ETACEPotential: Standalone calculator for simple use cases
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
#  SiteEnergyModel Interface
# ============================================================================
#
# Any model producing per-site (per-atom) energies can implement this interface:
#
#   site_energies(model, G::ETGraph, ps, st) -> Vector  # per-atom energies
#   site_energy_grads(model, G::ETGraph, ps, st) -> ∂G  # edge gradients for forces
#   cutoff_radius(model) -> Float64                     # in Ångström
#
# This enables composition via StackedCalculator for:
# - One-body reference energies (E0Model)
# - Pairwise interactions (PairModel)
# - Many-body ACE (WrappedETACE)
# - Future: dispersion, coulomb, etc.

"""
    site_energies(model, G, ps, st)

Compute per-site (per-atom) energies for the given interaction graph.
Returns a vector of length `nnodes(G)`.
"""
function site_energies end

"""
    site_energy_grads(model, G, ps, st)

Compute gradients of site energies w.r.t. edge positions.
Returns a named tuple with `edge_data` field containing gradient vectors.
"""
function site_energy_grads end

"""
    cutoff_radius(model)

Return the cutoff radius in Ångström for the model.
"""
function cutoff_radius end


# ============================================================================
#  E0Model - One-body reference energies
# ============================================================================

"""
    E0Model{T}

One-body reference energy model. Assigns constant energy per atomic species.
No forces (energy is position-independent).

# Example
```julia
E0 = E0Model(Dict(ChemicalSpecies(:Si) => -0.846, ChemicalSpecies(:O) => -2.15))
```
"""
struct E0Model{T<:Real}
   E0s::Dict{ChemicalSpecies, T}
end

# Constructor from element symbols
function E0Model(E0s::Dict{Symbol, T}) where T<:Real
   return E0Model(Dict(ChemicalSpecies(k) => v for (k, v) in E0s))
end

cutoff_radius(::E0Model) = 0.0  # No neighbors needed

function site_energies(model::E0Model, G::ET.ETGraph, ps, st)
   T = valtype(model.E0s)
   return T[model.E0s[node.z] for node in G.node_data]
end

function site_energy_grads(model::E0Model{T}, G::ET.ETGraph, ps, st) where T
   # Constant energy → zero gradients
   zero_grad = PState(𝐫 = zero(SVector{3, T}))
   return (edge_data = fill(zero_grad, length(G.edge_data)),)
end


# ============================================================================
#  WrappedETACE - ETACE model with SiteEnergyModel interface
# ============================================================================

"""
    WrappedETACE{MOD<:ETACE, PS, ST}

Wraps an ETACE model to implement the SiteEnergyModel interface.
Mutable to allow parameter updates during training.

# Fields
- `model::ETACE` - The underlying ETACE model
- `ps` - Model parameters (mutable for training)
- `st` - Model state
- `rcut::Float64` - Cutoff radius in Ångström
- `co_ps` - Optional committee parameters for uncertainty quantification
"""
mutable struct WrappedETACE{MOD<:ETACE, PS, ST}
   model::MOD
   ps::PS
   st::ST
   rcut::Float64
   co_ps::Any
end

# Constructor without committee parameters
function WrappedETACE(model::ETACE, ps, st, rcut::Real)
   return WrappedETACE(model, ps, st, Float64(rcut), nothing)
end

cutoff_radius(w::WrappedETACE) = w.rcut

function site_energies(w::WrappedETACE, G::ET.ETGraph, ps, st)
   # Use wrapper's ps/st, ignore passed ones (they're for StackedCalculator dispatch)
   Ei, _ = w.model(G, w.ps, w.st)
   return Ei
end

function site_energy_grads(w::WrappedETACE, G::ET.ETGraph, ps, st)
   return site_grads(w.model, G, w.ps, w.st)
end


# ============================================================================
#  WrappedSiteCalculator - Converts SiteEnergyModel to AtomsCalculators
# ============================================================================

"""
    WrappedSiteCalculator{M}

Wraps a SiteEnergyModel and provides the AtomsCalculators interface.
Converts site quantities (per-atom energies, edge gradients) to global
quantities (total energy, atomic forces, virial tensor).

# Example
```julia
E0 = E0Model(Dict(:Si => -0.846, :O => -2.15))
calc = WrappedSiteCalculator(E0, 5.5)  # cutoff for graph construction

E = potential_energy(sys, calc)
F = forces(sys, calc)
```

# Fields
- `model` - Model implementing SiteEnergyModel interface
- `rcut::Float64` - Cutoff radius for graph construction (Å)
"""
struct WrappedSiteCalculator{M}
   model::M
   rcut::Float64
end

function WrappedSiteCalculator(model)
   rcut = cutoff_radius(model)
   # Ensure minimum cutoff for graph construction (must be > 0 for neighbor list)
   # Use 3.0 Å as minimum - smaller than typical bond lengths
   rcut = max(rcut, 3.0)
   return WrappedSiteCalculator(model, rcut)
end

cutoff_radius(calc::WrappedSiteCalculator) = calc.rcut * u"Å"

function _wrapped_energy(calc::WrappedSiteCalculator, sys::AbstractSystem)
   G = ET.Atoms.interaction_graph(sys, calc.rcut * u"Å")
   Ei = site_energies(calc.model, G, nothing, nothing)
   return sum(Ei)
end

function _wrapped_forces(calc::WrappedSiteCalculator, sys::AbstractSystem)
   G = ET.Atoms.interaction_graph(sys, calc.rcut * u"Å")
   ∂G = site_energy_grads(calc.model, G, nothing, nothing)
   # Handle empty edge case (e.g., E0 model with small cutoff)
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
   ∂G = site_energy_grads(calc.model, G, nothing, nothing)
   # Handle empty edge case
   if isempty(∂G.edge_data)
      return zeros(SMatrix{3,3,Float64,9})
   end
   return _compute_virial(G, ∂G)
end

function _wrapped_energy_forces_virial(calc::WrappedSiteCalculator, sys::AbstractSystem)
   G = ET.Atoms.interaction_graph(sys, calc.rcut * u"Å")

   # Energy from site energies
   Ei = site_energies(calc.model, G, nothing, nothing)
   E = sum(Ei)

   # Forces and virial from edge gradients
   ∂G = site_energy_grads(calc.model, G, nothing, nothing)

   # Handle empty edge case (e.g., E0 model with small cutoff)
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
#  ETACEPotential - Type alias for WrappedSiteCalculator{WrappedETACE}
# ============================================================================

"""
    ETACEPotential

AtomsCalculators-compatible calculator wrapping an ETACE model.
This is a type alias for `WrappedSiteCalculator{<:WrappedETACE}`.

Access underlying components via:
- `calc.model` - The WrappedETACE wrapper
- `calc.model.model` - The ETACE model
- `calc.model.ps` - Model parameters
- `calc.model.st` - Model state
- `calc.rcut` - Cutoff radius in Ångström
- `calc.model.co_ps` - Committee parameters (optional)

# Example
```julia
calc = ETACEPotential(et_model, ps, st, 5.5)
E = potential_energy(sys, calc)
```
"""
const ETACEPotential{MOD<:ETACE, PS, ST} = WrappedSiteCalculator{WrappedETACE{MOD, PS, ST}}

# Constructor: creates WrappedSiteCalculator wrapping WrappedETACE
function ETACEPotential(model::ETACE, ps, st, rcut::Real)
   wrapped = WrappedETACE(model, ps, st, rcut)
   return WrappedSiteCalculator(wrapped, Float64(rcut))
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

# Accessor helpers for ETACEPotential (which is WrappedSiteCalculator{WrappedETACE})
_etace(calc::ETACEPotential) = calc.model.model      # Underlying ETACE model
_ps(calc::ETACEPotential) = calc.model.ps            # Model parameters
_st(calc::ETACEPotential) = calc.model.st            # Model state

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

   # Reshape and copy into ps (via the WrappedETACE which is mutable)
   ps = _ps(calc)
   new_W = reshape(θ, 1, nbasis, nspecies)
   calc.model.ps = merge(ps, (readout = merge(ps.readout, (W = new_W,)),))
   return calc
end

