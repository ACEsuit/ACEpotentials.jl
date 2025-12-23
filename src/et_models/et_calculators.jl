
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
    WrappedETACE{MOD<:ETACE, T}

Wraps an ETACE model to implement the SiteEnergyModel interface.

# Fields
- `model::ETACE` - The underlying ETACE model
- `ps` - Model parameters
- `st` - Model state
- `rcut::Float64` - Cutoff radius in Ångström
"""
struct WrappedETACE{MOD<:ETACE, PS, ST}
   model::MOD
   ps::PS
   st::ST
   rcut::Float64
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
#  ETACEPotential - Standalone calculator for ETACE models
# ============================================================================

"""
    ETACEPotential

AtomsCalculators-compatible calculator wrapping an ETACE model.

# Fields
- `model::ETACE` - The ETACE model
- `ps` - Model parameters
- `st` - Model state
- `rcut::Float64` - Cutoff radius in Ångström
- `co_ps` - Optional committee parameters for uncertainty quantification
"""
mutable struct ETACEPotential{MOD<:ETACE, T}
   model::MOD
   ps::T
   st::NamedTuple
   rcut::Float64
   co_ps::Any
end

# Constructor without committee parameters
function ETACEPotential(model::ETACE, ps, st, rcut::Real)
   return ETACEPotential(model, ps, st, Float64(rcut), nothing)
end

# Cutoff radius accessor
cutoff_radius(calc::ETACEPotential) = calc.rcut * u"Å"

# ============================================================================
#  Internal evaluation functions
# ============================================================================

function _compute_virial(G::ET.ETGraph, ∂G)
   # V = -∑ (∂E/∂𝐫ij) ⊗ 𝐫ij
   V = zeros(SMatrix{3,3,Float64,9})
   for (edge, ∂edge) in zip(G.edge_data, ∂G.edge_data)
      V -= ∂edge.𝐫 * edge.𝐫'
   end
   return V
end

function _evaluate_energy(calc::ETACEPotential, sys::AbstractSystem)
   G = ET.Atoms.interaction_graph(sys, calc.rcut * u"Å")
   Ei, _ = calc.model(G, calc.ps, calc.st)
   return sum(Ei)
end

function _evaluate_forces(calc::ETACEPotential, sys::AbstractSystem)
   G = ET.Atoms.interaction_graph(sys, calc.rcut * u"Å")
   ∂G = site_grads(calc.model, G, calc.ps, calc.st)
   # Note: forces_from_edge_grads returns +∇E, we need -∇E for forces
   return -ET.Atoms.forces_from_edge_grads(sys, G, ∂G.edge_data)
end

function _evaluate_virial(calc::ETACEPotential, sys::AbstractSystem)
   G = ET.Atoms.interaction_graph(sys, calc.rcut * u"Å")
   ∂G = site_grads(calc.model, G, calc.ps, calc.st)
   return _compute_virial(G, ∂G)
end

function _energy_forces_virial(calc::ETACEPotential, sys::AbstractSystem)
   G = ET.Atoms.interaction_graph(sys, calc.rcut * u"Å")

   # Forward pass for energy
   Ei, _ = calc.model(G, calc.ps, calc.st)
   E = sum(Ei)

   # Backward pass for gradients (forces and virial)
   ∂G = site_grads(calc.model, G, calc.ps, calc.st)

   # Forces from edge gradients (negate since forces_from_edge_grads returns +∇E)
   F = -ET.Atoms.forces_from_edge_grads(sys, G, ∂G.edge_data)

   # Virial from edge gradients
   V = _compute_virial(G, ∂G)

   return (energy=E, forces=F, virial=V)
end

# ============================================================================
#  AtomsCalculators interface
# ============================================================================

AtomsCalculators.@generate_interface function AtomsCalculators.potential_energy(
      sys::AbstractSystem, calc::ETACEPotential; kwargs...)
   return _evaluate_energy(calc, sys) * u"eV"
end

AtomsCalculators.@generate_interface function AtomsCalculators.forces(
      sys::AbstractSystem, calc::ETACEPotential; kwargs...)
   return _evaluate_forces(calc, sys) .* u"eV/Å"
end

AtomsCalculators.@generate_interface function AtomsCalculators.virial(
      sys::AbstractSystem, calc::ETACEPotential; kwargs...)
   return _evaluate_virial(calc, sys) * u"eV"
end

function AtomsCalculators.energy_forces_virial(
      sys::AbstractSystem, calc::ETACEPotential; kwargs...)
   efv = _energy_forces_virial(calc, sys)
   return (
      energy = efv.energy * u"eV",
      forces = efv.forces .* u"eV/Å",
      virial = efv.virial * u"eV"
   )
end

