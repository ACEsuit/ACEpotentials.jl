
# Calculator interfaces for ETACE models
# Provides AtomsCalculators-compatible energy/forces/virial evaluation

import AtomsCalculators
import AtomsBase: AbstractSystem
import EquivariantTensors as ET
using StaticArrays
using Unitful

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

