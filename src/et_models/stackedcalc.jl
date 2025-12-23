
# StackedCalculator - Combines multiple AtomsCalculators
#
# Generic utility for combining multiple calculators by summing their
# energy, forces, and virial contributions.

import AtomsCalculators
import AtomsBase: AbstractSystem
using StaticArrays
using Unitful

"""
    StackedCalculator{C<:Tuple}

Combines multiple AtomsCalculators by summing their energy, forces, and virial.
Each calculator in the tuple must implement the AtomsCalculators interface.

This allows combining site-based calculators (via WrappedSiteCalculator) with
calculators that don't have site decompositions (e.g., Coulomb, dispersion).

# Example
```julia
# Wrap site energy models
E0_calc = WrappedSiteCalculator(E0Model(Dict(:Si => -0.846)))
ace_calc = WrappedSiteCalculator(WrappedETACE(et_model, ps, st, 5.5))

# Stack them (could also add Coulomb, dispersion, etc.)
calc = StackedCalculator((E0_calc, ace_calc))

E = potential_energy(sys, calc)
F = forces(sys, calc)
```

# Fields
- `calcs::Tuple` - Tuple of calculators implementing AtomsCalculators interface
"""
struct StackedCalculator{C<:Tuple}
   calcs::C
end

# Get maximum cutoff from all calculators (for informational purposes)
function cutoff_radius(calc::StackedCalculator)
   rcuts = [ustrip(u"Å", cutoff_radius(c)) for c in calc.calcs]
   return maximum(rcuts) * u"Å"
end

# AtomsCalculators interface - sum contributions from all calculators
AtomsCalculators.@generate_interface function AtomsCalculators.potential_energy(
      sys::AbstractSystem, calc::StackedCalculator; kwargs...)
   E_total = 0.0 * u"eV"
   for c in calc.calcs
      E_total += AtomsCalculators.potential_energy(sys, c)
   end
   return E_total
end

AtomsCalculators.@generate_interface function AtomsCalculators.forces(
      sys::AbstractSystem, calc::StackedCalculator; kwargs...)
   F_total = nothing
   for c in calc.calcs
      F = AtomsCalculators.forces(sys, c)
      if F_total === nothing
         F_total = F
      else
         F_total = F_total .+ F
      end
   end
   return F_total
end

AtomsCalculators.@generate_interface function AtomsCalculators.virial(
      sys::AbstractSystem, calc::StackedCalculator; kwargs...)
   V_total = zeros(SMatrix{3,3,Float64,9}) * u"eV"
   for c in calc.calcs
      V_total += AtomsCalculators.virial(sys, c)
   end
   return V_total
end

function AtomsCalculators.energy_forces_virial(
      sys::AbstractSystem, calc::StackedCalculator; kwargs...)
   E_total = 0.0 * u"eV"
   F_total = nothing
   V_total = zeros(SMatrix{3,3,Float64,9}) * u"eV"

   for c in calc.calcs
      efv = AtomsCalculators.energy_forces_virial(sys, c)
      E_total += efv.energy
      V_total += efv.virial
      if F_total === nothing
         F_total = efv.forces
      else
         F_total = F_total .+ efv.forces
      end
   end

   return (energy=E_total, forces=F_total, virial=V_total)
end
