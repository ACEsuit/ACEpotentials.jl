
# StackedCalculator - Combines multiple AtomsCalculators
#
# Generic utility for combining multiple calculators by summing their
# energy, forces, and virial contributions.
#
# Uses @generated functions with Base.Cartesian for efficient
# compile-time loop unrolling when the number of calculators is known.

import AtomsCalculators
import AtomsBase: AbstractSystem
using StaticArrays
using Unitful
using Base.Cartesian: @nexprs, @ntuple, @ncall

"""
    StackedCalculator{N, C<:Tuple}

Combines multiple AtomsCalculators by summing their energy, forces, and virial.
Each calculator in the tuple must implement the AtomsCalculators interface.

This allows combining site-based calculators (via WrappedSiteCalculator) with
calculators that don't have site decompositions (e.g., Coulomb, dispersion).

The implementation uses compile-time loop unrolling for efficiency when
the number of calculators is small and known at compile time.

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
- `calcs::Tuple` - Tuple of N calculators implementing AtomsCalculators interface
"""
struct StackedCalculator{N, C<:Tuple}
   calcs::C
end

# Constructor that infers N from the tuple length
StackedCalculator(calcs::C) where {C<:Tuple} = StackedCalculator{length(C.parameters), C}(calcs)

# Get maximum cutoff from all calculators (for informational purposes)
@generated function cutoff_radius(calc::StackedCalculator{N}) where {N}
   quote
      rcuts = @ntuple $N i -> ustrip(u"Å", cutoff_radius(calc.calcs[i]))
      return maximum(rcuts) * u"Å"
   end
end

# ============================================================================
#  Efficient implementations using @generated for compile-time unrolling
# ============================================================================

@generated function _stacked_energy(sys::AbstractSystem, calc::StackedCalculator{N}) where {N}
   quote
      @nexprs $N i -> E_i = AtomsCalculators.potential_energy(sys, calc.calcs[i])
      return sum(@ntuple $N E)
   end
end

@generated function _stacked_forces(sys::AbstractSystem, calc::StackedCalculator{N}) where {N}
   quote
      @nexprs $N i -> F_i = AtomsCalculators.forces(sys, calc.calcs[i])
      return reduce(.+, @ntuple $N F)
   end
end

@generated function _stacked_virial(sys::AbstractSystem, calc::StackedCalculator{N}) where {N}
   quote
      @nexprs $N i -> V_i = AtomsCalculators.virial(sys, calc.calcs[i])
      return sum(@ntuple $N V)
   end
end

@generated function _stacked_efv(sys::AbstractSystem, calc::StackedCalculator{N}) where {N}
   quote
      @nexprs $N i -> efv_i = AtomsCalculators.energy_forces_virial(sys, calc.calcs[i])
      return (
         energy = sum(@ntuple $N i -> efv_i.energy),
         forces = reduce(.+, @ntuple $N i -> efv_i.forces),
         virial = sum(@ntuple $N i -> efv_i.virial)
      )
   end
end

# ============================================================================
#  AtomsCalculators interface
# ============================================================================

AtomsCalculators.@generate_interface function AtomsCalculators.potential_energy(
      sys::AbstractSystem, calc::StackedCalculator; kwargs...)
   return _stacked_energy(sys, calc)
end

AtomsCalculators.@generate_interface function AtomsCalculators.forces(
      sys::AbstractSystem, calc::StackedCalculator; kwargs...)
   return _stacked_forces(sys, calc)
end

AtomsCalculators.@generate_interface function AtomsCalculators.virial(
      sys::AbstractSystem, calc::StackedCalculator; kwargs...)
   return _stacked_virial(sys, calc)
end

function AtomsCalculators.energy_forces_virial(
      sys::AbstractSystem, calc::StackedCalculator; kwargs...)
   return _stacked_efv(sys, calc)
end
