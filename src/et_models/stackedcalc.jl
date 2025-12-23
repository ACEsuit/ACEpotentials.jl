
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

# Helper to generate sum expression: E_1 + E_2 + ... + E_N
function _gen_sum(N, prefix)
   if N == 1
      return Symbol(prefix, "_1")
   else
      ex = Symbol(prefix, "_1")
      for i in 2:N
         ex = :($ex + $(Symbol(prefix, "_", i)))
      end
      return ex
   end
end

# Helper to generate broadcast sum: F_1 .+ F_2 .+ ... .+ F_N
function _gen_broadcast_sum(N, prefix)
   if N == 1
      return Symbol(prefix, "_1")
   else
      ex = Symbol(prefix, "_1")
      for i in 2:N
         ex = :($ex .+ $(Symbol(prefix, "_", i)))
      end
      return ex
   end
end

@generated function _stacked_energy(sys::AbstractSystem, calc::StackedCalculator{N}) where {N}
   assignments = [:($(Symbol("E_", i)) = AtomsCalculators.potential_energy(sys, calc.calcs[$i])) for i in 1:N]
   sum_expr = _gen_sum(N, "E")
   quote
      $(assignments...)
      return $sum_expr
   end
end

@generated function _stacked_forces(sys::AbstractSystem, calc::StackedCalculator{N}) where {N}
   assignments = [:($(Symbol("F_", i)) = AtomsCalculators.forces(sys, calc.calcs[$i])) for i in 1:N]
   sum_expr = _gen_broadcast_sum(N, "F")
   quote
      $(assignments...)
      return $sum_expr
   end
end

@generated function _stacked_virial(sys::AbstractSystem, calc::StackedCalculator{N}) where {N}
   assignments = [:($(Symbol("V_", i)) = AtomsCalculators.virial(sys, calc.calcs[$i])) for i in 1:N]
   sum_expr = _gen_sum(N, "V")
   quote
      $(assignments...)
      return $sum_expr
   end
end

@generated function _stacked_efv(sys::AbstractSystem, calc::StackedCalculator{N}) where {N}
   # Generate assignments for each calculator
   assignments = [:($(Symbol("efv_", i)) = AtomsCalculators.energy_forces_virial(sys, calc.calcs[$i])) for i in 1:N]

   # Generate sum expressions
   E_exprs = [:($(Symbol("efv_", i)).energy) for i in 1:N]
   F_exprs = [:($(Symbol("efv_", i)).forces) for i in 1:N]
   V_exprs = [:($(Symbol("efv_", i)).virial) for i in 1:N]

   E_sum = N == 1 ? E_exprs[1] : reduce((a, b) -> :($a + $b), E_exprs)
   F_sum = N == 1 ? F_exprs[1] : reduce((a, b) -> :($a .+ $b), F_exprs)
   V_sum = N == 1 ? V_exprs[1] : reduce((a, b) -> :($a + $b), V_exprs)

   quote
      $(assignments...)
      E_total = $E_sum
      F_total = $F_sum
      V_total = $V_sum
      return (energy=E_total, forces=F_total, virial=V_total)
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
