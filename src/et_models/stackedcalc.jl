
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

# ============================================================================
#  Training Assembly Interface for StackedCalculator
# ============================================================================

import ACEfit

"""
    length_basis(calc::StackedCalculator)

Return total number of linear parameters across all stacked calculators.
"""
function length_basis(calc::StackedCalculator)
   return sum(length_basis(c) for c in calc.calcs)
end

ACEfit.basis_size(calc::StackedCalculator) = length_basis(calc)

"""
    energy_forces_virial_basis(sys::AbstractSystem, calc::StackedCalculator)

Compute concatenated basis for all stacked calculators.
"""
function energy_forces_virial_basis(sys::AbstractSystem, calc::StackedCalculator)
   # Collect basis from each calculator
   results = [energy_forces_virial_basis(sys, c) for c in calc.calcs]

   natoms = length(sys)

   # Concatenate results - energy is Vector of Quantity{Float64}
   E_basis = vcat([_strip_energy_units(r.energy) for r in results]...)

   # For forces, need to hcat the matrices
   # Strip units element by element for matrices of SVectors with units
   F_parts = [_strip_force_units(r.forces) for r in results]
   F_basis = isempty(F_parts) ? zeros(SVector{3, Float64}, natoms, 0) : hcat(F_parts...)

   # Virial is Vector of SMatrix with units
   V_basis = vcat([_strip_virial_units(r.virial) for r in results]...)

   return (
      energy = E_basis * u"eV",
      forces = F_basis .* u"eV/Å",
      virial = V_basis * u"eV"
   )
end

# Helper to strip units from energy (Vector of Quantity{Float64})
function _strip_energy_units(E)
   return map(e -> ustrip(e), E)
end

# Helper to strip units from force matrices (Matrix of SVector with units)
function _strip_force_units(F)
   # F is Matrix{SVector{3, Quantity}}
   # We need to strip the units from the inner SVectors
   return map(f -> SVector{3, Float64}(ustrip.(f)), F)
end

# Helper to strip units from virial (Vector of SMatrix with units)
function _strip_virial_units(V)
   # V is Vector{SMatrix{3,3, Quantity}}
   return map(v -> SMatrix{3, 3, Float64, 9}(ustrip.(v)), V)
end

"""
    potential_energy_basis(sys::AbstractSystem, calc::StackedCalculator)

Compute concatenated energy basis for all stacked calculators.
"""
function potential_energy_basis(sys::AbstractSystem, calc::StackedCalculator)
   results = [potential_energy_basis(sys, c) for c in calc.calcs]
   E_basis = vcat([ustrip.(u"eV", r) for r in results]...)
   return E_basis * u"eV"
end

"""
    get_linear_parameters(calc::StackedCalculator)

Get concatenated linear parameters from all stacked calculators.
"""
function get_linear_parameters(calc::StackedCalculator)
   return vcat([get_linear_parameters(c) for c in calc.calcs]...)
end

"""
    set_linear_parameters!(calc::StackedCalculator, θ::AbstractVector)

Set linear parameters for all stacked calculators from concatenated vector.
"""
function set_linear_parameters!(calc::StackedCalculator, θ::AbstractVector)
   offset = 0
   for c in calc.calcs
      n = length_basis(c)
      if n > 0
         set_linear_parameters!(c, θ[offset+1:offset+n])
      end
      offset += n
   end
   @assert offset == length(θ) "Parameter count mismatch"
   return calc
end
