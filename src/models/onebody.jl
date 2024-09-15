
# TODO: move this to AtomsCalculatorsUtilities or EmpiricalPotentials

using Unitful
import AtomsCalculatorsUtilities, AtomsCalculators 
import AtomsCalculators: energy_unit, length_unit, force_unit, potential_energy
import AtomsCalculatorsUtilities.SitePotentials: SitePotential, 
                                  cutoff_radius, 
                                  eval_site, 
                                  eval_grad_site 
import AtomsBase: atomic_number, AbstractSystem, ChemicalSpecies

_atomic_number(s) = atomic_number(s)
_atomic_number(s::String) = _atomic_number(Symbol(s))
_atomic_number(s::Symbol) = atomic_number(ChemicalSpecies(s))
_atomic_number(z::Integer) = z 
_atomic_number(s::ChemicalSpecies) = atomic_number(s)

"""
`mutable struct OneBody{T}`

this should not normally be constructed by a user, but instead E0 should be
passed to the relevant model constructor, which will construct it.
"""
mutable struct OneBody{T, TU, TL} <: SitePotential
   E0::Dict{Int, T}
   energy_unit::TU
   length_unit::TL
end

function OneBody(E0s; length_unit = u"Å", )
   pairs = [] 
   for key in keys(E0s) 
      z = Int(_atomic_number(key))
      push!(pairs, z => ustrip(E0s[key]))
   end 
   T = promote_type(typeof.([p[end] for p in pairs])...)
   energy_unit = unit(pairs[1][2])
   pairs = [ (k => convert(T, v)) for (k, v) in pairs ]
   return OneBody( Dict{Int, T}(pairs...), energy_unit, length_unit )
end

energy_unit(V::OneBody) = V.energy_unit
length_unit(V::OneBody) = V.length_unit

cutoff_radius(::OneBody{T}) where {T} = 
      sqrt(eps(T)) * u"Å"
      
eval_site(V::OneBody, Rs, Zs, zi::Integer) = 
      V.E0[zi]

eval_grad_site(V::OneBody{T}, Rs, Zs, zi) where {T} = 
      eval_site(V, Rs, Zs, zi), fill( zero(SVector{3, T}), length(Zs) )


function AtomsCalculators.potential_energy(sys::AbstractSystem, V::OneBody)
   energy = AtomsCalculators.zero_energy(sys, V)
   for i = 1:length(sys) 
      Zi = atomic_number(sys, i)
      energy += V.E0[Zi] * energy_unit(V)
   end
   return energy
end
