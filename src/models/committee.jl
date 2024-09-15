using Optimisers
using AtomsBase: AbstractSystem 
using AtomsCalculators: potential_energy
export @committee, set_committee!, co_length

# --------------------------------------------------------
#   Committee - prototype implementation with poor performance
#   this can be done much more efficiently; the canonical approach 
#   seems to be to make `W` a matrix or vector of SVectors. 

function set_committee!(model::ACEPotential, 
                       co_vec::AbstractVector{<: AbstractVector})
   ps_vec, _restruct = destructure(model.ps)
   co_nt = [ _restruct(co_vec[i]) for i in 1:length(co_vec) ]
   return set_committee!(model, co_nt)
end

function set_committee!(model::ACEPotential, 
                        co_nt::AbstractVector{<: NamedTuple})
   model.co_ps = co_nt
   return model
end

function co_length(model::ACEPotential)
   if isnothing(model.co_ps)
      return 0 
   else 
      return length(model.co_ps)
   end
end

function committee(f, sys::AbstractSystem, model::ACEPotential)
   if f == potential_energy
      return co_potential_energy(sys, model)
   end
   F = f(sys, model)
   ps0 = model.ps
   co_F = [ (model.ps = model.co_ps[i]; f(sys, model)) 
            for i = 1:length(model.co_ps) ]
   model.ps = ps0            
   return F, co_F
end

macro committee(ex)
   esc_args = esc.(ex.args)
   quote
      committee($(esc_args...))
   end
end

function co_potential_energy(sys::AbstractSystem, model::ACEPotential)
   basis_E = potential_energy_basis(sys, model)
   eref = potential_energy(sys, model.model.Vref) * u"eV"
   E = dot(basis_E, destructure(model.ps)[1]) + eref
   co_E = [ dot(basis_E, destructure(model.co_ps[i])[1]) + eref 
            for i = 1:length(model.co_ps) ]
   return E, co_E
end

function co_potential_energy_2(sys::AbstractSystem, model::ACEPotential)
   f = potential_energy 
   F = f(sys, model)
   ps0 = model.ps
   co_F = [ (model.ps = model.co_ps[i]; f(sys, model)) 
            for i = 1:length(model.co_ps) ]
   model.ps = ps0            
   return F, co_F
end