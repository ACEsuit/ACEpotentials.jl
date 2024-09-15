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
   E = f(sys, model)
   ps0 = model.ps
   co_E = [ (model.ps = model.co_ps[i]; f(sys, model)) 
            for i = 1:length(model.co_ps) ]
   model.ps = ps0            
   return E, co_E
end

macro committee(ex)
   esc_args = esc.(ex.args)
   quote
      committee($(esc_args...))
   end
end
