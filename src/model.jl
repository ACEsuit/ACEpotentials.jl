

import ACE1x 
import ACE1x: ACE1Model, acemodel, _set_params!

export acemodel, acefit!

import JuLIP: energy, forces, virial, cutoff
import ACE1.Utils: get_maxn

_mean(x) = sum(x) / length(x)


default_weights() = Dict("default"=>Dict("E"=>30.0, "F"=>1.0, "V"=>1.0))

"""
`function acefit!` : provides a simplified interface to fitting the 
parameters of a model specified via `ACE1Model`. The data should be 
provided as a collection (`AbstractVector`) of `JuLIP.Atoms` structures. 
The keyword arguments `energy_key`, `force_key`, `virial_key` specify 
the label of the data to which the parameters will be fitted. 
The final keyword argument is a `weights` dictionary. 
"""
function acefit!(model::ACE1Model, raw_data, solver; 
                weights = default_weights(),
                energy_key = "energy", 
                force_key = "force", 
                virial_key = "virial", )
   data = [ AtomsData(at, energy_key, force_key, virial_key, 
                      weights, model.Vref) for at in raw_data ] 
   result = ACEfit.linear_fit(data, model.basis, solver) 
   ACE1x._set_params!(model, result["C"])
   return model 
end



function linear_errors(data::AbstractVector{<: Atoms}, model::ACE1Model; 
                       energy_key = "energy", 
                       force_key = "force", 
                       virial_key = "virial", 
                       weights = default_weights())
   Vref = model.Vref                       
   _data = [ AtomsData(d, energy_key, force_key, virial_key, weights, Vref) for d in data ]
   return linear_errors(_data, model.potential)
end


using LinearAlgebra: Diagonal 

smoothness_prior(model::ACE1Model; p = 2) = 
      Diagonal(vcat(ACE1.scaling.(model.basis.BB, p)...))