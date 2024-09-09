
using LinearAlgebra: I, Diagonal, UniformScaling

import ACEpotentials.Models: ACEPotential

import ACEfit: assemble

export acefit!, assemble, linear_errors


# ---------------- some utilities and defaults 
# these are temporary monkey-patches to get things up and running but 
# should be cleaned up and replaced over time ... 

_get_Vref(model::ACEPotential) = model.model.Vref 

__set_params!(model::ACEPotential, coeffs) = ACEpotentials.Models.set_parameters!(model, coeffs)

default_weights() = Dict("default"=>Dict("E"=>30.0, "F"=>1.0, "V"=>1.0))

function _make_prior(model::ACEpotentials.Models.ACEPotential, smoothness, P)
   if P isa AbstractMatrix || P isa UniformScaling 
      return P 
   elseif smoothness isa Number 
      if smoothness >= 0 
         return ACEpotentials.Models.algebraic_smoothness_prior(model.model; p = smoothness)
      else
         error("smoothness must be >= 0")
      end
   end
end

# ---------------- the main fitting function

"""
   acefit!(rawdata, model; kwargs...)

provides a convenient interface to fitting the parameters of an ACE model. 
The data should be provided as a collection of `AbstractSystem` structures. 

Keyword arguments:
* `energy_key`, `force_key`, `virial_key` specify 
the label of the data to which the parameters will be fitted. 
* `weights` specifies the regression weights, default is 30 for energy, 1 for forces and virials
* `solver` specifies the lsq solver, default is `BLR` (BayesianLinearRegression)
* `smoothness` specifies the smoothness prior, i.e. how strongly damped 
   parameters corresponding to high polynomial degrees are; is 2.
* `prior` specifies a covariance of the prior, if `nothing` then a smoothness prior 
   is used, using the `smoothness` parameter 
* `repulsion_restraint` specifies whether to add artificial data to the training 
   set that effectively introduces a restraints encouraging repulsion 
   in the limit rij -> 0.
* `restraint_weight` specifies the weight of the repulsion restraint.
* `export_lammps` : path to a file to which the fitted potential will be exported 
   in a LAMMPS compatible format (yace)
* `export_json` : path to a file to which the fitted potential will be exported 
   in a JSON format, which can be read from Julia or Python
"""
function acefit!(raw_data::AbstractArray{<: AbstractSystem}, model;
                solver = ACEfit.BLR(),
                weights = default_weights(),
                energy_key = "energy", 
                force_key = "force", 
                virial_key = "virial", 
                smoothness = 4, 
                prior = nothing, 
                repulsion_restraint = false, 
                restraint_weight = 0.01, 
                export_lammps = nothing, 
                export_json = nothing, 
                verbose=true,
                kwargs...
   )

   # convert raw data to AtomsData, which ACEfit.jl understands 
   data = map( raw_data ) do d
      AtomsData(d;
         energy_key = energy_key, 
         force_key=force_key, 
         virial_key = virial_key, 
         weights = weights, 
         v_ref = _get_Vref(model)
      )
   end

   # print some information about the dataset 
   # (how many observations in each class)
   if verbose
      assess_dataset(
         data;
         energy_key = energy_key, 
         force_key  = force_key, 
         virial_key = virial_key,
         kwargs...
      )
   end 

   if repulsion_restraint 
      error("Repulsion restraint is currently not implemented")
      # if eltype(data) == AtomsData
      #    append!(data, _rep_dimer_data(model; weight = restraint_weight))
      # else
      #    tmp = _rep_dimer_data_atomsbase(
      #       model; 
      #       weight = restraint_weight,
      #       energy_key = Symbol(energy_key),
      #       kwargs...
      #       )
      #    append!(data, tmp)
      # end
   end
               
   # build a prior from the kw arguments 
   P = _make_prior(model, smoothness, prior)

   # actual assembly of the least square system 
   A, Y, W = ACEfit.assemble(data, model)

   # transform the system to incorporate the prior and the weights 
   # then solve the transformed problem 
   Ap = Diagonal(W) * (A / P) 
   Y = W .* Y
   result = ACEfit.solve(solver, Ap, Y)
   coeffs = P \ result["C"]

   # dispatch setting of parameters 
   __set_params!(model, coeffs)

   if haskey(result, "committee")
      @warn("Committees are not yet supported")
      #  co_coeffs = result["committee"]
      #  for i in 1:size(co_coeffs,2)
      #     co_coeffs[:,i] = P \ co_coeffs[:,i]
      #  end
      #  IP_com = ACE1.committee_potential(model.basis, coeffs, co_coeffs)
      #  (model.Vref != nothing) && (IP_com = JuLIP.MLIPs.SumIP(model.Vref, IP_com))
      #  # possibly too drastic to overwrite potential with committee potential?
      #  model.potential = IP_com
   end

   if export_lammps != nothing 
      error("automatic lammps export currently not supported")
      # export2lammps(export_lammps, model)
   end
   if export_json != nothing 
      error("automatic json export currently not supported")
      # export2json(export_json, model)
   end

   return model 
end




function linear_errors(raw_data::AbstractArray{<: AbstractSystem}, model; 
                       energy_key = "energy", 
                       force_key = "force", 
                       virial_key = "virial", 
                       weights = default_weights(), 
                       verbose = true )
   data = [ AtomsData(at; energy_key = energy_key, force_key=force_key, 
                          virial_key = virial_key, weights = weights, 
                          v_ref = _get_Vref(model)) 
            for at in raw_data ]
   return linear_errors(data, model; verbose=verbose)
end




function assemble(raw_data::AbstractArray{<: AbstractSystem}, model; 
                     weights = default_weights(),
                     energy_key = "energy", 
                     force_key = "force", 
                     virial_key = "virial", 
                     # smoothness = 4, 
                     # prior = nothing, 
                     repulsion_restraint = false, 
                     restraint_weight = 0.01, 
                     mode = :serial, 
                     weights_only = false)

   data = [ AtomsData(at; energy_key = energy_key, force_key=force_key, 
                  virial_key = virial_key, weights = weights, 
                  v_ref = model.Vref)  for at in raw_data ]

   if repulsion_restraint 
      error("Repulsion restraint is currently not implemented")
      # append!(data, _rep_dimer_data(model, weight = restraint_weight))
   end
        
   if weights_only
      W = ACEfit.assemble_weights(data)
      return W
   end 
      
   A, Y, W = assemble(data, model.basis, mode)
   return A, Y, W
end
