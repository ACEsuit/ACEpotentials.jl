# --------------- ACEfit.jl interface 

using JuLIP
using StaticArrays: SVector, SMatrix

import JuLIP: Atoms, energy, forces, JVec, JMat, mat, vecs 

import ACEfit: Dat, eval_obs, vec_obs, devec_obs, basis_obs 
import ACEfit

mutable struct ObsPotentialEnergy
    E                # value of a potential energy
    weight::Real     # regression weight 
    E_ref            # reference energy
end

ObsPotentialEnergy(E::Number) = ObsPotentialEnergy(E, 1.0, 0.0)

# evaluating an observation type on a model - 
# here we assume implicitly that `at = dat.config::Atoms` and that 
# `energy(model, at)` has been implemented. Different models could either 
# overload just `energy(model, at)` or overload ``
eval_obs(::Type{ObsPotentialEnergy}, model, config) = 
         ObsPotentialEnergy( energy(model, config) )


function basis_obs(::Type{ObsPotentialEnergy}, basis, at)
    E = energy(basis, at)::AbstractVector{<: Number}
    return ObsPotentialEnergy.(E)
end

# now given an observation we need to convert it 
vec_obs(obs::ObsPotentialEnergy) = [ obs.E ]
   
function devec_obs(obs::ObsPotentialEnergy, x::AbstractVector)
    @assert length(x) == 1
    return ObsPotentialEnergy(x[1])
end


# TODO: allow generalisation of regression weight for forces to
#       general matrices 
mutable struct ObsForces
    F::Vector{<: SVector}   # forces stored as vector of 3-vectors
    weight::Real            # regression weight
end

ObsForces(F::AbstractVector) = ObsForces(F, 1.0)

eval_obs(::Type{ObsForces}, model, cfg) = 
            ObsForces( forces(model, cfg) )

function basis_obs(::Type{ObsForces}, basis, at) 
    F = forces(basis, at)
    return ObsForces.(F)
end


# converts [ [F11 F12 F13] [F21 F22 F23] ... ] into 
#          [ F11, F12, F13, F21, F22, F23, ... ]
vec_obs(obs::ObsForces) = mat(obs.F)[:]
   
# and the converse; collect just gets rid of the re-interpret business... 
devec_obs(obs::ObsForces, x::AbstractVector) = 
         ObsForces(collect(vecs(x)))

mutable struct ObsVirial
    V::SMatrix      # virial stored as a matrix, but when we use it we convert it to a 6-vector
    weight::Real    # regression weight
end

ObsVirial(V::AbstractMatrix) = ObsVirial(JMat(V...), 1.0)

eval_obs(::Type{ObsVirial}, model, cfg) = 
               ObsVirial( virial(model, cfg) )

function basis_obs(::Type{ObsVirial}, basis, at) 
    V = virial(basis, at)
    return ObsVirial.(V)
end


const _IVst = SVector(1,5,9,6,3,2)

# converts [ V11 V12 V13; V21 V22 V23; V31 V32 V33] 
#       to [ V11, V22, V33, V32, V31, V21 ]
vec_obs(obs::ObsVirial) = obs.V[_IVst]
    
# and the converse; collect just gets rid of the re-interpret business... 
devec_obs(obs::ObsVirial, x::AbstractVector) = 
         ObsVirial( SMatrix{3,3}(x[1], x[6], x[5], 
                                 x[6], x[2], x[4], 
                                 x[5], x[4], x[3]) )


default_weighthooks = Dict{String, Any}("default" => Dict(
      ObsPotentialEnergy => (w, cfg) -> w / sqrt(length(cfg)), 
               ObsForces => (w, cfg) -> w, 
               ObsVirial => (w, cfg) -> w / sqrt(length(cfg)) ))

# err_weighthook(::ValV, d::Dat) = 1.0 / length(d.at)

# ------ new


struct AtomsData <: ACEfit.AbstractData
    atoms::Atoms
    energy_key
    force_key
    virial_key
    weights
    vref
    function AtomsData(atoms::Atoms,
                       energy_key, force_key, virial_key,
                       weights, vref)

        # set energy, force, and virial keys for this configuration
        # ("nothing" indicates data that are absent or ignored)
        ek, fk, vk = nothing, nothing, nothing
        for key in keys(atoms.data)
            if lowercase(key) == lowercase(energy_key)
                ek = key
            elseif lowercase(key) == lowercase(force_key)
                fk = key
            elseif lowercase(key) == lowercase(virial_key)
                vk = key
            end
        end

        # set weights for this configuration
        w = weights["default"]
        for (key, val) in atoms.data
            if lowercase(key)=="config_type" && lowercase(val.data) in lowercase.(keys(weights))
                w = weights[val.data]
            end
        end

        return new(atoms, ek, fk, vk, w, vref)
    end
end

function ACEfit.countrows(d::AtomsData)
    return !isnothing(d.energy_key) +
           3*length(d.atoms)*!isnothing(d.force_key) +
           6*!isnothing(d.virial_key)
end

function ACEfit.designmatrix(d::AtomsData, basis)
    dm = Array{Float64}(undef, ACEfit.countrows(d), length(basis))
    i = 1
    if !isnothing(d.energy_key)
        dm[i,:] .= energy(basis, d.atoms)
        i += 1
    end
    if !isnothing(d.force_key)
        f = mat.(forces(basis, d.atoms))
        for j in 1:length(basis)
            dm[i:i+3*length(d.atoms)-1,j] .= f[j][:]
        end
        i += 3*length(d.atoms)
    end
    if !isnothing(d.virial_key)
        v = virial(basis, d.atoms)
        for j in 1:length(basis)
            dm[i:i+5,j] .= v[j][SVector(1,5,9,6,3,2)]
        end
        i += 5
    end
    return dm
end

function ACEfit.targetvector(d::AtomsData)
    tv = Array{Float64}(undef, ACEfit.countrows(d))
    i = 1
    if !isnothing(d.energy_key)
        e = d.atoms.data[d.energy_key].data
        tv[i] = e - energy(d.vref, d.atoms)
        i += 1
    end
    if !isnothing(d.force_key)
        f = mat(d.atoms.data[d.force_key].data)
        tv[i:i+3*length(d.atoms)-1] .= f[:]
        i += 3*length(d.atoms)
    end
    if !isnothing(d.virial_key)
        v = vec(d.atoms.data[d.virial_key].data)
        tv[i:i+5] .= v[SVector(1,5,9,6,3,2)]
        i += 5
    end
    return tv
end

function ACEfit.weightvector(d::AtomsData)
    wv = Array{Float64}(undef, ACEfit.countrows(d))
    i = 1
    if !isnothing(d.energy_key)
        wv[i] = d.weights["E"] / sqrt(length(d.atoms))
        i += 1
    end
    if !isnothing(d.force_key)
        wv[i:i+3*length(d.atoms)-1] .= d.weights["F"]
        i += 3*length(d.atoms)
    end
    if !isnothing(d.virial_key)
        wv[i:i+5] .= d.weights["V"] / sqrt(length(d.atoms))
        i += 5
    end
    return wv
end

function error_llsq(data, approx, exact)

   errors = approx - exact
   config_types = String[]
   config_counts = Dict("set"=>Dict("E"=>0, "F"=>0, "V"=>0))
   config_errors = Dict("set"=>Dict("E"=>0.0, "F"=>0.0, "V"=>0.0))
   config_norms = Dict("set"=>Dict("E"=>0.0, "F"=>0.0, "V"=>0.0))
   for dat in data
       if !(dat.configtype in config_types)
          push!(config_types, dat.configtype)
          merge!(config_counts, Dict(dat.configtype=>Dict("E"=>0,   "F"=>0,   "V"=>0)))
          merge!(config_errors, Dict(dat.configtype=>Dict("E"=>0.0, "F"=>0.0, "V"=>0.0)))
          merge!(config_norms, Dict(dat.configtype=>Dict("E"=>0.0, "F"=>0.0, "V"=>0.0)))
       end
   end

   i = 1
   for dat in data
      for o in observations(dat)
         obs_len = length(vec_obs(o))
         obs_errors = errors[i:i+obs_len-1]
         obs_values = exact[i:i+obs_len-1]
         # TODO: we store the ref energy because it is used for the relrmse
         #       calculation ... but does it make sense to use the total energy?
         if hasproperty(o, :E)
            obs_values = obs_values .+ o.E_ref
         end
         if hasproperty(o, :E) || hasproperty(o, :V)
            obs_errors = obs_errors ./ length(dat.config)
            obs_values = obs_values ./ length(dat.config)
         end
         obs_error = sum(obs_errors.^2)
         obs_norm = sum(obs_values.^2)
         if hasproperty(o, :E)
            config_counts["set"]["E"] += obs_len
            config_errors["set"]["E"] += obs_error
            config_norms["set"]["E"] += obs_norm
            config_counts[dat.configtype]["E"] += obs_len
            config_errors[dat.configtype]["E"] += obs_error
            config_norms[dat.configtype]["E"] += obs_norm
         elseif hasproperty(o, :F)
            config_counts["set"]["F"] += obs_len
            config_errors["set"]["F"] += obs_error
            config_norms["set"]["F"] += obs_norm
            config_counts[dat.configtype]["F"] += obs_len
            config_errors[dat.configtype]["F"] += obs_error
            config_norms[dat.configtype]["F"] += obs_norm
         elseif hasproperty(o, :V)
            config_counts["set"]["V"] += obs_len
            config_errors["set"]["V"] += obs_error
            config_norms["set"]["V"] += obs_norm
            config_counts[dat.configtype]["V"] += obs_len
            config_errors[dat.configtype]["V"] += obs_error
            config_norms[dat.configtype]["V"] += obs_norm
         else
            println("something is wrong")
         end
         i += obs_len
      end
   end

   for i in keys(config_errors)
      for j in keys(config_errors[i])
         config_errors[i][j] = sqrt(config_errors[i][j] / config_counts[i][j])
         config_norms[i][j] = sqrt(config_norms[i][j] / config_counts[i][j])
         config_norms[i][j] = config_errors[i][j] / config_norms[i][j]
      end
   end

   return Dict("rmse"=>config_errors, "relrmse"=>config_norms)

end

function error_llsq_new(params, approx, exact)

   v_ref = OneBody(convert(Dict{String,Any},params["e0"]))
   energy_key = params["data"]["energy_key"]
   force_key = params["data"]["force_key"]
   virial_key = params["data"]["virial_key"]
   weights = params["weights"]

   errors = approx - exact

   config_types = String[]
   config_counts = Dict("set"=>Dict("E"=>0, "F"=>0, "V"=>0))
   config_errors = Dict("set"=>Dict("E"=>0.0, "F"=>0.0, "V"=>0.0))
   config_norms = Dict("set"=>Dict("E"=>0.0, "F"=>0.0, "V"=>0.0))
   for dict in ExtXYZ.iread_frames(params["data"]["fname"])
       atoms = JuLIP._extxyz_dict_to_atoms(dict)
       dat = _atoms_to_data(atoms, v_ref, weights, energy_key, force_key, virial_key)
       if !(dat.configtype in config_types)
          push!(config_types, dat.configtype)
          merge!(config_counts, Dict(dat.configtype=>Dict("E"=>0,   "F"=>0,   "V"=>0)))
          merge!(config_errors, Dict(dat.configtype=>Dict("E"=>0.0, "F"=>0.0, "V"=>0.0)))
          merge!(config_norms, Dict(dat.configtype=>Dict("E"=>0.0, "F"=>0.0, "V"=>0.0)))
       end
    end

   i = 1
   for dict in ExtXYZ.iread_frames(params["data"]["fname"])
      atoms = JuLIP._extxyz_dict_to_atoms(dict)
      dat = _atoms_to_data(atoms, v_ref, weights, energy_key, force_key, virial_key)
      for o in observations(dat)
         obs_len = length(vec_obs(o))
         obs_errors = errors[i:i+obs_len-1]
         obs_values = exact[i:i+obs_len-1]
         # TODO: we store the ref energy because it is used for the relrmse
         #       calculation ... but does it make sense to use the total energy?
         if hasproperty(o, :E)
            obs_values = obs_values .+ o.E_ref
         end
         if hasproperty(o, :E) || hasproperty(o, :V)
            obs_errors = obs_errors ./ length(dat.config)
            obs_values = obs_values ./ length(dat.config)
         end
         obs_error = sum(obs_errors.^2)
         obs_norm = sum(obs_values.^2)
         if hasproperty(o, :E)
            config_counts["set"]["E"] += obs_len
            config_errors["set"]["E"] += obs_error
            config_norms["set"]["E"] += obs_norm
            config_counts[dat.configtype]["E"] += obs_len
            config_errors[dat.configtype]["E"] += obs_error
            config_norms[dat.configtype]["E"] += obs_norm
         elseif hasproperty(o, :F)
            config_counts["set"]["F"] += obs_len
            config_errors["set"]["F"] += obs_error
            config_norms["set"]["F"] += obs_norm
            config_counts[dat.configtype]["F"] += obs_len
            config_errors[dat.configtype]["F"] += obs_error
            config_norms[dat.configtype]["F"] += obs_norm
         elseif hasproperty(o, :V)
            config_counts["set"]["V"] += obs_len
            config_errors["set"]["V"] += obs_error
            config_norms["set"]["V"] += obs_norm
            config_counts[dat.configtype]["V"] += obs_len
            config_errors[dat.configtype]["V"] += obs_error
            config_norms[dat.configtype]["V"] += obs_norm
         else
            println("something is wrong")
         end
         i += obs_len
      end
   end

   for i in keys(config_errors)
      for j in keys(config_errors[i])
         config_errors[i][j] = sqrt(config_errors[i][j] / config_counts[i][j])
         config_norms[i][j] = sqrt(config_norms[i][j] / config_counts[i][j])
         config_norms[i][j] = config_errors[i][j] / config_norms[i][j]
      end
   end

   return Dict("rmse"=>config_errors, "relrmse"=>config_norms)

end
