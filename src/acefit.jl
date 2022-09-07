# --------------- ACEfit.jl interface 

using JuLIP
using StaticArrays: SVector, SMatrix

import JuLIP: Atoms, energy, forces, JVec, JMat, mat, vecs 

###import ACEfit: Dat, eval_obs, vec_obs, devec_obs, basis_obs 
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
            if lowercase(key)=="config_type" && (lowercase(val.data) in lowercase.(keys(weights)))
                w = weights[val.data]
            end
        end

        return new(atoms, ek, fk, vk, w, vref)
    end
end

function ACEfit.count_observations(d::AtomsData)
    return !isnothing(d.energy_key) +
           3*length(d.atoms)*!isnothing(d.force_key) +
           6*!isnothing(d.virial_key)
end

function ACEfit.feature_matrix(d::AtomsData, basis)
    dm = Array{Float64}(undef, ACEfit.count_observations(d), length(basis))
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

function ACEfit.target_vector(d::AtomsData)
    y = Array{Float64}(undef, ACEfit.count_observations(d))
    i = 1
    if !isnothing(d.energy_key)
        e = d.atoms.data[d.energy_key].data
        y[i] = e - energy(d.vref, d.atoms)
        i += 1
    end
    if !isnothing(d.force_key)
        f = mat(d.atoms.data[d.force_key].data)
        y[i:i+3*length(d.atoms)-1] .= f[:]
        i += 3*length(d.atoms)
    end
    if !isnothing(d.virial_key)
        v = vec(d.atoms.data[d.virial_key].data)
        y[i:i+5] .= v[SVector(1,5,9,6,3,2)]
        i += 5
    end
    return y
end

function ACEfit.weight_vector(d::AtomsData)
    w = Array{Float64}(undef, ACEfit.count_observations(d))
    i = 1
    if !isnothing(d.energy_key)
        w[i] = d.weights["E"] / sqrt(length(d.atoms))
        i += 1
    end
    if !isnothing(d.force_key)
        w[i:i+3*length(d.atoms)-1] .= d.weights["F"]
        i += 3*length(d.atoms)
    end
    if !isnothing(d.virial_key)
        w[i:i+5] .= d.weights["V"] / sqrt(length(d.atoms))
        i += 5
    end
    return w
end

function config_type(d::AtomsData)
    config_type = missing
    for (k,v) in d.atoms.data
        if (lowercase(k)=="config_type")
            config_type = v.data
        end
    end
    return config_type
end

function llsq_errors(data, model)

   mae = Dict("E"=>0.0, "F"=>0.0, "V"=>0.0)
   rmse = Dict("E"=>0.0, "F"=>0.0, "V"=>0.0)
   num = Dict("E"=>0, "F"=>0, "V"=>0)

   config_types = []
   config_mae = Dict{String,Any}()
   config_rmse = Dict{String,Any}()
   config_num = Dict{String,Any}()

   for d in data

       ct = config_type(d)
       if !(ct in config_types)
          push!(config_types, ct)
          merge!(config_rmse, Dict(ct=>Dict("E"=>0.0, "F"=>0.0, "V"=>0.0)))
          merge!(config_mae, Dict(ct=>Dict("E"=>0.0, "F"=>0.0, "V"=>0.0)))
          merge!(config_num, Dict(ct=>Dict("E"=>0, "F"=>0, "V"=>0)))
       end

       # energy errors
       if !isnothing(d.energy_key)
           estim = energy(model, d.atoms) / length(d.atoms)
           exact = d.atoms.data[d.energy_key].data / length(d.atoms)
           mae["E"] += abs(estim-exact)
           rmse["E"] += (estim-exact)^2
           num["E"] += 1
           config_mae[ct]["E"] += abs(estim-exact)
           config_rmse[ct]["E"] += (estim-exact)^2
           config_num[ct]["E"] += 1
       end

       # force errors
       if !isnothing(d.force_key)
           estim = mat(forces(model, d.atoms))
           exact = mat(d.atoms.data[d.force_key].data)
           mae["F"] += sum(abs.(estim-exact))
           rmse["F"] += sum((estim-exact).^2)
           num["F"] += 3*length(d.atoms)
           config_mae[ct]["F"] += sum(abs.(estim-exact))
           config_rmse[ct]["F"] += sum((estim-exact).^2)
           config_num[ct]["F"] += 3*length(d.atoms)
       end

       # virial errors
       if !isnothing(d.virial_key)
           estim = virial(model, d.atoms)[SVector(1,5,9,6,3,2)] ./ length(d.atoms)
           exact = d.atoms.data[d.virial_key].data[SVector(1,5,9,6,3,2)] ./ length(d.atoms)
           mae["V"] += sum(abs.(estim-exact))
           rmse["V"] += sum((estim-exact).^2)
           num["V"] += 6
           config_mae[ct]["V"] += sum(abs.(estim-exact))
           config_rmse[ct]["V"] += sum((estim-exact).^2)
           config_num[ct]["V"] += 6
       end
    end

    # finalize errors
    for (k,n) in num
        (n==0) && continue
        rmse[k] = sqrt(rmse[k] / n)
        mae[k] = mae[k] / n
    end
    errors = Dict("mae"=>mae, "rmse"=>rmse)

    # finalize config errors
    for ct in config_types
        for (k,cn) in config_num[ct]
            (cn==0) && continue
            config_rmse[ct][k] = sqrt(config_rmse[ct][k] / cn)
            config_mae[ct][k] = config_mae[ct][k] / cn
        end
    end
    config_errors = Dict("mae"=>config_mae, "rmse"=>config_rmse)

    # merge errors into config_errors and return
    merge!(config_errors["mae"], Dict("set"=>mae))
    merge!(config_errors["rmse"], Dict("set"=>rmse))
    return config_errors
end
