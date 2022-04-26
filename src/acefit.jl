# --------------- ACEfit.jl interface 

using JuLIP
using StaticArrays: SVector, SMatrix

import JuLIP: Atoms, energy, forces, JVec, JMat, mat, vecs 

import ACEfit: Dat, eval_obs, vec_obs, devec_obs, basis_obs 

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

