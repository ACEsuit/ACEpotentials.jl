# --------------- ACEfit.jl interface 

using JuLIP

import JuLIP: Atoms, energy, forces, JVec, mat, vecs 

import ACEfit: Dat, eval_obs, vec_obs, devec_obs, basis_obs 

# export ObsPotentialEnergy, ObsForces 

struct ObsPotentialEnergy{T} 
    E::T             # value of a potential energy
    weight::Real     # regression weight 
end

ObsPotentialEnergy{T}(E::Number) where {T} = ObsPotentialEnergy{T}(E, 1.0)
ObsPotentialEnergy(E::Number) = ObsPotentialEnergy(E, 1.0)

# evaluating an observation type on a model - 
# here we assume implicitly that `at = dat.config::Atoms` and that 
# `energy(model, at)` has been implemented. Different models could either 
# overload just `energy(model, at)` or overload ``
eval_obs(::Type{TOBS}, model, config) where {TOBS <: ObsPotentialEnergy} = 
        TOBS( energy(model, config) )


function basis_obs(::Type{TOBS}, basis, at) where {TOBS <: ObsPotentialEnergy}
    E = energy(basis, at)::AbstractVector{<: Number}
    return TOBS.(E)
end

# now given an observation we need to convert it 
vec_obs(obs::ObsPotentialEnergy) = [ obs.E ]
   
function devec_obs(obs::TOBS, x::AbstractVector) where {TOBS <: ObsPotentialEnergy}
    @assert length(x) == 1
    return TOBS(x[1])
end


# TODO: allow generalisation of regression weight for forces to
#       general matrices 
struct ObsForces{T}
    F::Vector{JVec{T}}   # forces stored as vector of 3-vectors
    weight::Real         # regression weight
end

ObsForces{T}(F::AbstractVector) where {T} = ObsForces(F, 1.0)
ObsForces(F::AbstractVector) = ObsForces(F, 1.0)

eval_obs(::Type{TOBS}, model, cfg) where {TOBS <: ObsForces} = 
        TOBS( forces(model, cfg) )

function basis_obs(::Type{TOBS}, basis, at) where {TOBS <: ObsForces}
    F = forces(basis, at)
    return TOBS.(F)
end


# converts [ [F11 F12 F13] [F21 F22 F23] ... ] into 
#          [ F11, F12, F13, F21, F22, F23, ... ]
vec_obs(obs::ObsForces) = mat(obs.F)[:]
   
# and the converse; collect just gets rid of the re-interpret business... 
devec_obs(obs::TOBS, x::AbstractVector) where {TOBS <: ObsForces} = 
        TOBS(collect(vecs(x)))


struct ObsVirial{T}
    V::SMatrix{3, 3, T}     # virial stored as a matrix, but when we use it we convert it to a 6-vector
    weight::Real            # regression weight
end

ObsVirial{T}(V::AbstractMatrix) where {T} = ObsVirial(SMatrix{3, 3, T}(V...), 1.0)
ObsVirial{T}(V::AbstractMatrix) where {T} = ObsVirial(SMatrix{3, 3, T}(V...), 1.0)

eval_obs(::Type{TOBS}, model, cfg) where {TOBS <: ObsVirial} = 
        TOBS( virial(model, cfg) )

function basis_obs(::Type{TOBS}, basis, at) where {TOBS <: ObsVirial}
    V = virial(basis, at)
    return TOBS.(V)
end


const _IVst = SVector(1,5,9,6,3,2)

# converts [ V11 V12 V13; V21 V22 V23; V31 V32 V33] 
#       to [ V11, V22, V33, V32, V31, V21 ]
vec_obs(obs::ObsVirial) = obs.V[_IVst]
    
# and the converse; collect just gets rid of the re-interpret business... 
devec_obs(obs::TOBS, x::AbstractVector) where {TOBS <: ObsVirial} = 
        TOBS( SMatrix{3,3}(x[1], x[6], x[5], x[6], x[2], x[4], x[5], x[4], x[3]) )
        
# weighthook(::ValV, d::Dat) = 1.0 / sqrt(length(d.at))
# err_weighthook(::ValV, d::Dat) = 1.0 / length(d.at)

