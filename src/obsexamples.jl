

module ObsExamples

   using JuLIP, ACEfit 

   import JuLIP: Atoms, energy, forces, 
                 JVec, mat, vecs 

   import ACEfit: Dat, eval_obs, vec_obs, devec_obs, basis_obs 

   # export ObsPotentialEnergy, ObsForces 

   # note in ObsPotentialEnergy
   #  - o.E stores the value of a potential energy observation 
   #  - o.weight the regression weight 
   struct ObsPotentialEnergy{T} 
      E::T    
      weight::Real
      E0::T
   end

   ObsPotentialEnergy{T}(E::Number) where {T} = ObsPotentialEnergy{T}(E, 1.0, 0.0)
   ObsPotentialEnergy(E::Number) = ObsPotentialEnergy(E, 1.0, 0.0)

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


   struct ObsForces{T}
      F::Vector{JVec{T}}
      weight::Real
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
      V::AbstractVector{T}  # possible this should be a matrix...
      weight::Real
   end
   ObsVirial{T}(V::AbstractVector) where {T} = ObsVirial(V, 1.0)
   ObsVirial(V::AbstractVector) = ObsVirial(V, 1.0)
   ACEfit.vec_obs(obs::ObsVirial) = obs.V
   function ACEfit.basis_obs(::Type{TOBS}, basis, at) where {TOBS <: ObsVirial}
         all_virials = virial(basis, at)
         V = []
         for m in all_virials
            v = [m[1,1], m[2,2], m[3,3], m[3,2], m[3,1], m[2,1]]
            push!(V, TOBS(v))
         end
         return V
   end

end
