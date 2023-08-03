
using JuLIP: Atoms, energy, cutoff


"""
`function get_rdf(data::AbstractVector{<: Atoms}, r_cut; kwargs...)` : 

Produce a list of r values that occur in the dataset, restricted to the cutoff 
radius `r_cut`. Keyword arguments: 
* `rescale = true` : resample the data to account for volume scaling, i.e. a distance r will be kept with probability `min(1, (r0/r)^2)`.
* `r0 = :min` : parameter for resampling. If `:min` then the minimum r occuring in the dataset is taken. 
* `maxsamples = 100_000` : maximum number of samples to return. 
"""
function get_rdf(data::AbstractVector{<: Atoms}, r_cut; 
                 rescale = true, 
                 r0 = :min, 
                 maxsamples = 100_000)
   R = Float64[] 
   for at in data 
      nlist = JuLIP.neighbourlist(at, r_cut; recompute=true)
      r = [ norm(rr) for (i, j, rr) in pairs(nlist) ] 
      append!(R, r)
   end
   sort!(R) 

   R1 = Float64[]
   if rescale 
      # choose a minimum r value relative to which we resample. 
      _r0 = (r0 == :min) ? R[1] : r0
      for r in R 
         if rand() < min(1, (_r0/r)^2)
            push!(R1, r)
         end
      end
   else
      R1 = R 
   end 

   if length(R1) > maxsamples 
      Ikeep = floor.(Int, range(1, length(R1), length = maxsamples))
      R1 = R1[Ikeep]
   end

   return R1
end

"""
`function get_adf(data::AbstractVector{<: Atoms}, r_cut; kwargs...)` :

Angular distribution, i.e. list of angles in [0, π] between all pairs of bonds 
of length at most `r_cut`. Keyword arguments:
* `skip = 3` : only consider every `skip`th atom in the dataset.
* `maxsamples = 100_000` : maximum number of samples to return.
"""
function get_adf(data::AbstractVector{<: Atoms}, r_cut; 
                 skip = 3, 
                 maxsamples = 100_000)
   skip = max(skip, 1)                 
   A = Float64[] 
   ctr = -1
   for at in data 
      nlist = JuLIP.neighbourlist(at, r_cut; recompute=true)
      for i = 1:length(at)
         ctr = mod(ctr + 1, skip); if ctr != 0; continue; end 

         Js, Rs = neigs(nlist, i)
         for a1 = 1:length(Js)-1, a2 = a1+1:length(Js)
            r̂1 = Rs[a1] / norm(Rs[a1])
            r̂2 = Rs[a2] / norm(Rs[a2])
            push!(A, acos(dot(r̂1, r̂2)))
         end
      end 
   end 
   sort!(A) 

   if length(A) > maxsamples 
      Ikeep = floor.(Int, range(1, length(A), length = maxsamples))
      A = A[Ikeep]
   end

   return A
end
