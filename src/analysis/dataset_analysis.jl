using AtomsBase: AbstractSystem, ChemicalSpecies, atomic_number
import NeighbourLists
using AtomsCalculatorsUtilities.SitePotentials: cutoff_radius, PairList, get_neighbours
using AtomsCalculators: potential_energy, energy_unit 
using StaticArrays
using AtomsBuilder 
using LinearAlgebra: norm, dot 

function copy_zz_sym!(D::AbstractDict)
   _zz = collect(keys(D))
   for z12 in _zz
      sym12 = Symbol.( ChemicalSpecies.(z12) ) 
      D[sym12] = D[z12]
   end
end

"""
`function get_rdf(data::AbstractVector{<: Atoms}, r_cut; kwargs...)` : 

Produce a list of r values that occur in the dataset, restricted to the cutoff 
radius `r_cut`. Keyword arguments: 
* `rescale = true` : resample the data to account for volume scaling, i.e. a distance r will be kept with probability `min(1, (r0/r)^2)`.
* `r0 = :min` : parameter for resampling. If `:min` then the minimum r occuring in the dataset is taken. 
* `maxsamples = 100_000` : maximum number of samples to return. 
"""
function get_rdf(data::AbstractVector{<: AbstractSystem}, r_cut; 
                 rescale = true, 
                 r0 = :min, 
                 maxsamples = 100_000)

   zz = Int[] 
   for at in data 
      zz = unique(append!(zz, unique(atomic_number(at, :))))
   end        
            
   zz_pairs = [ (z0, z) for z0 in zz for z in zz ]
   R0 = Dict{Any, Vector{Float64}}([ z12 => Float64[] for z12 in zz_pairs ]...)
   R = deepcopy(R0)         

   for sys in data 
      nlist = PairList(sys, r_cut)
      Z = atomic_number(sys, :)
      for (i, j, rr) in pairs(nlist) 
         z12 = (Z[i], Z[j])
         push!(R[z12], norm(rr))
      end
   end
   for z12 in zz_pairs 
      sort!(R[z12])
   end

   # drop random samples with probability selected to adjust for 
   # volume scaling. 
   if rescale 
      R1 = deepcopy(R0)
      for z12 in keys(R)
         rr = R[z12]
         # choose a minimum r value relative to which we resample. 
         _r0 = (r0 == :min) ? rr[1] : r0
         for r in rr
            if rand() < min(1, (_r0/r)^2)
               push!(R1[z12], r)
            end
         end
      end
   else
      R1 = R 
   end 

   # sub-select uniformly to get only #maxsamples samples for each 
   # pair of atomic numbers.
   for z12 in keys(R1)
      rr = R1[z12]
      if length(rr) > maxsamples 
         Ikeep = floor.(Int, range(1, length(rr), length = maxsamples))
         R1[z12] = rr[Ikeep]
      end
   end

   ## allow access to the rdf via z or via symbols
   copy_zz_sym!(R1)

   return R1
end


"""
`function get_adf(data::AbstractVector{<: Atoms}, r_cut; kwargs...)` :

Angular distribution, i.e. list of angles in [0, π] between all pairs of bonds 
of length at most `r_cut`. Keyword arguments:
* `skip = 3` : only consider every `skip`th atom in the dataset.
* `maxsamples = 100_000` : maximum number of samples to return.
"""
function get_adf(data::AbstractVector{<: AbstractSystem}, r_cut; 
                 skip = 3, 
                 maxsamples = 100_000)
   skip = max(skip, 1)                 
   A = Float64[] 
   ctr = -1
   for sys in data 
      nlist = PairList(sys, r_cut)
      for i = 1:length(sys)
         ctr = mod(ctr + 1, skip); if ctr != 0; continue; end 

         Js, Rs = NeighbourLists.neigs(nlist, i)
         for a1 = 1:length(Js)-1, a2 = a1+1:length(Js)
            r̂1 = Rs[a1] / norm(Rs[a1])
            r̂2 = Rs[a2] / norm(Rs[a2])
            d = min(max(dot(r̂1, r̂2), -1.0), 1.0)
            push!(A, acos(d))
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
