import EquivariantModels

function _inv_list(l)
   d = Dict()
   for (i, x) in enumerate(l)
      d[x] = i
   end
   return d
end



# TODO : the `sparse_AA_spec` should be replaced with a `sparse_ace_spec`
#        which generates only a (n, l) spec. From that, we can then generate 
#        the corresponding (n, l, ) AA spec. This would be much more readable. 

"""
This is one of the most important functions to generate an ACE model with 
sparse AA basis. It generates the AA basis specification as a list (`Vector`)
of vectors of `@NamedTuple{n::Int, l::Int, m::Int}`.

### Parameters 

* `order` : maximum correlation order 
* `r_spec` : radial basis specification in the format `Vector{@NamedTuple{a::Int64, b::Int64}}`
* `max_level` : maximum level of the basis, either a single scalar, or an iterable (one for each order)
* `level` : a function that computes the level of a basis element; see e.g. `TotalDegree` and `EuclideanDegree`
"""
function sparse_AA_spec(; order = nothing, 
                          r_spec = nothing, 
                          max_level = nothing, 
                          level = nothing, )
   # compute the r levels
   r_level = [ level(b) for b in r_spec ]

   # generate the A basis spec from the radial basis spec. 
   NT_NLM = NamedTuple{(:n, :l, :m), Tuple{Int, Int, Int}}
   A_spec = NT_NLM[]
   A_spec_level = eltype(r_level)[]
   for br in r_spec
      for m = -br.l:br.l 
         b = (n = br.n, l = br.l, m = m)
         push!(A_spec, b) 
         push!(A_spec_level, level(b))
      end
   end
   p = sortperm(A_spec_level)
   A_spec = A_spec[p]
   A_spec_level = A_spec_level[p]
   inv_A_spec = _inv_list(A_spec)

   # generate the AA basis spec from the A basis spec
   tup2b = vv -> [ A_spec[v] for v in vv[vv .> 0]  ]
   admissible = bb -> (length(bb) == 0) || level(bb) <= max_level
   filter_ = EquivariantModels.RPE_filter_real(0)

   AA_spec = EquivariantModels.gensparse(; 
                        NU = order, tup2b = tup2b, filter = filter_, 
                        admissible = admissible,
                        minvv = fill(0, order), 
                        maxvv = fill(length(A_spec), order), 
                        ordered = true)

   AA_spec = [ vv[vv .> 0] for vv in AA_spec if !(isempty(vv[vv .> 0])) ]

   # map back to nlm
   AA_spec_nlm = Vector{NT_NLM}[]
   if length(AA_spec[1]) == 0 
      push!(AA_spec_nlm, NT_NLM[])
      idx0 = 2
   else
      idx0 = 1
   end
   for vv in AA_spec
      push!(AA_spec_nlm, [ A_spec[v] for v in vv ])
   end

   return AA_spec_nlm 
end


"""
Get the specification of the BBbasis as a list (`Vector`) of vectors of `@NamedTuple{n::Int, l::Int}`.

### Parameters 

* `model` : an ACEModel
"""
function get_nnll_spec(model::ACEModel)
   return get_nnll_spec(model.tensor)
end


import ACE1 

rand_atenv(model::ACEModel, Nat) = rand_atenv(model.rbasis, Nat)

function rand_atenv(rbasis::Union{LearnableRnlrzzBasis, SplineRnlrzzBasis}, Nat)
   z0 = rand(rbasis._i2z) 
   zs = rand(rbasis._i2z, Nat) 
   
   rs = Float64[] 
   for zj in zs 
      iz0 = _z2i(rbasis, z0)
      izj = _z2i(rbasis, zj)
      x = 2 * rand() - 1 
      t_ij = rbasis.transforms[iz0, izj] 
      r_ij = inv_transform(t_ij, x)
      push!(rs, r_ij)
   end
   Rs = [ r * ACE1.Random.rand_sphere() for r in rs ]
   return Rs, zs, z0 
end


using StaticArrays: @SMatrix 
using LinearAlgebra: qr

function rand_rot()
   A = @SMatrix randn(3, 3) 
   Q, _ = qr(A) 
   return Q 
end

rand_iso() = rand([-1,1]) * rand_rot()


