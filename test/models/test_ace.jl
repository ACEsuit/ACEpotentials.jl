
using Pkg; Pkg.activate(joinpath(@__DIR__(), "..", ".."))
# using TestEnv; TestEnv.activate();

using ACEpotentials
M = ACEpotentials.Models

using Polynomials4ML
P4ML = Polynomials4ML

using Random, LuxCore
rng = Random.MersenneTwister(1234)

function _inv_list(l)
   d = Dict()
   for (i, x) in enumerate(l)
      d[x] = i
   end
   return d
end

struct TotalDegree 
   wL::Float64
end 

TotalDegree() = TotalDegree(1.5)

(l::TotalDegree)(b::NamedTuple) = b.n + b.l
(l::TotalDegree)(bb::Vector{<: NamedTuple}) = sum(l(b) for b in bb)


function make_AA_spec(; order = nothing, 
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

function make_A_spec(AA_spec, level) 
   NT_NLM = NamedTuple{(:n, :l, :m), Tuple{Int, Int, Int}}
   A_spec = NT_NLM[]
   for bb in AA_spec 
      append!(A_spec, bb)
   end
   A_spec_level = [ level(b) for b in A_spec ]
   p = sortperm(A_spec_level)
   A_spec = A_spec[p]
   return A_spec
end 

##

elements = (:Si, :O)
level = TotalDegree()
max_level = 8
lmax = 4 

rbasis = M.ace_learnable_Rnlrzz(Dtot = Dtot, lmax = lmax, elements = elements)
r_spec = rbasis.spec

AA_spec = make_specs(order = 3, r_spec = r_spec, 
                     level = level, max_level = max_level)

##

import RepLieGroups
import EquivariantModels
import SpheriCart

cgen = EquivariantModels.Rot3DCoeffs_real(0)
AA2BB_map = EquivariantModels._rpi_A2B_matrix(cgen, AA_spec)

keep_AA_idx = findall(sum(abs, AA2BB_map; dims = 1)[:] .> 0)

AA_spec = AA_spec[keep_AA_idx]
AA2BB_map = AA2BB_map[:, keep_AA_idx]

A_spec = make_A_spec(AA_spec, level)

maxl = maximum([ b.l for b in A_spec ])

ybasis = SpheriCart.SolidHarmonics(maxl)

## 
# now we need to take the human-readable specs and convert them into 
# the layer-readable specs 

r_spec = rbasis.spec 

# this should go into sphericart or P4ML 
NT_LM = NamedTuple{(:l, :m), Tuple{Int, Int}}
y_spec = NT_LM[] 
for i = 1:SpheriCart.sizeY(maxl)
   l, m = SpheriCart.idx2lm(i)
   push!(y_spec, (l = l, m = m))
end

# get the idx version of A_spec 
inv_r_spec = _inv_list(r_spec)
inv_y_spec = _inv_list(y_spec)
A_spec_idx = [ (inv_r_spec[(n=b.n, l=b.l)], inv_y_spec[(l=b.l, m=b.m)]) 
               for b in A_spec ]
a_basis = P4ML.PooledSparseProduct(A_spec_idx)
a_basis.meta["A_spec"] = A_spec

inv_A_spec = _inv_list(A_spec)
AA_spec_idx = [ [ inv_A_spec[b] for b in bb ] for bb in AA_spec ]
sort!.(AA_spec_idx)
aa_basis = P4ML.SparseSymmProdDAG(AA_spec_idx)
aa_basis.meta["AA_spec"] = AA_spec

length(aa_basis)