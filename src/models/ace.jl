
using LuxCore: AbstractExplicitLayer, 
               AbstractExplicitContainerLayer,
               initialparameters, 
               initialstates

using Lux: glorot_normal

using Random: AbstractRNG
using SparseArrays: SparseMatrixCSC     
using StaticArrays: SVector
using LinearAlgebra: norm, dot

import SpheriCart
using SpheriCart: SolidHarmonics, SphericalHarmonics
import RepLieGroups
import EquivariantModels
import Polynomials4ML

# ------------------------------------------------------------
#    ACE MODEL SPECIFICATION 


struct ACEModel{NZ, TRAD, TY, TA, TAA, T} <: AbstractExplicitContainerLayer{(:rbasis,)}
   _i2z::NTuple{NZ, Int}
   rbasis::TRAD
   ybasis::TY
   abasis::TA
   aabasis::TAA
   A2Bmap::SparseMatrixCSC{T, Int}
   # -------------- 
   # we can add a nonlinear embedding here 
   # --------------
   bparams::NTuple{NZ, Vector{T}}
   aaparams::NTuple{NZ, Vector{T}}
   # --------------
   meta::Dict{String, Any}
end

# ------------------------------------------------------------
#    CONSTRUCTORS AND UTILITIES

const NT_NLM = NamedTuple{(:n, :l, :m), Tuple{Int, Int, Int}}

function _make_Y_basis(Ytype, lmax) 
   if Ytype == :solid 
      return SolidHarmonics(lmax)
   elseif Ytype == :spherical
      return SphericalHarmonics(lmax)
   end 

   error("unknown `Ytype` = $Ytype - I don't know how to generate a spherical basis from this.")
end

# can we ignore the level function here? 
function _make_A_spec(AA_spec, level)
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

# this should go into sphericart or P4ML 
function _make_Y_spec(maxl::Integer)
   NT_LM = NamedTuple{(:l, :m), Tuple{Int, Int}}
   y_spec = NT_LM[] 
   for i = 1:SpheriCart.sizeY(maxl)
      l, m = SpheriCart.idx2lm(i)
      push!(y_spec, (l = l, m = m))
   end
   return y_spec 
end


function _generate_ace_model(rbasis, Ytype::Symbol, AA_spec::AbstractVector, 
                             level = TotalDegree())
   # generate the coupling coefficients 
   cgen = EquivariantModels.Rot3DCoeffs_real(0)
   AA2BB_map = EquivariantModels._rpi_A2B_matrix(cgen, AA_spec)

   # find which AA basis functions are actually used and discard the rest 
   keep_AA_idx = findall(sum(abs, AA2BB_map; dims = 1)[:] .> 0)
   AA_spec = AA_spec[keep_AA_idx]
   AA2BB_map = AA2BB_map[:, keep_AA_idx]

   # generate the corresponding A basis spec
   A_spec = _make_A_spec(AA_spec, level)

   # from the A basis we can generate the Y basis since we now know the 
   # maximum l value (though we probably already knew that from r_spec)
   maxl = maximum([ b.l for b in A_spec ])   
   ybasis = _make_Y_basis(Ytype, maxl)
   
   # now we need to take the human-readable specs and convert them into 
   # the layer-readable specs 
   r_spec = rbasis.spec
   y_spec = _make_Y_spec(maxl)

   # get the idx version of A_spec 
   inv_r_spec = _inv_list(r_spec)
   inv_y_spec = _inv_list(y_spec)
   A_spec_idx = [ (inv_r_spec[(n=b.n, l=b.l)], inv_y_spec[(l=b.l, m=b.m)]) 
                  for b in A_spec ]
   # from this we can now generate the A basis layer                   
   a_basis = Polynomials4ML.PooledSparseProduct(A_spec_idx)
   a_basis.meta["A_spec"] = A_spec  #(also store the human-readable spec)

   # get the idx version of AA_spec
   inv_A_spec = _inv_list(A_spec)
   AA_spec_idx = [ [ inv_A_spec[b] for b in bb ] for bb in AA_spec ]
   sort!.(AA_spec_idx)
   # from this we can now generate the AA basis layer
   aa_basis = Polynomials4ML.SparseSymmProdDAG(AA_spec_idx)
   aa_basis.meta["AA_spec"] = AA_spec  # (also store the human-readable spec)
   
   NZ = _get_nz(rbasis)
   n_B_params, n_AA_params = size(AA2BB_map)
   return ACEModel(rbasis._i2z, rbasis, ybasis, a_basis, aa_basis, AA2BB_map, 
                   ntuple(_ -> zeros(n_B_params), NZ),
                   ntuple(_ -> zeros(n_AA_params), NZ),
                   Dict{String, Any}() )
end

# TODO: it is not entirely clear that the `level` is really needed here 
#       since it is implicitly already encoded in AA_spec. We need a 
#       function `auto_level` that generates level automagically from AA_spec.
function ace_model(rbasis, Ytype, AA_spec::AbstractVector, level)
   return _generate_ace_model(rbasis, Ytype, AA_spec, level)
end 

# NOTE : a nicer convenience constructor is also provided in `ace_heuristics.jl`
#        this is where we should move all defaults, heuristics and other things 
#        that make life good.

# ------------------------------------------------------------
#   Lux stuff 

function initialparameters(rng::AbstractRNG, 
                           model::ACEModel)
   NZ = _get_nz(model)
   n_B_params, n_AA_params = size(model.A2Bmap)

   # only the B params are parameters, the AA params are uniquely defined 
   # via the B params. 

   # there are different ways to initialize parameters 
   if model.meta["init_WB"] == "zeros"
      winit = zeros 
   elseif model.meta["init_WB"] == "glorot_normal"
      winit = glorot_normal
   else
      error("unknown `init_WB` = $(model.meta["init_WB"])")
   end

   return (WB = [ winit(Float64, n_B_params) for _=1:NZ ], 
           rbasis = initialparameters(rng, model.rbasis), )
end

function initialstates(rng::AbstractRNG, 
                       model::ACEModel)
   return ( rbasis = initialstates(rng, model.rbasis), )
end

(l::ACEModel)(args...) = evaluate(l, args...)


# ------------------------------------------------------------
#   Model Evaluation 
#   this should possibly be moved to a separate file once it 
#   gets more complicated.

# these _getlmax and _length should be moved into SpheriCart 
_getlmax(ybasis::SolidHarmonics{L}) where {L} = L 
_length(ybasis::SolidHarmonics) = SpheriCart.sizeY(_getlmax(ybasis))

function evaluate(model::ACEModel, 
                  Rs::AbstractVector{SVector{3, T}}, Zs, Z0, 
                  ps, st) where {T}
   # get the radii 
   rs = [ norm(r) for r in Rs ]   # use Bumper 

   # evaluate the radial basis
   # use Bumper to pre-allocate 
   Rnl, _st = evaluate_batched(model.rbasis, rs, Z0, Zs, 
                              ps.rbasis, st.rbasis)

   # evaluate the Y basis
   Ylm = zeros(T, length(Rs), _length(model.ybasis))    # use Bumper here
   SpheriCart.compute!(Ylm, model.ybasis, Rs)

   # evaluate the A basis
   TA = promote_type(T, eltype(Rnl))
   A = zeros(T, length(model.abasis))
   Polynomials4ML.evaluate!(A, model.abasis, (Rnl, Ylm))

   # evaluate the AA basis
   _AA = zeros(T, length(model.aabasis))     # use Bumper here
   Polynomials4ML.evaluate!(_AA, model.aabasis, A)
   # project to the actual AA basis 
   proj = model.aabasis.projection
   AA = _AA[proj]     # use Bumper here, or view; needs experimentation. 

   # evaluate the coupling coefficients
   B = model.A2Bmap * AA

   # contract with params 
   i_z0 = _z2i(model.rbasis, Z0)
   val = dot(B, ps.WB[i_z0])
            
   return val, st 
end