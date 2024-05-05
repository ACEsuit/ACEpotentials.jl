
import LuxCore: AbstractExplicitLayer, 
                initialparameters, 
                initialstates
using StaticArrays: SMatrix 
using Random: AbstractRNG

abstract type AbstractRnlzzBasis <: AbstractExplicitLayer end

# NOTEs: 
#  each smatrix in the types below indexes (i, j) 
#  where i is the center, j is neighbour


struct LearnableRnlrzzBasis{NZ, TPOLY, TT, TENV, TW} <: AbstractRnlzzBasis
   _i2z::NTuple{NZ, Int}
   polys::TPOLY
   transforms::SMatrix{NZ, NZ, TT}
   envelopes::SMatrix{NZ, NZ, TENV}
   # rcut::SMatrix{NZ, NZ, T}            # matrix of (rin, rout)
   weights::SMatrix{NZ, NZ, TW}        # learnable weights, nothing when using Lux
   #--------------
   # meta should contain spec, rin0cuts
   meta::Dict{String, Any} 
end


# struct SplineRnlrzzBasis{NZ, SPL, ENV} <: AbstractRnlzzBasis
#    _i2z::NTuple{NZ, Int}                 # iz -> z mapping
#    splines::SMatrix{NZ, NZ, SPL}         # matrix of splined radial bases
#    envelopes::SMatrix{NZ, NZ, ENV}       # matrix of radial envelopes
#    rincut::SMatrix{NZ, NZ, Tuple{T, T}}  # matrix of (rin, rout)

#    #-------------- 
#    # meta should contain spec 
#    meta::Dict{String, Any} 
# end

# a few getter functions for convenient access to those fields of matrices
_rincut_zz(obj, zi, zj) = obj.rincut[_z2i(obj, zi), _z2i(obj, zj)]
_envelope_zz(obj, zi, zj) = obj.envelopes[_z2i(obj, zi), _z2i(obj, zj)]
_spline_zz(obj, zi, zj) = obj.splines[_z2i(obj, zi), _z2i(obj, zj)]
_transform_zz(obj, zi, zj) = obj.transforms[_z2i(obj, zi), _z2i(obj, zj)]
_poly_zz(obj, zi, zj) = obj.poly[_z2i(obj, zi), _z2i(obj, zj)]


# ------------------------------------------------------------ 
#      CONSTRUCTORS AND UTILITIES 
# ------------------------------------------------------------ 

# these _auto_trans are very poor and need to take care of a lot more 
# cases, e.g. we may want to pass in the objects as a Matrix rather than 
# SMatrix ... 

_auto_trans(t, NZ) = (t isa SMatrix) ? t : SMatrix{NZ, NZ}(fill(t, (NZ, NZ)))

_auto_envel(env, NZ) = (env isa SMatrix) ? env : SMatrix{NZ, NZ}(fill(env, (NZ, NZ)))

_auto_rincut(rincut, NZ) = (rincut isa SMatrix) ? rincut : SMatrix{NZ, NZ}(fill(rincut, (NZ, NZ)))

_auto_weights(weights, NZ) = (weights isa SMatrix) ? weights : SMatrix{NZ, NZ}(fill(weights, (NZ, NZ)))


function LearnableRnlrzzBasis(
            zlist, polys, transforms, envelopes, 
            rin0cuts, 
            spec::Vector{<: NamedTuple}; 
            weights=nothing, 
            meta=Dict{String, Any}())
   meta["rin0cuts"] = rin0cuts
   meta["spec"] = spec          
   LearnableRnlrzzBasis(_convert_zlist(zlist), polys, 
                     _auto_trans(transforms, length(zlist)), 
                     _auto_envel(envelopes, length(zlist)), 
                     # _auto_rincut(rincut, length(zlist)), 
                     _auto_weights(weights, length(zlist)), 
                     meta)
end

Base.length(basis::LearnableRnlrzzBasis) = length(basis.meta["spec"])

function initialparameters(rng::AbstractRNG, 
                           basis::LearnableRnlrzzBasis)
   NZ = _get_nz(basis) 
   len_nl = length(basis)
   len_q = length(basis.polys)

   function _W()
      W = randn(rng, len_nl, len_q)
      W  = W ./ sqrt.(sum(W.^2, dims = 2))
   end

   return (W = [ _W() for i = 1:NZ, j = 1:NZ ], )
end

function initialstates(rng::AbstractRNG, 
                       basis::LearnableRnlrzzBasis)
   return NamedTuple()                       
end
                  

# function learnable_Rnlrzz_basis(zlist; 
#                 polys = :auto, 
#                 transforms = :auto, 
#                 envelopes = :auto, 
#                 rincut = :auto, 
#                 weight = :auto)
   
# end                

function splinify(basis::LearnableRnlrzzBasis)

end

# ------------------------------------------------------------ 
#      EVALUATION INTERFACE
# ------------------------------------------------------------ 

import Polynomials4ML

(l::LearnableRnlrzzBasis)(args...) = evaluate(l, args...)

function evaluate(basis::LearnableRnlrzzBasis, r, Zi, Zj, ps, st)
   iz = _z2i(basis, Zi)
   jz = _z2i(basis, Zj)
   Wij = ps.W[iz, jz]
   trans_ij = basis.transforms[iz, jz]
   x = trans_ij(r)
   P = Polynomials4ML.evaluate(basis.polys, x)
   env_ij = basis.envelopes[iz, jz]
   e = evaluate(env_ij, x)   
   return Wij * (P .* e), st 
end