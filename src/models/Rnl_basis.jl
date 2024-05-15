
import LuxCore: AbstractExplicitLayer, 
                initialparameters, 
                initialstates
using StaticArrays: SMatrix, SVector
using Random: AbstractRNG

import OffsetArrays, Interpolations
using Interpolations: cubic_spline_interpolation, BSpline, Cubic, Line, OnGrid


# NOTEs: 
#  each smatrix in the Rnl types indexes (i, j) 
#  where i is the center, j is neighbour

const NT_RIN0CUTS{T} = NamedTuple{(:rin, :r0, :rcut), Tuple{T, T, T}}
const NT_NL_SPEC = NamedTuple{(:n, :l), Tuple{Int, Int}}
const SPL_OF_SVEC{DIM, T} = 
      Interpolations.Extrapolation{SVector{DIM, T}, 1, 
         Interpolations.ScaledInterpolation{SVector{DIM, T}, 1, 
            Interpolations.BSplineInterpolation{SVector{DIM, T}, 1, 
               OffsetArrays.OffsetVector{SVector{DIM, T}, Vector{SVector{DIM, T}}}, 
               BSpline{Cubic{Line{OnGrid}}}, Tuple{Base.OneTo{Int}}}, 
            BSpline{Cubic{Line{OnGrid}}}, 
            Tuple{StepRangeLen{T, Base.TwicePrecision{T}, Base.TwicePrecision{T}, Int}}}, 
         BSpline{Cubic{Line{OnGrid}}}, Interpolations.Throw{Nothing}
      }


struct LearnableRnlrzzBasis{NZ, TPOLY, TT, TENV, TW, T} <: AbstractExplicitLayer
   _i2z::NTuple{NZ, Int}
   polys::TPOLY
   transforms::SMatrix{NZ, NZ, TT}
   envelopes::SMatrix{NZ, NZ, TENV}
   # -------------- 
   weights::Array{TW, 4}                      # learnable weights, `nothing` when using Lux
   rin0cuts::SMatrix{NZ, NZ, NT_RIN0CUTS{T}}  # matrix of (rin, rout, rcut)
   spec::Vector{NT_NL_SPEC}       
   # --------------
   # meta
   meta::Dict{String, Any} 
end

function set_params(basis::LearnableRnlrzzBasis, ps)
   return LearnableRnlrzzBasis(basis._i2z, 
                               basis.polys, 
                               basis.transforms, 
                               basis.envelopes, 
                               # ---------------
                               ps.Wnlq, 
                               basis.rin0cuts, 
                               basis.spec, 
                               # ---------------
                               basis.meta)
end


struct SplineRnlrzzBasis{NZ, TT, TENV, LEN, T} <: AbstractExplicitLayer
   _i2z::NTuple{NZ, Int}
   transforms::SMatrix{NZ, NZ, TT}
   envelopes::SMatrix{NZ, NZ, TENV}
   splines::SMatrix{NZ, NZ, SPL_OF_SVEC{LEN, T}}
   # -------------- 
   rin0cuts::SMatrix{NZ, NZ, NT_RIN0CUTS{T}}  # matrix of (rin, rout, rcut)
   spec::Vector{NT_NL_SPEC}       
   # --------------
   meta::Dict{String, Any} 
end



# a few getter functions for convenient access to those fields of matrices
_rincut_zz(obj, zi, zj) = obj.rin0cuts[_z2i(obj, zi), _z2i(obj, zj)]
_rin0cuts_zz(obj, zi, zj) = obj.rin0cuts[_z2i(obj, zi), _z2i(obj, zj)]
_rcut_zz(obj, zi, zj) = obj.rin0cuts[_z2i(obj, zi), _z2i(obj, zj)].rcut
_rin_zz(obj, zi, zj) = obj.rin0cuts[_z2i(obj, zi), _z2i(obj, zj)].rin
_r0_zz(obj, zi, zj) = obj.rin0cuts[_z2i(obj, zi), _z2i(obj, zj)].r0
_envelope_zz(obj, zi, zj) = obj.envelopes[_z2i(obj, zi), _z2i(obj, zj)]
_spline_zz(obj, zi, zj) = obj.splines[_z2i(obj, zi), _z2i(obj, zj)]
_transform_zz(obj, zi, zj) = obj.transforms[_z2i(obj, zi), _z2i(obj, zj)]

_get_T(basis::LearnableRnlrzzBasis) = typeof(basis.rin0cuts[1,1].rin)

splinify(basis::SplineRnlrzzBasis; kwargs...) = basis 
    

function splinify(basis::LearnableRnlrzzBasis; nnodes = 100)

   # transform : r ∈ [rin, rcut] -> x
   # and then Rnl =  Wnl_q * Pq(x) * env(x) gives the basis. 
   # The problem with this is that we cannot evaluate the envelope from just 
   # r coordinates. We therefore keep the transform inside the splinified 
   # basis and only splinify the last operation, x -> Rnl(x)
   # this also has the potential advantage that few spline points are needed, 
   # and that we get access to the same meta-information about the model building.
   #
   # in the following we assume all transforms map [rin, rcut] -> [-1, 1]

   NZ = _get_nz(basis)
   T = _get_T(basis)
   LEN = size(basis.weights, 1)
   _splines = Matrix{SPL_OF_SVEC{LEN, T}}(undef, (NZ, NZ))
   x_nodes = range(-1.0, 1.0, length = nnodes)
   polys = basis.polys

   for iz0 = 1:NZ, iz1 = 1:NZ
      rin0cut = basis.rin0cuts[iz0, iz1]
      rin, rcut = rin0cut.rin, rin0cut.rcut
      
      Tij = basis.transforms[iz0, iz1]
      Wnlq_ij = @view basis.weights[:, :, iz0, iz1] 
      Rnl = [ SVector{LEN}( Wnlq_ij * Polynomials4ML.evaluate(polys, x) )
              for x in x_nodes ]

      # now we need to spline the Rnl
      splines_ij = cubic_spline_interpolation(x_nodes, Rnl)
      _splines[iz0, iz1] = splines_ij
   end

   splines = SMatrix{NZ, NZ, SPL_OF_SVEC{LEN, T}}(_splines)

   spl_basis = SplineRnlrzzBasis(basis._i2z, 
                                 basis.transforms,
                                 basis.envelopes, 
                                 splines, 
                                 basis.rin0cuts,
                                 basis.spec, 
                                 basis.meta)

   spl_basis.meta["info"] = "constructed from LearnableRnlrzzBasis via `splinify`"

   # we should probably store more meta-data from which the splines can be 
   # easily reconstructed. 

   return spl_basis
end

