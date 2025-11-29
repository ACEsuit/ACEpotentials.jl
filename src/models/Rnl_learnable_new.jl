
import EquivariantTensors as ET
using StaticArrays
using Lux

# In ET we currently store an edge xij as a NamedTuple, e.g, 
#    xij = (𝐫ij = ..., zi = ..., zj = ...)
# The NTtransform is a wrapper for mapping xij -> y 
# (in this case y = transformed distance) adding logic to enable 
# differentiation through this operation. 
#
# In ET.Atoms edges are of the form xij = (𝐫 = ..., s0 = ..., s1 = ...)


# build a pure Lux Rnl basis 100% compatible with LearnableRnlrzz
#

function _convert_Rnl_learnable(basis; zlist = basis._i2z, 
                                       rfun = x -> x.r )

   # number of species 
   NZ = length(zlist)

   # species z -> index i mapping 
   __z2i = let _i2z = (_i2z = zlist,) 
      z -> _z2i(_i2z, z)
   end

   # __zz2i maps a `(Zi, Zj)` pair to a single index `a` representing 
   # (Zi, Zj) in a flattened array
   __zz2ii = (zi, zj) -> (__z2i(zi) - 1) * NZ + __z2i(zj)
   selector = xij -> __zz2ii(xij.s0, xij.s1)

   # construct the transform to be a Lux layer that behaves a bit 
   # like a WrappedFunction, but with additional support for 
   # named-tuple inputs 
   #
   et_trans = let transforms = basis.transforms
      ET.NTtransform( xij -> begin
            trans_ij = transforms[__z2i(xij.s0), __z2i(xij.s1)]
            return trans_ij(rfun(xij))
         end )
      end 

   # the envelope is always a simple quartic (1 -x^2)^2
   # otherwise make this transform fail. 
   #  ( note the transforms is normalized to map to [-1, 1]
   #    y outside [-1, 1] maps to 1 or -1. )  
   # this obviously needs to be relaxed if we want compatibility 
   # with older versions of the code 
   for env in basis.envelopes
      @assert env isa PolyEnvelope2sX
      @assert env.p1 == env.p2 == 2 
      @assert env.x1 == -1
      @assert env.x2 == 1
   end

   et_env = y -> (1 - y^2)^2

   # the polynomial basis just stays the same 
   #
   et_polys = basis.polys
    
   # the linear layer transformation  
   #   P(yij) -> W[(Zi, Zj)] * P(yij) 
   # with W[a] learnable weights 
   #                                          
   et_linl = ET.SelectLinL(length(et_polys),         # indim
                           length(basis.spec),       # outdim
                           NZ^2,                     # num (Zi,Zj) pairs
                           selector)

   et_rbasis = SkipConnection(   # input is (rij, zi, zj)
            Chain(y = et_trans,  # transforms yij 
                  P = SkipConnection(
                        et_polys, 
                        WrappedFunction( Py -> et_env.(Py[2]) .* Py[1] )
                     )
                  ),   # r -> y -> P = e(y) * polys(y)
            et_linl    # P -> W(Zi, Zj) * P 
         )
        
   return et_rbasis 
end

