
using StaticArrays
using Lux

import EquivariantTensors as ET
import Polynomials4ML as P4ML

import ACEpotentials.Models: LearnableRnlrzzBasis, PolyEnvelope2sX, 
         _i2z, GeneralizedAgnesiTransform 

using LinearAlgebra: norm, dot 


function convert2et(model)
   # TODO: add checks that the model we are importing is of the format 
   #       that we can actually import and then raise errors if not.
   #       but since we might just drop this import functionality entirely it
   #       is not so clear we should waste our time on that. 

   # extract species information from the ACE model 
   rbasis = model.rbasis
   et_i2z = ChemicalSpecies.(rbasis._i2z)

   # ---------------------------- REMBED
   # convert the radial basis 
   et_rbasis = _convert_Rnl_learnable(rbasis) 
   et_rspec = rbasis.spec
   # convert the radial basis into an edge embedding layer which has some 
   # additional logic for handling the ETGraph input correctly 
   rembed = ET.EdgeEmbed( et_rbasis; name = "Rnl" )

   # ---------------------------- YEMBED 
   # convert the angular basis
   ybasis = model.ybasis
   et_ybasis = ET.EmbedDP( ET.NTtransform(x -> x.𝐫), 
                           ybasis )
   et_yspec = P4ML.natural_indices(et_ybasis.basis)
   yembed = ET.EdgeEmbed( et_ybasis; name = "Ylm" )

   # ---------------------------- MANY-BODY BASIS
   # Convert AA_spec from (n,l,m) format to (n,l) format for mb_spec
   AA_spec = model.tensor.meta["𝔸spec"] 
   et_mb_spec = unique([[(n=b.n, l=b.l) for b in bb] for bb in AA_spec])

   et_mb_basis = ET.sparse_equivariant_tensor(
         L = 0,  # Invariant (scalar) output only
         mb_spec = et_mb_spec,
         Rnl_spec = et_rspec,
         Ylm_spec = et_yspec,
         basis = real
      )

   # ---------------------------- READOUT LAYER
   # readout layer : need to select which linear operator to apply 
   # based on the center atom species
   selector = let zlist = et_i2z
      x -> ET.cat2idx(zlist, x.z)
   end
   readout = ET.SelectLinL(
                  et_mb_basis.lens[1],  # input dim (mb basis length)
                  1,                    # output dim (only one site energy per atom)
                  length(et_i2z),       # number of categories = num species 
                  selector)            

   # generate the model and return it 
   et_model = ETACE(rembed, yembed, et_mb_basis, readout)
   return et_model
end


# In ET we currently store an edge xij as a NamedTuple, e.g, 
#    xij = (𝐫ij = ..., zi = ..., zj = ...)
# The NTtransform is a wrapper for mapping xij -> y 
# (in this case y = transformed distance) adding logic to enable 
# differentiation through this operation. 
#
# In ET.Atoms edges are of the form xij = (𝐫 = ..., s0 = ..., s1 = ...)


# build a pure Lux Rnl basis 100% compatible with LearnableRnlrzz

function _convert_Rnl_learnable(basis; zlist = ChemicalSpecies.(basis._i2z), 
                                       rfun = x -> norm(x.𝐫) )

   # number of species 
   NZ = length(zlist)

   # species z -> index i mapping 
   __z2i = let _i2z = (_i2z = zlist,) 
      z -> _z2i(_i2z, z)
   end

   # __zz2i maps a `(Zi, Zj)` pair to a single index `a` representing 
   # (Zi, Zj) in a flattened array
   __zz2ii = (zi, zj) -> (__z2i(zi) - 1) * NZ + __z2i(zj)

   selector = let zlist = tuple(zlist...)
      xij -> ET.catcat2idx(zlist, xij.z0, xij.z1)
   end

   # construct the transform to be a Lux layer that behaves a bit 
   # like a WrappedFunction, but with additional support for 
   # named-tuple or DP inputs 
   #
   et_trans = _convert_agnesi(basis)
   
   # OLD VERSION - KEEP FOR DEBUGGING then remove 
   # et_trans = let transforms = basis.transforms
   #    ET.NTtransform( xij -> begin
   #          trans_ij = transforms[__z2i(xij.s0), __z2i(xij.s1)]
   #          return trans_ij(rfun(xij))
   #       end )
   #    end 

   # the envelope is always a simple quartic y -> (1 - y^2)^2
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
   # but needs to be wrapped due to the envelope being applied 
   # 
   et_polys = basis.polys
   Penv = P4ML.wrapped_basis( BranchLayer(
            et_polys,   # y -> P
            WrappedFunction( y -> et_env.(y) ),  # y -> fₑₙᵥ
            fusion = WrappedFunction( Pe -> Pe[2] .* Pe[1] )  
         ), rand() ) 

   # the linear layer transformation  
   #   P(yij) -> W[(Zi, Zj)] * P(yij) 
   # with W[a] learnable weights 
   #                                          
   et_linl = ET.SelectLinL(length(et_polys),         # indim
                           length(basis.spec),       # outdim
                           NZ^2,                     # num (Zi,Zj) pairs
                           selector)

   et_rbasis = ET.EmbedDP(et_trans, Penv, et_linl)
   return et_rbasis 
end



# important auxiliary function to convert the transforms 

function _agnesi_et_params(trans) 
   @assert trans.trans isa GeneralizedAgnesiTransform
   a = trans.trans.a 
   pcut = trans.trans.p 
   pin = trans.trans.q 
   req = trans.trans.r0 
   rin = trans.trans.rin 
   rcut = trans.rcut 

   params = ET.agnesi_params(pcut, pin, rin, req, rcut)
   @assert params.a ≈ a 

   # ----- for debugging -----------
   # r = rin + rand() * (rcut - rin)
   # y1 = trans(r) 
   # y2 = ET.eval_agnesi(r, params)
   # @assert y1 ≈ y2
   # -------------------------------

   return params
end 


function _convert_agnesi(rbasis::LearnableRnlrzzBasis)
   transforms = rbasis.transforms 
   @assert transforms isa SMatrix 
   NZ = size(transforms, 1) 
   params = [] 
   for i = 1:NZ, j = i:NZ 
      push!(params, _agnesi_et_params(transforms[i,j]))
   end
   st = (zlist = ChemicalSpecies.(rbasis._i2z), 
         params = SVector{length(params)}(identity.(params)) )
    
   f_agnesi = let 
      (x, st) -> begin
         r = norm(x.𝐫)
         idx = ET.catcat2idx_sym(st.zlist, x.z0, x.z1)
         return ET.eval_agnesi(r, st.params[idx])
      end   
   end
         
   return ET.NTtransformST(f_agnesi, st)
end 