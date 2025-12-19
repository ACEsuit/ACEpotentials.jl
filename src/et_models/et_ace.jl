
import EquivariantTensors as ET 
import Polynomials4ML as P4ML

import LuxCore: AbstractLuxContainerLayer
import AtomsBase: ChemicalSpecies 
using ConcreteStructs: @concrete
using LinearAlgebra: norm, dot 


@concrete struct ETACE  <: AbstractLuxContainerLayer{(:rembed, :yembed, :basis, :readout)}
   rembed     # radial embedding layer
   yembed     # angular embedding layer
   basis      # many-body basis layer
   readout    # selectlinl readout layer
end 


(l::ETACE)(X::ET.ETGraph, ps, st) = _apply_etace(l, X, ps, st), st 
      
      
function _apply_etace(l::ETACE, X::ET.ETGraph, ps, st)      
   # embed edges 
   Rnl, _ = l.rembed(X, ps.rembed, st.rembed)
   Ylm, _ = l.yembed(X, ps.yembed, st.yembed)

   # many-body basis 
   𝔹, _ = l.basis((Rnl, Ylm), ps.basis, st.basis)

   # readout layer 
   φ, _ = l.readout((𝔹[1], X.node_data), ps.readout, st.readout)

   # TODO: return site energies or total energy? 
   #       for THIS layer probably site energies, then write all 
   #       the summation and differentiation in the calculator layer. 
   #       so this is only temporary for testing. 

   return sum(φ)
end


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