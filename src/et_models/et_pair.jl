#
# This is a temporary model implementation needed due to the fact that
# ETACEModel has Rnl, Ylm hard-coded. In the future it could be tested
# whether the pair model could simply be taken as another ACE model
# with a single embedding rather than several, This would need generalization
# of a fair few methods in both ACEpotentials and EquivariantTensors.
#


import EquivariantTensors as ET 
import Zygote 
import LuxCore: AbstractLuxContainerLayer
using ConcreteStructs: @concrete


@concrete struct ETPairModel  <: AbstractLuxContainerLayer{(:rembed, :readout)}
   rembed     # radial embedding layer = basis 
   readout    # normally a selectlinl readout layer
end 


(l::ETPairModel)(X::ET.ETGraph, ps, st) = _apply_etpairmodel(l, X, ps, st), st 
      
      
function _apply_etpairmodel(l::ETPairModel, X::ET.ETGraph, ps, st)      
   # evaluate the basis 
   𝔹 = site_basis(l, X, ps, st)

   # readout layer 
   φ, _ = l.readout((𝔹, X.node_data), ps.readout, st.readout)

   return φ
end

# ----------------------------------------------------------- 


# analytic edge gradients (see _site_grads_analytic in et_calculators.jl);
# replaces the previous allocation-heavy Zygote.gradient implementation.
site_grads(l::ETPairModel, X::ET.ETGraph, ps, st) = _site_grads_analytic(l, X, ps, st)


# ----------------------------------------------------------- 
#    basis and jacobian evaluation 


function site_basis(l::ETPairModel, X::ET.ETGraph, ps, st)      
   # embed edges 
   Rnl, _ = l.rembed(X, ps.rembed, st.rembed)

   # the basis is obtain by summing over the neighbours of each node, 
   # which is just a sum over the first dimension of Rnl 
   𝔹 = dropdims(sum(Rnl, dims=1), dims=1)

   return 𝔹
end


function site_basis_jacobian(l::ETPairModel, X::ET.ETGraph, ps, st)    
   (R, ∂R), _ = ET.evaluate_ed(l.rembed, X, ps.rembed, st.rembed)
   𝔹 = dropdims(sum(R, dims=1), dims=1)
   # ∂𝔹 == ∂R
   return 𝔹, ∂R
end

