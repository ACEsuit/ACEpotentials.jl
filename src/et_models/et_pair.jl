
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
   # embed edges 
   Rnl, _ = l.rembed(X, ps.rembed, st.rembed)

   # readout layer 
   φ, _ = l.readout((Rnl, X.node_data), ps.readout, st.readout)

   return φ
end

# ----------------------------------------------------------- 


function site_grads(l::ETPairModel, X::ET.ETGraph, ps, st)
   ∂X = Zygote.gradient( X -> sum(_apply_etpairmodel(l, X, ps, st)), X)[1]
   return ∂X
end


# ----------------------------------------------------------- 
#    basis and jacobian evaluation 


function site_basis(l::ETPairModel, X::ET.ETGraph, ps, st)      
   # embed edges 
   Rnl, _ = l.rembed(X, ps.rembed, st.rembed)

   return Rnl 
end


function site_basis_jacobian(l::ETPairModel, X::ET.ETGraph, ps, st)    
   (R, ∂R), _ = ET.evaluate_ed(l.rembed, X, ps.rembed, st.rembed)
   return R, ∂R
end

