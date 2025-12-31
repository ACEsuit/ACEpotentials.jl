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
   # embed edges (inline to avoid Zygote thunk issues with site_basis)
   Rnl, _ = l.rembed(X, ps.rembed, st.rembed)

   # sum over neighbours for each node
   𝔹 = dropdims(sum(Rnl, dims=1), dims=1)

   # readout layer
   φ, _ = l.readout((𝔹, X.node_data), ps.readout, st.readout)

   return φ
end

# ----------------------------------------------------------- 


function site_grads(l::ETPairModel, X::ET.ETGraph, ps, st)
   # Use evaluate_ed to get basis and derivatives, avoiding Zygote thunk issues
   (R, ∂R), _ = ET.evaluate_ed(l.rembed, X, ps.rembed, st.rembed)

   # R has shape (maxneigs, nnodes, nbasis) after embedding
   # 𝔹 = sum over neighbours: shape (nnodes, nbasis)
   𝔹 = dropdims(sum(R, dims=1), dims=1)

   # Get readout weights
   iZ = l.readout.selector.(X.node_data)
   WW = ps.readout.W

   # ∂E/∂R = W[1, :, iZ[i]] for each node, broadcast over neighbours
   # ∂R has shape (maxneigs, nnodes, nbasis)
   nnodes = length(X.node_data)
   ∂E_∂𝔹 = reduce(hcat, WW[1, :, iZ[i]] for i in 1:nnodes)'  # (nnodes, nbasis)

   # ∂E/∂R[j, i, k] = ∂E/∂𝔹[i, k] (same for all neighbours j)
   ∂E_∂R = reshape(∂E_∂𝔹, 1, size(∂E_∂𝔹)...)  # (1, nnodes, nbasis)

   # Chain rule: ∂E/∂X = sum over k of (∂E/∂R * ∂R/∂X)
   # ∂R has shape (maxneigs, nnodes, nbasis), contains VState gradients
   ∂E_edges = dropdims(sum(∂E_∂R .* ∂R, dims=3), dims=3)  # (maxneigs, nnodes)

   # Reshape to match edge_data format
   ∂E_edges_vec = ET.rev_reshape_embedding(∂E_edges, X)

   return (; edge_data = ∂E_edges_vec)
end


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

