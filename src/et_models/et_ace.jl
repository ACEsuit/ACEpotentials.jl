
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
   (𝔹,), _ = l.basis((Rnl, Ylm), ps.basis, st.basis)

   # readout layer 
   φ, _ = l.readout((𝔹, X.node_data), ps.readout, st.readout)

   # TODO: return site energies or total energy? 
   #       for THIS layer probably site energies, then write all 
   #       the summation and differentiation in the calculator layer. 

   return φ
end

# -----------------------------------------------------------

import Zygote

#
# At first glance this looks like we are computing ∂E / ∂ri but this is not
# actually true. Because E = ∑ Ei and by interpreting G as a list of edges
# we are differentiating E w.r.t. 𝐫ij which is the same as Ei w.r.t. 𝐫ij.
#
# NB: an analytic VJP (via ET._ka_pullback) was prototyped but only ~10-15%
# faster here — ET's many-body kernel intermediates dominate the cost, not
# Zygote's overhead — and it coupled ACEpotentials tightly to ET internals.
# We therefore keep the Zygote gradient for the many-body model. The pair model
# (et_pair.jl) does use the analytic gradient, since there it is a trivial,
# low-coupling contraction.
#
function site_grads(l::ETACE, X::ET.ETGraph, ps, st)
   ∂X = Zygote.gradient( X -> sum(_apply_etace(l, X, ps, st)), X)[1]
   return ∂X
end


# ----------------------------------------------------------- 
#    basis and jacobian evaluation 


function site_basis(l::ETACE, X::ET.ETGraph, ps, st)      
   # embed edges 
   Rnl, _ = l.rembed(X, ps.rembed, st.rembed)
   Ylm, _ = l.yembed(X, ps.yembed, st.yembed)

   # many-body basis 
   𝔹, _ = l.basis((Rnl, Ylm), ps.basis, st.basis)

   return 𝔹[1] 
end


function site_basis_jacobian(l::ETACE, X::ET.ETGraph, ps, st)
   (R, ∂R), _ = ET.evaluate_ed(l.rembed, X, ps.rembed, st.rembed)
   (Y, ∂Y), _ = ET.evaluate_ed(l.yembed, X, ps.yembed, st.yembed)
   # _jacobian_X for SparseACEbasis takes (basis, Rnl, Ylm, dRnl, dYlm, ps, st)
   # Requires EquivariantTensors >= 0.4.2
   (𝔹,), (∂𝔹,) = ET._jacobian_X(l.basis, R, Y, ∂R, ∂Y, ps.basis, st.basis)
   return 𝔹, ∂𝔹
end
