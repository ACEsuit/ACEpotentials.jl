#
# NOTES 
# First draft of a one-body function that fits into the 
# et_model interface. At the moment the E0s are forced to be 
# stored as an SVector, to be accessed via the selector. 
# But with a little bit of extra work we could add the option 
# of making the E0s either constants or learnable parameters. 
#

using Random: AbstractRNG
import EquivariantTensors as ET
using DecoratedParticles: VState
using StaticArrays: SVector


"""
   one_body(D::Dict, catfun)

Create a one-body energy model that assigns to each atom an energy based on 
a categorical variable that is extracted from the atom state via 
`catfun`. The dictionary `D` contains category-value pairs. The one-body 
energy assigned to an atom with state `x` is `D[catfun(x)]`.
"""
function one_body(D::Dict, catfun) 
   categories = SVector(collect(keys(D))...)
   E0s = SVector(collect(values(D))...)
   NZ = length(categories)
   selector = x -> ET.cat2idx(categories, catfun(x))
   return ETOneBody{NZ, eltype(E0s), eltype(categories), typeof(selector)}(
               E0s, categories, selector)
end


using StaticArrays: SVector 
import LuxCore: AbstractLuxLayer, initialparameters, initialstates


struct ETOneBody{NZ, T, CAT, TSEL}  <: AbstractLuxLayer
   E0s::SVector{NZ, T}
   categories::SVector{NZ, CAT}
   selector::TSEL
end 

initialstates(rng::AbstractRNG, l::ETOneBody) = (; E0s = l.E0s)



(l::ETOneBody)(x, ps, st) = _apply_onebody(l, x, st), st
(l::ETOneBody)(x) = _apply_onebody(l, x, (; E0s = l.E0s))

_apply_onebody(l::ETOneBody, X::ET.ETGraph, st) = 
      _apply_onebody(l, X.node_data, st)

_apply_onebody(l::ETOneBody, X::AbstractVector, st) = 
         ___apply_onebody(l.selector, X, st.E0s)

___apply_onebody(selector, X::AbstractVector, E0s) = 
         map(x -> E0s[selector(x)], X)


# ETOneBody energy only depends on atom types (categorical), not positions.
# Gradient w.r.t. positions is always zero.
# Return empty edge_data array since there are no position-dependent gradients.
function site_grads(l::ETOneBody, X::ET.ETGraph, ps, st)
   return (; edge_data = VState[])
end

site_basis(l::ETOneBody, X::ET.ETGraph, ps, st) = 
         fill(zero(eltype(st.E0s)), (ET.nnodes(X), 0))

function site_basis_jacobian(l::ETOneBody, X::ET.ETGraph, ps, st) 
   𝔹 = site_basis(l, X, ps, st) 
   ∂𝔹 = fill(VState(), (ET.maxneigs(X), ET.nnodes(X), 0))
   return 𝔹, ∂𝔹
end

