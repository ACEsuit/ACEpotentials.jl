

import EquivariantTensors as ET 
import Polynomials4ML as P4ML

import DecoratedParticles: XState

import LuxCore: AbstractLuxLayer 
using ConcreteStructs: @concrete


@concrete struct TransSelSplineBasis  <: AbstractLuxLayer
   trans      # transform 
   envelope   # envelope
   selector   # selector 
   ref_spl    # reference spline basis (ignore the stored parameters)
   states     # reference spline parameters (frozen hence states)
end 


(l::TransSelSplineBasis)(x, ps, st) = _apply_etsplinebasis(l, x, ps, st), st 
      
      
function _apply_etsplinebasis(l::TransSelSplineBasis, 
                              X::AbstractVector{<: XState}, 
                              ps, st)
   # transform 
   Y = l.trans(X) 
   # select the spline parameters 
   i_sel = map(l.selector, X)
   # allocate 
   S = similar(Y, eltype(Y), (length(X), length(l.ref_spl)))

   for (idx, y) in enumerate(Y)
      spl_idx = st.states[i_sel[idx]]
      S[idx, :] = P4ML.evaluate(l.ref_spl, y, spl_idx)
   end

   if envelope != nothing 
      ee, _ = l.envelope(X, ps.envelope, st.envelope)
      S .= ee .* S
   end 
   
   return S
end

