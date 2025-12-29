

using ConcreteStructs
import Polynomials4ML: evaluate, evaluate_ed 
import LuxCore: AbstractLuxContainerLayer
import ChainRulesCore: NoTangent, rrule, unthunk

"""
   struct EnvRBranchL

An auxiliary layer that is basically a branch layer needed to build  
radial bases, with additional evaluate_ed functionality, needed for 
Jacobians. 
"""
@concrete struct EnvRBranchL <: AbstractLuxContainerLayer{(:envelope, :rbasis)}
   envelope 
   rbasis 
end 

(l::EnvRBranchL)(X, ps, st) = _apply_envrbranchl(l, X, ps, st), st

evaluate(l::EnvRBranchL, X, ps, st) = l(X, ps, st)

function _apply_envrbranchl(l::EnvRBranchL, X, ps, st)
   ee, _ = l.envelope(X, ps.envelope, st.envelope)
   P, _ = l.rbasis(X, ps.rbasis, st.rbasis)
   return ee .* P
end

function evaluate_ed(l::EnvRBranchL, X, ps, st)
   (ee, d_ee), _ = evaluate_ed(l.envelope, X, ps.envelope, st.envelope)
   (P, d_P), _ = evaluate_ed(l.rbasis,X, ps.rbasis, st.rbasis)

   # product rule 
   pP = ee .* P
   ∂_pP = d_ee .* P .+ ee .* d_P

   return (pP, ∂_pP), st
end

function rrule(::typeof(_apply_envrbranchl), 
               l::EnvRBranchL, X, ps, st)

   (P, dP), st = evaluate_ed(l, X, ps, st)

   function _pb_embeddp(_∂P)
      ∂P = unthunk(_∂P)
      ∂X = dropdims( sum(∂P .* dP, dims = 2), dims = 2) 
      return NoTangent(), NoTangent(), ∂X, NoTangent(), NoTangent()
   end

   return P, _pb_embeddp
end

