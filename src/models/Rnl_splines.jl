

# ------------------------------------------------------------ 
#      CONSTRUCTORS AND UTILITIES 
# ------------------------------------------------------------ 


Base.length(basis::SplineRnlrzzBasis) = length(basis.spec)

function initialparameters(rng::AbstractRNG, 
                           basis::SplineRnlrzzBasis)
   return NamedTuple()
end                           

function initialstates(rng::AbstractRNG, 
                       basis::SplineRnlrzzBasis)
   return NamedTuple()                       
end
                  

# ------------------------------------------------------------ 
#      EVALUATION INTERFACE
# ------------------------------------------------------------ 

(l::SplineRnlrzzBasis)(args...) = evaluate(l, args...)


# function evaluate!(Rnl, basis::SplineRnlrzzBasis, r::Real, Zi, Zj, ps, st)
#    Rnl[:] .= evaluate(basis, r, Zi, Zj, ps, st)
#    return Rnl, st 
# end


function evaluate(basis::SplineRnlrzzBasis, r::Real, Zi, Zj, ps, st)
   iz = _z2i(basis, Zi)
   jz = _z2i(basis, Zj)
   T_ij = basis.transforms[iz, jz]
   env_ij = basis.envelopes[iz, jz]
   spl_ij = basis.splines[iz, jz]

   x_ij = T_ij(r)
   e_ij = evaluate(env_ij, r, x_ij)

   return spl_ij(x_ij) * e_ij
end


function evaluate_batched!(Rnl, basis::SplineRnlrzzBasis, 
                           rs, zi, zjs, ps, st)

   @assert length(rs) == length(zjs)                          
   # evaluate the first one to get the types and size
   Rnl_1 = evaluate(basis, rs[1], zi, zjs[1], ps, st)
   # ... and then allocate storage
   # Rnl = zeros(eltype(Rnl_1), (length(rs), length(Rnl_1)))
   Rnl[1, :] .= Rnl_1

   # then evaluate the rest in-place 
   for j = 2:length(rs)
      Rnl[j, :] = evaluate(basis, rs[j], zi, zjs[j], ps, st)
   end

   return Rnl
end

function whatalloc(::typeof(evaluate_batched!), 
                   basis::SplineRnlrzzBasis, 
                   rs, zi, zjs, ps, st)
   T = eltype(rs)
   return (T, length(rs), length(basis))
end                   


function evaluate_batched(basis::SplineRnlrzzBasis, 
                           rs, zi, zjs, ps, st)
   Rnl = zeros(whatalloc(evaluate_batched!, basis, rs, zi, zjs, ps, st)...)
   return evaluate_batched!(Rnl, basis, rs, zi, zjs, ps, st)
end

# ----- gradients 
# because the typical scenario is that we have few r, then moderately 
# many q and then many (n, l), this seems to be best done in Forward-mode. 


import ForwardDiff
using ForwardDiff: Dual

function evaluate_ed(basis::SplineRnlrzzBasis, r::T, Zi, Zj, ps, st) where {T <: Real}
   d_r = Dual{T}(r, one(T))
   d_Rnl = evaluate(basis, d_r, Zi, Zj, ps, st)
   Rnl = ForwardDiff.value.(d_Rnl)
   Rnl_d = ForwardDiff.extract_derivative(T, d_Rnl) 
   return Rnl, Rnl_d 
end



function evaluate_ed_batched!(Rnl, Rnl_d, 
                             basis::SplineRnlrzzBasis, 
                             rs::AbstractVector{T}, Zi, Zs, ps, st
                             ) where {T <: Real}
   
   @assert length(rs) == length(Zs)            
   Rnl1, ∇Rnl1 = evaluate_ed(basis, rs[1], Zi, Zs[1], ps, st)

   for j = 1:length(rs)
      Rnl_j, ∇Rnl_j = evaluate_ed(basis, rs[j], Zi, Zs[j], ps, st)
      Rnl[j, :] = Rnl_j
      Rnl_d[j, :] = ∇Rnl_j
   end       

   return Rnl, Rnl_d 
end


function whatalloc(::typeof(evaluate_ed_batched!), 
                  basis::SplineRnlrzzBasis, 
                  rs::AbstractVector, Zi, Zs, ps, st)
   T = eltype(rs)
   return (T, length(rs), length(basis)), (T, length(rs), length(basis))
end


function evaluate_ed_batched(basis::SplineRnlrzzBasis, 
                             rs::AbstractVector, Zi, Zs, ps, st)
   alc_Rnl, alc_Rnl_d = whatalloc(evaluate_ed_batched!, basis, rs, Zi, Zs, ps, st)
   Rnl = zeros(alc_Rnl...)
   Rnl_d = zeros(alc_Rnl_d...)
   return evaluate_ed_batched!(Rnl, Rnl_d, basis, rs, Zi, Zs, ps, st)
end


function rrule(::typeof(evaluate_batched), 
               basis::SplineRnlrzzBasis, 
               rs, zi, zjs, ps, st)
   Rnl, st = evaluate_batched(basis, rs, zi, zjs, ps, st)

   return (Rnl, st), 
         Δ -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), 
              NamedTuple(), NoTangent())
end
