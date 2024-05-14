


# ------------------------------------------------------------ 
#      CONSTRUCTORS AND UTILITIES 
# ------------------------------------------------------------ 

function LearnableRnlrzzBasis(
            zlist, polys, transforms, envelopes, rin0cuts, 
            spec::AbstractVector{NT_NL_SPEC}; 
            weights=nothing, 
            meta=Dict{String, Any}())
   NZ = length(zlist) 
   if isnothing(weights) 
      weights = fill(nothing, (1,1,1,1)) 
   end 
   LearnableRnlrzzBasis(_convert_zlist(zlist), 
                        polys, 
                        _make_smatrix(transforms, NZ), 
                        _make_smatrix(envelopes, NZ), 
                        # --------------
                        weights, 
                        _make_smatrix(rin0cuts, NZ),
                        collect(spec), 
                        meta)
end

Base.length(basis::LearnableRnlrzzBasis) = length(basis.spec)

function initialparameters(rng::AbstractRNG, 
                           basis::LearnableRnlrzzBasis)
   NZ = _get_nz(basis) 
   len_nl = length(basis)
   len_q = length(basis.polys)

   Wnlq = zeros(len_nl, len_q, NZ, NZ)
   for i = 1:NZ, j = 1:NZ
      Wnlq[:, :, i, j] .= glorot_normal(rng, Float64, len_nl, len_q)
   end

   return (Wnlq = Wnlq, )
end

function initialstates(rng::AbstractRNG, 
                       basis::LearnableRnlrzzBasis)
   return NamedTuple()                       
end
      

function parameterlength(basis::LearnableRnlrzzBasis)
   NZ = _get_nz(basis) 
   len_nl = length(basis)
   len_q = length(basis.polys)
   return len_nl * len_q * NZ * NZ
end


# ------------------------------------------------------------ 
#      EVALUATION INTERFACE
# ------------------------------------------------------------ 

import Polynomials4ML

(l::LearnableRnlrzzBasis)(args...) = evaluate(l, args...)

function evaluate!(Rnl, basis::LearnableRnlrzzBasis, r::Real, Zi, Zj, ps, st)
   iz = _z2i(basis, Zi)
   jz = _z2i(basis, Zj)
   Wij = @view ps.Wnlq[:, :, iz, jz]
   trans_ij = basis.transforms[iz, jz]
   x = trans_ij(r)
   P = Polynomials4ML.evaluate(basis.polys, x)
   env_ij = basis.envelopes[iz, jz]
   e = evaluate(env_ij, x)   
   Rnl[:] .= Wij * (P .* e)
   return Rnl, st 
end

function evaluate(basis::LearnableRnlrzzBasis, r::Real, Zi, Zj, ps, st)
   iz = _z2i(basis, Zi)
   jz = _z2i(basis, Zj)
   Wij = @view ps.Wnlq[:, :, iz, jz]
   trans_ij = basis.transforms[iz, jz]
   x = trans_ij(r)
   P = Polynomials4ML.evaluate(basis.polys, x)
   env_ij = basis.envelopes[iz, jz]
   e = evaluate(env_ij, x)   
   return Wij * (P .* e), st 
end


function evaluate_batched(basis::LearnableRnlrzzBasis, 
                           rs, zi, zjs, ps, st)

   @assert length(rs) == length(zjs)                          
   # evaluate the first one to get the types and size
   Rnl_1, st = evaluate(basis, rs[1], zi, zjs[1], ps, st)
   # ... and then allocate storage
   Rnl = zeros(eltype(Rnl_1), (length(rs), length(Rnl_1)))

   # then evaluate the rest in-place 
   for j = 1:length(rs)
      iz = _z2i(basis, zi)
      jz = _z2i(basis, zjs[j])
      trans_ij = basis.transforms[iz, jz]
      x = trans_ij(rs[j])
      env_ij = basis.envelopes[iz, jz]
      e = evaluate(env_ij, x)   
      P = Polynomials4ML.evaluate(basis.polys, x) .* e
      Rnl[j, :] = (@view ps.Wnlq[:, :, iz, jz]) * P
   end

   return Rnl, st
end




# ----- gradients 
# because the typical scenario is that we have few r, then moderately 
# many q and then many (n, l), this seems to be best done in Forward-mode. 
# in initial tests it seems the performance is very near optimal 
# so there is little sense trying to do something manual. 

import ForwardDiff
using ForwardDiff: Dual

function evaluate_ed(basis::LearnableRnlrzzBasis, r::T, Zi, Zj, ps, st) where {T <: Real}
   d_r = Dual{T}(r, one(T))
   d_Rnl, st = evaluate(basis, d_r, Zi, Zj, ps, st)
   Rnl = ForwardDiff.value.(d_Rnl)
   Rnl_d = ForwardDiff.extract_derivative(T, d_Rnl) 
   return Rnl, Rnl_d, st 
end


function evaluate_ed_batched(basis::LearnableRnlrzzBasis, 
                             rs::AbstractVector{T}, Zi, Zs, ps, st
                             ) where {T <: Real}
   
   @assert length(rs) == length(Zs)            
   Rnl1, st = evaluate(basis, rs[1], Zi, Zs[1], ps, st) 
   Rnl = zeros(T, length(rs), length(Rnl1))
   Rnl_d = zeros(T, length(rs), length(Rnl1))

   for j = 1:length(rs)
      d_r = Dual{T}(rs[j], one(T))   
      d_Rnl, st = evaluate(basis, d_r, Zi, Zs[j], ps, st)  # should reuse memory here 
      map!(ForwardDiff.value, (@view Rnl[j, :]), d_Rnl)
      map!(d -> ForwardDiff.extract_derivative(T, d), (@view Rnl_d[j, :]), d_Rnl)
   end       

   return Rnl, Rnl_d, st 
end




# -------- RRULES 

import ChainRulesCore: rrule, NotImplemented, NoTangent 

# NB : iz = īz = _z2i(z0) throughout
#
# Rnl[j, nl] = Wnlq[iz, jz] * Pq * e
# ∂_Wn̄l̄q̄[īz,j̄z] { ∑_jnl Δ[j,nl] * Rnl[j, nl] } 
#    = ∑_jnl Δ[j,nl] * Pq * e * δ_q̄q * δ_l̄l * δ_n̄n * δ_{īz,iz} * δ_{j̄z,jz}
#    = ∑_{jz = j̄z} Δ[j̄z, n̄l̄] * P_q̄ * e
#
function pullback_evaluate_batched(Δ, basis::LearnableRnlrzzBasis, 
                                   rs, zi, zjs, ps, st)
   @assert length(rs) == length(zjs)                          
   # evaluate the first one to get the types and size
   Rnl_1, _ = evaluate(basis, rs[1], zi, zjs[1], ps, st)
   # ... and then allocate storage
   Rnl = zeros(eltype(Rnl_1), (length(rs), length(Rnl_1)))
   
   # output storage for the gradients
   T_∂Wnlq = promote_type(eltype(Δ), eltype(rs))
   NZ = _get_nz(basis)
   ∂Wnlq = zeros(T_∂Wnlq, size(ps.Wnlq))

   # then evaluate the rest in-place 
   for j = 1:length(rs)
      iz = _z2i(basis, zi)
      jz = _z2i(basis, zjs[j])
      trans_ij = basis.transforms[iz, jz]
      x = trans_ij(rs[j])
      env_ij = basis.envelopes[iz, jz]
      e = evaluate(env_ij, x)   
      P = Polynomials4ML.evaluate(basis.polys, x) .* e
      # TODO: the P shouuld be stored inside a closure in the 
      #       forward pass and then resused. 

      # TODO:  ... and obviously this part here needs to be moved 
      # to a SIMD loop.
      ∂Wnlq[:, :, iz, jz] .+= Δ[j, :] * P'
   end

   return (Wnql = ∂Wnlq,)
end


function rrule(::typeof(evaluate_batched), 
               basis::LearnableRnlrzzBasis, 
               rs, zi, zjs, ps, st)
   Rnl, st = evaluate_batched(basis, rs, zi, zjs, ps, st)

   return (Rnl, st), 
         Δ -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), 
              pullback_evaluate_batched(Δ, basis, rs, zi, zjs, ps, st), 
              NoTangent())
end