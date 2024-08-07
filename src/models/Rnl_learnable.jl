import LuxCore


# ------------------------------------------------------------ 
#      CONSTRUCTORS AND UTILITIES 
# ------------------------------------------------------------ 

function LearnableRnlrzzBasis(
            zlist, polys, transforms, envelopes, rin0cuts, 
            spec::AbstractVector{NT_NL_SPEC}; 
            Winit = :glorot_normal, 
            meta=Dict{String, Any}())
   NZ = length(zlist) 
   meta["Winit"] = string(Winit)
   LearnableRnlrzzBasis(_convert_zlist(zlist), 
                        polys, 
                        _make_smatrix(transforms, NZ), 
                        _make_smatrix(envelopes, NZ), 
                        # --------------
                        _make_smatrix(rin0cuts, NZ),
                        collect(spec), 
                        meta)
end

Base.length(basis::LearnableRnlrzzBasis) = length(basis.spec)

function initialparameters(rng::Union{AbstractRNG, Nothing}, 
                           basis::LearnableRnlrzzBasis)
   NZ = _get_nz(basis) 
   len_nl = length(basis)
   len_q = length(basis.polys)

   Wnlq = zeros(len_nl, len_q, NZ, NZ)
   ps = (Wnlq = Wnlq, )

   if !haskey(basis.meta, "Winit") 
      @warn("No key Winit found for radial basis, use glorot_normal to initialize.")
      basis.meta["Winit"] = "glorot_normal"
   end
   
   if basis.meta["Winit"] == "glorot_normal"
      for i = 1:NZ, j = 1:NZ
         Wnlq[:, :, i, j] .= glorot_normal(rng, Float64, len_nl, len_q)
      end

   elseif basis.meta["Winit"] == "onehot" 
      set_onehot_weights!(basis, ps)

   elseif basis.meta["Winit"] == "zeros"
      @warn("Setting inner basis weights to zero.")
      Wnlq[:] .= 0 

   else 
      error("Unknown key Winit = $(basis.meta["Winit"]) to initialize radial basis weights.")
   end 

   return ps 
end

function initialstates(rng::AbstractRNG, 
                       basis::LearnableRnlrzzBasis)
   return NamedTuple()                       
end
      

function LuxCore.parameterlength(basis::LearnableRnlrzzBasis)
   NZ = _get_nz(basis) 
   len_nl = length(basis)
   len_q = length(basis.polys)
   return len_nl * len_q * NZ * NZ
end

"""
Set the radial weights as they would be in a linear ACE model. 
"""
function set_onehot_weights!(rbasis::LearnableRnlrzzBasis, ps)
   # Rnl(r, Z1, Z2) = ∑_q W[(nl), q, Z1, Z2] * P_q(r)
   # For linear models this becomes R(n'z')l(r, Z1, Z2) = Pn'(r) * δ_{z',Z2}
   # here, Z1 is the center atom, Z2 the neighbour atom. 
   NZ = _get_nz(rbasis)
   ps.Wnlq[:] .= 0 
   for iz1 = 1:NZ, iz2 = 1:NZ
      for (i_nl, nl) in enumerate(rbasis.spec)
         # n    | 1    2    3    4    5    6    7    8    ...
         # n'z' | 1,1  1,2  1,3  2,1  2,2  2,3  3,1  3,2  ...
         # => z' = mod1(n, NZ), n' = div(n-1, NZ) + 1
         z_ = mod1(nl.n, NZ)
         n_ = div(nl.n - 1, NZ) + 1
         if z_ == iz2 && n_ <= size(ps.Wnlq, 2) 
            ps.Wnlq[i_nl, n_, iz1, iz2] = 1
         end
      end
   end
   return ps 
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
   e = evaluate(env_ij, r, x)   
   Rnl[:] .= Wij * (P .* e)
   return Rnl 
end

function evaluate(basis::LearnableRnlrzzBasis, r::Real, Zi, Zj, ps, st)
   iz = _z2i(basis, Zi)
   jz = _z2i(basis, Zj)
   Wij = @view ps.Wnlq[:, :, iz, jz]
   trans_ij = basis.transforms[iz, jz]
   x = trans_ij(r)
   P = Polynomials4ML.evaluate(basis.polys, x)
   env_ij = basis.envelopes[iz, jz]
   e = evaluate(env_ij, r, x)   
   return Wij * (P .* e) 
end


function evaluate_batched!(Rnl, 
                           basis::LearnableRnlrzzBasis, 
                           rs, zi, zjs, ps, st)

   @assert length(rs) == length(zjs)    
   @assert size(Rnl, 1) >= length(rs) 
   @assert size(Rnl, 2) >= length(basis)  

   # evaluate the first one to get the types and size
   Rnl_1 = evaluate(basis, rs[1], zi, zjs[1], ps, st)

   # then evaluate the rest in-place 
   for j = 1:length(rs)
      iz = _z2i(basis, zi)
      jz = _z2i(basis, zjs[j])
      trans_ij = basis.transforms[iz, jz]
      x = trans_ij(rs[j])
      env_ij = basis.envelopes[iz, jz]
      e = evaluate(env_ij, rs[j], x)   
      P = Polynomials4ML.evaluate(basis.polys, x) .* e
      Rnl[j, :] = (@view ps.Wnlq[:, :, iz, jz]) * P
   end

   return Rnl
end

function whatalloc(::typeof(evaluate_batched!), 
                    basis::LearnableRnlrzzBasis, 
                    rs::AbstractVector{T}, zi, zjs, ps, st) where {T}
   T1 = promote_type(eltype(ps.Wnlq), T)
   return (T1, length(rs), length(basis))
end

function evaluate_batched(basis::LearnableRnlrzzBasis, 
                          rs, zi, zjs, ps, st)
   Rnl = zeros(whatalloc(evaluate_batched!, basis, rs, zi, zjs, ps, st)...)
   return evaluate_batched!(Rnl, basis, rs, zi, zjs, ps, st)
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
   d_Rnl = evaluate(basis, d_r, Zi, Zj, ps, st)
   Rnl = ForwardDiff.value.(d_Rnl)
   Rnl_d = ForwardDiff.extract_derivative(T, d_Rnl) 
   return Rnl, Rnl_d 
end


function evaluate_ed_batched!(Rnl, Rnl_d, 
                             basis::LearnableRnlrzzBasis, 
                             rs::AbstractVector{T}, Zi, Zs, ps, st
                             ) where {T <: Real}
   
   @assert length(rs) == length(Zs)            
   for j = 1:length(rs)
      d_r = Dual{T}(rs[j], one(T))   
      d_Rnl = evaluate(basis, d_r, Zi, Zs[j], ps, st)  # should reuse memory here 
      for t = 1:size(Rnl, 2) 
         Rnl[j, t] = ForwardDiff.value(d_Rnl[t])
         Rnl_d[j, t] = ForwardDiff.extract_derivative(T, d_Rnl[t])
      end
   end       

   return Rnl, Rnl_d 
end

function whatalloc(::typeof(evaluate_ed_batched!), 
                    basis::LearnableRnlrzzBasis, 
                    rs::AbstractVector{T}, Zi, Zs, ps, st) where {T}
   T1 = promote_type(eltype(ps.Wnlq), T)
   return (T1, length(rs), length(basis)), (T1, length(rs), length(basis))                                  
end

function evaluate_ed_batched(basis::LearnableRnlrzzBasis, 
                        rs::AbstractVector{T}, Zi, Zs, ps, st
                        ) where {T <: Real}
   allocinfo = whatalloc(evaluate_ed_batched!, basis, rs, Zi, Zs, ps, st)
   Rnl = zeros(allocinfo[1]...)
   Rnl_d = zeros(allocinfo[2]...)
   return evaluate_ed_batched!(Rnl, Rnl_d, basis, rs, Zi, Zs, ps, st)
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
      e = evaluate(env_ij, rs[j], x)   
      P = Polynomials4ML.evaluate(basis.polys, x) .* e
      # TODO: the P shouuld be stored inside a closure in the 
      #       forward pass and then resused. 

      # TODO:  ... and obviously this part here needs to be moved 
      # to a SIMD loop.
      ∂Wnlq[:, :, iz, jz] .+= Δ[j, :] * P'
   end

   return (Wnql = ∂Wnlq,)
end


# function rrule(::typeof(evaluate_batched), 
#                basis::LearnableRnlrzzBasis, 
#                rs, zi, zjs, ps, st)
#    Rnl, st = evaluate_batched(basis, rs, zi, zjs, ps, st)

#    return (Rnl, st), 
#          Δ -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), 
#               pullback_evaluate_batched(Δ, basis, rs, zi, zjs, ps, st), 
#               NoTangent())
# end

function rrule(::typeof(evaluate_batched), 
               basis::LearnableRnlrzzBasis, 
               rs, zi, zjs, ps, st)
   Rnl = evaluate_batched(basis, rs, zi, zjs, ps, st)

   return Rnl, Δ -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), 
                     pullback_evaluate_batched(Δ, basis, rs, zi, zjs, ps, st), 
                     NoTangent())
end
