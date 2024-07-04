using SparseArrays: SparseMatrixCSC     
import Polynomials4ML


struct SparseEquivTensor{T, TA, TAA}
   abasis::TA
   aabasis::TAA
   A2Bmap::SparseMatrixCSC{T, Int}
   # ------- 
   meta::Dict{String, Any}
end

Base.length(tensor::SparseEquivTensor) = size(tensor.A2Bmap, 1) 


function evaluate!(B, _AA, tensor::SparseEquivTensor{T}, Rnl, Ylm) where {T}
   # evaluate the A basis
   TA = promote_type(T, eltype(Rnl), eltype(eltype(Ylm)))
   A = zeros(TA, length(tensor.abasis))
   Polynomials4ML.evaluate!(A, tensor.abasis, (Rnl, Ylm))

   # evaluate the AA basis
   _AA = zeros(TA, length(tensor.aabasis))     # use Bumper here
   Polynomials4ML.evaluate!(_AA, tensor.aabasis, A)
   # project to the actual AA basis 
   proj = tensor.aabasis.projection
   AA = _AA[proj]     # use Bumper here, or view; needs experimentation. 

   # evaluate the coupling coefficients
   # B = tensor.A2Bmap * AA
   mul!(B, tensor.A2Bmap, AA)   

   return B, (_AA = _AA, )
end

function whatalloc(::typeof(evaluate!), tensor::SparseEquivTensor, Rnl, Ylm)
   TA = promote_type(eltype(Rnl), eltype(eltype(Ylm)))
   TB = promote_type(TA, eltype(tensor.A2Bmap))
   return (TB, size(tensor.A2Bmap, 1),), (TA, length(tensor.abasis),)
end

function evaluate(tensor::SparseEquivTensor, Rnl, Ylm)
   allocinfo = whatalloc(evaluate!, tensor, Rnl, Ylm)
   B = zeros(allocinfo[1]...)
   AA = zeros(allocinfo[2]...)
   return evaluate!(B, AA, tensor, Rnl, Ylm)
end


# ---------


function pullback!(∂Rnl, ∂Ylm, 
                   ∂B, tensor::SparseEquivTensor, Rnl, Ylm, 
                   intermediates)
   _AA = intermediates._AA
   proj = tensor.aabasis.projection
   T_∂AA = promote_type(eltype(∂B), eltype(tensor.A2Bmap))
   T_∂A = promote_type(T_∂AA, eltype(_AA))

   @no_escape begin 
   #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                           
   # ∂Ei / ∂AA = ∂Ei / ∂B * ∂B / ∂AA = (WB[i_z0]) * A2Bmap
   # ∂AA = tensor.A2Bmap' * ∂B   
   ∂AA = @alloc(T_∂AA, size(tensor.A2Bmap, 2))
   mul!(∂AA, tensor.A2Bmap', ∂B)
   _∂AA = @alloc(T_∂AA, length(_AA))
   fill!(_∂AA, zero(T_∂AA))
   _∂AA[proj] = ∂AA

   # ∂Ei / ∂A = ∂Ei / ∂AA * ∂AA / ∂A = pullback(aabasis, ∂AA)
   ∂A = @alloc(T_∂A, length(tensor.abasis))
   P4ML.unsafe_pullback!(∂A, _∂AA, tensor.aabasis, _AA)
   
   # ∂Ei / ∂Rnl, ∂Ei / ∂Ylm = pullback(abasis, ∂A)
   P4ML.pullback!((∂Rnl, ∂Ylm), ∂A, tensor.abasis, (Rnl, Ylm))

   #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   end # no_escape

   return ∂Rnl, ∂Ylm
end

function whatalloc(::typeof(pullback!),  
                   ∂B, tensor::SparseEquivTensor{T}, Rnl, Ylm, 
                   intermediates) where {T} 
   TA = promote_type(T, eltype(intermediates._AA), eltype(∂B), 
                     eltype(Rnl), eltype(eltype(Ylm)))
   return (TA, size(Rnl)...), (TA, size(Ylm)...)   
end

function pullback(∂B, tensor::SparseEquivTensor{T}, Rnl, Ylm, 
                           intermediates) where {T} 
   alc_∂Rnl, alc_∂Ylm = whatalloc(pullback!, ∂B, tensor, Rnl, Ylm, intermediates)
   ∂Rnl = zeros(alc_∂Rnl...)
   ∂Ylm = zeros(alc_∂Ylm...)
   return pullback!(∂Rnl, ∂Ylm, ∂B, tensor, Rnl, Ylm, intermediates)
end

# ----------------------------------------
#  utilities 

"""
Get the specification of the BBbasis as a list (`Vector`) of vectors of `@NamedTuple{n::Int, l::Int}`.

### Parameters 

* `tensor` : a SparseEquivTensor, possibly from ACEModel
"""
function get_nnll_spec(tensor::SparseEquivTensor{T}) where {T}
   _nl(bb) = [(n = b.n, l = b.l) for b in bb]
   # assume the new ACE model NEVER has the z channel
   spec = tensor.aabasis.meta["AA_spec"]
   nBB = size(tensor.A2Bmap, 1)
   nnll_list = Vector{NT_NL_SPEC}[]
   for i in 1:nBB
      AAidx_nnz = tensor.A2Bmap[i, :].nzind
      bbs = spec[AAidx_nnz]
      @assert all([bb == _nl(bbs[1]) for bb in _nl.(bbs)])
      push!(nnll_list, _nl(bbs[1]))
   end
   @assert length(nnll_list) == nBB
   return nnll_list
end

