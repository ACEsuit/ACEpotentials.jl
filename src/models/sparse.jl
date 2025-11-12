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
   P4ML.evaluate!(A, tensor.abasis, (Rnl, Ylm))

   # evaluate the AA basis
   # _AA = zeros(TA, length(tensor.aabasis))     # use Bumper here
   P4ML.evaluate!(_AA, tensor.aabasis, A)

   # evaluate the coupling coefficients
   # B = tensor.A2Bmap * _AA
   # Note: SparseSymmProd no longer needs projection; it evaluates directly to correct size
   mul!(B, tensor.A2Bmap, _AA)

   return B, (_AA = _AA, _A = A)
end

function whatalloc(::typeof(evaluate!), tensor::SparseEquivTensor, Rnl, Ylm)
   TA = promote_type(eltype(Rnl), eltype(eltype(Ylm)))
   TB = promote_type(TA, eltype(tensor.A2Bmap))
   return (TB, length(tensor),), (TA, length(tensor.aabasis),)
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
   _A = intermediates._A
   T_∂AA = promote_type(eltype(∂B), eltype(tensor.A2Bmap))
   T_∂A = promote_type(T_∂AA, eltype(_AA))

   @no_escape begin
   #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   # ∂Ei / ∂AA = ∂Ei / ∂B * ∂B / ∂AA = (WB[i_z0]) * A2Bmap
   # Note: SparseSymmProd no longer needs projection; gradient maps directly
   _∂AA = @alloc(T_∂AA, size(tensor.A2Bmap, 2))
   mul!(_∂AA, tensor.A2Bmap', ∂B)

   # ∂Ei / ∂A = ∂Ei / ∂AA * ∂AA / ∂A = pullback(aabasis, ∂AA)
   # Note: pullback! takes A (input) not AA (output) in new API
   ∂A = @alloc(T_∂A, length(tensor.abasis))
   P4ML.pullback!(∂A, _∂AA, tensor.aabasis, _A)
   
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
   spec = tensor.meta["AA_spec"]
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



# ----------------------------------------
#  experimental pushforwards 

function _pfwd(tensor::SparseEquivTensor{T}, Rnl, Ylm, ∂Rnl, ∂Ylm) where {T}
   A, ∂A = _pfwd(tensor.abasis, (Rnl, Ylm), (∂Rnl, ∂Ylm))
   _AA, _∂AA = _pfwd(tensor.aabasis, A, ∂A)

   # evaluate the coupling coefficients
   # Note: SparseSymmProd no longer needs projection; use _AA and _∂AA directly
   B = tensor.A2Bmap * _AA
   ∂B = tensor.A2Bmap * _∂AA
   return B, ∂B 
end


function _pfwd(abasis::EquivariantTensors.PooledSparseProduct{2}, RY, ∂RY) 
   R, Y = RY 
   TA = typeof(R[1] * Y[1])
   ∂R, ∂Y = ∂RY
   ∂TA = typeof(R[1] * ∂Y[1] + ∂R[1] * Y[1])

   # check lengths 
   nX = size(R, 1)
   @assert nX == size(R, 1) == size(∂R, 1) == size(Y, 1) == size(∂Y, 1)

   A = zeros(TA, length(abasis.spec))
   ∂A = zeros(∂TA, size(∂R, 1), length(abasis.spec))

   for i = 1:length(abasis.spec)
      @inbounds begin 
         n1, n2 = abasis.spec[i]
         ai = zero(TA)
         @simd ivdep for α = 1:nX 
            ai += R[α, n1] * Y[α, n2]
            ∂A[α, i] = R[α, n1] * ∂Y[α, n2] + ∂R[α, n1] * Y[α, n2]
         end 
         A[i] = ai
      end 
   end 
   return A, ∂A
end 


function _pfwd(aabasis::EquivariantTensors.SparseSymmProd, A, ∂A)
   n∂ = size(∂A, 1)
   num1 = aabasis.num1 
   nodes = aabasis.nodes 
   AA = zeros(eltype(A), length(nodes))
   T∂AA = typeof(A[1] * ∂A[1])
   ∂AA = zeros(T∂AA, length(nodes), size(∂A, 1))
   for i = 1:num1 
      AA[i] = A[i] 
      for α = 1:n∂
         ∂AA[i, α] = ∂A[α, i]
      end
   end 
   for iAA = num1+1:length(nodes)
      n1, n2 = nodes[iAA]
      AA_n1 = AA[n1]
      AA_n2 = AA[n2]
      AA[iAA] = AA_n1 * AA_n2
      for α = 1:n∂
         ∂AA[iAA, α] = AA_n2 * ∂AA[n1, α] + AA_n1 * ∂AA[n2, α]
      end
   end
   return AA, ∂AA
end


