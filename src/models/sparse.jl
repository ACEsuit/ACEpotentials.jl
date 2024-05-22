using StrideArrays, Bumper
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

function whatalloc(tensor::SparseEquivTensor, Rnl, Ylm)
   TB = promote_type(eltype(Rnl), eltype(Ylm), eltype(tensor.A2Bmap))
   sz = length(tensor)
   return (TB, sz), (_AA = (TB, length(tensor.aabasis)), )
end

function evaluate(tensor::SparseEquivTensor{T}, Rnl, Ylm) where {T} 
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
   B = tensor.A2Bmap * AA

   return B, (_AA = _AA, )
end

# BB::tuple of matrices 
function Polynomials4ML.evaluate!(A, 
                  basis::Polynomials4ML.PooledSparseProduct{2}, 
                  BB::Polynomials4ML.TupMat, 
                  nX = size(BB[1], 1))
   @assert length(BB) == 2 
   @assert size(BB[2], 1) >= nX
   TA = eltype(A) 
   spec = basis.spec
   fill!(A, zero(TA))
   @inbounds for iA = 1:length(spec) 
      ϕ = spec[iA]; ϕ1 = ϕ[1]; ϕ2 = ϕ[2]
      a = zero(TA)
      @simd ivdep for j = 1:nX
         a = muladd(BB[1][j, ϕ1], BB[2][j, ϕ2], a) 
      end
      A[iA] = a
   end
   return nothing
end

@noinline function evaluate!(B, tensor::SparseEquivTensor{T}, Rnl, Ylm, intm) where {T} 

   @no_escape begin 
      # evaluate the A basis
      TA = promote_type(T, eltype(Rnl), eltype(eltype(Ylm)))
      A = @alloc(TA, length(tensor.abasis))
      Polynomials4ML.evaluate!(A, tensor.abasis, (Rnl, Ylm))

      # evaluate the AA basis
      _AA = intm._AA
      Polynomials4ML.evaluate!(_AA, tensor.aabasis, A)

      # project to the actual AA basis 
      proj = tensor.aabasis.projection
      AA = @alloc(TA, length(proj))
      for (i, ip) in enumerate(proj)
         AA[i] = _AA[ip]
      end

      # evaluate the coupling coefficients
      mul!(B, tensor.A2Bmap, AA)
      nothing  # this seems needed so that Bumper doesn't get confused 
               # it otherwise thinks that B::PtrArray gets returned ?wtf?
   end  # @no_escape

   return nothing 
end


function pullback_evaluate(∂B, tensor::SparseEquivTensor{T}, Rnl, Ylm, 
                           intermediates) where {T} 
   _AA = intermediates._AA
   proj = tensor.aabasis.projection
                           
   # ∂Ei / ∂AA = ∂Ei / ∂B * ∂B / ∂AA = (WB[i_z0]) * A2Bmap
   ∂AA = tensor.A2Bmap' * ∂B   # TODO: make this in-place 
   _∂AA = zeros(T, length(_AA)) 
   _∂AA[proj] = ∂AA

   # ∂Ei / ∂A = ∂Ei / ∂AA * ∂AA / ∂A = pullback(aabasis, ∂AA)
   TA = promote_type(T, eltype(_AA), eltype(∂B), 
                        eltype(Rnl), eltype(eltype(Ylm)))
   ∂A = zeros(TA, length(tensor.abasis))
   Polynomials4ML.pullback_arg!(∂A, _∂AA, tensor.aabasis, _AA)
   
   # ∂Ei / ∂Rnl, ∂Ei / ∂Ylm = pullback(abasis, ∂A)
   ∂Rnl = zeros(TA, size(Rnl))
   ∂Ylm = zeros(TA, size(Ylm))
   Polynomials4ML._pullback_evaluate!((∂Rnl, ∂Ylm), ∂A, tensor.abasis, (Rnl, Ylm))

   return ∂Rnl, ∂Ylm
end

