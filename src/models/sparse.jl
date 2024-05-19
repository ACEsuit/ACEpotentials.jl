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

