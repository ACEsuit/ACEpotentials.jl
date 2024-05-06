
using LuxCore: AbstractExplicitLayer, 
               AbstractExplicitContainerLayer,
               initialparameters, 
               initialstates

using SparseArrays: SparseMatrixCSC     

using SpheriCart: SolidHarmonics, SphericalHarmonics

# ------------------------------------------------------------
#    ACE MODEL SPECIFICATION 


struct ACEModel{NZ, TRAD, TY, TA, TAA, T} <: AbstractExplicitContainerLayer
   _i2z::NTuple{NZ, Int}
   rbasis::TRAD
   ybasis::TY
   abasis::TA
   aabasis::TAA
   A2Bmap::SparseMatrixCSC{T, Int}
   # -------------- 
   # we can add a FS embedding here 
   # --------------
   bparams::NTuple{NZ, Vector{T}}
   aaparams::NTuple{NZ, Vector{T}}
   # --------------
   meta::Dict{String, Any}
end

# ------------------------------------------------------------
#    CONSTRUCTORS AND UTILITIES

const NT_NLM = NamedTuple{(:n, :l, :m), Tuple{Int, Int, Int}}

function _A_from_AA_spec(AA_spec) 
   A_spec = NT_NLM[] 
   for bb in AA_spec 
      append!(A_spec, bb)
   end
   return unique(sort(A_spec))
end

function make_Y_basis(Ytype, lmax) 
   if Ytype == :solid 
      return SolidHarmonics(lmax)
   elseif Ytype == :spherical
      return SphericalHarmonics(lmax)
   end 

   error("unknown `Ytype` = $Ytype - I don't know how to generate a spherical basis from this.")
end

function sort_and_filter_AA_spec(AA_spec) 
   
   return unique(sort(AA_spec))
end


function generate_A2B_map(AA_spec)
   function _rpi_A2B_matrix(cgen::Union{Rot3DCoeffs{L,T}, Rot3DCoeffs_real{L,T}, Rot3DCoeffs_long{L,T}}, spec) where {L,T}

end

function ace_model(rbasis, Ytype, AA_spec) 

   A_spec = _A_from_AA_spec(AA_spec)
   lmax = maximum([b.l for b in a_spec])
   ybasis = make_Y_basis(Ytype, lmax)

end 

