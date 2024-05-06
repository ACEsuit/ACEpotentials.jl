# --------------------------------------------------
#   different notions of "level" / total degree.

abstract type AbstractLevel end 
struct TotalDegree <: AbstractLevel
   wn::Float64
   wl::Float64
end 

TotalDegree() = TotalDegree(1.0, 0.66)

(l::TotalDegree)(b::NamedTuple) = b.n/l.wn + b.l/l.wl
(l::TotalDegree)(bb::AbstractVector{<: NamedTuple}) = sum(l(b) for b in bb)


struct EuclideanDegree <: AbstractLevel
   wn::Float64
   wl::Float64
end

EuclideanDegree() = EuclideanDegree(1.0, 0.66)

(l::EuclideanDegree)(b::NamedTuple) = sqrt( (b.n/l.wn)^2 + (b.l/l.wl)^2 )
(l::EuclideanDegree)(bb::AbstractVector{<: NamedTuple}) = sqrt( sum(l(b)^2 for b in bb) )


# -------------------------------------------------------
#  construction of Rnlrzz bases with lots of defaults
#
# TODO: offer simple option to initialize ACE1-like or trACE-like
#

function ace_learnable_Rnlrzz(; 
               max_level = nothing, 
               level = nothing, 
               maxl = nothing, 
               maxn = nothing,
               elements = nothing, 
               spec = nothing, 
               rin0cuts = _default_rin0cuts(elements),
               transforms = agnesi_transform.(rin0cuts, 2, 2), 
               polys = :legendre, 
               envelopes = PolyEnvelope2sX(-1.0, 1.0, 2, 2)
               )
   if elements == nothing
      error("elements must be specified!")
   end
   if (spec == nothing) && (level == nothing || max_level == nothing)
      error("Must specify either `spec` or `level, max_level`!")
   end

   zlist =_convert_zlist(elements)

   if spec == nothing
      spec = [ (n = n, l = l) for n = 1:maxn, l = 0:maxl
                              if level((n = n, l = l)) <= max_level ]
   end

   # now the actual maxn is the maximum n in the spec
   actual_maxn = maximum([ s.n for s in spec ])

   if polys isa Symbol 
      if polys == :legendre
         polys = Polynomials4ML.legendre_basis(actual_maxn) 
      else
         error("unknown polynomial type : $polys")
      end
   end

   if actual_maxn > length(polys)
      error("actual_maxn > length of polynomial basis")
   end

   return LearnableRnlrzzBasis(zlist, polys, transforms, envelopes, rin0cuts, spec)
end 



function ace_model(; elements = nothing, 
                     order = nothing, 
                     Ytype = :solid,  
                     # radial basis 
                     rbasis = nothing, 
                     rbasis_type = :learnable, 
                     maxl = 30, # maxl, max are fairly high defaults 
                     maxn = 50, # that we will likely never reach 
                     # basis size parameters 
                     level = nothing, 
                     max_level = nothing, 
                     
                     )
   # construct an rbasis if needed
   if isnothing(rbasis)
      if rbasis_type == :learnable
         rbasis = ace_learnable_Rnlrzz(; max_level = max_level, level = level, 
                                         maxl = maxl, maxn = maxn, 
                                         elements = elements)
      else
         error("unknown rbasis_type = $rbasis_type")
      end
   end

   AA_spec = sparse_AA_spec(; order = order, r_spec = rbasis.spec, 
                              level = level, max_level = max_level)

   return ace_model(rbasis, Ytype, AA_spec, level)                              
end


# -------------------------------------------------------
