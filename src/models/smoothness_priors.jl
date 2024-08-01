

# --------------------------------------------------
#   different notions of "level" / total degree.
#   selecting the basis in this way is assumed smoothness of the target 
#   and is closely related to the choice of smoothness prior. 

abstract type AbstractLevel end 
struct TotalDegree <: AbstractLevel
   wn::Float64
   wl::Float64
end 

TotalDegree() = TotalDegree(1.0, 2/3)

(l::TotalDegree)(b::NamedTuple) = b.n/l.wn + b.l/l.wl
(l::TotalDegree)(bb::AbstractVector{<: NamedTuple}) = sum(l(b) for b in bb)


struct EuclideanDegree <: AbstractLevel
   wn::Float64
   wl::Float64
end

EuclideanDegree() = EuclideanDegree(1.0, 2/3)

(l::EuclideanDegree)(b::NamedTuple) = sqrt( (b.n/l.wn)^2 + (b.l/l.wl)^2 )
(l::EuclideanDegree)(bb::AbstractVector{<: NamedTuple}) = sqrt( sum(l(b)^2 for b in bb) )


struct BasisSelector
   order::Int 
   maxlevels::AbstractVector{<: Number}
   level
end

# --------------------------------------------------
