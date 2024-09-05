using LinearAlgebra: Diagonal
import ACEpotentials.Models: ACEPotential

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

(l::TotalDegree)(b::NamedTuple) = b.n / l.wn + b.l/l.wl
(l::TotalDegree)(bb::AbstractVector{<: NamedTuple}) = sum(l(b) for b in bb)


struct EuclideanDegree <: AbstractLevel
   wn::Float64
   wl::Float64
end

EuclideanDegree() = EuclideanDegree(1.0, 2/3)

(l::EuclideanDegree)(b::NamedTuple) = sqrt( (b.n/l.wn)^2 + (b.l/l.wl)^2 )
(l::EuclideanDegree)(bb::AbstractVector{<: NamedTuple}) = sqrt( sum(l(b)^2 for b in bb) )


function oneparticle_spec(level::Union{TotalDegree, EuclideanDegree}, maxlevel)
   maxn1 = ceil(Int, maxlevel * level.wn)
   maxl1 = ceil(Int, maxlevel * level.wl)
   spec = [ (n = n, l = l) for n = 1:maxn1, l = 0:maxl1
                  if level((n = n, l = l)) <= maxlevel ]
   return sort(spec; by = x -> (x.l, x.n))
end

# --------------------------------------------------

   
# this should maybe be moved elsewhere, but for now it can live here. 

function _basis_length(model) 
   len_tensor = length(get_nnll_spec(model.tensor))
   len_pair = length(model.pairbasis.spec) 
   return (len_tensor + len_pair) * _get_nz(model)
end

function _nnll_basis(model)
   NTNL = typeof((n = 1, l = 0)) 
   TBB = Vector{NTNL}
   
   global_spec = Vector{TBB}(undef, _basis_length(model))

   nnll_tensor = get_nnll_spec(model.tensor)
   nn_pair = [ [b,] for b in model.pairbasis.spec]
   
   for iz = 1:_get_nz(model) 
      z = _i2z(model, iz)
      global_spec[get_basis_inds(model, z)] = nnll_tensor
      global_spec[get_pairbasis_inds(model, z)] = nn_pair
   end

   return global_spec
end

function _coupling_scalings(model) 
   scal = ones(_basis_length(model))
   for iz = 1:_get_nz(model)
      z = _i2z(model, iz)
      mb_inds = get_basis_inds(model, z)
      @assert length(mb_inds) == size(model.tensor.A2Bmap, 1) 
      for i = 1:length(mb_inds)
         scal[mb_inds[i]] = sqrt(sum(abs2, model.tensor.A2Bmap[i,:]))
      end
   end
   return scal 
end

smoothness_prior(model::ACEPotential, f; kwargs...) = 
      smoothness_prior(model.model, f; kwargs...)

function smoothness_prior(model, f)
   nnll = _nnll_basis(model)
   γ = zeros(length(nnll))
   for (i, bb) in enumerate(nnll)
      γ[i] = f(bb)
   end
   return Diagonal(γ) #  .* _coupling_scalings(model))
end

algebraic_smoothness_prior(model; p = 4, wl = 2/3, wn = 1.0) = 
      smoothness_prior(model, bb -> sum((b.l/wl)^p + (b.n/wn)^p for b in bb))

exp_smoothness_prior(model; wn = 1.0, wl = 2/3) = 
      smoothness_prior(model, bb -> exp( sum(b.l / wl + b.n / wn for b in bb) ))

gaussian_smoothness_prior(model; wl = 1/sqrt(2), wn = 1/sqrt(2)) = 
      smoothness_prior(model, bb -> exp( sum( (b.l/wl)^2 + (b.n/wn)^2 for b in bb) ))


algebraic_smoothness_prior_ace1(model::ACEPotential; kwargs...) = 
         algebraic_smoothness_prior_ace1(model.model, kwargs...)

function algebraic_smoothness_prior_ace1(model; p = 4, wL = 3/2) 
   nnll = _nnll_basis(model)
   γ = zeros(length(nnll))
   for (i, bb) in enumerate(nnll)
      γ[i] = sum(b.n^p + wL * b.l^p * (1 + b.l/(p+1)) for b in bb)
   end
   scal = _coupling_scalings(model)
   return Diagonal(γ .* scal .+ 1)
end 