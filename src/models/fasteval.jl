
using AtomsCalculators, WithAlloc, SpheriCart

import AtomsCalculators: energy_unit, length_unit

import AtomsCalculatorsUtilities.SitePotentials: SitePotential, 
                            cutoff_radius, 
                            eval_site, 
                            eval_grad_site

import AtomsCalculatorsUtilities.PairPotentials: PairPotential, eval_pair                            

struct FastACEinner{TRAD, TY, TA, TAA}
   rbasis::TRAD
   ybasis::TY
   abasis::TA 
   aadot::TAA
end

struct FastACEpair{NZ, TRANS, ENV, SPL} <: PairPotential
   _i2z::NTuple{NZ, Int}
   transforms::SMatrix{NZ, NZ, TRANS}
   envelopes::SMatrix{NZ, NZ, ENV}
   spl::SMatrix{NZ, NZ, SPL}
end

struct FastACE{NZ, TINNER, TPAIR, TREF, T} <: SitePotential
   _i2z::NTuple{NZ, Int}
   inner::NTuple{NZ, TINNER}
   pair::TPAIR
   Vref::TREF 
   rcut::T
end


function FastACEInner(model::ACEPotential{<: ACEModel}, iz; 
                      aa_static = false)
   _get_L(ybasis::SphericalHarmonics{L}) where {L} = L 
   _get_L(ybasis::Polynomials4ML.RealSCWrapper) = _get_L(ybasis.scbasis)
   
   rbasis = model.model.rbasis
   ybasis = model.model.ybasis
   abasis = model.model.tensor.abasis
   aabasis = model.model.tensor.aabasis
   A2Bmap = model.model.tensor.A2Bmap
   
   wB = model.ps.WB
   wAA = A2Bmap' * wB[:, iz]
   
   # construct reduced A basis and AA basis 
   Inz = findall(!iszero, wAA)
   aa_spec_new = aabasis.meta["AA_spec"][Inz]
   a_spec_new = unique(reduce(vcat, aa_spec_new))
   
   # take human-readable specs and convert into layer-readable specs 
   # TODO: here we could try to make the y and r bases smaller 
   # maxl = maximum([ b.l for b in a_spec_new ])   
   maxl = _get_L(ybasis)
   r_spec = rbasis.spec
   y_spec = _make_Y_spec(maxl)
   A_spec_idx = _make_idx_A_spec(a_spec_new, r_spec, y_spec)
   a_basis = Polynomials4ML.PooledSparseProduct(A_spec_idx)
   AA_spec_idx = _make_idx_AA_spec(aa_spec_new, a_spec_new) 
   aa_basis = Polynomials4ML.SparseSymmProdDAG(AA_spec_idx)
   
   if aa_static 
      # generate a static evaluator
      aadot = generate_AA_dot(AA_spec_idx, wAA[Inz])
   else 
      # generate a standard evaluator
      wAA_rec = zeros(length(aa_basis))
      wAA_rec[aa_basis.projection] = wAA[Inz]
      aadot = AADot(wAA_rec, aa_basis)
   end 
   
   return FastACEinner(rbasis, ybasis, a_basis, aadot)   
end   

function fast_evaluator(model::ACEPotential{<: ACEModel}; 
                        aa_static = :auto)
   if aa_static == :auto 
      aa_static = (length(model.model.tensor.aabasis) < 1200)
   end

   _i2z = model.model._i2z 
   inner = ntuple( iz -> FastACEInner(model, iz; aa_static = aa_static),  
                   length(_i2z) )

   pairbasis = model.model.pairbasis                   
   spl = _make_pair_splines(pairbasis, model.ps.Wpair)
   pair = FastACEpair(_i2z, pairbasis.transforms, pairbasis.envelopes, spl)

   rcut = ustrip(u"Å", cutoff_radius(model))

   return FastACE(_i2z, inner, pair, model.model.Vref, rcut)
end


function _make_pair_splines(basis, W)

   function _make_pair_spline(splb, w)
      nodes = splb.itp.ranges[1]
      valb = [splb(n) for n in nodes]
      val = [ dot(v, w) for v in valb ]
      return cubic_spline_interpolation(nodes, val)
   end

   NZ = length(basis._i2z)
   spl = SMatrix{NZ, NZ}(
         [ _make_pair_spline(basis.splines[iz0, iz], W[:, iz0])
           for iz0 = 1:NZ, iz = 1:NZ]  )
   return spl   
end


# ----------------------------------------- 
#  AtomsCalculators Interface 

energy_unit(pot::FastACE) = u"eV" 
length_unit(pot::FastACE) = u"Å"
cutoff_radius(pot::FastACE) = pot.rcut * length_unit(pot)

function eval_site(pot::FastACE, Rs, Zs, z0) 
   iz0 = findfirst(isequal(z0), pot._i2z)
   iz0 == nothing && error("z0 = $z0 not found in the model")
   rs = norm.(Rs)
   return eval_site(pot.inner[iz0], Rs, Zs, z0) + 
          eval_site(pot.pair, rs, Zs, z0) + 
          eval_site(pot.Vref, Rs, Zs, z0)
end

function eval_pair(pot::FastACEpair, r, z1, z0) 
   iz = findfirst(isequal(z0), pot._i2z)
   jz = findfirst(isequal(z1), pot._i2z)
   T_ij = pot.transforms[iz, jz]
   env_ij = pot.envelopes[iz, jz]
   spl_ij = pot.spl[iz, jz]
   x_ij = T_ij(r)
   e_ij = evaluate(env_ij, r, x_ij)
   # The 2 is to take into account that AtomsCalculatorsUtilities 
   # divides again by 2. 
   return 2 * spl_ij(x_ij) * e_ij
end

function eval_site(pot::FastACEinner, Rs, Zs, z0) 
   @no_escape begin
   rs = @withalloc radii!(Rs)
   Rnl = @withalloc evaluate_batched!(pot.rbasis, rs, z0, Zs, 
                                         NamedTuple(), NamedTuple())
   Ylm = @withalloc P4ML.evaluate!(pot.ybasis, Rs)
   A = @withalloc P4ML.evaluate!(pot.abasis, (Rnl, Ylm))
   out = pot.aadot(A)
   end 
   return out 
end


# ----------------------------------------- 
#  standard evaluator for AA ⋅ θ

using LinearAlgebra: dot 

"""
Implementation of AA ⋅ θ; for easier use within the FastACE.
"""
struct AADot{T, TAA}
   cc::Vector{T} 
   aabasis::TAA
end

function (aadot::AADot)(A)
   @no_escape begin 
      AA = @alloc(eltype(A), length(aadot.aabasis))
      P4ML.evaluate!(AA, aadot.aabasis, A)
      out = dot(aadot.cc, AA)
   end
   return out 
end

function eval_and_grad!(∇φ_A, aadot::AADot, A)
   φ = aadot(A)
   P4ML.pullback!(∇φ_A, aadot.cc, aadot.aabasis, A)
   return φ
end


# ------------------------------------------ 
# an experimental fast evaluator for AA ⋅ θ

using StaticArrays
using StaticPolynomials: Polynomial, evaluate_and_gradient
using DynamicPolynomials: @polyvar

"""
This naive code is not supposed to be fast, it is only used to generate a 
dynamic polynomial representating the operation AA ⋅ c -> εᵢ 

The generated (giant) polynomial is then used to generate optimized 
evaluation and gradient code. 
"""
function _AA_dot(A, spec, c)
   T = promote_type(eltype(A), eltype(c))
   out = zero(T)
   for (i, kk) in enumerate(spec)
      out += c[i] * prod(A[kk[t]] for t = 1:length(spec[i]))
   end
   return out 
end


function generate_AA_dot(spec, c)
   nA = maximum(maximum, spec)
   @polyvar A[1:nA]
   dynamic_poly = _AA_dot(A, spec, c)
   return Polynomial(dynamic_poly)
end

function eval_and_grad!(∇φ_A, aadot, A)
   # evaluate_and_gradient!(∇_A, aadot, A)
   φ, ∇φ_A_1 = evaluate_and_gradient(aadot, A)
   for n = 1:length(A)
      ∇φ_A[n] = ∇φ_A_1[n]
   end
   return φ 
end
