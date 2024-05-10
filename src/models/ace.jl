
using LuxCore: AbstractExplicitLayer, 
               AbstractExplicitContainerLayer,
               initialparameters, 
               initialstates

using Lux: glorot_normal

using Random: AbstractRNG
using SparseArrays: SparseMatrixCSC     
using StaticArrays: SVector
using LinearAlgebra: norm, dot

import SpheriCart
using SpheriCart: SolidHarmonics, SphericalHarmonics
import RepLieGroups
import EquivariantModels
import Polynomials4ML

# ------------------------------------------------------------
#    ACE MODEL SPECIFICATION 


struct ACEModel{NZ, TRAD, TY, TA, TAA, T} <: AbstractExplicitContainerLayer{(:rbasis,)}
   _i2z::NTuple{NZ, Int}
   rbasis::TRAD
   ybasis::TY
   abasis::TA
   aabasis::TAA
   A2Bmap::SparseMatrixCSC{T, Int}
   # -------------- 
   # we can add a nonlinear embedding here 
   # --------------
   bparams::NTuple{NZ, Vector{T}}
   aaparams::NTuple{NZ, Vector{T}}
   # --------------
   meta::Dict{String, Any}
end

# ------------------------------------------------------------
#    CONSTRUCTORS AND UTILITIES

const NT_NLM = NamedTuple{(:n, :l, :m), Tuple{Int, Int, Int}}

function _make_Y_basis(Ytype, lmax) 
   if Ytype == :solid 
      return SolidHarmonics(lmax)
   elseif Ytype == :spherical
      return SphericalHarmonics(lmax)
   end 

   error("unknown `Ytype` = $Ytype - I don't know how to generate a spherical basis from this.")
end

# can we ignore the level function here? 
function _make_A_spec(AA_spec, level)
   NT_NLM = NamedTuple{(:n, :l, :m), Tuple{Int, Int, Int}}
   A_spec = NT_NLM[]
   for bb in AA_spec 
      append!(A_spec, bb)
   end
   A_spec_level = [ level(b) for b in A_spec ]
   p = sortperm(A_spec_level)
   A_spec = A_spec[p]
   return A_spec
end 

# this should go into sphericart or P4ML 
function _make_Y_spec(maxl::Integer)
   NT_LM = NamedTuple{(:l, :m), Tuple{Int, Int}}
   y_spec = NT_LM[] 
   for i = 1:SpheriCart.sizeY(maxl)
      l, m = SpheriCart.idx2lm(i)
      push!(y_spec, (l = l, m = m))
   end
   return y_spec 
end


function _generate_ace_model(rbasis, Ytype::Symbol, AA_spec::AbstractVector, 
                             level = TotalDegree())
   # generate the coupling coefficients 
   cgen = EquivariantModels.Rot3DCoeffs_real(0)
   AA2BB_map = EquivariantModels._rpi_A2B_matrix(cgen, AA_spec)

   # find which AA basis functions are actually used and discard the rest 
   keep_AA_idx = findall(sum(abs, AA2BB_map; dims = 1)[:] .> 0)
   AA_spec = AA_spec[keep_AA_idx]
   AA2BB_map = AA2BB_map[:, keep_AA_idx]

   # generate the corresponding A basis spec
   A_spec = _make_A_spec(AA_spec, level)

   # from the A basis we can generate the Y basis since we now know the 
   # maximum l value (though we probably already knew that from r_spec)
   maxl = maximum([ b.l for b in A_spec ])   
   ybasis = _make_Y_basis(Ytype, maxl)
   
   # now we need to take the human-readable specs and convert them into 
   # the layer-readable specs 
   r_spec = rbasis.spec
   y_spec = _make_Y_spec(maxl)

   # get the idx version of A_spec 
   inv_r_spec = _inv_list(r_spec)
   inv_y_spec = _inv_list(y_spec)
   A_spec_idx = [ (inv_r_spec[(n=b.n, l=b.l)], inv_y_spec[(l=b.l, m=b.m)]) 
                  for b in A_spec ]
   # from this we can now generate the A basis layer                   
   a_basis = Polynomials4ML.PooledSparseProduct(A_spec_idx)
   a_basis.meta["A_spec"] = A_spec  #(also store the human-readable spec)

   # get the idx version of AA_spec
   inv_A_spec = _inv_list(A_spec)
   AA_spec_idx = [ [ inv_A_spec[b] for b in bb ] for bb in AA_spec ]
   sort!.(AA_spec_idx)
   # from this we can now generate the AA basis layer
   aa_basis = Polynomials4ML.SparseSymmProdDAG(AA_spec_idx)
   aa_basis.meta["AA_spec"] = AA_spec  # (also store the human-readable spec)
   
   NZ = _get_nz(rbasis)
   n_B_params, n_AA_params = size(AA2BB_map)
   return ACEModel(rbasis._i2z, rbasis, ybasis, a_basis, aa_basis, AA2BB_map, 
                   ntuple(_ -> zeros(n_B_params), NZ),
                   ntuple(_ -> zeros(n_AA_params), NZ),
                   Dict{String, Any}() )
end

# TODO: it is not entirely clear that the `level` is really needed here 
#       since it is implicitly already encoded in AA_spec. We need a 
#       function `auto_level` that generates level automagically from AA_spec.
function ace_model(rbasis, Ytype, AA_spec::AbstractVector, level)
   return _generate_ace_model(rbasis, Ytype, AA_spec, level)
end 

# NOTE : a nicer convenience constructor is also provided in `ace_heuristics.jl`
#        this is where we should move all defaults, heuristics and other things 
#        that make life good.

# ------------------------------------------------------------
#   Lux stuff 

function initialparameters(rng::AbstractRNG, 
                           model::ACEModel)
   NZ = _get_nz(model)
   n_B_params, n_AA_params = size(model.A2Bmap)

   # only the B params are parameters, the AA params are uniquely defined 
   # via the B params. 

   # there are different ways to initialize parameters 
   if model.meta["init_WB"] == "zeros"
      winit = zeros 
   elseif model.meta["init_WB"] == "glorot_normal"
      winit = glorot_normal
   else
      error("unknown `init_WB` = $(model.meta["init_WB"])")
   end

   return (WB = [ winit(Float64, n_B_params) for _=1:NZ ], 
           rbasis = initialparameters(rng, model.rbasis), )
end

function initialstates(rng::AbstractRNG, 
                       model::ACEModel)
   return ( rbasis = initialstates(rng, model.rbasis), )
end

(l::ACEModel)(args...) = evaluate(l, args...)


# ------------------------------------------------------------
#   Model Evaluation 
#   this should possibly be moved to a separate file once it 
#   gets more complicated.

import Zygote

# these _getlmax and _length should be moved into SpheriCart 
_getlmax(ybasis::SolidHarmonics{L}) where {L} = L 
_length(ybasis::SolidHarmonics) = SpheriCart.sizeY(_getlmax(ybasis))

function evaluate(model::ACEModel, 
                  Rs::AbstractVector{SVector{3, T}}, Zs, Z0, 
                  ps, st) where {T}
   # get the radii 
   rs = [ norm(r) for r in Rs ]   # use Bumper 

   # evaluate the radial basis
   # use Bumper to pre-allocate 
   Rnl, _st = evaluate_batched(model.rbasis, rs, Z0, Zs, 
                              ps.rbasis, st.rbasis)

   # evaluate the Y basis
   Ylm = zeros(T, length(Rs), _length(model.ybasis))    # use Bumper here
   SpheriCart.compute!(Ylm, model.ybasis, Rs)

   # evaluate the A basis
   TA = promote_type(T, eltype(Rnl))
   A = zeros(T, length(model.abasis))
   Polynomials4ML.evaluate!(A, model.abasis, (Rnl, Ylm))

   # evaluate the AA basis
   _AA = zeros(T, length(model.aabasis))     # use Bumper here
   Polynomials4ML.evaluate!(_AA, model.aabasis, A)
   # project to the actual AA basis 
   proj = model.aabasis.projection
   AA = _AA[proj]     # use Bumper here, or view; needs experimentation. 

   # evaluate the coupling coefficients
   B = model.A2Bmap * AA

   # contract with params 
   i_z0 = _z2i(model.rbasis, Z0)
   val = dot(B, ps.WB[i_z0])
            
   return val, st 
end



function evaluate_ed(model::ACEModel, 
                     Rs::AbstractVector{SVector{3, T}}, Zs, Z0, 
                     ps, st) where {T}

   # ---------- EMBEDDINGS ------------
   # (these are done in forward mode, so not part of the fwd, bwd passes)

   # get the radii 
   rs = [ norm(r) for r in Rs ]   # TODO: use Bumper 

   # evaluate the radial basis
   # TODO: use Bumper to pre-allocate 
   Rnl, dRnl, _st = evaluate_ed_batched(model.rbasis, rs, Z0, Zs, 
                                        ps.rbasis, st.rbasis)
   # evaluate the Y basis
   Ylm = zeros(T, length(Rs), _length(model.ybasis))    # TODO: use Bumper
   dYlm = zeros(SVector{3, T}, length(Rs), _length(model.ybasis))
   SpheriCart.compute_with_gradients!(Ylm, dYlm, model.ybasis, Rs)

   # ---------- FORWARD PASS ------------

   # evaluate the A basis
   TA = promote_type(T, eltype(Rnl))
   A = zeros(T, length(model.abasis))
   Polynomials4ML.evaluate!(A, model.abasis, (Rnl, Ylm))

   # evaluate the AA basis
   _AA = zeros(T, length(model.aabasis))     # TODO: use Bumper here
   Polynomials4ML.evaluate!(_AA, model.aabasis, A)
   # project to the actual AA basis 
   proj = model.aabasis.projection
   AA = _AA[proj]     # TODO: use Bumper here, or view; needs experimentation. 

   # evaluate the coupling coefficients 
   # TODO: use Bumper and do it in-place 
   B = model.A2Bmap * AA

   # contract with params 
   # (here we can insert another nonlinearity instead of the simple dot)
   i_z0 = _z2i(model.rbasis, Z0)
   Ei = dot(B, ps.WB[i_z0])

   # ---------- BACKWARD PASS ------------

   # ∂Ei / ∂B = WB[i_z0]
   ∂B = ps.WB[i_z0]

   # ∂Ei / ∂AA = ∂Ei / ∂B * ∂B / ∂AA = (WB[i_z0]) * A2Bmap
   ∂AA = model.A2Bmap' * ∂B   # TODO: make this in-place 
   _∂AA = zeros(T, length(_AA)) 
   _∂AA[proj] = ∂AA

   # ∂Ei / ∂A = ∂Ei / ∂AA * ∂AA / ∂A = pullback(aabasis, ∂AA)
   ∂A = zeros(T, length(model.abasis))
   Polynomials4ML.pullback_arg!(∂A, _∂AA, model.aabasis, _AA)
   
   # ∂Ei / ∂Rnl, ∂Ei / ∂Ylm = pullback(abasis, ∂A)
   ∂Rnl = zeros(T, size(Rnl))
   ∂Ylm = zeros(T, size(Ylm))
   Polynomials4ML._pullback_evaluate!((∂Rnl, ∂Ylm), ∂A, model.abasis, (Rnl, Ylm))
   
   # ---------- ASSEMBLE DERIVATIVES ------------
   # The ∂Ei / ∂𝐫ⱼ can now be obtained from the ∂Ei / ∂Rnl, ∂Ei / ∂Ylm 
   # as follows: 
   #    ∂Ei / ∂𝐫ⱼ = ∑_nl ∂Ei / ∂Rnl[j] * ∂Rnl[j] / ∂𝐫ⱼ 
   #              + ∑_lm ∂Ei / ∂Ylm[j] * ∂Ylm[j] / ∂𝐫ⱼ
   ∇Ei = zeros(SVector{3, T}, length(Rs))
   for j = 1:length(Rs)
      ∇Ei[j] = dot(∂Rnl[j, :], dRnl[j, :]) * (Rs[j] / rs[j]) + 
               sum(∂Ylm[j, :] .* dYlm[j, :])
   end

   return Ei, ∇Ei, st 
end


function grad_params(model::ACEModel, 
                     Rs::AbstractVector{SVector{3, T}}, Zs, Z0, 
                     ps, st) where {T}

   # ---------- EMBEDDINGS ------------
   # (these are done in forward mode, so not part of the fwd, bwd passes)

   # get the radii 
   rs = [ norm(r) for r in Rs ]   # TODO: use Bumper 

   # evaluate the radial basis
   # TODO: use Bumper to pre-allocate 
   (Rnl, _st), pb_Rnl = rrule(evaluate_batched, model.rbasis, 
                              rs, Z0, Zs, ps.rbasis, st.rbasis)
   # evaluate the Y basis
   Ylm = zeros(T, length(Rs), _length(model.ybasis))    # TODO: use Bumper
   dYlm = zeros(SVector{3, T}, length(Rs), _length(model.ybasis))
   SpheriCart.compute_with_gradients!(Ylm, dYlm, model.ybasis, Rs)

   # ---------- FORWARD PASS ------------

   # evaluate the A basis
   TA = promote_type(T, eltype(Rnl))
   A = zeros(T, length(model.abasis))
   Polynomials4ML.evaluate!(A, model.abasis, (Rnl, Ylm))

   # evaluate the AA basis
   _AA = zeros(T, length(model.aabasis))     # TODO: use Bumper here
   Polynomials4ML.evaluate!(_AA, model.aabasis, A)
   # project to the actual AA basis 
   proj = model.aabasis.projection
   AA = _AA[proj]     # TODO: use Bumper here, or view; needs experimentation. 

   # evaluate the coupling coefficients 
   # TODO: use Bumper and do it in-place 
   B = model.A2Bmap * AA

   # contract with params 
   # (here we can insert another nonlinearity instead of the simple dot)
   i_z0 = _z2i(model.rbasis, Z0)
   Ei = dot(B, ps.WB[i_z0])

   # ---------- BACKWARD PASS ------------

   # we need ∂WB = ∂Ei/∂WB -> this goes into the gradient 
   # but we also need ∂B = ∂Ei / ∂B = WB[i_z0] to backpropagate
   ∂WB_i = B 
   ∂B = ps.WB[i_z0]

   # ∂Ei / ∂AA = ∂Ei / ∂B * ∂B / ∂AA = (WB[i_z0]) * A2Bmap
   ∂AA = model.A2Bmap' * ∂B   # TODO: make this in-place 
   _∂AA = zeros(T, length(_AA)) 
   _∂AA[proj] = ∂AA

   # ∂Ei / ∂A = ∂Ei / ∂AA * ∂AA / ∂A = pullback(aabasis, ∂AA)
   ∂A = zeros(T, length(model.abasis))
   Polynomials4ML.pullback_arg!(∂A, _∂AA, model.aabasis, _AA)
   
   # ∂Ei / ∂Rnl, ∂Ei / ∂Ylm = pullback(abasis, ∂A)
   ∂Rnl = zeros(T, size(Rnl))
   ∂Ylm = zeros(T, size(Ylm))   # we could make this a black hole since we don't need it. 
   Polynomials4ML._pullback_evaluate!((∂Rnl, ∂Ylm), ∂A, model.abasis, (Rnl, Ylm))
   
   # ---------- ASSEMBLE DERIVATIVES ------------
   # the first grad_param is ∂WB, which we already have but it needs to be 
   # written into a vector of vectors 
   ∂WB = [ zeros(eltype(∂WB_i), size(∂WB_i)) for _=1:_get_nz(model) ]
   ∂WB[i_z0] = ∂WB_i

   # the second one is the gradient with respect to Rnl params 
   #
   # ∂Ei / ∂Wn̄l̄q̄ 
   #   = ∑_nlj ∂Ei / ∂Rnl[j] * ∂Rnl[j] / ∂Wn̄l̄q̄
   #   = pullback(∂Rnl, rbasis, args...)
   _, _, _, _, ∂Wqnl, _ = pb_Rnl(∂Rnl)  # this should be a named tuple already.

   return Ei, (WB = ∂WB, rbasis = ∂Wqnl), st
end


using Optimisers: destructure 
using ForwardDiff: Dual, value, extract_derivative

function pullback_2_mixed(Δ, Δd, model::ACEModel, 
            Rs::AbstractVector{SVector{3, T}}, Zs, Z0, ps, st) where {T}
   # this is implemented as a directional derivative 
   # following a wonderful discussion on Discourse with 
   # Steven G. Johnson and Avik Pal 
   #
   # we want the pullback for the pair (Ei, ∇Ei) computed via evaluate_ed 
   # i.e. for Δ, Δd the output sensitivities we want to compute 
   #    ∂_w { Δ * Ei + Δd * ∇Ei }
   # where ∂_w = gradient with respect to parameters. We first rewrite this as 
   #    Δ * ∂_w Ei + d/dt ∂_w Ei( x + t Δd) |_{t = 0}
   # Which we can compute from 
   #      ∂_w Ei( x + Dual(0, 1) * Δd )
   # Beautiful.

   Rs_d = Rs + Dual{T}(0, 1) * Δd
   Ei_d, ∂Ei_d, st = grad_params(model, Rs_d, Zs, Z0, ps, st)

   # To make our life easy we can hack the named-tuples a bit 
   # TODO: this can and probably should be made more efficient 
   ∂Ei_d_vec, _restruct = destructure(∂Ei_d)
   # extract the gradient w.r.t. parameters 
   ∂Ei = value.(∂Ei_d_vec)
   # extract the directional derivative w.r.t. positions 
   ∂∇Ei_Δd = extract_derivative.(T, ∂Ei_d_vec) 
   
   # combine to produce the output 
   return _restruct(Δ * ∂Ei + ∂∇Ei_Δd)
end

# ------------------------------------------------------------
#  ACE basis evaluation 


function get_basis_inds(model::ACEModel, Z)
   len_Bi = size(model.A2Bmap, 1)
   i_z = _z2i(model.rbasis, Z)
   return (i_z - 1) * len_Bi .+ (1:len_Bi)
end

function evaluate_basis(model::ACEModel, 
                        Rs::AbstractVector{SVector{3, T}}, Zs, Z0, 
                        ps, st) where {T}
   # get the radii 
   rs = [ norm(r) for r in Rs ]   # use Bumper 

   # evaluate the radial basis
   # use Bumper to pre-allocate 
   Rnl, _st = evaluate_batched(model.rbasis, rs, Z0, Zs, 
                              ps.rbasis, st.rbasis)

   # evaluate the Y basis
   Ylm = zeros(T, length(Rs), _length(model.ybasis))    # use Bumper here
   SpheriCart.compute!(Ylm, model.ybasis, Rs)

   # evaluate the A basis
   TA = promote_type(T, eltype(Rnl))
   A = zeros(T, length(model.abasis))
   Polynomials4ML.evaluate!(A, model.abasis, (Rnl, Ylm))

   # evaluate the AA basis
   _AA = zeros(T, length(model.aabasis))     # use Bumper here
   Polynomials4ML.evaluate!(_AA, model.aabasis, A)
   # project to the actual AA basis 
   proj = model.aabasis.projection
   AA = _AA[proj]     # use Bumper here, or view; needs experimentation. 

   # evaluate the coupling coefficients 
   # TODO: use Bumper and do it in-place 
   Bi = model.A2Bmap * AA
   B = zeros(eltype(Bi), length(Bi) * _get_nz(model))
   B[get_basis_inds(model, Z0)] .= Bi
   
   return B, st
end

__vec(Rs::AbstractVector{SVector{3, T}}) where {T} = reinterpret(T, Rs)
__svecs(Rsvec::AbstractVector{T}) where {T} = reinterpret(SVector{3, T}, Rsvec)

function evaluate_basis_ed(model::ACEModel, 
                           Rs::AbstractVector{SVector{3, T}}, Zs, Z0, 
                           ps, st) where {T}

   B, st = evaluate_basis(model, Rs, Zs, Z0, ps, st)

   dB_vec = ForwardDiff.jacobian( 
            _Rs -> evaluate_basis(model, __svecs(_Rs),  Zs, Z0, ps, st)[1],
            __vec(Rs))
   dB1 = __svecs(collect(dB_vec')[:])
   dB = collect( permutedims( reshape(dB1, length(Rs), length(B)), 
                               (2, 1) ) )

   return B, dB, st         
end



function jacobian_grad_params(model::ACEModel, 
                              Rs::AbstractVector{SVector{3, T}}, Zs, Z0, 
                              ps, st) where {T}

   Ei, ∂Ei, st = grad_params(model, Rs, Zs, Z0, ps, st)
   ∂∂Ei_vec = ForwardDiff.jacobian( _Rs -> (
            destructure( grad_params(model, __svecs(_Rs), Zs, Z0, ps, st)[2] )[1]
         ), 
         __vec(Rs))
   ∂Ei_vec = destructure(∂Ei)[1]
   ∂∂Ei = collect( permutedims( 
               reshape( __svecs((∂∂Ei_vec')[:]), length(Rs), length(∂Ei_vec) ), 
               (2, 1) ) )
   return Ei, ∂Ei_vec, ∂∂Ei, st
end

