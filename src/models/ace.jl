

using Lux: glorot_normal

using StaticArrays: SVector
using LinearAlgebra: norm, dot
using Polynomials4ML: real_sphericalharmonics, real_solidharmonics

import RepLieGroups
import EquivariantTensors


# ------------------------------------------------------------
#    ACE MODEL SPECIFICATION 

mutable struct ACEModel{NZ, TRAD, TY, TTEN, TPAIR, TVREF} <: AbstractExplicitContainerLayer{(:rbasis,)}
   _i2z::NTuple{NZ, Int}
   # --------------
   # embeddings of the particles 
   rbasis::TRAD
   ybasis::TY
   # --------------
   # the tensor format
   tensor::TTEN 
   # -------------- 
   # we can add a nonlinear embedding here 
   # --------------
   #   pair potential & Vref 
   pairbasis::TPAIR 
   Vref::TVREF    # E0s::NTuple{NZ, T}   
   # --------------
   meta::Dict{String, Any}
end

# ------------------------------------------------------------
#    CONSTRUCTORS AND UTILITIES

# # this is terrible : I'm assuming here that there is a unique 
# # output type, which is of course not the case. It is needed temporarily 
# # to make things work with AtomsCalculators and EmpiricalPotentials 
# fl_type(::ACEModel{NZ, TRAD, TY, TTEN, T, TPAIR}) where {NZ, TRAD, TY, TTEN, T, TPAIR} = T 

const NT_NLM = NamedTuple{(:n, :l, :m), Tuple{Int, Int, Int}}

function _make_Y_basis(Ytype, lmax) 
   if Ytype == :solid 
      return real_solidharmonics(lmax)
   elseif Ytype == :spherical
      return real_sphericalharmonics(lmax)
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
   A_spec = unique(A_spec)
   A_spec_level = [ level(b) for b in A_spec ]
   p = sortperm(A_spec_level)
   A_spec = A_spec[p]
   return A_spec
end 

# TODO: this should go into sphericart or P4ML 
function _make_Y_spec(maxl::Integer)
   NT_LM = NamedTuple{(:l, :m), Tuple{Int, Int}}
   y_spec = NT_LM[] 
   for i = 1:P4ML.SpheriCart.sizeY(maxl)
      l, m = P4ML.SpheriCart.idx2lm(i)
      push!(y_spec, (l = l, m = m))
   end
   return y_spec 
end

function _make_idx_A_spec(A_spec, r_spec, y_spec)
   inv_r_spec = _inv_list(r_spec)
   inv_y_spec = _inv_list(y_spec)
   A_spec_idx = [ (inv_r_spec[(n=b.n, l=b.l)], inv_y_spec[(l=b.l, m=b.m)]) 
                  for b in A_spec ]
   return A_spec_idx                  
end

function _make_idx_AA_spec(AA_spec, A_spec) 
   inv_A_spec = _inv_list(A_spec)
   AA_spec_idx = [ [ inv_A_spec[b] for b in bb ] for bb in AA_spec ]
   sort!.(AA_spec_idx)
   return AA_spec_idx
end 


function _generate_ace_model(rbasis, Ytype::Symbol, AA_spec::AbstractVector, 
                             Vref, 
                             level = TotalDegree(), 
                             pair_basis = nothing, 
                             ) 

   # # storing E0s with unit
   # model_meta = Dict{String, Any}("E0s" => deepcopy(E0s))
   model_meta = Dict{String, Any}()

   # generate the coupling coefficients
   # Migrated from EquivariantModels._rpi_A2B_matrix to EquivariantTensors.symmetrisation_matrix
   # Note: symmetrisation_matrix returns (matrix, pruned_spec), we only need the matrix
   AA2BB_map, _ = EquivariantTensors.symmetrisation_matrix(0, AA_spec;
                                                            prune = true,
                                                            PI = true,
                                                            basis = real)

   # find which AA basis functions are actually used and discard the rest 
   keep_AA_idx = findall(sum(abs, AA2BB_map; dims = 1)[:] .> 1e-12)
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
   A_spec_idx = _make_idx_A_spec(A_spec, r_spec, y_spec)

   # from this we can now generate the A basis layer                   
   a_basis = Polynomials4ML.PooledSparseProduct(A_spec_idx)
   a_basis.meta["A_spec"] = A_spec  #(also store the human-readable spec)

   # get the idx version of AA_spec
   AA_spec_idx = _make_idx_AA_spec(AA_spec, A_spec) 

   # from this we can now generate the AA basis layer
   aa_basis = Polynomials4ML.SparseSymmProdDAG(AA_spec_idx)
   aa_basis.meta["AA_spec"] = AA_spec  # (also store the human-readable spec)

   tensor = SparseEquivTensor(a_basis, aa_basis, AA2BB_map, 
                              Dict{String, Any}())

   return ACEModel(rbasis._i2z, rbasis, ybasis, 
                   tensor, pair_basis, Vref, 
                   model_meta )
end

# TODO: it is not entirely clear that the `level` is really needed here 
#       since it is implicitly already encoded in AA_spec. We need a 
#       function `auto_level` that generates level automagically from AA_spec.
function ace_model(rbasis, Ytype, AA_spec::AbstractVector, level, 
                   pair_basis, Vref)
   return _generate_ace_model(rbasis, Ytype, AA_spec, Vref, level, pair_basis)
end 

# NOTE : a nicer convenience constructor is also provided in `ace_heuristics.jl`
#        this is where we should move all defaults, heuristics and other things 
#        that make life good.

# ------------------------------------------------------------
#   Lux stuff 

function _W_init(str)
   if str == "zeros"
      return (rng, T, args...) -> zeros(T, args...)
   elseif str == "glorot_normal"
      return glorot_normal
   else
      error("unknown `init_WB` = $str")
   end
end

function initialparameters(rng::AbstractRNG, 
                           model::ACEModel)
   NZ = _get_nz(model)
   n_B_params = length(model.tensor)

   # only the B params are parameters, the AA params are uniquely defined 
   # via the B params. 

   # there are different ways to initialize parameters 
   winit = _W_init(model.meta["init_WB"])
   WB = zeros(n_B_params, NZ)
   for iz = 1:NZ 
      WB[:, iz] .= winit(rng, Float64, n_B_params)
   end

   # generate pair basis parameters 
   n_pair = length(model.pairbasis)
   Wpair = zeros(n_pair, NZ)
   winit_pair = _W_init(model.meta["init_WB"])

   for iz = 1:NZ 
      Wpair[:, iz] .= winit_pair(rng, Float64, n_pair)
   end

   return (WB = WB, Wpair = Wpair, 
           rbasis = initialparameters(rng, model.rbasis),
           pairbasis = initialparameters(rng, model.pairbasis), ) 
end

function initialstates(rng::AbstractRNG, 
                       model::ACEModel)
   return ( rbasis = initialstates(rng, model.rbasis), 
            pairbasis = initialstates(rng, model.pairbasis), ) 
end

(l::ACEModel)(args...) = evaluate(l, args...)

function LuxCore.parameterlength(model::ACEModel)
   # this layer stores the pair basis parameters and the B basis parameters 
   NZ = _get_nz(model)
   return NZ^2 * length(model.pairbasis) + NZ * length(model.tensor)
end

function splinify(model::ACEModel, ps::NamedTuple; kwargs...)
   rbasis_spl = splinify(model.rbasis, ps.rbasis; kwargs...)
   pairbasis_spl = splinify(model.pairbasis, ps.pairbasis; kwargs...)
   return ACEModel(model._i2z, 
                     rbasis_spl, 
                     model.ybasis, 
                     model.tensor, 
                     pairbasis_spl, 
                     model.Vref,
                     model.meta)
end

# ------------------------------------------------------------
#  utilities 

function radii!(rs, Rs::AbstractVector{SVector{D, T}}) where {D, T <: Real}
   @assert length(rs) >= length(Rs)
   @inbounds for i = 1:length(Rs)
      rs[i] = norm(Rs[i])
   end
   return rs   
end

function whatalloc(::typeof(radii!), Rs::AbstractVector{SVector{D, T}}) where {D, T <: Real} 
   return (T, length(Rs))
end

function radii_ed!(rs, ∇rs, Rs::AbstractVector{SVector{D, T}}) where {D, T <: Real}
   @assert length(rs) >= length(Rs)
   @assert length(∇rs) >= length(Rs)
   @inbounds for i = 1:length(Rs)
      rs[i] = norm(Rs[i])
      ∇rs[i] = Rs[i] / rs[i]
   end
   return rs, ∇rs
end

function whatalloc(::typeof(radii_ed!), Rs::AbstractVector{SVector{D, T}}) where {D, T <: Real} 
   return (T, length(Rs)), (SVector{D, T}, length(Rs))
end



# ------------------------------------------------------------
#   Model Evaluation 
#   this should possibly be moved to a separate file once it 
#   gets more complicated.

function evaluate(model::ACEModel, 
                  Rs::AbstractVector{SVector{3, T}}, Zs, Z0, 
                  ps, st) where 
                  {T}
   i_z0 = _z2i(model.rbasis, Z0)

   if length(Rs) == 0 
      return model.Vref.E0[Z0]
   end 

   @no_escape begin 

   # get the radii 
   rs = @withalloc radii!(Rs) 

   # evaluate the radial basis
   Rnl = @withalloc evaluate_batched!(model.rbasis, rs, Z0, Zs, 
                                      ps.rbasis, st.rbasis)

   # evaluate the Y basis
   Ylm = @withalloc P4ML.evaluate!(model.ybasis, Rs)

   # equivariant tensor product 
   B, _ = @withalloc evaluate!(model.tensor, Rnl, Ylm)

   # contract with params 
   val = dot(B, (@view ps.WB[:, i_z0]))

   
   # ------------------- 
   #  pair potential 
   if model.pairbasis != nothing 
      Rpair = evaluate_batched(model.pairbasis, rs, Z0, Zs, 
                               ps.pairbasis, st.pairbasis)
      Apair = sum(Rpair, dims=1)[:]
      val += dot(Apair, (@view ps.Wpair[:, i_z0]))
   end
   # ------------------- 
   #  Vref
   val += eval_site(model.Vref, Rs, Zs, Z0)
   # ------------------- 

   end # @no_escape
            
   return val
end



function evaluate_ed(model::ACEModel, 
                     Rs::AbstractVector{SVector{3, T}}, Zs, Z0, 
                     ps, st) where {T}

   i_z0 = _z2i(model.rbasis, Z0)

   if length(Rs) == 0 
      return model.Vref.E0[Z0], SVector{3, T}[] 
   end 


   @no_escape begin 
   # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   
   # ---------- EMBEDDINGS ------------
   # (these are done in forward mode, so not part of the fwd, bwd passes)

   # get the radii 
   rs, ∇rs = @withalloc radii_ed!(Rs)

   # evaluate the radial basis
   Rnl, dRnl = @withalloc evaluate_ed_batched!(model.rbasis, rs, Z0, Zs, 
                                               ps.rbasis, st.rbasis)

   # evaluate the Y basis
   Ylm, dYlm = @withalloc P4ML.evaluate_ed!(model.ybasis, Rs)

   # Forward Pass through the tensor 
   # keep intermediates to be used in backward pass 
   B, intermediates = @withalloc evaluate!(model.tensor, Rnl, Ylm)

   # contract with params 
   # (here we can insert another nonlinearity instead of the simple dot)
   Ei = dot(B, (@view ps.WB[:, i_z0]))

   # Start the backward pass 
   # ∂Ei / ∂B = WB[i_z0]
   ∂B = @view ps.WB[:, i_z0]
   
   # backward pass through tensor 
   ∂Rnl, ∂Ylm = @withalloc pullback!(∂B, model.tensor, Rnl, Ylm, intermediates)
   
   # ---------- ASSEMBLE DERIVATIVES ------------
   # The ∂Ei / ∂𝐫ⱼ can now be obtained from the ∂Ei / ∂Rnl, ∂Ei / ∂Ylm 
   # as follows: 
   #    ∂Ei / ∂𝐫ⱼ = ∑_nl ∂Ei / ∂Rnl[j] * ∂Rnl[j] / ∂𝐫ⱼ 
   #              + ∑_lm ∂Ei / ∂Ylm[j] * ∂Ylm[j] / ∂𝐫ⱼ
   ∇Ei = zeros(SVector{3, T}, length(Rs))
   for t = 1:size(∂Rnl, 2)
      for j = 1:size(∂Rnl, 1)
         ∇Ei[j] += (∂Rnl[j, t] * dRnl[j, t]) * ∇rs[j]
      end
   end
   for t = 1:size(∂Ylm, 2)
      for j = 1:size(∂Ylm, 1)
         ∇Ei[j] += ∂Ylm[j, t] * dYlm[j, t]
      end
   end

   # ------------------- 
   #  pair potential 
   if model.pairbasis != nothing 
      Rpair, dRpair = evaluate_ed_batched(model.pairbasis, rs, Z0, Zs, 
                                             ps.pairbasis, st.pairbasis)
      Apair = sum(Rpair, dims=1)[:]
      Wp_i = @view ps.Wpair[:, i_z0]
      Ei += dot(Apair, Wp_i)

      # pullback --- I'm now assuming that the pair basis is not learnable.
      # if !( ps.pairbasis == NamedTuple() ) 
      #    error("I'm currently assuming the pair basis is not learnable.")
      # end

      for j = 1:length(Rs)
         ∇Ei[j] += dot(Wp_i, (@view dRpair[j, :])) * (Rs[j] / rs[j])
      end
   end
   # ------------------- 
   #  TODO - generiv Vref, for now assume it is a OneBody 
   @assert model.Vref isa OneBody
   Ei += model.Vref.E0[Z0]
   # ------------------- 

   # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   end # @no_escape

   return Ei, ∇Ei
end


function grad_params(model::ACEModel, 
                     Rs::AbstractVector{SVector{3, T}}, Zs, Z0, 
                     ps, st) where {T}

   @no_escape begin 
   # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                     

   # ---------- EMBEDDINGS ------------
   # (these are done in forward mode, so not part of the fwd, bwd passes)

   # get the radii 
   rs = @withalloc radii!(Rs) 

   # evaluate the radial basis
   # TODO: use Bumper to pre-allocate 
   Rnl, pb_Rnl = rrule(evaluate_batched, model.rbasis, 
                              rs, Z0, Zs, ps.rbasis, st.rbasis)
   
   # evaluate the Y basis
   Ylm = @withalloc P4ML.evaluate!(model.ybasis, Rs)

   # Ylm = zeros(T, length(Rs), length(model.ybasis))    # TODO: use Bumper
   # dYlm = zeros(SVector{3, T}, length(Rs), length(model.ybasis))
   # SpheriCart.compute_with_gradients!(Ylm, dYlm, model.ybasis, Rs)

   # Forward Pass through the tensor 
   # keep intermediates to be used in backward pass 
   # B, intermediates = evaluate(model.tensor, Rnl, Ylm)
   B, intermediates = @withalloc evaluate!(model.tensor, Rnl, Ylm)

   # contract with params 
   # (here we can insert another nonlinearity instead of the simple dot)
   i_z0 = _z2i(model.rbasis, Z0)
   Ei = dot(B, (@view ps.WB[:, i_z0]))

   # ---------- BACKWARD PASS ------------

   # we need ∂WB = ∂Ei/∂WB -> this goes into the gradient 
   # but we also need ∂B = ∂Ei / ∂B = WB[i_z0] to backpropagate
   ∂WB_i = B 
   ∂B = @view ps.WB[:, i_z0]

   # backward pass through tensor 
   # ∂Rnl, ∂Ylm = pullback_evaluate(∂B, model.tensor, Rnl, Ylm, intermediates)
   ∂Rnl, ∂Ylm = @withalloc pullback!(∂B, model.tensor, Rnl, Ylm, intermediates)
   
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


   # ------------------- 
   #  pair potential 
   if model.pairbasis != nothing 
      Rpair = evaluate_batched(model.pairbasis, rs, Z0, Zs, 
                                    ps.pairbasis, st.pairbasis)
      Apair = sum(Rpair, dims=1)[:]
      Wp_i = @view ps.Wpair[:, i_z0]
      Ei += dot(Apair, Wp_i)

      # pullback --- I'm now assuming that the pair basis is not learnable.
      if !( ps.pairbasis == NamedTuple() ) 
         error("I'm currently assuming the pair basis is not learnable.")
      end

      ∂Wpair = zeros(eltype(Apair), size(ps.Wpair))
      ∂Wpair[:, i_z0] = Apair
   end
   # ------------------- 
   #  TODO - generic Vref, assume OneBody for now
   @assert model.Vref isa OneBody
   Ei += model.Vref.E0[Z0]
   # ------------------- 


   # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   end # @no_escape

   return Ei, (WB = ∂WB, Wpair = ∂Wpair, rbasis = ∂Wqnl, 
               pairbasis = NamedTuple()), st
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
   len_Bi = length(model.tensor)
   i_z = _z2i(model.rbasis, Z)
   return (i_z - 1) * len_Bi .+ (1:len_Bi)
end

function get_pairbasis_inds(model::ACEModel, Z)
   len_Bi = length(model.tensor)
   NZ = _get_nz(model)
   len_B = NZ * len_Bi 

   len_pair = length(model.pairbasis)
   i_z = _z2i(model, Z)
   return (len_B + (i_z - 1) * len_pair) .+ (1:len_pair)
end

function length_basis(model::ACEModel)
   len_Bi = length(model.tensor)
   len_pair = length(model.pairbasis)
   NZ = _get_nz(model)
   return (len_Bi + len_pair) * NZ 
end

function len_basis(model::ACEModel)
   @warn("len_basis is deprecated, use length_basis instead")
   return length_basis(model)
end


function get_basis_params(model::ACEModel, ps, )
   # this is magically given by the basis ordering we picked
   return vcat(ps.WB[:], ps.Wpair[:])   
end

function evaluate_basis(model::ACEModel, 
                        Rs::AbstractVector{SVector{3, T}}, Zs, Z0, 
                        ps, st) where {T}
   if length(Rs) == 0 
      return zeros(T, length_basis(model))
   end

   # get the radii 
   rs = @withalloc radii!(Rs) 

   # evaluate the radial basis
   Rnl = evaluate_batched(model.rbasis, rs, Z0, Zs, 
                           ps.rbasis, st.rbasis)

   # evaluate the Y basis
   Ylm = @withalloc P4ML.evaluate!(model.ybasis, Rs)

   # equivariant tensor product 
   Bi, _ = @withalloc evaluate!(model.tensor, Rnl, Ylm)

   B = zeros(eltype(Bi), length_basis(model))
   B[get_basis_inds(model, Z0)] .= Bi
   
   # ------------------- 
   #  pair potential 
   if model.pairbasis != nothing 
      Rpair = evaluate_batched(model.pairbasis, rs, Z0, Zs, 
                                    ps.pairbasis, st.pairbasis)
      Apair = sum(Rpair, dims=1)[:]
      B[get_pairbasis_inds(model, Z0)] .= Apair
   end 

   return B
end

__vec(Rs::AbstractVector{SVector{3, T}}) where {T} = reinterpret(T, Rs)
__svecs(Rsvec::AbstractVector{T}) where {T} = reinterpret(SVector{3, T}, Rsvec)

function evaluate_basis_ed_old(model::ACEModel, 
                           Rs::AbstractVector{SVector{3, T}}, Zs, Z0, 
                           ps, st) where {T}

   if length(Rs) == 0 
      B = zeros(T, length_basis(model))
      dB = zeros(SVector{3, T}, (0, length_basis(model)))
   else

      B = evaluate_basis(model, Rs, Zs, Z0, ps, st)

      dB_vec = ForwardDiff.jacobian( 
               _Rs -> evaluate_basis(model, __svecs(_Rs),  Zs, Z0, ps, st),
               __vec(Rs))
      dB1 = __svecs(collect(dB_vec')[:])
      dB = collect( permutedims( reshape(dB1, length(Rs), length(B)), 
                                 (2, 1) ) )
   end 

   return B, dB         
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



# ---------------------------------------------------------
#  experimental pushforwards 

function evaluate_basis_ed(model::ACEModel, 
                            Rs::AbstractVector{SVector{3, T}}, Zs, Z0, 
                            ps, st) where {T}

   TB = T
   ∂TB = SVector{3, T}
   B = zeros(TB, length_basis(model))
   ∂B = zeros(∂TB, length_basis(model), length(Rs))

   if length(Rs) == 0
      return B, ∂B
   end
   
   @no_escape begin 
   # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                     

   # get the radii 
   rs, ∇rs = @withalloc radii_ed!(Rs)

   # evaluate the radial basis
   Rnl, dRnl = @withalloc evaluate_ed_batched!(model.rbasis, rs, Z0, Zs, 
                                               ps.rbasis, st.rbasis)

   # evaluate the Y basis
   Ylm, dYlm = @withalloc P4ML.evaluate_ed!(model.ybasis, Rs)

   # compute vectorial dRnl 
   ∂Ylm = dYlm 
   ∂Rnl = @alloc(eltype(dYlm), size(dRnl)...)
   for nl = 1:size(dRnl, 2)
      # @inbounds begin 
      # @simd ivdep 
         for j = 1:size(dRnl, 1)
            ∂Rnl[j, nl] = dRnl[j, nl] * ∇rs[j]
         end
      # end
   end

   # pushfoward through the sparse tensor - this completes the MB jacobian. 
   Bmb_i, ∂Bmb_i = _pfwd(model.tensor, Rnl, Ylm, ∂Rnl, ∂Ylm)

   # ------------------- 
   #  pair potential 
   if model.pairbasis != nothing    
      Rnl2, dRnl2 = @withalloc evaluate_ed_batched!(model.pairbasis, 
                                 rs, Z0, Zs, 
                                 ps.pairbasis, st.pairbasis)
      B2_i = sum(Rnl2, dims=1)[:]
      ∂B2_i = zeros(eltype(∂Bmb_i), size(Rnl2, 2), size(Rnl2, 1))
      for nl = 1:size(dRnl2, 2)
         for j = 1:size(dRnl2, 1)
            ∂B2_i[nl, j] = dRnl2[j, nl] * ∇rs[j]
         end
      end
   else 
      B2_i = zeros(eltype(Bmb_i), 0)
      ∂B2_i = zeros(eltype(∂Bmb_i), 0, size(Bmb_i, 2))
   end 

   # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                     
   end  # @no_escape

   B[get_basis_inds(model, Z0)] .= Bmb_i
   B[get_pairbasis_inds(model, Z0)] .= B2_i
   ∂B[get_basis_inds(model, Z0), :] .= ∂Bmb_i
   ∂B[get_pairbasis_inds(model, Z0), :] .= ∂B2_i

   return B, ∂B 
end