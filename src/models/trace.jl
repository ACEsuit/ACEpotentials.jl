# This file implements contructors of TRACE type models. 
# J. P. Darby, D. P. Kovács, I. Batatia, M. A. Caro, G. L. W. Hart, 
# C. Ortner, and G. Csányi. Tensor-reduced atomic density representations. 
# Phys. Rev. Lett., 131, 2023

# This implementation treats TRACE style models are simply special choices of 
# the sparse ACE model. This is likely a non-optimal implementation but allows 
# for a simple implementation for testing purposes. Optimized kernels can 
# then be provided over time.


struct TraceLevel
   maxn::Int 
end 

(l::TraceLevel)(b::NamedTuple) = b.l 
(l::TraceLevel)(bb::AbstractVector{<: NamedTuple}) = sum(l(b) for b in bb)
(l::TraceLevel)(l1::Integer) = l1
(l::TraceLevel)(ll::AbstractVector{<: Integer}) = sum(ll)


"""
   trace_model(; kwargs...)

Most important arguments to play with:    
- `maxn` : number of uncoupled radial channel for each ll tuple 
- `max_level` : keep all ll s.t. ∑ l_t <= max_level 
- `maxq` : number of radial embedding functions
- `pair_maxn` : number of radial basis functions per species 
- `tensor_format` : :cp or :symcp, default is :cp 
"""
function trace_model(; elements = nothing, 
                     order = nothing, 
                     Ytype = :spherical,  
                     E0s = nothing,
                     rin0cuts = :auto,
                     tensor_format = :symcp,  # :cp or :symcp
                     # radial basis 
                     #  we could allow passing an rbasis but for trace this 
                     #  doesn't sound like a good idea (for now anyhow...)
                     # rbasis_type = :auto, 
                     maxl = 30, # maxl, is a fairly high defaults 
                     maxn = 32, # that we will likely never reach due to the level restriction  
                     level = TraceLevel(maxn), 
                     max_level = nothing, 
                     maxq_fact = 2, 
                     maxq = maxq_fact * max_level, 
                     init_Wradial = :glorot_normal, 
                     radial_scaling = (:algebraic, 2),
                     init_WB = :zeros, 
                     # pair basis 
                     pair_maxn = nothing, 
                     pair_basis = :auto, 
                     pair_learnable = false, 
                     pair_transform = (:agnesi, 1, 4), 
                     init_Wpair = :onehot, 
                     rng = Random.default_rng(), 
                     )
   if tensor_format != :symcp
      error("only `tensor_format = :symcp` is supported for now")
   end                     

   if rin0cuts == :auto
      rin0cuts = _default_rin0cuts(elements)
   else
      NZ = length(elements)
      @assert rin0cuts isa SMatrix && size(rin0cuts) == (NZ, NZ)
   end

   # construct an rbasis
   rbasis_ = ace_learnable_Rnlrzz(; max_level = max_level, level = level, 
                                   maxl = maxl, maxn = maxn, 
                                   maxq = maxq, 
                                   elements = elements, 
                                   rin0cuts = rin0cuts, 
                                   Winit = init_Wradial)

   # for linear trace, this radial basis should be made non-learnable 
   rbasis_.meta["radial_scaling"] = radial_scaling
   ps_rb = initialparameters(rng, rbasis_)
   rbasis = splinify(rbasis_, ps_rb)

   # construct a pair basis if needed 
   if pair_basis == :auto
      @assert pair_maxn isa Integer 

      pair_basis = ace_learnable_Rnlrzz(; 
               elements = rbasis._i2z, 
               level = TotalDegree(), 
               max_level = pair_maxn, 
               maxl = 0, 
               maxn = pair_maxn, 
               rin0cuts = rbasis.rin0cuts,
               transforms = pair_transform, 
               envelopes = :poly1sr )

      pair_basis.meta["Winit"] = init_Wpair 

      if !pair_learnable
         ps_pair = initialparameters(rng, pair_basis)
         pair_basis = splinify(pair_basis, ps_pair)
      end
   end

   # this is a naive spec generator and the AA_spec is therefore much 
   # too big. But it will be sparsified in the symmetrization step 
   AA_spec = sym_trace_spec(; order = order, r_spec = rbasis.spec, 
                              level = level, max_level = max_level)

   model = ace_model(rbasis, Ytype, AA_spec, level, pair_basis, E0s)
   model.meta["init_WB"] = String(init_WB)
   model.meta["init_Wpair"] = String(init_Wpair)
   model.meta["model_type"] = "trace"
   model.meta["tensor_format"] = "symcp"

   return model
end


# generate the list of all ll tuples 
function _generate_LL(maxl::Integer, N::Integer, lvl_N::Integer, level = ll -> sum(ll)) 
   LL_max = CartesianIndices( ntuple(_ -> 0:maxl, N) ) 
   LL1 = [ ll.I for ll in LL_max ] 
   LL2 = unique([ sort([ l for l in ll])  for ll in LL1 ])
   LL3 = filter( ll -> (level(ll) <= lvl_N) && iseven(sum(ll)), LL2 ) 
   return sort(LL3; by = level)
end

function _generate_LL(maxl::Integer, order::Integer, 
                      lvls::Union{AbstractVector, Tuple}, 
                      level = ll -> sum(ll)) 
   return vcat([ _generate_LL(maxl, N, lvls[N], level) 
                 for N = 1:order ]... )
end

function _mrange(ll) 
   MM1 = CartesianIndices( ntuple(i -> -ll[i]:ll[i], length(ll)) )
   MM2 = [ mm.I for mm in MM1 ] 
   # MM3 = filter( mm -> sum(mm) == 0, MM2 ) 
   return MM2
end


function sym_trace_spec(; order = nothing, 
                          r_spec = nothing, 
                          max_level = nothing, 
                          level = TraceLevel(0), )

   # convert the max_level to a list 
   if max_level isa Number 
      max_levels = fill(max_level, order)
   else
      max_levels = max_level
   end

   # extract some info from the r basis 
   maxn = maximum( b.n for b in r_spec )
   maxl = maximum( b.l for b in r_spec )

   # get all possible ll tuples 
   LL = _generate_LL(maxl, order, max_levels, level)

   NT_NLM = NamedTuple{(:n, :l, :m), Tuple{Int, Int, Int}}
   AA_spec = Vector{NT_NLM}[] 
   
   for ll in LL 
      for mm in _mrange(ll) 
         for n = 1:maxn
            bb = [ (n = n, l = ll[i], m = mm[i], ) for i = 1:length(ll) ]
            push!(AA_spec, bb)
         end
      end 
   end

   return AA_spec
end
