import Random
import ACEpotentials: DefaultHypers
import EmpiricalPotentials 


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
               maxq_fact = 1.5, 
               maxq = :auto, 
               elements = nothing, 
               spec = nothing, 
               rin0cuts = _default_rin0cuts(elements),
               transforms = agnesi_transform.(rin0cuts, 2, 2), 
               polys = :legendre, 
               envelopes = :poly2sx, 
               Winit = :glorot_normal, 
               )
   if elements == nothing
      error("elements must be specified!")
   end
   if (spec == nothing) && (level == nothing || max_level == nothing)
      error("Must specify either `spec` or `level, max_level`!")
   end

   zlist =_convert_zlist(elements)
   NZ = length(zlist)

   if spec == nothing
      _max_lvl = maximum(max_level)
      spec = [ (n = n, l = l) for n = 1:maxn, l = 0:maxl
                              if level((n = n, l = l)) <= _max_lvl ]
   end

   # now the actual maxn is the maximum n in the spec
   actual_maxn = maximum([ s.n for s in spec ])
   
   if maxq == :auto 
      maxq = ceil(Int, actual_maxn * maxq_fact)
   end 

   if maxq < actual_maxn / NZ 
      @warn("maxq < actual_maxn / NZ; likely linear dependence")
   end 

   if polys isa Symbol 
      if polys == :legendre
         polys = Polynomials4ML.legendre_basis(ceil(Int, maxq)) 
      else
         error("unknown polynomial type : $polys")
      end
   elseif polys isa Tuple 
      if polys[1] == :jacobi 
         α = polys[2]
         β = polys[3]
         polys = Polynomials4ML.jacobi_basis(ceil(Int, maxq), α, β)
      else
         error("unknown polynomial type : $polys")
      end
   end

   if transforms isa Tuple && transforms[1] == :agnesi 
      p = transforms[2] 
      q = transforms[3]
      transforms = agnesi_transform.(rin0cuts, p, q)
   end

   if envelopes == :poly2sx
      envelopes = PolyEnvelope2sX(-1.0, 1.0, 2, 2)
   elseif envelopes == :poly1sr
      envelopes = [ PolyEnvelope1sR(rin0cuts[iz, jz].rcut, 1) 
                    for iz = 1:NZ, jz = 1:NZ ]
   elseif envelopes isa Tuple && envelopes[1] == :x 
      @assert length(envelopes) == 3 
      envelopes = PolyEnvelope2sX(-1.0, 1.0, envelopes[2], envelopes[3])
   elseif envelopes isa Tuple && envelopes[1] == :r 
      envelopes = [ PolyEnvelope1sR(rin0cuts[iz, jz].rcut, envelopes[2]) 
                     for iz = 1:NZ, jz = 1:NZ ]
   elseif envelopes isa Tuple && envelopes[1] == :r_ace1
      envelopes = [ ACE1_PolyEnvelope1sR(rin0cuts[iz, jz].rcut, rin0cuts[iz, jz].r0, envelopes[2])
                     for iz = 1:NZ, jz = 1:NZ ]
   else
      error("cannot read envelope : $envelopes")
   end

   if actual_maxn > length(polys) * NZ 
      @warn("actual_maxn/NZ > maxq; likely linear dependence")
   end

   return LearnableRnlrzzBasis(zlist, polys, transforms, envelopes, 
                               rin0cuts, spec; 
                               Winit = Winit)
end 



_convert_E0s(E0s::Union{Dict, NamedTuple}) = E0s 
_convert_E0s(E0s::Union{AbstractVector, Tuple}) = Dict(E0s...)
_convert_E0s(E0s) = error("E0s must be nothing, a NamedTuple, Dict or list of pairs")

# E0s can be anything with (key, value) pairs 
_make_Vref_E0s(elements, E0s) = OneBody(_convert_E0s(E0s))

function _make_Vref_E0s(elements, E0s::Nothing)
   NZ = length(elements)
   zz = _convert_zlist(elements)
   return _make_Vref_E0s(elements, [ z => 0.0 for z in zz ] )
end


function _make_Vref(elements, E0s, ZBL, rcut = nothing)

   if !(isnothing(E0s)) 
      E0s = _convert_E0s(E0s) 
      if string.((sort([elements...]))) != string.(sort(collect(keys(E0s))))
         @error("E0s keys must be the same as the list of elements")
         display(E0s) 
         display(elements)
         error("")
      end
   end 

   Vref_E0s = _make_Vref_E0s(elements, E0s)

   if ZBL 
      Vref_zbl = EmpiricalPotentials.ZBL(rcut*u"Å")
      return SitePotentialStack((Vref_E0s, Vref_zbl))
   else
      return Vref_E0s
   end
end



function ace_model(; elements = nothing, 
                     order = nothing, 
                     Ytype = :solid,  
                     ZBL = false, 
                     E0s = nothing,
                     rin0cuts = :auto,
                     # radial basis 
                     rbasis = nothing, 
                     rbasis_type = :learnable, 
                     maxl = 30, # maxl, max are fairly high defaults 
                     maxn = 50, # that we will likely never reach 
                     maxq_fact = 1.5, 
                     maxq = :auto, 
                     init_Wradial = :glorot_normal, 
                     # basis size parameters 
                     level = nothing, 
                     max_level = nothing, 
                     init_WB = :zeros, 
                     # pair basis 
                     pair_maxn = nothing, 
                     pair_basis = :auto, 
                     pair_learnable = false, 
                     pair_transform = (:agnesi, 1, 4), 
                     init_Wpair = :onehot, 
                     rng = Random.default_rng(), 
                     )

   if rin0cuts == :auto
      rin0cuts = _default_rin0cuts(elements)
   else
      NZ = length(elements)
      @assert rin0cuts isa SMatrix && size(rin0cuts) == (NZ, NZ)
   end

   # construct an rbasis if needed
   if isnothing(rbasis)
      if rbasis_type == :learnable
         rbasis = ace_learnable_Rnlrzz(; max_level = max_level, level = level, 
                                         maxl = maxl, maxn = maxn, 
                                         maxq_fact = maxq_fact, maxq = maxq, 
                                         elements = elements, 
                                         rin0cuts = rin0cuts, 
                                         Winit = init_Wradial)
      else
         error("unknown rbasis_type = $rbasis_type")
      end
   end

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

   AA_spec = sparse_AA_spec(; order = order, r_spec = rbasis.spec, 
                              level = level, max_level = max_level)

   rcut = maximum([ x.rcut for x in rin0cuts ])
   Vref = _make_Vref(elements, E0s, ZBL, rcut)

   model = ace_model(rbasis, Ytype, AA_spec, level, pair_basis, Vref)
   model.meta["init_WB"] = String(init_WB)
   model.meta["init_Wpair"] = String(init_Wpair)

   return model 
end


# -------------------------------------------------------


function _default_rin0cuts(zlist; rinfactor = 0.0, rcutfactor = 2.5)
   function rin0cut(zi, zj) 
      r0 = DefaultHypers.bond_len(zi, zj)
      return (rin = r0 * rinfactor, r0 = r0, rcut = r0 * rcutfactor)
   end
   NZ = length(zlist)
   return SMatrix{NZ, NZ}([ rin0cut(zi, zj) for zi in zlist, zj in zlist ])
end

