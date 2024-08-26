# This module is a translation of most ACE1x options to the new ACE 
# kernels. It is used to provide compatibility with the old ACE1 / ACE1x. 

module ACE1compat

using NamedTupleTools, StaticArrays, Unitful
import ACEpotentials: DefaultHypers, Models 

using ACEpotentials.Models: agnesi_transform, 
                            SplineRnlrzzBasis, 
                            ace_learnable_Rnlrzz

ace1_defaults() = deepcopy(_kw_defaults)

const _kw_defaults = Dict(:elements => nothing,
                          :order => nothing,
                          :totaldegree => nothing,
                          :wL => 1.5,
                          #
                          :rin => 0.0,
                          :r0 => :bondlen,
                          :rcut => (:bondlen, 2.5),
                          :transform => (:agnesi, 2, 4),
                          :envelope => (:x, 2, 2),
                          :rbasis => :legendre,
                          #
                          :pure2b => false,   # TODO: ALLOW TRUE!!!
                          :delete2b => false, #       here too
                          :pure => false,
                          #
                          :pair_rin => :rin,
                          :pair_rcut => :rcut,
                          :pair_degree => :totaldegree,
                          :pair_transform => (:agnesi, 1, 3),
                          :pair_basis => :legendre,
                          :pair_envelope => (:r, 2),
                          #
                          :Eref => missing,
                          #
                          :variable_cutoffs => false,
                          )

const _kw_aliases = Dict( :N => :order,
                          :species => :elements,
                          :trans => :transform,
                        )


function _clean_args(kwargs)
   dargs = Dict{Symbol, Any}()
   for key in keys(kwargs)
      if haskey(_kw_aliases, key)
         dargs[_kw_aliases[key]] = kwargs[key]
      else
         dargs[key] = kwargs[key]
      end
   end
   for key in keys(_kw_defaults)
      if !haskey(dargs, key)
         dargs[key] = _kw_defaults[key]
      end
   end

   if dargs[:pair_rcut] == :rcut
      dargs[:pair_rcut] = dargs[:rcut]
   end

   if kwargs[:pure2b] || kwargs[:pure]
      error("ACE1compat current does not support `pure2b` or `pure` options.")
   end

   return namedtuple(dargs)
end

function _get_order(kwargs)
   if haskey(kwargs, :order)
      return kwargs[:order]
   elseif haskey(kwargs, :bodyorder)
      return kwargs[:bodyorder] - 1
   end
   error("Cannot determine correlation order or body order of ACE basis from the arguments provided.")
end

function _get_degrees(kwargs)

   if haskey(kwargs, :totaldegree)
      deg = kwargs[:totaldegree]
      cor_order = _get_order(kwargs)

      if deg isa Number 
         maxlevels = [deg for i in 1:cor_order]
      elseif deg isa AbstractVector{<: Number}
         @assert length(deg) == cor_order
         maxlevels = deg
      else
         error("Cannot determine total degree of ACE basis from the arguments provided.")
      end

      wL = kwargs[:wL]
      NZ = length(_get_elements(kwargs))

      return Models.TotalDegree(1.0*NZ, 1/wL), maxlevels
   end

   error("Cannot determine total degree of ACE basis from the arguments provided.")
end

function _get_r0(kwargs, z1, z2)
   if kwargs[:r0] == :bondlen
      return DefaultHypers.bond_len(z1, z2)
   elseif kwargs[:r0] isa Number
      return kwargs[:r0]
   elseif kwargs[:r0] isa Dict
      return kwargs[:r0][(z1, z2)]
   end
   error("Unable to determine r0($z1, $z2) from the arguments provided.")
end

function _get_elements(kwargs)
   return [ kwargs[:elements]... ]
end

function _get_all_r0(kwargs)
   elements = _get_elements(kwargs) 
   r0 = Dict( [ (s1, s2) => _get_r0(kwargs, s1, s2)
                   for s1 in elements, s2 in elements]... )
end

function _get_rcut(kwargs, s1, s2; _rcut = kwargs[:rcut])
   if _rcut isa Tuple  
      if _rcut[1] == :bondlen   # rcut = (:bondlen, rcut_factor)
         return _rcut[2] * _get_r0(kwargs, s1, s2)
      end
   elseif _rcut isa Number   # rcut = explicit value 
      return _rcut
   elseif _rcut isa Dict     # explicit values for each pair 
      return _rcut[(s1, s2)]
   end
   error("Unable to determine rcut($s1, $s2) from the arguments provided.")
end

function _get_all_rcut(kwargs; _rcut = kwargs[:rcut])
   if _rcut isa Number
      return _rcut
   end
   elements = _get_elements(kwargs) 
   rcut = Dict( [ (s1, s2) => _get_rcut(kwargs, s1, s2; _rcut = _rcut)
                   for s1 in elements, s2 in elements]... )
   if !kwargs[:variable_cutoffs]
      rcut = maximum(values(rcut))
   end        
   return rcut
end


function _rin0cuts_rcut(zlist, cutoffs::Dict)
   function rin0cut(zi, zj) 
      r0 = DefaultHypers.bond_len(zi, zj)
      rin, rcut = cutoffs[zi, zj]
      return (rin = rin, r0 = r0, rcut = rcut)
   end
   NZ = length(zlist)
   return SMatrix{NZ, NZ}([ rin0cut(zi, zj) for zi in zlist, zj in zlist ])
end


function _ace1_rin0cuts(kwargs; rcutkey = :rcut) 
   elements = _get_elements(kwargs) 
   rcut = _get_all_rcut(kwargs; _rcut = kwargs[rcutkey])
   if rcut isa Number 
      cutoffs = Dict([ (s1, s2) => (0.0, rcut) for s1 in elements, s2 in elements]...)
   else
      cutoffs = Dict([ (s1, s2) => (0.0, rcut[(s1, s2)]) for s1 in elements, s2 in elements]...)
   end
   # rcut = maximum(values(rcut))  # multitransform wants a single cutoff.

   # construct the rin0cut structures 
   rin0cuts = _rin0cuts_rcut(elements, cutoffs)
end


function _transform(kwargs; transform = kwargs[:transform], 
                            rcutkey = :rcut)
   elements = _get_elements(kwargs) 

   if transform isa Tuple
      if transform[1] == :agnesi
         if length(transform) != 3
            error("The ACE1 compatibility only supports (:agnesi, p, q) type transforms.")
         end
         p = transform[2]
         q = transform[3]
         rin0cuts = _ace1_rin0cuts(kwargs; rcutkey = rcutkey)
         transforms = agnesi_transform.(rin0cuts, p, q)
         return transforms 
         
         # transforms = Dict([ (s1, s2) => agnesi_transform(r0[(s1, s2)], p, q)
         #                    for s1 in elements, s2 in elements]... )
         # trans_ace = multitransform(transforms; rin = 0.0, rcut = rcut, cutoffs=cutoffs)
         # return trans_ace
       end
   end

   error("Unable to determine transform from the arguments provided.")
end


function _get_Rnl_spec(kwargs, 
                     maxdeg = maximum(kwargs[:totaldegree]) )
   wL = kwargs[:wL] 
   NZ = length(_get_elements(kwargs))
   lvl = Models.TotalDegree(1.0*NZ, 1/wL)
   return Models.oneparticle_spec(lvl, maxdeg)   
end


function _radial_basis(kwargs)
   rbasis = kwargs[:rbasis]
   elements = _get_elements(kwargs)

   if rbasis isa SplineRnlrzzBasis
      return rbasis

   elseif rbasis == :legendre

      trans_ace = _transform(kwargs)
      rin0cuts = _ace1_rin0cuts(kwargs)
      Rnl_spec = _get_Rnl_spec(kwargs)

      envelope = kwargs[:envelope]
      # this is the default envelope
      # envelopes = PolyEnvelope2sX(-1.0, 1.0, 2, 2)
      # just check that it hasn't been changed 
      if envelope != (:x, 2, 2)
         error("The ACE1 compatibility only supports (:x, 2, 2) type envelopes for the radial basis")
      end

      # finally we need to specify a polynomial basis. ACE1 incorporates 
      # the envelope into the orthogonality. This corresponds to 
      #  ∫ Pq(x) Pq(x) env(x)^2 dx = δ_{pq}
      # which results in a Jacobi basis 
      pin = envelope[2]
      pcut = envelope[3]
      polys = (:jacobi, Float64(2*pin), Float64(2*pcut))


      # This is to be revisited if we re-introduce pure2b
      # if envelope isa Tuple && envelope[1] == :x
      #    if (kwargs[:pure2b] || kwargs[:pure])
      #       maxn += (pin + pcut) * (cor_order-1)
      #    end
      # else
      #    error("Cannot construct the radial basis automatically without knowing the envelope.")
      # end
      # Rn_basis = transformed_jacobi(maxn, trans_ace; pcut = pcut, pin = pin)

      Rn_basis = ace_learnable_Rnlrzz(; spec = Rnl_spec, 
                                        maxq = maximum(b.n for b in Rnl_spec),  
                                        elements = elements, 
                                        rin0cuts = rin0cuts,
                                        transforms = trans_ace, 
                                        polys = polys, 
                                        Winit = :onehot)

      ps_Rn = Models.initialparameters(nothing, Rn_basis)
      Rn_spl = Models.splinify(Rn_basis, ps_Rn)
      return Rn_spl 
   end

   error("Unable to determine the radial basis from the arguments provided.")
end


function _pair_basis(kwargs)
   rbasis = kwargs[:pair_basis]
   elements = _get_elements(kwargs) 
   NZ = length(elements)
   rin0cuts = _ace1_rin0cuts(kwargs; rcutkey = :pair_rcut)

   if rbasis == :legendre

      # SPECIFICATION 
      if kwargs[:pair_degree] == :totaldegree
         maxq = ceil(Int, maximum(kwargs[:totaldegree]))
         maxn = maxq * NZ 
      elseif kwargs[:pair_degree] isa Integer
         maxq = ceil(Int, kwargs[:pair_degree])
         maxn = maxq * NZ 
      else
         error("Cannot determine `maxn` for pair basis from information provided.")
      end
      pair_spec = [ (n = n, l = 0) for n in 1:maxn ]

      # TRANSFORM 
      trans_pair = _transform(kwargs, transform = kwargs[:pair_transform], 
                                      rcutkey = :pair_rcut)

      # ENVELOPE 
      # here we use a similar convention, just need to convert to ace1-style 
      envelope = kwargs[:pair_envelope]
      if envelope isa Tuple && envelope[1] == :r 
         envelope = (:r_ace1, envelope[2])
      end 
      
      pair_basis = ace_learnable_Rnlrzz(; spec = pair_spec, 
                                    maxq = maxq,  
                                    elements = elements, 
                                    rin0cuts = rin0cuts,
                                    transforms = trans_pair, 
                                    envelopes = envelope, 
                                    polys = :legendre, 
                                    Winit = :onehot ) 
      ps_pair = Models.initialparameters(nothing, pair_basis)
      pair_spl = Models.splinify(pair_basis, ps_pair)
      return pair_spl
   end
   
   error("Cannot determine the pair basis from the arguments provided.")
end


function ace1_model(; kwargs...)

   kwargs = _clean_args(kwargs)

   elements = _get_elements(kwargs) 
   cor_order = _get_order(kwargs)
   rbasis = _radial_basis(kwargs)
   pairbasis = _pair_basis(kwargs)
   lvl, maxlvl = _get_degrees(kwargs)

   # if pure2b && kwargs[:pure]
      # error("Cannot use both `pure2b` and `pure` options.")
      # @info("Option `pure = true` overrides `pure2b=true`")
      # pure2b = false
   # end

   if kwargs[:pure2b] || kwargs[:pure]
      error("ACE1compat does not yet support the `pure2b` or `pure` options.")
   end


   # if pure2b
   #    rpibasis = Pure2b.pure2b_basis(species = AtomicNumber.(elements),
   #                            Rn=rbasis,
   #                            D=Deg,
   #                            maxdeg=maxdeg,
   #                            order=cor_order,
   #                            delete2b = kwargs[:delete2b])
   # elseif kwargs[:pure]
   #    dirtybasis = ACE1.ace_basis(species = AtomicNumber.(elements),
   #                             rbasis=rbasis,
   #                             D=Deg,
   #                             maxdeg=maxdeg,
   #                             N = cor_order, )
   #    _rem = kwargs[:delete2b] ? 1 : 0
   #    # remove all zero-basis functions that we might have accidentally created so that we purify less extra basis
   #    dirtybasis = ACE1.RPI.remove_zeros(dirtybasis)
   #    # and finally cleanup the rest of the basis 
   #    dirtybasis = ACE1._cleanup(dirtybasis)
   #    # finally purify
   #    rpibasis = ACE1x.Purify.pureRPIBasis(dirtybasis; remove = _rem)
   # else

   Eref = kwargs[:Eref]
   if ismissing(Eref) 
      E0s = nothing
   else 
      E0s = Dict([ key => val * u"eV" for (key, val) in Eref]...) 
   end

   model = Models.ace_model(; elements=elements, 
                       order = cor_order, 
                       Ytype = :spherical, 
                       E0s = E0s, 
                       rbasis = rbasis, 
                       pair_basis = pairbasis, 
                       rin0cuts = rbasis.rin0cuts, 
                       level = lvl, 
                       max_level = maxlvl,
                       init_WB = :zeros,)

   return model 
end



end 

