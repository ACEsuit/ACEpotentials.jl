# This module is a translation of most ACE1x options to the new ACE 
# kernels. It is used to provide compatibility with the old ACE1 / ACE1x. 

module ACE1compat

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
                          #temporary variable to specify whether using the variable cutoffs or not
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
      basis_selector = BasisSelector(cor_order, maxlevels, 
                                     TotalDegree(1.0, wL))
      maxn = maximum(maxlevels) 

      # return basis_selector, maxdeg, maxn      
      return basis_selector 
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
         return _rcut[2] * get_r0(s1, s2)
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
   return rcut
end


function _rin0cuts_rcut(zlist, cutoffs::Dict)
   function rin0cut(zi, zj) 
      r0 = DefaultHypers.bond_len(zi, zj)
      return (rin = 0.0, r0 = r0, rcut = cutoffs[zi, zj])
   end
   NZ = length(zlist)
   return SMatrix{NZ, NZ}([ rin0cut(zi, zj) for zi in zlist, zj in zlist ])
end



function _transform(kwargs; transform = kwargs[:transform])
   elements = _get_elements(kwargs) 

   if transform isa Tuple
      if transform[1] == :agnesi
         if length(transform) != 3
            error("The ACE1 compatibility only supports (:agnesi, p, q) type transforms.")
         end

         p = transform[2]
         q = transform[3]
         r0 = _get_all_r0(kwargs)
         rcut = _get_all_rcut(kwargs)
         if rcut isa Number || ! kwargs[:variable_cutoffs]
            cutoffs = nothing
         else
            cutoffs = Dict([ (s1, s2) => (0.0, rcut[(s1, s2)]) for s1 in elements, s2 in elements]...)
         end
         # rcut = maximum(values(rcut))  # multitransform wants a single cutoff.

         # construct the rin0cut structures 
         rin0cuts = _rin0cuts_rcut(elements, cutoffs)
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

#=

function _radial_basis(kwargs)
   rbasis = kwargs[:rbasis]

   if rbasis isa ACE1.ScalarBasis
      return rbasis

   elseif rbasis == :legendre
      Deg, maxdeg, maxn = _get_degrees(kwargs)
      cor_order = _get_order(kwargs)
      envelope = kwargs[:envelope]
      if envelope isa Tuple && envelope[1] == :x
         pin = envelope[2]
         pcut = envelope[3]
         if (kwargs[:pure2b] || kwargs[:pure])
            maxn += (pin + pcut) * (cor_order-1)
         end
      else
         error("Cannot construct the radial basis automatically without knowing the envelope.")
      end

      trans_ace = _transform(kwargs)

      Rn_basis = transformed_jacobi(maxn, trans_ace; pcut = pcut, pin = pin)
      # println("pcut is", pcut, "pin is", pin, "trans_ace is", trans_ace)
      # println(kwargs)
      #Rn_basis = transformed_jacobi(maxn, trans_ace, kwargs[:rcut], kwargs[:rin];)
      return Rn_basis
   end

   error("Unable to determine the radial basis from the arguments provided.")
end




function _pair_basis(kwargs)
   rbasis = kwargs[:pair_basis]
   elements = _get_elements(kwargs) 
   #elements has to be sorted becuase PolyPairBasis (see end of function) assumes sorted.
   if kwargs[:variable_cutoffs]
      elements = [chemical_symbol(z) for z in JuLIP.Potentials.ZList(elements, static=true).list]
   end

   if rbasis isa ACE1.ScalarBasis
      return rbasis

   elseif rbasis == :legendre
      if kwargs[:pair_degree] == :totaldegree
         Deg, maxdeg, maxn = _get_degrees(kwargs)
      elseif kwargs[:pair_degree] isa Integer
         maxn = kwargs[:pair_degree]
      else
         error("Cannot determine `maxn` for pair basis from information provided.")
      end

      allrcut = _get_all_rcut(kwargs; _rcut = kwargs[:pair_rcut])
      if allrcut isa Number
         allrcut = Dict([(s1, s2) => allrcut for s1 in elements, s2 in elements]...)
      end

      trans_pair = _transform(kwargs, transform = kwargs[:pair_transform])
      _s2i(s) = z2i(trans_pair.zlist, AtomicNumber(s))
      alltrans = Dict([(s1, s2) => trans_pair.transforms[_s2i(s1), _s2i(s2)].t
                       for s1 in elements, s2 in elements]...)

      allr0 = _get_all_r0(kwargs)

      function _r_basis(s1, s2, penv)
         _env = ACE1.PolyEnvelope(penv, allr0[(s1, s2)], allrcut[(s1, s2)] )
         return transformed_jacobi_env(maxn, alltrans[(s1, s2)], _env, allrcut[(s1, s2)])
      end

      _x_basis(s1, s2, pin, pcut)  = transformed_jacobi(maxn, alltrans[(s1, s2)], allrcut[(s1, s2)];
                                             pcut = pcut, pin = pin)

      envelope = kwargs[:pair_envelope]
      if envelope isa Tuple
         if envelope[1] == :x
            pin = envelope[2]
            pcut = envelope[3]
            rbases = [ _x_basis(s1, s2, pin, pcut) for s1 in elements, s2 in elements ]
         elseif envelope[1] == :r
            penv = envelope[2]
            rbases = [ _r_basis(s1, s2, penv) for s1 in elements, s2 in elements ]
         end
      end
   end

   return PolyPairBasis(rbases, elements)
end



function mb_ace_basis(kwargs)
   elements = _get_elements(kwargs) 
   cor_order = _get_order(kwargs)
   Deg, maxdeg, maxn = _get_degrees(kwargs)
   rbasis = _radial_basis(kwargs)
   pure2b = kwargs[:pure2b]

   if pure2b && kwargs[:pure]
      # error("Cannot use both `pure2b` and `pure` options.")
      @info("Option `pure = true` overrides `pure2b=true`")
      pure2b = false
   end

   if pure2b
      rpibasis = Pure2b.pure2b_basis(species = AtomicNumber.(elements),
                              Rn=rbasis,
                              D=Deg,
                              maxdeg=maxdeg,
                              order=cor_order,
                              delete2b = kwargs[:delete2b])
   elseif kwargs[:pure]
      dirtybasis = ACE1.ace_basis(species = AtomicNumber.(elements),
                               rbasis=rbasis,
                               D=Deg,
                               maxdeg=maxdeg,
                               N = cor_order, )
      _rem = kwargs[:delete2b] ? 1 : 0
      # remove all zero-basis functions that we might have accidentally created so that we purify less extra basis
      dirtybasis = ACE1.RPI.remove_zeros(dirtybasis)
      # and finally cleanup the rest of the basis 
      dirtybasis = ACE1._cleanup(dirtybasis)
      # finally purify
      rpibasis = ACE1x.Purify.pureRPIBasis(dirtybasis; remove = _rem)
   else
      rpibasis = ACE1.ace_basis(species = AtomicNumber.(elements),
                               rbasis=rbasis,
                               D=Deg,
                               maxdeg=maxdeg,
                               N = cor_order, )
   end

   return rpibasis
end

function ace_basis(; kwargs...)
   kwargs = _clean_args(kwargs)
   rpiB = mb_ace_basis(kwargs)
   pairB = _pair_basis(kwargs)
   return JuLIP.MLIPs.IPSuperBasis([pairB, rpiB]);
end
=#

end 