
export acemodel, acefit!

import JuLIP: energy, forces, virial, cutoff
import ACE1.Utils: get_maxn

_mean(x) = sum(x) / length(x)


"""
`struct ACE1Model` : this specifies an affine `ACE1.jl` model via a basis, the 
parameters and a reference potential `Vref`. The generic constructor is 
```
ACE1Model(basis, params, Vref) 
``` 
where `basis` is typically a `ACE1.RPIBasis` or a `JuLIP.MLIPs.IPSuperBasis` 
object, `params` is a vector of model parameters and `Vref` a reference 
potential, typically a `JuLIP.OneBody`. Setting new parameters is done 
via `ACE1pack.set_params!`. 

A convenience constructor that exposes the most commonly used options to 
construct ACE1 models is provided by `ACE1pack.acemodel`. 
"""
mutable struct ACE1Model 
   basis 
   params 
   Vref  
   potential
end

# TODO: 
# - some of this replicates functionality from ACE1.jl and we should 
#   consider having it in just one of the two places. 
# - better defaults for transform
# - more documentation 

"""
required parameters: 
* `species`
* `maxdeg`
* `rcut`
Recommended parameters: 
* `maxdeg2`
* `rcut2`
* `Eref`

The pair basis is activated by either providing `maxdeg2` or `rbasis2`. 
The reference potential is activated by either providing `Eref` or `Vref`.
"""
function acemodel(;  species = nothing, 
                     N = nothing, 
                     # default transform parameters
                     r0 = _mean( rnn.(species) ),
                     trans = PolyTransform(2, r0),
                     # degree parameters
                     wL = 1.5, 
                     D = SparsePSHDegree(; wL = wL),
                     maxdeg = nothing,
                     # radial basis parameters
                     rcut = nothing,
                     rin = 0.5 * r0,
                     pcut = 2,
                     pin = 2,
                     constants = false,
                     rbasis = nothing,
                     # pair basis parameters 
                     trans2 = PolyTransform(2, r0),
                     maxdeg2 = nothing, 
                     rcut2 = rcut, 
                     rin2 = 0.0, 
                     pin2 = 0, 
                     pcut2 = 2, 
                     rbasis2 = nothing, 
                     # reference energies 
                     Eref = nothing, 
                     Vref = nothing, 
                     # .... more stuff  
                     warn = true, 
                     Basis1p = BasicPSH1pBasis
                    )
   if rbasis == nothing    
      if (pcut < 2) && warn 
         @warn("`pcut` should normally be ≥ 2.")
      end
      if (pin < 2) && (pin != 0) && warn 
         @warn("`pin` should normally be ≥ 2 or 0.")
      end

      rbasis = ACE1.transformed_jacobi(get_maxn(D, maxdeg, species), trans, rcut, rin;
                                    pcut=pcut, pin=pin)
   end

   if rbasis2 == nothing && maxdeg2 != nothing 
      if (pcut2 < 2) && warn 
         @warn("`pcut2` should normally be ≥ 2.")
      end
      if (pin2 != 0)  && warn 
         @warn("`pin2` should normally be equal to 0 to allow repulsion.")
      end
      rbasis2 = ACE1.transformed_jacobi(maxdeg2, trans2, rcut2, rin2; pcut=pcut2, pin=pin2)
   end

   if Vref == nothing && Eref != nothing 
      Vref = JuLIP.OneBody(Eref...)
   end

   # construct the many-body basis   
   basis1p = Basis1p(rbasis; species = species, D = D)   
   ace_basis = RPIBasis(basis1p, N, D, maxdeg, constants)

   # construct the pair basis and combined basis if needed
   if rbasis2 != nothing 
      pair_basis = PolyPairBasis(rbasis2, species)
      basis = JuLIP.MLIPs.IPSuperBasis(ace_basis, pair_basis)
   else
      basis = ace_basis
   end

   # construct a model without parameters and without an evaluator 
   model = ACE1Model(basis, nothing, Vref, nothing)

   # set some random parameters -> this will also generate the evaluator 
   params = randn(length(basis))
   params = params ./ (1:length(basis)).^4
   set_params!(model, params)
   return model 
end


_sumip(pot1, pot2) = 
      JuLIP.MLIPs.SumIP([pot1, pot2])

_sumip(pot1::JuLIP.MLIPs.SumIP, pot2) = 
      JuLIP.MLIPs.SumIP([pot1.components..., pot2])
                                      
function set_params!(model, params)
   model.params = params
   model.potential = JuLIP.MLIPs.combine(model.basis, model.params)
   if model.Vref != nothing 
      model.potential = _sumip(model.potential, model.Vref)
   end
   return model 
end


function acefit!(model, raw_data, solver; 
             weights = Dict("default"=>Dict("E"=>30.0, "F"=>1.0, "V"=>1.0)),
             energy_key = "energy", 
             force_key = "force", 
             virial_key = "virial")
   data = [ AtomsData(at, energy_key, force_key, virial_key, 
                      weights, model.Vref) for at in raw_data ] 
   result = ACEfit.linear_fit(data, model.basis, solver) 
   set_params!(model, result["C"])
   return model 
end