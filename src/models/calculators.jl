
import EmpiricalPotentials 
import EmpiricalPotentials: SitePotential, 
                            cutoff_radius, 
                            eval_site, 
                            eval_grad_site, 
                            site_virial, 
                            PairList, 
                            get_neighbours, 
                            atomic_number

import AtomsCalculators
import AtomsCalculators: energy_forces_virial

using Folds, ChunkSplitters, Unitful, NeighbourLists

using ComponentArrays: ComponentArray

using ObjectPools: release! 

struct ACEPotential{MOD} <: SitePotential
   model::MOD
end

# TODO: allow user to specify what units the model is working with

energy_unit(::ACEPotential) = 1.0u"eV"
distance_unit(::ACEPotential) = 1.0u"Å"
force_unit(V) = energy_unit(V) / distance_unit(V)
Base.zero(V::ACEPotential) =  zero(energy_unit(V))


# --------------------------------------------------------------- 
#   EmpiricalPotentials / SitePotential based implementation 
#
#   this currently doesn't know how to handle ps and st 
#   it assumes implicitly without checking that the model is 
#   storing its own parameters. 

cutoff_radius(V::ACEPotential{<: ACEModel}) = 
      maximum(x.rcut for x in V.model.rbasis.rin0cuts) * distance_unit(V)

eval_site(V::ACEPotential{<: ACEModel}, Rs, Zs, z0) = 
      evaluate(V.model, Rs, Zs, z0) * energy_unit(V)

eval_grad_site(V::ACEPotential{<: ACEModel}, Rs, Zs, z0) = 
      evaluate_ed(V.model, Rs, Zs, z0) * force_unit(V)


# --------------------------------------------------------------- 
#   manual implementation allowing parameters  
#   but basically copied from the EmpiricalPotentials implementation 

import JuLIP
import AtomsBase

AtomsBase.atomic_number(at::JuLIP.Atoms, iat::Integer) = at.Z[iat]

function energy_forces_virial(
         at, V::ACEPotential{<: ACEModel}, ps, st;
         domain   = 1:length(at), 
         executor = ThreadedEx(),
         ntasks   = Threads.nthreads(),
         nlist    = JuLIP.neighbourlist(at, cutoff_radius(V)/distance_unit(V)),
         kwargs...
         )

   T = fl_type(V.model) # this is ACE specific 
   init_e() = zero(T) * energy_unit(V)
   init_f() = zeros(SVector{3, T}, length(at)) * force_unit(V)
   init_v() = zero(SMatrix{3, 3, T}) * energy_unit(V)

   # TODO: each task needs its own state if that is where  
   #       the temporary arrays will be stored? 
   #       but if we use bumper then there is no issue

         
   E_F_V = Folds.sum(collect(chunks(domain, ntasks)), 
                     executor;
                     init = [init_e(), init_f(), init_v()],
                     ) do (sub_domain, _)

      energy = init_e()
      forces = init_f()
      virial = init_v()

      for i in sub_domain
         Js, Rs, Zs, z0 = get_neighbours(at, V, nlist, i) 
         v, dv, st = evaluate_ed(V.model, Rs, Zs, z0, ps, st)
         energy += v * energy_unit(V)
         forces[Js] -= (dv * force_unit(V))
         forces[i]  += sum(dv) * force_unit(V)
         virial += JuLIP.Potentials.site_virial(dv, Rs) * energy_unit(V)
         release!(Js); release!(Rs); release!(Zs)
      end
      [energy, forces, virial]
   end
   return (energy = E_F_V[1], forces = E_F_V[2], virial = E_F_V[3])
end
