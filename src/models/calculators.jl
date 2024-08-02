
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

using Folds, ChunkSplitters, Unitful, NeighbourLists, 
      Optimisers, LuxCore, ChainRulesCore 

import ChainRulesCore: rrule, NoTangent, ZeroTangent

mutable struct ACEPotential{MOD} <: SitePotential
   model::MOD
   ps
   st 
end

ACEPotential(model) = ACEPotential(model, nothing, nothing)

# TODO: allow user to specify what units the model is working with

energy_unit(::ACEPotential) = 1.0u"eV"
distance_unit(::ACEPotential) = 1.0u"Å"
force_unit(V) = energy_unit(V) / distance_unit(V)
Base.zero(V::ACEPotential) =  zero(energy_unit(V))

initialparameters(rng::AbstractRNG, V::ACEPotential) = initialparameters(rng, V.model) 
initialstates(rng::AbstractRNG, V::ACEPotential) = initialstates(rng, V.model)

set_parameters!(V::ACEPotential, ps) = (V.ps = ps; V)
set_states!(V::ACEPotential, st) = (V.st = st; V)
set_psst!(V::ACEPotential, ps, st) = (V.ps = ps; V.st = st; V)

splinify(V::ACEPotential) = splinify(V, V.ps) 
splinify(V::ACEPotential, ps) = ACEPotential(splinify(V.model, ps), nothing, nothing) 

function set_parameters!(V::ACEPotential, θ::AbstractVector{<: Number})
   ps_vec, _restruct = destructure(V.ps)
   ps = _restruct(θ)
   return set_parameters!(V, ps)
end

# --------------------------------------------------------------- 
#   EmpiricalPotentials / SitePotential based implementation 
#
#   this currently doesn't know how to handle ps and st 
#   it assumes implicitly without checking that the model is 
#   storing its own parameters. 

cutoff_radius(V::ACEPotential{<: ACEModel}) = 
      maximum(x.rcut for x in V.model.rbasis.rin0cuts) * distance_unit(V)

eval_site(V::ACEPotential{<: ACEModel}, Rs, Zs, z0) = 
      evaluate(V.model, Rs, Zs, z0, V.ps, V.st) 

function eval_grad_site(V::ACEPotential{<: ACEModel}, Rs, Zs, z0) 
   v, dv = evaluate_ed(V.model, Rs, Zs, z0, V.ps, V.st) 
   return v, dv
end


# --------------------------------------------------------------- 
#   manual implementation allowing parameters  
#   but basically copied from the EmpiricalPotentials implementation 

import AtomsBase
using EmpiricalPotentials: PairList, get_neighbours, site_virial

using Unitful: ustrip
_ustrip(x) = ustrip(x)
_ustrip(x::ZeroTangent) = x

_site_virial(dV::AbstractVector{SVector{3, T1}}, 
                  Rs::AbstractVector{SVector{3, T2}}) where {T1, T2} =  
      (
         - sum( dVi * Ri' for (dVi, Ri) in zip(dV, Rs); 
               init = zero(SMatrix{3, 3, promote_type(T1, T2)}) )
      )

function energy_forces_virial_serial(
         at, V::ACEPotential{<: ACEModel}, ps, st;
         domain   = 1:length(at), 
         nlist    = PairList(at, cutoff_radius(V)), 
         )

   T = fl_type(V.model) # this is ACE specific 
   energy = zero(T)
   forces = zeros(SVector{3, T}, length(at))
   virial = zero(SMatrix{3, 3, T})

   for i in domain
      Js, Rs, Zs, z0 = get_neighbours(at, V, nlist, i) 
      v, dv = evaluate_ed(V.model, Rs, Zs, z0, ps, st)
      energy += v
      for α = 1:length(Js) 
         forces[Js[α]] -= dv[α]
         forces[i]     += dv[α]
      end
      virial += _site_virial(dv, Rs)
   end
   return (energy = energy * energy_unit(V), 
           forces = forces * force_unit(V), 
           virial = virial * energy_unit(V) )
end


function energy_forces_virial(
         at, V::ACEPotential{<: ACEModel}, ps, st;
         domain   = 1:length(at), 
         executor = ThreadedEx(),
         ntasks   = Threads.nthreads(),
         nlist    = PairList(at, cutoff_radius(V)), 
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
         v, dv = evaluate_ed(V.model, Rs, Zs, z0, ps, st)
         energy += v * energy_unit(V)
         for α = 1:length(Js) 
            forces[Js[α]] -= dv[α] * force_unit(V)
            forces[i]     += dv[α] * force_unit(V)
         end
         virial += _site_virial(dv, Rs) * energy_unit(V)
      end
      [energy, forces, virial]
   end
   return (energy = E_F_V[1], forces = E_F_V[2], virial = E_F_V[3])
end


function pullback_EFV(Δefv, 
               at, V::ACEPotential{<: ACEModel}, ps, st;
               domain   = 1:length(at), 
               executor = ThreadedEx(),
               ntasks   = Threads.nthreads(),
               nlist    = PairList(at, cutoff_radius(V)), 
               kwargs...
               )

   T = fl_type(V.model) 
   ps_vec, _restruct = destructure(ps)
   TP = promote_type(eltype(ps_vec), T) 

   # We resolve the pullback through the summation-over-sites manually, e.g., 
   # E = ∑_i E_i 
   #     ∂ (Δe * E) = ∑_i ∂( Δe * E_i ) 

   # TODO : There is a lot of ustrip hacking which implicitly 
   #        assumes that the loss is dimensionless and that the 
   #        gradient w.r.t. parameters therefore must also be dimensionless 

   g_vec = Folds.sum(collect(chunks(domain, ntasks)), 
                     executor;
                     init = zeros(TP, length(ps_vec)),
                     ) do (sub_domain, _)

      g_loc = zeros(TP, length(ps_vec))
      for i in sub_domain                     
         Js, Rs, Zs, z0 = get_neighbours(at, V, nlist, i)
                           
         Δei = _ustrip(Δefv.energy)

         # the adjoint for dV needs combination of the virial and forces pullback
         Δdi = [ - _ustrip.(Δefv.virial * rj) for rj in Rs ]
         if eltype(Δdi) != ZeroTangent
            for α = 1:length(Js) 
               # F[Js[α]] -= dV[α], F[i] += dV[α] 
               # ∂_dvj { Δf[Js[α]] * F[Js[α]] } -> 
               Δdi[α] -= _ustrip.( Δefv.forces[Js[α]] )
               Δdi[α] += _ustrip.( Δefv.forces[i]     )
            end
         end

         # now we can apply the pullback through evaluate_ed 
         # (maybe this needs to be renamed, it sounds a bit cryptic) 
         if eltype(Δdi) == ZeroTangent 
            g_nt = grad_params(V.model, Rs, Zs, z0, ps, st)[2]
            mult = Δei
         else 
            g_nt = pullback_2_mixed(Δei, Δdi, V.model, Rs, Zs, z0, ps, st)
            mult = one(TP)
         end

         # convert it back to a vector so we can accumulate it in the sum. 
         # this is quite bad - in the call to pullback_2_mixed we just 
         # converted it from a vector to a named tuple. We need to look into 
         # using something like ComponentArrays.jl to avoid this.
         g_loc += destructure(g_nt)[1] * mult
      end
      g_loc
   end

   return _restruct(g_vec)
end                


function rrule(::typeof(energy_forces_virial), 
               at, V::ACEPotential{<: ACEModel}, ps, st;
               domain   = 1:length(at), 
               executor = ThreadedEx(),
               ntasks   = Threads.nthreads(),
               nlist    = PairList(at, cutoff_radius(V)), 
               kwargs...
               )

   # TODO : analyze this code flow carefully and see if we can 
   #        re-use any of the computations done in the EFV evaluation                 
   EFV = energy_forces_virial(at, V, ps, st; 
                              domain = domain, 
                              executor = executor, 
                              ntasks = ntasks, 
                              nlist = nlist, 
                              kwargs...)

   return EFV, Δefv -> ( NoTangent(), NoTangent(), NoTangent(), 
                       pullback_EFV(Δefv, at, V, ps, st; 
                                    domain = domain, 
                                    executor = executor, 
                                    ntasks = ntasks, 
                                    nlist = nlist, 
                                    kwargs...), NoTangent() )
end


# --------------------------------------------------------
#   Basis evaluation 


function energy_forces_virial_basis(
            at, calc::ACEPotential{<: ACEModel}, ps, st;
            domain   = 1:length(at), 
            executor = ThreadedEx(),
            ntasks   = Threads.nthreads(),
            nlist    = PairList(at, cutoff_radius(calc)), 
            kwargs...
            )
   
   Js, Rs, Zs, z0 = get_neighbours(at, calc, nlist, 1)            
   E1 = evaluate_basis(calc.model, Rs, Zs, z0, ps, st)
   N_basis = length(E1)
   T = fl_type(calc.model) # this is ACE specific 

   E = fill(zero(T) * energy_unit(calc), N_basis)
   F = fill(zero(SVector{3, T}) * force_unit(calc), length(at), N_basis)
   V = fill(zero(SMatrix{3, 3, T}) * energy_unit(calc), N_basis)

   for i in domain
      Js, Rs, Zs, z0 = get_neighbours(at, V, nlist, i) 
      v, dv = evaluate_basis_ed(calc.model, Rs, Zs, z0, ps, st)

      for k = 1:N_basis
         E[k] += v[k] * energy_unit(calc) 
         for α = 1:length(Js) 
            F[Js[α], k] -= dv[k, α] * force_unit(calc)
            F[i, k]     += dv[k, α] * force_unit(calc)
         end
         V[k] += _site_virial(dv[k, :], Rs) * energy_unit(calc)
      end
   end
         
   return (energy = E, forces = F, virial = V)
end
