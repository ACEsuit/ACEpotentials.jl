
import AtomsCalculators 
import AtomsCalculatorsUtilities

import AtomsCalculatorsUtilities.SitePotentials: SitePotential, 
                     cutoff_radius, eval_site, eval_grad_site, 
                     energy_unit, length_unit 


struct SitePotentialStack{TP} <: SitePotential
   pots::TP 
end

function energy_unit(pot::SitePotentialStack) 
   return energy_unit(pot.pots[1])
end

function length_unit(pot::SitePotentialStack) 
   return length_unit(pot.pots[1])
end

function cutoff_radius(pot::SitePotentialStack) 
   return maximum(cutoff_radius, pot.pots)
end

function eval_site(pot::SitePotentialStack, Rs, Zs, z0)
   return sum( eval_site(p, Rs, Zs, z0) for p in pot.pots )
end

function eval_grad_site(pot::SitePotentialStack, Rs, Zs, z0)
   return eval_site(pot, Rs, Zs, z0),
            sum( eval_grad_site(p, Rs, Zs, z0)[2] for p in pot.pots )
end

