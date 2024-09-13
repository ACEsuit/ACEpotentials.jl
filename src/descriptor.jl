export site_descriptors

using ACEpotentials.Models: evaluate_basis

import AtomsCalculatorsUtilities.SitePotentials: SitePotential, 
                            cutoff_radius, 
                            eval_site, 
                            PairList, 
                            get_neighbours


# I'm RETIRING THIS FOR NOW BECAUSE IT IS HIGHLY INEFFICIENT
# """
#     site_descriptor(basis, atoms::AbstractSystem, i::Integer)

# Compute the site descriptor for the `i`th atom in `atoms`.
# """
# site_descriptor(model::ACEPotential, args...) = site_descriptor(model.model, args...)

# function site_descriptor(model::ACEModel, atoms::AbstractSystem, i::Integer) 
# end


"""
    site_descriptors(system::AbstractSystem, model::ACEPotential;
                     domain, nlist)

Compute site descriptors for all atoms in `system`, returning them as
a vector of vectors. If the optional kw argument `domain` is passed as a list of 
integers (atom indices), then only the site descriptors for those atoms are 
computed and returned. The neighbourlist `nlist` can be supplied optionally
as a kw arg, otherwise it is recomputed. 
"""
function site_descriptors(system::AbstractSystem, model::ACEPotential;  
                          domain = 1:length(system), 
                          nlist = PairList(system, cutoff_radius(model)))
    
    function _site_descriptor(system, model, i, nlist)
        Js, Rs, Zs, z0 = get_neighbours(system, model, nlist, i) 
        return evaluate_basis(model.model, Rs, Zs, z0, model.ps, model.st)
    end                    

    return [ _site_descriptor(system, model, i, nlist) 
             for i in domain ]
end
