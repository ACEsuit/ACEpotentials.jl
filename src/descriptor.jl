export site_descriptor, site_descriptors


"""
    site_descriptor(basis, atoms::AbstractAtoms, i::Integer)

Compute the site descriptor for the `i`th atom in `atoms`.
"""
function site_descriptor(basis, atoms::AbstractAtoms, i::Integer)
   return site_energy(basis, atoms, i)
end


"""
    site_descriptors(basis, atoms::AbstractAtoms[, domain])

Compute site descriptors for all atoms in `atoms`, returning them as
a vector of vectors. If the optional argument `domain` is passed as a list of 
integers (atom indices), then only the site descriptors for those atoms are 
computed and returned. 
"""
function site_descriptors(basis, atoms::AbstractAtoms, 
                          domain = 1:length(atoms))
    return [site_descriptor(basis, atoms, i) for i in domain]
end
