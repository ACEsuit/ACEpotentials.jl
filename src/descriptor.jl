export site_descriptor, site_descriptors


"""
    site_descriptor(basis, atoms::AbstractAtoms, i::Integer)

Compute the site descriptor for the `i`th atom in `atoms`.
"""
function site_descriptor(basis, atoms::AbstractAtoms, i::Integer)
   return site_energy(basis, atoms, i)
end


"""
    site_descriptors(basis, atoms::AbstractAtoms)

Compute site descriptors for all atoms in `atoms`, returning them as a list.
"""
function site_descriptors(basis, atoms::AbstractAtoms)
    return [site_descriptor(basis, atoms, i) for i in 1:length(atoms)]
end
