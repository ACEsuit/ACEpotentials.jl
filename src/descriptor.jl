export descriptor, descriptors


"""
    descriptor(basis, atoms::AbstractAtoms, i::Integer)

Compute the site descriptor for the `i`th atom in `atoms`.
"""
function descriptor(basis, atoms::AbstractAtoms, i::Integer)
   return site_energy(basis, atoms, i)
end


"""
    descriptor(basis, atoms::AbstractAtoms)

Compute site descriptors for all atoms in `atoms`, returning them as a list.
"""
function descriptors(basis, atoms::AbstractAtoms)
    return [descriptor(basis, atoms, i) for i in 1:length(atoms)]
end
