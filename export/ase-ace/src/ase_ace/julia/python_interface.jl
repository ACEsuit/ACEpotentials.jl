# Julia interface functions for Python/JuliaCall integration
# Loaded once via `include()` instead of repeated `seval` calls
#
# This module provides helper functions for:
# - ASE to AtomsBase conversion
# - Result extraction (forces, virial)
# - Descriptor computation using ACEpotentials functions

module ACEPythonInterface

using ACEpotentials
using ACEpotentials: site_descriptors
using ACEpotentials.Models: energy_forces_virial_basis, cutoff_radius, length_basis
using AtomsBase
using AtomsCalculators
using Unitful
using Unitful: ustrip
using StaticArrays

export make_positions, make_cell, make_atoms, make_system
export extract_forces, extract_virial
export compute_site_descriptors, compute_force_virial_descriptors

# ============================================================================
# ASE to AtomsBase conversion helpers
# ============================================================================

"""
    make_positions(pos_flat, n)

Convert flat position array from Python to vector of SVector with units.
Expects column-major (Fortran) order: [x1,y1,z1,x2,y2,z2,...].
"""
function make_positions(pos_flat, n)
    pos = reshape(collect(Float64, pos_flat), 3, n)
    return [SVector{3, Float64}(pos[:, i]...) * u"Å" for i in 1:n]
end

"""
    make_cell(cell_flat)

Convert flat cell array from Python to vector of SVector lattice vectors with units.
Expects column-major order for 3x3 matrix.
"""
function make_cell(cell_flat)
    c = reshape(collect(Float64, cell_flat), 3, 3)
    return [SVector{3, Float64}(c[:, i]...) * u"Å" for i in 1:3]
end

"""
    make_atoms(positions, numbers)

Create vector of AtomsBase.Atom from positions and atomic numbers.
"""
function make_atoms(positions, numbers)
    return [AtomsBase.Atom(Int(z), r) for (z, r) in zip(numbers, positions)]
end

"""
    make_system(atoms, cell, pbc)

Create AtomsBase periodic system from atoms, cell, and boundary conditions.
"""
function make_system(atoms, cell, pbc)
    return AtomsBase.periodic_system(atoms, cell; boundary_conditions=pbc)
end

# ============================================================================
# Result extraction helpers
# ============================================================================

"""
    extract_forces(forces)

Extract forces from Julia result to (natoms, 3) matrix for Python.
Strips units automatically.
"""
function extract_forces(forces)
    natoms = length(forces)
    result = zeros(Float64, natoms, 3)
    for i in 1:natoms
        f = ustrip.(forces[i])
        result[i, 1] = f[1]
        result[i, 2] = f[2]
        result[i, 3] = f[3]
    end
    return result
end

"""
    extract_virial(virial)

Extract virial tensor from Julia result, stripping units.
Returns 3x3 matrix.
"""
function extract_virial(virial)
    return ustrip.(virial)
end

# ============================================================================
# Descriptor computation using existing ACEpotentials functions
# ============================================================================

"""
    compute_site_descriptors(at, model)

Compute site energy descriptors for all atoms using ACEpotentials.site_descriptors.
Returns (natoms, n_basis) matrix for Python.
"""
function compute_site_descriptors(at, model)
    # Use ACEpotentials.site_descriptors instead of duplicating code
    descriptors_vec = site_descriptors(at, model)
    natoms = length(descriptors_vec)

    # Handle case of no atoms or empty descriptors
    if natoms == 0
        return zeros(Float64, 0, length_basis(model))
    end

    n_basis = length(descriptors_vec[1])

    # Convert vector of vectors to matrix for Python
    descriptors = zeros(Float64, natoms, n_basis)
    for i in 1:natoms
        descriptors[i, :] = descriptors_vec[i]
    end
    return descriptors
end

"""
    compute_force_virial_descriptors(at, model)

Compute per-basis force and virial contributions using energy_forces_virial_basis.

Returns named tuple with:
- energy: (n_basis,) per-basis energy contributions
- forces: (natoms, n_basis, 3) per-basis force contributions per atom
- virial: (n_basis, 3, 3) per-basis virial tensors
"""
function compute_force_virial_descriptors(at, model)
    result = energy_forces_virial_basis(at, model)

    # Extract and format for Python
    # Energy descriptors: per-basis energy contributions
    energy_desc = ustrip.(result.energy)  # Shape: (n_basis,)

    # Force descriptors: (natoms, n_basis) of SVector{3} -> (natoms, n_basis, 3)
    forces = result.forces
    natoms, n_basis = size(forces)
    force_desc = zeros(Float64, natoms, n_basis, 3)
    for i in 1:natoms
        for k in 1:n_basis
            f = ustrip.(forces[i, k])
            force_desc[i, k, 1] = f[1]
            force_desc[i, k, 2] = f[2]
            force_desc[i, k, 3] = f[3]
        end
    end

    # Virial descriptors: (n_basis,) of 3x3 matrices -> (n_basis, 3, 3)
    virial_desc = zeros(Float64, n_basis, 3, 3)
    for k in 1:n_basis
        virial_desc[k, :, :] = ustrip.(result.virial[k])
    end

    return (energy = energy_desc, forces = force_desc, virial = virial_desc)
end

end # module ACEPythonInterface
