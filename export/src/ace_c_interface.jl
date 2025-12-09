# C Interface for ACE Potentials
#
# Two API levels:
#
# 1. SITE-LEVEL (for LAMMPS): Works with pre-computed neighbor lists
#    - ace_site_energy(z0, nneigh, neighbor_z, neighbor_R) -> energy
#    - ace_site_energy_forces(z0, nneigh, neighbor_z, neighbor_R, forces) -> energy
#    - ace_site_energy_forces_virial(z0, nneigh, neighbor_z, neighbor_R, forces, virial) -> energy
#
# 2. SYSTEM-LEVEL (for Python/ASE): Computes neighbor list internally
#    - ace_energy(natoms, species, positions, cell, pbc) -> energy
#    - ace_energy_forces(..., forces) -> energy
#    - ace_energy_forces_virial(..., forces, virial) -> energy
#
# All arrays use C layout (row-major for 2D, contiguous for 1D).

using StaticArrays

# ============================================================================
# SITE-LEVEL C INTERFACE (for LAMMPS)
# ============================================================================
# These work directly with LAMMPS neighbor lists.
# Forces returned are forces ON the neighbors (not on the center atom).
# LAMMPS handles force accumulation via Newton's 3rd law.

@inline function c_read_Rij(ptr::Ptr{Cdouble}, nneigh::Int)::Vector{SVector{3, Float64}}
    Rs = Vector{SVector{3, Float64}}(undef, nneigh)
    @inbounds for j in 1:nneigh
        x = unsafe_load(ptr, 3*(j-1) + 1)
        y = unsafe_load(ptr, 3*(j-1) + 2)
        z = unsafe_load(ptr, 3*(j-1) + 3)
        Rs[j] = SVector(x, y, z)
    end
    return Rs
end

@inline function c_read_species(ptr::Ptr{Cint}, n::Int)::Vector{Int}
    species = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        species[i] = unsafe_load(ptr, i)
    end
    return species
end

@inline function c_write_forces!(ptr::Ptr{Cdouble}, forces::Vector{SVector{3, Float64}})
    @inbounds for j in 1:length(forces)
        unsafe_store!(ptr, forces[j][1], 3*(j-1) + 1)
        unsafe_store!(ptr, forces[j][2], 3*(j-1) + 2)
        unsafe_store!(ptr, forces[j][3], 3*(j-1) + 3)
    end
end

@inline function c_write_virial!(ptr::Ptr{Cdouble}, virial::SMatrix{3,3,Float64,9})
    # Voigt notation: xx, yy, zz, yz, xz, xy (LAMMPS convention)
    unsafe_store!(ptr, virial[1,1], 1)  # xx
    unsafe_store!(ptr, virial[2,2], 2)  # yy
    unsafe_store!(ptr, virial[3,3], 3)  # zz
    unsafe_store!(ptr, virial[2,3], 4)  # yz
    unsafe_store!(ptr, virial[1,3], 5)  # xz
    unsafe_store!(ptr, virial[1,2], 6)  # xy
end

"""
    ace_site_energy(z0, nneigh, neighbor_z, neighbor_Rij) -> Float64

Compute site energy for a single atom.

Arguments:
- z0: Atomic number of center atom
- nneigh: Number of neighbors
- neighbor_z: Ptr{Cint} to [nneigh] neighbor atomic numbers
- neighbor_Rij: Ptr{Cdouble} to [nneigh*3] displacement vectors (Rj - Ri)

Returns: Site energy in eV
"""
Base.@ccallable function ace_site_energy(
    z0::Cint,
    nneigh::Cint,
    neighbor_z::Ptr{Cint},
    neighbor_Rij::Ptr{Cdouble}
)::Cdouble
    if nneigh == 0
        # Return E0 for isolated atom
        iz0 = z2i(z0)
        if iz0 == 1
            return E0_1
        end
        return 0.0
    end

    Zs = c_read_species(neighbor_z, nneigh)
    Rs = c_read_Rij(neighbor_Rij, nneigh)

    return site_energy(Rs, Zs, Int(z0))
end

"""
    ace_site_energy_forces(z0, nneigh, neighbor_z, neighbor_Rij, forces) -> Float64

Compute site energy and forces for a single atom.

Arguments:
- z0, nneigh, neighbor_z, neighbor_Rij: Same as ace_site_energy
- forces: Ptr{Cdouble} to [nneigh*3] output array for forces ON neighbors

Returns: Site energy in eV
Note: forces[j] is the force on neighbor j due to this site (for LAMMPS accumulation)
"""
Base.@ccallable function ace_site_energy_forces(
    z0::Cint,
    nneigh::Cint,
    neighbor_z::Ptr{Cint},
    neighbor_Rij::Ptr{Cdouble},
    forces::Ptr{Cdouble}
)::Cdouble
    if nneigh == 0
        iz0 = z2i(z0)
        if iz0 == 1
            return E0_1
        end
        return 0.0
    end

    Zs = c_read_species(neighbor_z, nneigh)
    Rs = c_read_Rij(neighbor_Rij, nneigh)

    Ei, Fi = site_energy_forces(Rs, Zs, Int(z0))

    # Write forces (these are -dE/dRj, the force ON neighbor j)
    c_write_forces!(forces, Fi)

    return Ei
end

"""
    ace_site_energy_forces_virial(z0, nneigh, neighbor_z, neighbor_Rij, forces, virial) -> Float64

Compute site energy, forces, and virial contribution for a single atom.

Arguments:
- z0, nneigh, neighbor_z, neighbor_Rij, forces: Same as ace_site_energy_forces
- virial: Ptr{Cdouble} to [6] output array for virial in Voigt notation (xx,yy,zz,yz,xz,xy)

Returns: Site energy in eV
"""
Base.@ccallable function ace_site_energy_forces_virial(
    z0::Cint,
    nneigh::Cint,
    neighbor_z::Ptr{Cint},
    neighbor_Rij::Ptr{Cdouble},
    forces::Ptr{Cdouble},
    virial::Ptr{Cdouble}
)::Cdouble
    if nneigh == 0
        iz0 = z2i(z0)
        # Zero virial for isolated atom
        for k in 1:6
            unsafe_store!(virial, 0.0, k)
        end
        if iz0 == 1
            return E0_1
        end
        return 0.0
    end

    Zs = c_read_species(neighbor_z, nneigh)
    Rs = c_read_Rij(neighbor_Rij, nneigh)

    Ei, Fi, Vi = site_energy_forces_virial(Rs, Zs, Int(z0))

    c_write_forces!(forces, Fi)
    c_write_virial!(virial, Vi)

    return Ei
end

# ============================================================================
# SYSTEM-LEVEL FUNCTIONS (for Python/ASE)
# ============================================================================
# These compute neighbor lists internally - convenient for Python but not for LAMMPS.

# Simple neighbor list computation
function compute_neighbor_list(
    positions::Vector{SVector{3, Float64}},
    cell::Union{Nothing, SMatrix{3,3,Float64,9}},
    pbc::SVector{3, Bool},
    rcut::Float64
)
    natoms = length(positions)
    neighbors = Vector{Tuple{Int, Vector{Int}, Vector{SVector{3, Float64}}}}()

    # Determine search range for periodic images
    if cell !== nothing && any(pbc)
        cell_lengths = SVector(norm(cell[:, 1]), norm(cell[:, 2]), norm(cell[:, 3]))
        n_images = ceil.(Int, rcut ./ cell_lengths) .* Int.(pbc)
    else
        n_images = SVector(0, 0, 0)
    end

    for i in 1:natoms
        neigh_idx = Int[]
        neigh_R = SVector{3, Float64}[]

        for na in -n_images[1]:n_images[1]
            for nb in -n_images[2]:n_images[2]
                for nc in -n_images[3]:n_images[3]
                    if cell !== nothing
                        shift = cell * SVector(Float64(na), Float64(nb), Float64(nc))
                    else
                        shift = zero(SVector{3, Float64})
                    end

                    for j in 1:natoms
                        if na == 0 && nb == 0 && nc == 0 && i == j
                            continue
                        end
                        Rij = positions[j] + shift - positions[i]
                        r = norm(Rij)
                        if r < rcut && r > 1e-10
                            push!(neigh_idx, j)
                            push!(neigh_R, Rij)
                        end
                    end
                end
            end
        end
        push!(neighbors, (i, neigh_idx, neigh_R))
    end
    return neighbors
end

function compute_total_energy(
    species::Vector{Int},
    positions::Vector{SVector{3, Float64}},
    cell::Union{Nothing, SMatrix{3,3,Float64,9}},
    pbc::SVector{3, Bool}
)
    rcut = RIN0CUT_1_1.rcut
    neighbors = compute_neighbor_list(positions, cell, pbc, rcut)

    total_energy = 0.0
    for (i, neigh_idx, neigh_R) in neighbors
        Zs = [species[j] for j in neigh_idx]
        Ei = site_energy(neigh_R, Zs, species[i])
        total_energy += Ei
    end
    return total_energy
end

function compute_total_energy_forces(
    species::Vector{Int},
    positions::Vector{SVector{3, Float64}},
    cell::Union{Nothing, SMatrix{3,3,Float64,9}},
    pbc::SVector{3, Bool}
)
    natoms = length(species)
    rcut = RIN0CUT_1_1.rcut
    neighbors = compute_neighbor_list(positions, cell, pbc, rcut)

    forces = zeros(SVector{3, Float64}, natoms)
    total_energy = 0.0

    for (i, neigh_idx, neigh_R) in neighbors
        Zs = [species[j] for j in neigh_idx]
        Ei, Fi = site_energy_forces(neigh_R, Zs, species[i])
        total_energy += Ei

        # Fi[k] = force on neighbor k from site i
        # Newton's 3rd law: force on i = -sum(Fi)
        for (k, j) in enumerate(neigh_idx)
            forces[i] -= Fi[k]
            forces[j] += Fi[k]
        end
    end
    return total_energy, forces
end

function compute_total_energy_forces_virial(
    species::Vector{Int},
    positions::Vector{SVector{3, Float64}},
    cell::Union{Nothing, SMatrix{3,3,Float64,9}},
    pbc::SVector{3, Bool}
)
    natoms = length(species)
    rcut = RIN0CUT_1_1.rcut
    neighbors = compute_neighbor_list(positions, cell, pbc, rcut)

    forces = zeros(SVector{3, Float64}, natoms)
    virial = zeros(SMatrix{3, 3, Float64, 9})
    total_energy = 0.0

    for (i, neigh_idx, neigh_R) in neighbors
        Zs = [species[j] for j in neigh_idx]
        Ei, Fi, Vi = site_energy_forces_virial(neigh_R, Zs, species[i])
        total_energy += Ei
        virial += Vi

        for (k, j) in enumerate(neigh_idx)
            forces[i] -= Fi[k]
            forces[j] += Fi[k]
        end
    end
    return total_energy, forces, virial
end

# C-callable system-level functions
@inline function c_read_positions(ptr::Ptr{Cdouble}, n::Int)::Vector{SVector{3, Float64}}
    positions = Vector{SVector{3, Float64}}(undef, n)
    @inbounds for i in 1:n
        x = unsafe_load(ptr, 3*(i-1) + 1)
        y = unsafe_load(ptr, 3*(i-1) + 2)
        z = unsafe_load(ptr, 3*(i-1) + 3)
        positions[i] = SVector(x, y, z)
    end
    return positions
end

@inline function c_read_cell(ptr::Ptr{Cdouble})::SMatrix{3,3,Float64,9}
    # Row-major input -> column-major SMatrix
    vals = ntuple(i -> unsafe_load(ptr, i), 9)
    return SMatrix{3,3}(vals[1], vals[4], vals[7],
                        vals[2], vals[5], vals[8],
                        vals[3], vals[6], vals[9])
end

@inline function c_read_pbc(ptr::Ptr{Cint})::SVector{3, Bool}
    return SVector(unsafe_load(ptr, 1) != 0,
                   unsafe_load(ptr, 2) != 0,
                   unsafe_load(ptr, 3) != 0)
end

@inline function c_write_forces_system!(ptr::Ptr{Cdouble}, forces::Vector{SVector{3, Float64}})
    @inbounds for i in 1:length(forces)
        unsafe_store!(ptr, forces[i][1], 3*(i-1) + 1)
        unsafe_store!(ptr, forces[i][2], 3*(i-1) + 2)
        unsafe_store!(ptr, forces[i][3], 3*(i-1) + 3)
    end
end

@inline function c_write_virial_system!(ptr::Ptr{Cdouble}, virial::SMatrix{3,3,Float64,9})
    # Row-major 3x3 output
    unsafe_store!(ptr, virial[1,1], 1)
    unsafe_store!(ptr, virial[1,2], 2)
    unsafe_store!(ptr, virial[1,3], 3)
    unsafe_store!(ptr, virial[2,1], 4)
    unsafe_store!(ptr, virial[2,2], 5)
    unsafe_store!(ptr, virial[2,3], 6)
    unsafe_store!(ptr, virial[3,1], 7)
    unsafe_store!(ptr, virial[3,2], 8)
    unsafe_store!(ptr, virial[3,3], 9)
end

"""
    ace_energy(natoms, species, positions, cell, pbc) -> Float64

Compute total energy of a system (computes neighbor list internally).

Arguments:
- natoms: Number of atoms
- species: Ptr{Cint} to [natoms] atomic numbers
- positions: Ptr{Cdouble} to [natoms*3] positions (x1,y1,z1,x2,...)
- cell: Ptr{Cdouble} to [9] cell vectors (row-major), or C_NULL
- pbc: Ptr{Cint} to [3] periodic flags, or C_NULL

Returns: Total energy in eV
"""
Base.@ccallable function ace_energy(
    natoms::Cint,
    species_ptr::Ptr{Cint},
    positions_ptr::Ptr{Cdouble},
    cell_ptr::Ptr{Cdouble},
    pbc_ptr::Ptr{Cint}
)::Cdouble
    species = c_read_species(species_ptr, natoms)
    positions = c_read_positions(positions_ptr, natoms)

    if cell_ptr == C_NULL || pbc_ptr == C_NULL
        cell = nothing
        pbc = SVector(false, false, false)
    else
        cell = c_read_cell(cell_ptr)
        pbc = c_read_pbc(pbc_ptr)
    end

    return compute_total_energy(species, positions, cell, pbc)
end

"""
    ace_energy_forces(natoms, species, positions, cell, pbc, forces) -> Float64

Compute total energy and forces.
"""
Base.@ccallable function ace_energy_forces(
    natoms::Cint,
    species_ptr::Ptr{Cint},
    positions_ptr::Ptr{Cdouble},
    cell_ptr::Ptr{Cdouble},
    pbc_ptr::Ptr{Cint},
    forces_ptr::Ptr{Cdouble}
)::Cdouble
    species = c_read_species(species_ptr, natoms)
    positions = c_read_positions(positions_ptr, natoms)

    if cell_ptr == C_NULL || pbc_ptr == C_NULL
        cell = nothing
        pbc = SVector(false, false, false)
    else
        cell = c_read_cell(cell_ptr)
        pbc = c_read_pbc(pbc_ptr)
    end

    energy, forces = compute_total_energy_forces(species, positions, cell, pbc)
    c_write_forces_system!(forces_ptr, forces)

    return energy
end

"""
    ace_energy_forces_virial(natoms, species, positions, cell, pbc, forces, virial) -> Float64

Compute total energy, forces, and virial.
"""
Base.@ccallable function ace_energy_forces_virial(
    natoms::Cint,
    species_ptr::Ptr{Cint},
    positions_ptr::Ptr{Cdouble},
    cell_ptr::Ptr{Cdouble},
    pbc_ptr::Ptr{Cint},
    forces_ptr::Ptr{Cdouble},
    virial_ptr::Ptr{Cdouble}
)::Cdouble
    species = c_read_species(species_ptr, natoms)
    positions = c_read_positions(positions_ptr, natoms)

    if cell_ptr == C_NULL || pbc_ptr == C_NULL
        cell = nothing
        pbc = SVector(false, false, false)
    else
        cell = c_read_cell(cell_ptr)
        pbc = c_read_pbc(pbc_ptr)
    end

    energy, forces, virial = compute_total_energy_forces_virial(species, positions, cell, pbc)
    c_write_forces_system!(forces_ptr, forces)
    c_write_virial_system!(virial_ptr, virial)

    return energy
end

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

"""
    ace_get_cutoff() -> Float64

Get the cutoff radius of the potential.
"""
Base.@ccallable function ace_get_cutoff()::Cdouble
    return RIN0CUT_1_1.rcut
end

"""
    ace_get_n_species() -> Int32

Get the number of species supported by the potential.
"""
Base.@ccallable function ace_get_n_species()::Cint
    return Cint(NZ)
end

"""
    ace_get_species(idx) -> Int32

Get the atomic number for species index (1-based).
"""
Base.@ccallable function ace_get_species(idx::Cint)::Cint
    if idx < 1 || idx > NZ
        return Cint(-1)
    end
    return Cint(I2Z[idx])
end
