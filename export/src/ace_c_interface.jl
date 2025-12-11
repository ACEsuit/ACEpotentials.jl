# C Interface for ACE Potentials
#
# SITE-LEVEL API (for LAMMPS and Python via matscipy):
#    - ace_site_energy(z0, nneigh, neighbor_z, neighbor_R) -> energy
#    - ace_site_energy_forces(z0, nneigh, neighbor_z, neighbor_R, forces) -> energy
#    - ace_site_energy_forces_virial(z0, nneigh, neighbor_z, neighbor_R, forces, virial) -> energy
#
# The Python calculator uses matscipy for O(N) neighbor list construction
# and calls these site-level functions for evaluation.
#
# All arrays use C layout (row-major for 2D, contiguous for 1D).

using StaticArrays

# ============================================================================
# SITE-LEVEL C INTERFACE
# ============================================================================
# These work directly with pre-computed neighbor lists (from LAMMPS or matscipy).
# Forces returned are forces ON the neighbors (not on the center atom).
# The caller handles force accumulation via Newton's 3rd law.

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
Note: forces[j] is the force on neighbor j due to this site (for accumulation via Newton's 3rd law)
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
