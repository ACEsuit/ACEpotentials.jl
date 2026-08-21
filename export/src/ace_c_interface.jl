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
# BATCH API WITH THREADING
# ============================================================================
# These functions process multiple atoms at once with optional multi-threading.
# Use JULIA_NUM_THREADS environment variable to control thread count.
# Threading is enabled when nthreads > 1 (check with ace_get_nthreads).

"""
    ace_get_nthreads() -> Int32

Get the number of Julia threads available.
Set JULIA_NUM_THREADS before loading the library to change this.
"""
Base.@ccallable function ace_get_nthreads()::Cint
    return Cint(Threads.nthreads())
end

"""
    ace_batch_energy_forces_virial(
        natoms, z, neighbor_counts, neighbor_offsets,
        neighbor_z, neighbor_Rij,
        energies, forces, virials
    )

Compute energies, forces, and virials for multiple atoms with threading.

Arguments:
- natoms: Number of atoms to process
- z: Ptr{Cint} to [natoms] center atom atomic numbers
- neighbor_counts: Ptr{Cint} to [natoms] neighbor counts per atom
- neighbor_offsets: Ptr{Cint} to [natoms] offsets into neighbor arrays (0-indexed)
- neighbor_z: Ptr{Cint} to [total_neighbors] neighbor atomic numbers
- neighbor_Rij: Ptr{Cdouble} to [total_neighbors*3] displacement vectors
- energies: Ptr{Cdouble} to [natoms] output energies
- forces: Ptr{Cdouble} to [total_neighbors*3] output forces ON neighbors
- virials: Ptr{Cdouble} to [natoms*6] output virials in Voigt notation

Note: Uses Threads.@threads for parallelism when JULIA_NUM_THREADS > 1
"""
Base.@ccallable function ace_batch_energy_forces_virial(
    natoms::Cint,
    z::Ptr{Cint},
    neighbor_counts::Ptr{Cint},
    neighbor_offsets::Ptr{Cint},
    neighbor_z::Ptr{Cint},
    neighbor_Rij::Ptr{Cdouble},
    energies::Ptr{Cdouble},
    forces::Ptr{Cdouble},
    virials::Ptr{Cdouble}
)::Cvoid
    # Process atoms in parallel
    Threads.@threads for i in 1:natoms
        z0 = unsafe_load(z, i)
        nneigh = unsafe_load(neighbor_counts, i)
        offset = unsafe_load(neighbor_offsets, i)  # 0-indexed from C

        if nneigh == 0
            # Isolated atom
            iz0 = z2i(z0)
            E0 = (iz0 == 1) ? E0_1 : 0.0
            unsafe_store!(energies, E0, i)
            # Zero virial
            for k in 1:6
                unsafe_store!(virials, 0.0, (i-1)*6 + k)
            end
        else
            # Read neighbor data for this atom (offset is 0-indexed)
            # Use pointer indexing: base + offset + 1 for 1-indexed Julia
            Zs = Vector{Int}(undef, nneigh)
            Rs = Vector{SVector{3, Float64}}(undef, nneigh)

            @inbounds for j in 1:nneigh
                idx = offset + j  # 1-indexed from offset (which is 0-indexed)
                Zs[j] = unsafe_load(neighbor_z, idx)
                x = unsafe_load(neighbor_Rij, 3*(idx-1) + 1)
                y = unsafe_load(neighbor_Rij, 3*(idx-1) + 2)
                z_coord = unsafe_load(neighbor_Rij, 3*(idx-1) + 3)
                Rs[j] = SVector(x, y, z_coord)
            end

            # Compute
            Ei, Fi, Vi = site_energy_forces_virial(Rs, Zs, Int(z0))

            # Write outputs
            unsafe_store!(energies, Ei, i)

            # Forces for this atom's neighbors
            @inbounds for j in 1:nneigh
                idx = offset + j
                unsafe_store!(forces, Fi[j][1], 3*(idx-1) + 1)
                unsafe_store!(forces, Fi[j][2], 3*(idx-1) + 2)
                unsafe_store!(forces, Fi[j][3], 3*(idx-1) + 3)
            end

            # Virial in Voigt notation: xx, yy, zz, yz, xz, xy
            vbase = (i-1)*6
            unsafe_store!(virials, Vi[1,1], vbase + 1)
            unsafe_store!(virials, Vi[2,2], vbase + 2)
            unsafe_store!(virials, Vi[3,3], vbase + 3)
            unsafe_store!(virials, Vi[2,3], vbase + 4)
            unsafe_store!(virials, Vi[1,3], vbase + 5)
            unsafe_store!(virials, Vi[1,2], vbase + 6)
        end
    end
    return nothing
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
