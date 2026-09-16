# C interface and main entry point writing functions
# Split from export_ace_model.jl for maintainability
#
# Note: These functions use _emit_species_dispatch and _emit_species_dispatch_multi
# which must be defined before including this file.

function _write_main(io, NZ)
    println(io, """
# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

function (@main)(ARGS)
    println(Core.stdout, "=== ACE Potential Evaluation ===")
    println(Core.stdout, "Number of species: ", NZ)
    println(Core.stdout, "Basis size: ", N_BASIS)
    println(Core.stdout, "Radial basis size: ", N_RNL)
    println(Core.stdout, "Spherical harmonics: L=", MAXL, " (", N_YLM, " functions)")

    # Example evaluation with test data
    Rs = [
        SVector(2.35, 0.0, 0.0),
        SVector(-0.78, 2.22, 0.0),
        SVector(-0.78, -1.11, 1.92),
    ]
    Zs = fill(I2Z[1], length(Rs))  # Same species as center
    Z0 = I2Z[1]  # Center atom species

    println(Core.stdout, "")
    println(Core.stdout, "Test evaluation:")
    println(Core.stdout, "  Center species: Z=", Z0)
    println(Core.stdout, "  Number of neighbors: ", length(Rs))

    # Energy only
    E = site_energy(Rs, Zs, Z0)
    println(Core.stdout, "  Site energy: ", E, " eV")

    # Analytic forces
    println(Core.stdout, "")
    println(Core.stdout, "Analytic forces:")
    E2, F = site_energy_forces(Rs, Zs, Z0)
    for (j, f) in enumerate(F)
        println(Core.stdout, "  F[", j, "] = [", f[1], ", ", f[2], ", ", f[3], "]")
    end

    # Forces + Virial
    println(Core.stdout, "")
    println(Core.stdout, "With virial stress:")
    E3, F3, V = site_energy_forces_virial(Rs, Zs, Z0)
    println(Core.stdout, "  Energy: ", E3, " eV")
    println(Core.stdout, "  Virial tensor:")
    println(Core.stdout, "    [", V[1,1], ", ", V[1,2], ", ", V[1,3], "]")
    println(Core.stdout, "    [", V[2,1], ", ", V[2,2], ", ", V[2,3], "]")
    println(Core.stdout, "    [", V[3,1], ", ", V[3,2], ", ", V[3,3], "]")

    # Verify analytic vs finite difference forces
    println(Core.stdout, "")
    println(Core.stdout, "Force verification (analytic vs finite difference):")
    h = 1e-5
    max_err = 0.0
    for j in 1:length(Rs)
        f_fd = zeros(3)
        for α in 1:3
            Rs_p = copy(Rs)
            Rs_m = copy(Rs)
            e_α = zeros(3); e_α[α] = h
            Rs_p[j] = Rs[j] + SVector{3}(e_α)
            Rs_m[j] = Rs[j] - SVector{3}(e_α)
            Ep = site_energy(Rs_p, Zs, Z0)
            Em = site_energy(Rs_m, Zs, Z0)
            f_fd[α] = -(Ep - Em) / (2h)
        end
        err = sqrt(sum((F[j] - SVector{3}(f_fd)).^2))
        max_err = max(max_err, err)
        println(Core.stdout, "  F[", j, "] err = ", err)
    end
    println(Core.stdout, "  Max force error: ", max_err)

    println(Core.stdout, "")
    println(Core.stdout, "Evaluation successful!")

    return 0
end
""")
end

function _write_c_interface(io, NZ)
    println(io, """
# ============================================================================
# C INTERFACE FOR SHARED LIBRARY
# ============================================================================
#
# Two API levels:
# 1. SITE-LEVEL (for LAMMPS): Works with pre-computed neighbor lists
# 2. SYSTEM-LEVEL (for Python/ASE): Computes neighbor list internally
#
# RE-ENTRANCY.  Every hot entry point takes an opaque workspace handle as its FIRST argument,
# obtained from ace_workspace_new() and released with ace_workspace_free().  The library holds
# no mutable global state, so N threads with N workspaces produce bitwise the serial result.
# One workspace may NOT be used by two threads at once.

# ============================================================================
# WORKSPACE HANDLES
# ============================================================================
#
# `pointer_from_objref` does not root anything, so every live workspace is kept in WORKSPACES
# until it is freed; without that the GC is free to collect a workspace the C caller still
# holds a pointer to.  new/free take a lock because a plugin may allocate its per-thread
# workspaces from inside a parallel region; the HOT entries below take none -- they only
# convert the pointer back, which touches no shared state.

const WORKSPACES = Workspace[]
const WORKSPACES_LOCK = ReentrantLock()

Base.@ccallable function ace_workspace_new()::Ptr{Cvoid}
    ws = new_workspace()
    lock(WORKSPACES_LOCK)
    try
        push!(WORKSPACES, ws)
    finally
        unlock(WORKSPACES_LOCK)
    end
    return pointer_from_objref(ws)
end

Base.@ccallable function ace_workspace_free(p::Ptr{Cvoid})::Cvoid
    p == C_NULL && return nothing
    ws = unsafe_pointer_to_objref(p)::Workspace
    lock(WORKSPACES_LOCK)
    try
        i = findfirst(w -> w === ws, WORKSPACES)
        i === nothing || deleteat!(WORKSPACES, i)
    finally
        unlock(WORKSPACES_LOCK)
    end
    return nothing
end

# ============================================================================
# HELPER FUNCTIONS FOR C INTERFACE
# ============================================================================

@inline _ws(p::Ptr{Cvoid}) = unsafe_pointer_to_objref(p)::Workspace

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

# Force output written straight through the caller's buffer: `unsafe_wrap` + `reinterpret`
# gives an AbstractVector{SVector{3,Float64}} aliasing it, so the kernel's `forces[j] = -f`
# stores land in the C array with no intermediate Vector{SVector} to allocate and copy.
@inline function c_force_view(ptr::Ptr{Cdouble}, nneigh::Int)
    flat = unsafe_wrap(Array, ptr, 3 * nneigh; own = false)
    return reinterpret(SVector{3, Float64}, flat)
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

# ============================================================================
# SITE-LEVEL C INTERFACE (for LAMMPS)
# ============================================================================
# These work directly with LAMMPS neighbor lists.
# Forces returned are forces ON the neighbors (not on the center atom).
# LAMMPS handles force accumulation via Newton's 3rd law.

Base.@ccallable function ace_site_energy(
    ws::Ptr{Cvoid},
    z0::Cint,
    nneigh::Cint,
    neighbor_z::Ptr{Cint},
    neighbor_Rij::Ptr{Cdouble}
)::Cdouble
    if nneigh == 0
        return E0_of(z2i(z0))
    end

    Zs = c_read_species(neighbor_z, Int(nneigh))
    Rs = c_read_Rij(neighbor_Rij, Int(nneigh))

    return site_energy!(_ws(ws), Rs, Zs, Int(z0))
end

Base.@ccallable function ace_site_energy_forces(
    ws::Ptr{Cvoid},
    z0::Cint,
    nneigh::Cint,
    neighbor_z::Ptr{Cint},
    neighbor_Rij::Ptr{Cdouble},
    forces::Ptr{Cdouble}
)::Cdouble
    if nneigh == 0
        return E0_of(z2i(z0))
    end

    Zs = c_read_species(neighbor_z, Int(nneigh))
    Rs = c_read_Rij(neighbor_Rij, Int(nneigh))
    F = c_force_view(forces, Int(nneigh))

    return site_energy_forces!(_ws(ws), Rs, Zs, Int(z0), F)
end

Base.@ccallable function ace_site_energy_forces_virial(
    ws::Ptr{Cvoid},
    z0::Cint,
    nneigh::Cint,
    neighbor_z::Ptr{Cint},
    neighbor_Rij::Ptr{Cdouble},
    forces::Ptr{Cdouble},
    virial::Ptr{Cdouble}
)::Cdouble
    if nneigh == 0
        for k in 1:6
            unsafe_store!(virial, 0.0, k)
        end
        return E0_of(z2i(z0))
    end

    Zs = c_read_species(neighbor_z, Int(nneigh))
    Rs = c_read_Rij(neighbor_Rij, Int(nneigh))
    F = c_force_view(forces, Int(nneigh))

    Ei, Vi = site_energy_forces_virial!(_ws(ws), Rs, Zs, Int(z0), F)
    c_write_virial!(virial, Vi)

    return Ei
end

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

Base.@ccallable function ace_get_cutoff()::Cdouble
    return RCUT_MAX
end

Base.@ccallable function ace_get_n_species()::Cint
    return Cint(NZ)
end

Base.@ccallable function ace_get_species(idx::Cint)::Cint
    if idx < 1 || idx > NZ
        return Cint(-1)
    end
    return Cint(I2Z[idx])
end

Base.@ccallable function ace_get_n_basis()::Cint
    return Cint(N_BASIS)
end

# ============================================================================
# BASIS EVALUATION (for descriptor computation)
# ============================================================================

Base.@ccallable function ace_site_basis(
    ws::Ptr{Cvoid},
    z0::Cint,
    nneigh::Cint,
    neighbor_z::Ptr{Cint},
    neighbor_Rij::Ptr{Cdouble},
    basis_out::Ptr{Cdouble}
)::Cint
    if nneigh == 0
        # Return zeros for isolated atom
        for k in 1:N_BASIS
            unsafe_store!(basis_out, 0.0, k)
        end
        return Cint(0)
    end

    Zs = c_read_species(neighbor_z, Int(nneigh))
    Rs = c_read_Rij(neighbor_Rij, Int(nneigh))

    B = site_basis!(_ws(ws), Rs, Zs, Int(z0))

    for k in 1:N_BASIS
        unsafe_store!(basis_out, B[k], k)
    end

    return Cint(0)  # Success
end

# ============================================================================
# BATCH API
# ============================================================================
# Process multiple atoms at once, reducing Python-Julia FFI call overhead.
# Sequential within one call: pass one workspace per THREAD and split the atom range in the
# caller to evaluate concurrently (Threads.@threads does not survive --trim=safe).

Base.@ccallable function ace_batch_energy_forces_virial(
    ws::Ptr{Cvoid},
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
    w = _ws(ws)
    for i in 1:Int(natoms)
        z0 = unsafe_load(z, i)
        nneigh = Int(unsafe_load(neighbor_counts, i))
        offset = Int(unsafe_load(neighbor_offsets, i))  # 0-indexed from C

        if nneigh == 0
            unsafe_store!(energies, E0_of(z2i(z0)), i)
            for k in 1:6
                unsafe_store!(virials, 0.0, (i-1)*6 + k)
            end
        else
            Zs = Vector{Int}(undef, nneigh)
            Rs = Vector{SVector{3, Float64}}(undef, nneigh)

            @inbounds for j in 1:nneigh
                idx = offset + j  # 1-indexed from 0-indexed offset
                Zs[j] = unsafe_load(neighbor_z, idx)
                x = unsafe_load(neighbor_Rij, 3*(idx-1) + 1)
                y = unsafe_load(neighbor_Rij, 3*(idx-1) + 2)
                z_coord = unsafe_load(neighbor_Rij, 3*(idx-1) + 3)
                Rs[j] = SVector(x, y, z_coord)
            end

            F = c_force_view(forces + 3 * offset * sizeof(Cdouble), nneigh)
            Ei, Vi = site_energy_forces_virial!(w, Rs, Zs, Int(z0), F)

            unsafe_store!(energies, Ei, i)

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

""")
end
