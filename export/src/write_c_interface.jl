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
# WORKSPACE POOL AND HANDLES
# ============================================================================
#
# The pool is built WHEN THE IMAGE IS BUILT and the handle is a 1-BASED INDEX into it, cast to
# a pointer.  Both halves of that are load-bearing, and both were arrived at by measurement
# rather than by taste:
#
#  1. NO JULIA OBJECT ADDRESS CROSSES THE C BOUNDARY.  `pointer_from_objref(ws)` +
#     `unsafe_pointer_to_objref(p)::Workspace` is the obvious implementation.  In a juliac
#     --trim library it fails: a 2048-atom, 100-step LAMMPS run survives the first force
#     evaluation and then dies with a TypeError inside the pointer conversion at
#     `Allocations: 98048 ... GC: 1` -- on the FIRST garbage collection.
#
#  2. THE WORKSPACES ARE CREATED AT IMAGE BUILD TIME, not at runtime.  Replacing the raw
#     address with an index into a `const WORKSPACES = Workspace[]` that is `push!`ed to at
#     runtime did NOT fix it: the same run then dies with a smashed malloc arena (SEGV in
#     glibc's `unlink_chunk` from `_int_malloc`), i.e. the workspace's buffers were collected
#     and reused underneath the library.  Whatever roots a runtime-allocated object pushed
#     into an image-baked global in a trimmed image, it is not sufficient.  The pattern that
#     demonstrably DOES work in these libraries is the one the pre-B2 code used: fixed-size
#     arrays created when the image is built and never reallocated.  So the pool is built
#     here, at image build time, and `ace_workspace_new` only hands out a slot.
#
#     This is also why `Workspace` has no variable-length field: the per-neighbour cache that
#     the first version of the kernel carried would have had to be `resize!`d, which is
#     exactly the operation this constraint forbids.  `_forces_from_∂A!` re-evaluates the edge
#     instead.  Neither the image nor the workspace depends on the neighbour count, so there
#     is still NO neighbour cap.
#
#  3. The pool is FIXED SIZE.  MAX_WORKSPACES bounds the number of CONCURRENT workspaces (one
#     per OpenMP thread in the plugin, one per calculator instance in Python) -- it is not a
#     bound on anything physical.  `ace_workspace_new` returns NULL when the pool is
#     exhausted, and both callers check for it.
#
# THE HANDLE IS TAGGED.  It is `WORKSPACE_TAG | idx`, not a bare index.  A bare index would
# have been strictly WORSE than the raw pointer it replaced at catching caller error: every
# value in 1:MAX_WORKSPACES is a valid slot, so a caller that passes `z0` (14, say) where the
# handle belongs -- exactly what a library compiled before this ABI does -- gets a VALID
# workspace and a plausible wrong answer.  With the tag, 14 is rejected.  The check costs one
# AND and two compares ONCE PER SITE, against ~57 us of evaluation.
#
# A freed slot is marked free and reused; the handle carries the index, so slots never move.
# Handle 0 (NULL) is never valid, which matches the C convention that `ace_workspace_new`
# returns non-NULL on success.
#
# new/free take a lock (they mutate the shared free list); the HOT entries take none.

const MAX_WORKSPACES = 32
# High bits of every handle.  Chosen so that a small integer (a species number, a neighbour
# count, 0, -1) can never be mistaken for one.
const WORKSPACE_TAG = UInt(0x0ace0000)
const WORKSPACES = Workspace[new_workspace() for _ in 1:MAX_WORKSPACES]
const WORKSPACE_TAKEN = fill(false, MAX_WORKSPACES)
const WORKSPACES_LOCK = ReentrantLock()

Base.@ccallable function ace_workspace_new()::Ptr{Cvoid}
    idx = 0
    lock(WORKSPACES_LOCK)
    try
        @inbounds for i in 1:MAX_WORKSPACES
            if !WORKSPACE_TAKEN[i]
                WORKSPACE_TAKEN[i] = true
                idx = i
                break
            end
        end
    finally
        unlock(WORKSPACES_LOCK)
    end
    idx == 0 && return Ptr{Cvoid}(UInt(0))       # NULL == pool exhausted
    return Ptr{Cvoid}(WORKSPACE_TAG | UInt(idx))
end

Base.@ccallable function ace_workspace_free(p::Ptr{Cvoid})::Cvoid
    idx = _ws_index(p)
    if idx != 0
        lock(WORKSPACES_LOCK)
        try
            @inbounds WORKSPACE_TAKEN[idx] = false
        finally
            unlock(WORKSPACES_LOCK)
        end
    end
    return nothing
end

Base.@ccallable function ace_max_workspaces()::Cint
    return Cint(MAX_WORKSPACES)
end

# ============================================================================
# HELPER FUNCTIONS FOR C INTERFACE
# ============================================================================

# Handle -> slot index, or 0 if this is not one of our handles.  See the WORKSPACE_TAG note.
@inline function _ws_index(p::Ptr{Cvoid})
    u = UInt(p)
    (u & ~UInt(0xffff)) == WORKSPACE_TAG || return 0
    idx = Int(u & UInt(0xffff))
    (idx < 1 || idx > MAX_WORKSPACES) && return 0
    return idx
end

@inline _ws(idx::Int) = @inbounds WORKSPACES[idx]

# Reported once per bad call, to stderr, and then the entry point returns a sentinel.  It does
# NOT `throw`: a Julia exception unwinding out of a `Base.@ccallable` in a --trim image and
# into LAMMPS' C++ frames is undefined behaviour, and this is only reachable from a caller bug
# (a handle that was never `ace_workspace_new`ed, or an argument list off by one).  A NaN
# energy propagates visibly through LAMMPS' thermo output; a longjmp through a foreign stack
# does not.
@inline function _bad_handle()
    println(Core.stderr,
            "ace: invalid workspace handle -- obtain one from ace_workspace_new() and pass ",
            "it as the FIRST argument of every ace_site_*/ace_batch_* call")
    return nothing
end

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

# Force output.  The kernel writes into a Julia `Vector{SVector{3,Float64}}` and this copies
# it out; it does NOT alias the caller's buffer.
#
# Aliasing it was tried and REVERTED.  `unsafe_wrap(Array, ptr, 3n; own = false)` followed by
# `reinterpret(SVector{3,Float64}, ...)` gives a zero-copy view and is the obvious
# optimisation -- and in a juliac --trim library it corrupts the C heap: a 2048-atom,
# 100-step LAMMPS run dies inside glibc's `unlink_chunk` from `_int_malloc`, i.e. with a
# smashed malloc arena, after surviving the short `run 0` that every accuracy gate in this
# project uses.  The copy costs 3n stores per site against ~200 us of evaluation; it is not
# worth re-litigating without a specific diagnosis of what `unsafe_wrap` does to a foreign
# pointer under `--trim`.
@inline function c_write_forces!(ptr::Ptr{Cdouble}, forces::Vector{SVector{3, Float64}})
    @inbounds for j in 1:length(forces)
        unsafe_store!(ptr, forces[j][1], 3*(j-1) + 1)
        unsafe_store!(ptr, forces[j][2], 3*(j-1) + 2)
        unsafe_store!(ptr, forces[j][3], 3*(j-1) + 3)
    end
end

@inline _force_buffer(nneigh::Int) = Vector{SVector{3, Float64}}(undef, nneigh)

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
    wi = _ws_index(ws)
    wi == 0 && (_bad_handle(); return NaN)
    if nneigh == 0
        return E0_of(z2i(z0))
    end

    Zs = c_read_species(neighbor_z, Int(nneigh))
    Rs = c_read_Rij(neighbor_Rij, Int(nneigh))

    return site_energy!(_ws(wi), Rs, Zs, Int(z0))
end

Base.@ccallable function ace_site_energy_forces(
    ws::Ptr{Cvoid},
    z0::Cint,
    nneigh::Cint,
    neighbor_z::Ptr{Cint},
    neighbor_Rij::Ptr{Cdouble},
    forces::Ptr{Cdouble}
)::Cdouble
    wi = _ws_index(ws)
    wi == 0 && (_bad_handle(); return NaN)
    if nneigh == 0
        return E0_of(z2i(z0))
    end

    Zs = c_read_species(neighbor_z, Int(nneigh))
    Rs = c_read_Rij(neighbor_Rij, Int(nneigh))
    F = _force_buffer(Int(nneigh))
    Ei = site_energy_forces!(_ws(wi), Rs, Zs, Int(z0), F)
    c_write_forces!(forces, F)

    return Ei
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
    wi = _ws_index(ws)
    wi == 0 && (_bad_handle(); return NaN)
    if nneigh == 0
        for k in 1:6
            unsafe_store!(virial, 0.0, k)
        end
        return E0_of(z2i(z0))
    end

    Zs = c_read_species(neighbor_z, Int(nneigh))
    Rs = c_read_Rij(neighbor_Rij, Int(nneigh))
    F = _force_buffer(Int(nneigh))
    Ei, Vi = site_energy_forces_virial!(_ws(wi), Rs, Zs, Int(z0), F)
    c_write_forces!(forces, F)
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
    wi = _ws_index(ws)
    wi == 0 && (_bad_handle(); return Cint(-1))
    if nneigh == 0
        # Return zeros for isolated atom
        for k in 1:N_BASIS
            unsafe_store!(basis_out, 0.0, k)
        end
        return Cint(0)
    end

    Zs = c_read_species(neighbor_z, Int(nneigh))
    Rs = c_read_Rij(neighbor_Rij, Int(nneigh))

    B = site_basis!(_ws(wi), Rs, Zs, Int(z0))

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
    wi = _ws_index(ws)
    wi == 0 && (_bad_handle(); return nothing)
    w = _ws(wi)
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

            F = _force_buffer(nneigh)
            Ei, Vi = site_energy_forces_virial!(w, Rs, Zs, Int(z0), F)
            c_write_forces!(forces + 3 * offset * sizeof(Cdouble), F)

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
