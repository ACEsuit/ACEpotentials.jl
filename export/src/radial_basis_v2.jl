# Refactored radial basis export using data tables
#
# This version replaces per-pair function generation with:
# 1. Parameter tables (tuples of NamedTuples)
# 2. Generic kernel functions
# 3. Single dispatcher using table lookup
#
# Benefits:
# - Significantly reduced code generation (~80% less radial code)
# - Still fully trim=safe (tuple indexing is static dispatch)
# - Easier to maintain and audit

"""
    _write_etace_radial_basis_v2(io, etace, ps, agnesi_params, NZ, rcut)

Write ETACE radial basis using data-table approach instead of per-pair code generation.
This produces ~80% less code while maintaining identical functionality.
"""
function _write_etace_radial_basis_v2(io, etace, ps, agnesi_params, NZ, rcut)
    println(io, """
# ============================================================================
# RADIAL BASIS (ETACE: Data-table approach - reduced code generation)
# ============================================================================
""")

    # Write cutoff
    println(io, "const RCUT_MAX = $(rcut)")
    println(io)

    # Write species pair index helper
    println(io, """
# Symmetric species pair indexing: (iz, jz) -> pair index
# For NZ=2: (1,1)->1, (1,2)->2, (2,2)->3
@inline function zz2pair_sym(iz::Int, jz::Int)::Int
    i, j = min(iz, jz), max(iz, jz)
    return (i - 1) * NZ - (i - 1) * (i - 2) ÷ 2 + (j - i + 1)
end
""")

    # Extract polynomial basis info
    rembed_layer = etace.rembed.layer
    poly_basis = rembed_layer.basis.l.layers.layer_1
    n_polys = length(poly_basis)

    # Extract polynomial coefficients
    poly_refstate = poly_basis.refstate
    poly_A = poly_refstate.A
    poly_B = poly_refstate.B
    poly_C = poly_refstate.C

    println(io, "# Polynomial basis (orthonormalized Chebyshev)")
    println(io, "const N_POLYS = $(n_polys)")
    println(io, "const POLY_A = SVector{$(n_polys), Float64}($(repr(collect(poly_A))))")
    println(io, "const POLY_B = SVector{$(n_polys), Float64}($(repr(collect(poly_B))))")
    println(io, "const POLY_C = SVector{$(n_polys), Float64}($(repr(collect(poly_C))))")
    println(io)

    # Write polynomial evaluation (same as before, this is already efficient)
    println(io, """
# Polynomial evaluation via 3-term recurrence
@inline function eval_polys(y::T) where {T}
    P = MVector{N_POLYS, T}(undef)
    @inbounds begin
        P[1] = T(POLY_A[1])
        if N_POLYS >= 2
            P[2] = POLY_A[2] * y + POLY_B[2]
        end
        for n = 3:N_POLYS
            P[n] = (POLY_A[n] * y + POLY_B[n]) * P[n-1] + POLY_C[n] * P[n-2]
        end
    end
    return P
end

@inline function eval_polys_ed(y::T) where {T}
    P = MVector{N_POLYS, T}(undef)
    dP = MVector{N_POLYS, T}(undef)
    @inbounds begin
        P[1] = T(POLY_A[1])
        dP[1] = zero(T)
        if N_POLYS >= 2
            P[2] = POLY_A[2] * y + POLY_B[2]
            dP[2] = T(POLY_A[2])
        end
        for n = 3:N_POLYS
            P[n] = (POLY_A[n] * y + POLY_B[n]) * P[n-1] + POLY_C[n] * P[n-2]
            dP[n] = POLY_A[n] * P[n-1] + (POLY_A[n] * y + POLY_B[n]) * dP[n-1] + POLY_C[n] * dP[n-2]
        end
    end
    return P, dP
end
""")

    # Write radial weights
    W_radial = ps.rembed.post.W
    n_rnl = size(W_radial, 1)
    n_pairs = size(W_radial, 3)

    println(io, "const N_RNL = $(n_rnl)")
    println(io)

    # Write weights as tuple of SMatrix (or convert to appropriate form)
    # For trim-safe tuple indexing, we need weights as a tuple
    println(io, "# Radial basis weights per species pair (tuple for trim-safe indexing)")
    println(io, "const RBASIS_W = (")
    for pair_idx in 1:n_pairs
        W_pair = W_radial[:, :, pair_idx]
        W_vec = vec(W_pair)  # Flatten for repr
        println(io, "    SMatrix{$(n_rnl), $(n_polys), Float64, $(n_rnl * n_polys)}($(repr(collect(W_vec)))),")
    end
    println(io, ")")
    println(io)

    # ==========================================================================
    # DATA TABLE: Agnesi transform parameters
    # ==========================================================================
    # Build parameter table for all species pairs
    println(io, "# ============================================================================")
    println(io, "# TRANSFORM PARAMETERS (data table approach)")
    println(io, "# ============================================================================")
    println(io)

    # Write the parameter table as a tuple of NamedTuples
    println(io, "# Agnesi transform parameters per species pair")
    println(io, "const TRANSFORM_PARAMS = (")

    for asym_pair_idx in 1:n_pairs
        # Convert asymmetric pair index to (iz, jz)
        iz = (asym_pair_idx - 1) ÷ NZ + 1
        jz = (asym_pair_idx - 1) % NZ + 1
        # Get symmetric pair index
        sym_i = min(iz, jz)
        sym_j = max(iz, jz)
        sym_pair_idx = (sym_i - 1) * NZ - (sym_i - 1) * (sym_i - 2) ÷ 2 + (sym_j - sym_i + 1)
        p = agnesi_params[sym_pair_idx]

        println(io, "    (rin=$(Float64(p.rin)), req=$(Float64(p.req)), rcut=$(Float64(rcut)), " *
                    "pin=$(Float64(p.pin)), pcut=$(Float64(p.pcut)), a=$(Float64(p.a)), " *
                    "b0=$(Float64(p.b0)), b1=$(Float64(p.b1))),  # pair $asym_pair_idx: ($iz, $jz)")
    end
    println(io, ")")
    println(io)

    # ==========================================================================
    # GENERIC KERNEL FUNCTIONS (embedded, not external dependency)
    # ==========================================================================
    println(io, """
# ============================================================================
# GENERIC KERNEL FUNCTIONS (embedded for self-containment)
# ============================================================================

# Quartic envelope: (1 - y²)²
@inline envelope_quartic(y::T) where {T} = max(zero(T), one(T) - y^2)^2
@inline function envelope_quartic_d(y::T) where {T}
    one_minus_y2 = max(zero(T), one(T) - y^2)
    return one_minus_y2^2, -4 * y * one_minus_y2
end

# Generic Agnesi transform using parameter tuple
@inline function agnesi_transform(r::T, p) where {T}
    if r <= p.rin
        return one(T)
    end
    if r >= p.rcut
        return -one(T)
    end
    s = (r - p.rin) / (p.req - p.rin)
    s_pin = s^p.pin
    s_diff = s^(p.pin - p.pcut)
    denom = one(T) + s_diff
    x = one(T) / (one(T) + p.a * s_pin / denom)
    y = p.b1 * x + p.b0
    return clamp(y, -one(T), one(T))
end

# Agnesi transform with derivative
@inline function agnesi_transform_d(r::T, p) where {T}
    if r <= p.rin
        return one(T), zero(T)
    end
    if r >= p.rcut
        return -one(T), zero(T)
    end

    ds_dr = one(T) / (p.req - p.rin)
    s = (r - p.rin) * ds_dr

    if s <= T(1e-12)
        return one(T), zero(T)
    end

    s_pin = s^p.pin
    s_diff = s^(p.pin - p.pcut)
    denom = one(T) + s_diff

    x = one(T) / (one(T) + p.a * s_pin / denom)
    y = p.b1 * x + p.b0

    # Derivative via chain rule
    dg_ds = p.a * s^(p.pin - 1) * (p.pin + p.pcut * s_diff) / (denom^2)
    dx_ds = -x^2 * dg_ds
    dy_ds = p.b1 * dx_ds
    dy_dr = dy_ds * ds_dr

    y_clamped = clamp(y, -one(T), one(T))
    if y_clamped != y
        return y_clamped, zero(T)
    end
    return y, dy_dr
end
""")

    # ==========================================================================
    # GENERIC RADIAL BASIS FUNCTIONS (using parameter tables)
    # ==========================================================================
    println(io, """
# ============================================================================
# RADIAL BASIS EVALUATION (generic, using parameter tables)
# ============================================================================

# Generic radial basis evaluation for any pair
@inline function _evaluate_Rnl_pair(r::T, pair_idx::Int)::SVector{N_RNL, T} where {T}
    @inbounds p = TRANSFORM_PARAMS[pair_idx]
    y = agnesi_transform(r, p)

    env = envelope_quartic(y)
    if env <= zero(T)
        return zero(SVector{N_RNL, T})
    end

    P = eval_polys(y)
    @inbounds W = RBASIS_W[pair_idx]
    return W * SVector{N_POLYS, T}(env .* P)
end

# Generic radial basis with derivatives
@inline function _evaluate_Rnl_d_pair(r::T, pair_idx::Int)::Tuple{SVector{N_RNL, T}, SVector{N_RNL, T}} where {T}
    @inbounds p = TRANSFORM_PARAMS[pair_idx]
    y, dy_dr = agnesi_transform_d(r, p)

    env, denv_dy = envelope_quartic_d(y)
    denv_dr = denv_dy * dy_dr

    if env <= zero(T)
        return zero(SVector{N_RNL, T}), zero(SVector{N_RNL, T})
    end

    P, dP = eval_polys_ed(y)
    dP_dr = dP .* dy_dr

    P_env = env .* P
    dP_env_dr = denv_dr .* P .+ env .* dP_dr

    @inbounds W = RBASIS_W[pair_idx]
    Rnl = W * SVector{N_POLYS, T}(P_env)
    dRnl = W * SVector{N_POLYS, T}(dP_env_dr)

    return Rnl, dRnl
end

# Public API: dispatch by species indices
@inline function evaluate_Rnl(r::T, iz::Int, jz::Int)::SVector{N_RNL, T} where {T}
    pair_idx = (iz - 1) * NZ + jz  # Asymmetric indexing for weights
    return _evaluate_Rnl_pair(r, pair_idx)
end

@inline function evaluate_Rnl_d(r::T, iz::Int, jz::Int)::Tuple{SVector{N_RNL, T}, SVector{N_RNL, T}} where {T}
    pair_idx = (iz - 1) * NZ + jz
    return _evaluate_Rnl_d_pair(r, pair_idx)
end
""")
end
