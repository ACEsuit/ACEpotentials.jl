# Radial basis and spherical harmonics writing functions
# Split from export_ace_model.jl for maintainability

function _write_spline_radial_basis_header(io, rcut)
    println(io, """
# ============================================================================
# RADIAL BASIS CONFIGURATION
# ============================================================================

const RCUT_MAX = $(rcut)

# Symmetric species pair indexing: (iz, jz) -> pair index
# For NZ=2: (1,1)->1, (1,2)->2, (2,2)->3
@inline function zz2pair_sym(iz::Int, jz::Int)::Int
    i, j = min(iz, jz), max(iz, jz)
    return (i - 1) * NZ - (i - 1) * (i - 2) ÷ 2 + (j - i + 1)
end
""")
end

# Write ETACE radial basis using data-table approach for reduced code generation.
# Uses parameter tables and generic kernel functions instead of per-pair code generation.
function _write_etace_radial_basis(io, etace, ps, agnesi_params, NZ, rcut)
    println(io, """
# ============================================================================
# RADIAL BASIS (ETACE: Data-table approach)
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

    # Write polynomial evaluation
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

    # Write weights as tuple for trim-safe indexing
    println(io, "# Radial basis weights per species pair (tuple for trim-safe indexing)")
    println(io, "const RBASIS_W = (")
    for pair_idx in 1:n_pairs
        W_pair = W_radial[:, :, pair_idx]
        W_vec = vec(W_pair)
        println(io, "    SMatrix{$(n_rnl), $(n_polys), Float64, $(n_rnl * n_polys)}($(repr(collect(W_vec)))),")
    end
    println(io, ")")
    println(io)

    # Write parameter table for all species pairs
    println(io, "# ============================================================================")
    println(io, "# TRANSFORM PARAMETERS (data table approach)")
    println(io, "# ============================================================================")
    println(io)

    println(io, "# Agnesi transform parameters per species pair")
    println(io, "const TRANSFORM_PARAMS = (")

    for asym_pair_idx in 1:n_pairs
        iz = (asym_pair_idx - 1) ÷ NZ + 1
        jz = (asym_pair_idx - 1) % NZ + 1
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

    # Write generic kernel functions
    println(io, """
# ============================================================================
# GENERIC KERNEL FUNCTIONS
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

    # Write generic radial basis functions
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

# ============================================================================
# PAIR POTENTIAL (ETPairModel)
# ============================================================================
#
# Model (src/et_models/et_pair.jl + src/et_models/convert.jl:`convertpair`):
#
#   rembed  = EdgeEmbed( EnvRBranchL(envelope, EmbedDP(agnesi, polys, SelectLinL)) )
#   readout = SelectLinL(n_pairbasis -> 1, NZ, selector = centre species)
#
#   Rnl_pair[edge, n] = env(r_ij) * Σ_q W[n, q, (iz0,jz)] * P_q(y_ij)
#   𝔹[i, n]           = Σ_{j ∈ N(i)} Rnl_pair[edge, n]          (et_pair.jl:48-57)
#   E_pair(i)         = Σ_n Wread[1, n, iz0] * 𝔹[i, n]          (et_pair.jl:25-33)
#
# so, folding the readout into the polynomial coefficients at export time,
#
#   E_pair(i) = Σ_{j ∈ N(i)} env(r_ij) * dot( PAIR_C[(iz0,jz)], P(y_ij) )
#   PAIR_C[(iz0,jz)][q] = Σ_n Wread[1, n, iz0] * W[n, q, (iz0,jz)]
#
# Index conventions, all verified against the sources rather than guessed:
#  * `ET.catcat2idx` (utils/selector.jl) = (i1-1)*NZ + i2 with i1 the *centre* species
#    (the graph stores z0 = species(i), z1 = species(j); EquivariantTensors
#    ext/NeighbourListsExt.jl:19-21), so the SelectLinL weights W and PAIR_C are indexed by
#    the ORDERED pair (iz0, jz) -- same convention as the many-body RBASIS_W above.
#  * the Agnesi transform parameters are stored per SYMMETRIC pair
#    (`_convert_agnesi` loops `for i = 1:NZ, j = i:NZ` and the selector is
#    `catcat2idx_sym`), i.e. NZ*(NZ+1)/2 entries addressed by `symidx`.  The ordered ->
#    symmetric mapping below is the same one `_write_etace_radial_basis` uses for
#    TRANSFORM_PARAMS (lines ~126-136).
#  * the readout weight Wread is per CENTRE species only (shape (1, n_pairbasis, NZ)).
#
# Numerics:
#  * the polynomials are the raw P4ML `OrthPolyBasis1D3T` -- unlike the many-body radial
#    basis the pair basis has NO quartic envelope wrapped around them; the envelope is the
#    separate `PolyEnvelope1sR` branch.
#  * the envelope is `_eval_env_1sr` (src/et_models/convert.jl:233-237):
#        env(r) = (s^-p - 1) * (1 - s) * (s < 1),  s = r / rcut
#  * the transform is `ET.eval_agnesi` (EquivariantTensors src/transforms/agnesi.jl:53-61).
#    A dedicated `_pair_transform_d` is emitted rather than reusing `agnesi_transform_d`
#    because the latter carries `r <= rin -> +1` / `r >= rcut -> -1` shortcuts that
#    `eval_agnesi` does not have (it only clamps), and because the stored parameter tuple
#    has no `rcut` field of its own.  `pin`/`pcut` are kept as `Int` so that `s^pin` is the
#    same integer power `eval_agnesi` evaluates, bit for bit.
function _write_pair_basis(io, pair_calc, NZ)
    pm, ps = pair_calc.model, pair_calc.ps

    branch = pm.rembed.layer            # EnvRBranchL(envelope, rbasis)
    rb     = branch.rbasis              # EmbedDP(trans, basis, post)
    polys  = rb.basis                   # Polynomials4ML.OrthPolyBasis1D3T
    pA, pB, pC = polys.refstate.A, polys.refstate.B, polys.refstate.C
    nq = length(pA)

    W     = ps.rembed.rbasis.post.W     # (n_pairbasis, n_pairpolys, NZ^2)
    Wr    = ps.readout.W                # (1, n_pairbasis, NZ)
    env   = branch.envelope.refstate    # (rcut, p) of the PolyEnvelope1sR branch
    trans = rb.trans.refstate.params    # SVector{NZ(NZ+1)/2} of Agnesi parameters

    n_pairbasis = size(W, 1)
    @assert size(W, 2) == nq "pair SelectLinL in_dim $(size(W,2)) != n polys $nq"
    @assert size(W, 3) == NZ^2 "pair SelectLinL has $(size(W,3)) categories, expected NZ^2 = $(NZ^2)"
    @assert size(Wr) == (1, n_pairbasis, NZ) "pair readout W has size $(size(Wr)), expected (1, $n_pairbasis, $NZ)"
    @assert length(trans) == (NZ * (NZ + 1)) ÷ 2 "pair transform params: $(length(trans)) entries, expected $((NZ*(NZ+1))÷2) (per symmetric pair)"

    println(io, """
# ============================================================================
# PAIR POTENTIAL (ETPairModel; readout folded into per-ordered-pair coefficients)
# ============================================================================
""")

    println(io, "# Orthogonal polynomial basis of the pair term (3-term recurrence)")
    println(io, "const N_PAIRPOLYS = $(nq)")
    println(io, "const PAIRPOLY_A = SVector{$(nq), Float64}($(repr(collect(pA))))")
    println(io, "const PAIRPOLY_B = SVector{$(nq), Float64}($(repr(collect(pB))))")
    println(io, "const PAIRPOLY_C = SVector{$(nq), Float64}($(repr(collect(pC))))")
    println(io)

    println(io, "# PolyEnvelope1sR:  env(r) = (s^-p - 1) * (1 - s) * (s < 1),  s = r / rcut")
    println(io, "const PAIR_ENV_RCUT = $(Float64(env.rcut))")
    println(io, "const PAIR_ENV_P = $(Int(env.p))")
    println(io)

    println(io, "# Readout-folded polynomial coefficients, one entry per ORDERED pair (iz0, jz)")
    println(io, "const PAIR_C = (")
    for iz0 in 1:NZ, jz in 1:NZ
        k = (iz0 - 1) * NZ + jz
        c = vec(transpose(Wr[1, :, iz0]) * W[:, :, k])
        @assert length(c) == nq
        println(io, "    SVector{$(nq), Float64}($(repr(collect(c)))),  # pair $k: ($iz0, $jz)")
    end
    println(io, ")")
    println(io)

    println(io, "# Agnesi transform parameters, expanded from the symmetric-pair storage")
    println(io, "# to one entry per ORDERED pair (iz0, jz) so no runtime mapping is needed.")
    println(io, "const PAIR_TRANSFORM_PARAMS = (")
    for iz0 in 1:NZ, jz in 1:NZ
        k = (iz0 - 1) * NZ + jz
        sym_i, sym_j = min(iz0, jz), max(iz0, jz)
        sym_idx = (sym_i - 1) * NZ - (sym_i - 1) * (sym_i - 2) ÷ 2 + (sym_j - sym_i + 1)
        p = trans[sym_idx]
        println(io, "    (pin=$(Int(p.pin)), pcut=$(Int(p.pcut)), a=$(Float64(p.a)), " *
                    "b0=$(Float64(p.b0)), b1=$(Float64(p.b1)), rin=$(Float64(p.rin)), " *
                    "req=$(Float64(p.req))),  # pair $k: ($iz0, $jz) -> sym $sym_idx")
    end
    println(io, ")")
    println(io)

    println(io, raw"""
# Pair envelope with derivative d/dr
@inline function _pair_env_d(r::Float64)
    s = r / PAIR_ENV_RCUT
    s >= 1.0 && return 0.0, 0.0
    sp = s^(-PAIR_ENV_P)
    e = (sp - 1.0) * (1.0 - s)
    de = (-PAIR_ENV_P * sp / s) * (1.0 - s) - (sp - 1.0)
    return e, de / PAIR_ENV_RCUT
end

# Generalized Agnesi transform of the pair term, with derivative d/dr.
# Mirrors ET.eval_agnesi exactly: no rin/rcut shortcuts, clamp to [-1, 1] only.
@inline function _pair_transform_d(r::Float64, p)
    ds_dr = 1.0 / (p.req - p.rin)
    s = (r - p.rin) * ds_dr
    s_pin = s^p.pin
    s_diff = s^(p.pin - p.pcut)
    denom = 1.0 + s_diff
    x = 1.0 / (1.0 + p.a * s_pin / denom)
    y = p.b1 * x + p.b0
    dg_ds = p.a * s^(p.pin - 1) * (p.pin + p.pcut * s_diff) / (denom * denom)
    dy_dr = p.b1 * (-x * x * dg_ds) * ds_dr
    y_clamped = clamp(y, -1.0, 1.0)
    y_clamped != y && return y_clamped, 0.0
    return y, dy_dr
end

# Pair site-energy contribution of one neighbour, and its derivative w.r.t. r.
#   (iz0, jz) is the ORDERED pair: centre species first.
@inline function pair_energy_d(r::Float64, iz0::Int, jz::Int)
    e, de = _pair_env_d(r)
    e == 0.0 && return 0.0, 0.0
    k = (iz0 - 1) * NZ + jz
    @inbounds p = PAIR_TRANSFORM_PARAMS[k]
    y, dy_dr = _pair_transform_d(r, p)
    @inbounds c = PAIR_C[k]
    @inbounds begin
        P1 = PAIRPOLY_A[1]
        dP1 = 0.0
        P2 = PAIRPOLY_A[2] * y + PAIRPOLY_B[2]
        dP2 = PAIRPOLY_A[2]
        v = c[1] * P1 + c[2] * P2
        dv = c[2] * dP2
        for n = 3:N_PAIRPOLYS
            Pn = (PAIRPOLY_A[n] * y + PAIRPOLY_B[n]) * P2 + PAIRPOLY_C[n] * P1
            dPn = PAIRPOLY_A[n] * P2 + (PAIRPOLY_A[n] * y + PAIRPOLY_B[n]) * dP2 +
                  PAIRPOLY_C[n] * dP1
            v += c[n] * Pn
            dv += c[n] * dPn
            P1, P2, dP1, dP2 = P2, Pn, dP2, dPn
        end
    end
    return e * v, de * v + e * dv * dy_dr
end

@inline pair_energy(r::Float64, iz0::Int, jz::Int) = pair_energy_d(r, iz0, jz)[1]
""")
end

# Stub emitted when the exported model has no ETPairModel term, so that the generated
# evaluation functions are identical in both cases.  Returning literal zeros makes every
# pair contribution an exact no-op (x + 0.0 == x, x + 0.0 * r̂ == x).
function _write_no_pair_basis(io)
    println(io, """
# ============================================================================
# PAIR POTENTIAL: none in this model (many-body + E0 only)
# ============================================================================

@inline pair_energy_d(r::Float64, iz0::Int, jz::Int) = (0.0, 0.0)
@inline pair_energy(r::Float64, iz0::Int, jz::Int) = 0.0
""")
end


function _write_radial_basis(io, rbasis::ACEpotentials.Models.SplineRnlrzzBasis, NZ)
    println(io, """
# ============================================================================
# RADIAL BASIS (Spline-based)
# ============================================================================
""")

    # Write rin0cuts
    println(io, "# Cutoff parameters: (rin, r0, rcut) for each species pair")
    rcut_max = 0.0
    for iz in 1:NZ
        for jz in 1:NZ
            rin0cut = rbasis.rin0cuts[iz, jz]
            println(io, "const RIN0CUT_$(iz)_$(jz) = (rin=$(rin0cut.rin), r0=$(rin0cut.r0), rcut=$(rin0cut.rcut))")
            rcut_max = max(rcut_max, rin0cut.rcut)
        end
    end
    # Maximum cutoff across all species pairs (used for neighbor list construction)
    println(io, "const RCUT_MAX = $(rcut_max)")
    println(io)

    # Write spec
    println(io, "const R_SPEC = $(repr(rbasis.spec))")
    println(io, "const N_RNL = $(length(rbasis.spec))")
    println(io)

    # Write transforms
    println(io, "# Radial transforms")
    for iz in 1:NZ
        for jz in 1:NZ
            trans = rbasis.transforms[iz, jz]
            _write_transform(io, trans, iz, jz)
        end
    end

    # Write envelopes
    println(io, "# Radial envelopes")
    for iz in 1:NZ
        for jz in 1:NZ
            env = rbasis.envelopes[iz, jz]
            _write_envelope(io, env, iz, jz)
        end
    end

    # Write splines - this is the key data for evaluation
    println(io, "# Spline knots and coefficients")
    for iz in 1:NZ
        for jz in 1:NZ
            spl = rbasis.splines[iz, jz]
            _write_spline(io, spl, iz, jz)
        end
    end

    # Write evaluation function
    println(io, """

# Radial basis evaluation (returns SVector{N_RNL, T})
@inline function evaluate_Rnl(r::T, iz::Int, jz::Int)::SVector{N_RNL, T} where {T}
    # Get transform and envelope for this species pair
    # (Dispatched at compile time for small NZ)
    """)

    for iz in 1:NZ
        for jz in 1:NZ
            cond = iz == 1 && jz == 1 ? "if" : "elseif"
            println(io, "    $cond iz == $iz && jz == $jz")
            println(io, "        rcut = RIN0CUT_$(iz)_$(jz).rcut")
            println(io, "        if r >= rcut; return zero(SVector{N_RNL, T}); end")
            println(io, "        x = transform_$(iz)_$(jz)(r)")
            println(io, "        env = envelope_$(iz)_$(jz)(r, x)")
            println(io, "        return env .* spline_$(iz)_$(jz)(x)")
        end
    end
    println(io, "    end")
    println(io, "    # Fallback for unknown species pair")
    println(io, "    return zero(SVector{N_RNL, T})")
    println(io, "end")
    println(io)

    # Write evaluation with derivative function
    println(io, """
# Radial basis evaluation with derivative dRnl/dr
@inline function evaluate_Rnl_d(r::T, iz::Int, jz::Int)::Tuple{SVector{N_RNL, T}, SVector{N_RNL, T}} where {T}
    """)

    for iz in 1:NZ
        for jz in 1:NZ
            cond = iz == 1 && jz == 1 ? "if" : "elseif"
            println(io, "    $cond iz == $iz && jz == $jz")
            println(io, "        rcut = RIN0CUT_$(iz)_$(jz).rcut")
            println(io, "        if r >= rcut; return zero(SVector{N_RNL, T}), zero(SVector{N_RNL, T}); end")
            println(io, "        # Transform with derivative")
            println(io, "        x, dx_dr = transform_d_$(iz)_$(jz)(r)")
            println(io, "        # Envelope with derivative")
            println(io, "        env, denv_dx = envelope_d_$(iz)_$(jz)(r, x)")
            println(io, "        # Spline with derivative")
            println(io, "        spl, dspl_dx = spline_d_$(iz)_$(jz)(x)")
            println(io, "        # Rnl = env * spl")
            println(io, "        Rnl = env .* spl")
            println(io, "        # dRnl/dr = denv/dr * spl + env * dspl/dr")
            println(io, "        #         = denv/dx * dx/dr * spl + env * dspl/dx * dx/dr")
            println(io, "        dRnl_dr = (denv_dx * dx_dr) .* spl .+ env .* dspl_dx .* dx_dr")
            println(io, "        return Rnl, dRnl_dr")
        end
    end
    println(io, "    end")
    println(io, "    # Fallback for unknown species pair")
    println(io, "    return zero(SVector{N_RNL, T}), zero(SVector{N_RNL, T})")
    println(io, "end")
    println(io)
end

function _write_transform(io, trans::ACEpotentials.Models.NormalizedTransform, iz, jz)
    inner = trans.trans
    println(io, """
# Transform $(iz)-$(jz): NormalizedTransform
const TRANS_$(iz)_$(jz)_YIN = $(trans.yin)
const TRANS_$(iz)_$(jz)_YCUT = $(trans.ycut)
const TRANS_$(iz)_$(jz)_P = $(inner.p)
const TRANS_$(iz)_$(jz)_Q = $(inner.q)
const TRANS_$(iz)_$(jz)_R0 = $(inner.r0)
const TRANS_$(iz)_$(jz)_RIN = $(inner.rin)
const TRANS_$(iz)_$(jz)_A = $(inner.a)

@inline function transform_$(iz)_$(jz)(r::T) where {T}
    rin = TRANS_$(iz)_$(jz)_RIN
    if r <= rin
        return one(T)
    end
    r0, q, p, a = TRANS_$(iz)_$(jz)_R0, TRANS_$(iz)_$(jz)_Q, TRANS_$(iz)_$(jz)_P, TRANS_$(iz)_$(jz)_A
    s = (r - rin) / (r0 - rin)
    y = (a + s^q) / (a + s^p)
    # Normalize
    yin, ycut = TRANS_$(iz)_$(jz)_YIN, TRANS_$(iz)_$(jz)_YCUT
    return clamp(-one(T) + 2 * (y - yin) / (ycut - yin), -one(T), one(T))
end

# Transform with derivative dx/dr
@inline function transform_d_$(iz)_$(jz)(r::T) where {T}
    rin = TRANS_$(iz)_$(jz)_RIN
    if r <= rin
        return one(T), zero(T)
    end
    r0, q, p, a = TRANS_$(iz)_$(jz)_R0, TRANS_$(iz)_$(jz)_Q, TRANS_$(iz)_$(jz)_P, TRANS_$(iz)_$(jz)_A
    yin, ycut = TRANS_$(iz)_$(jz)_YIN, TRANS_$(iz)_$(jz)_YCUT

    s = (r - rin) / (r0 - rin)
    ds_dr = one(T) / (r0 - rin)

    # y = (a + s^q) / (a + s^p)
    num = a + s^q
    den = a + s^p
    y = num / den

    # dy/ds = (q*s^(q-1)*den - p*s^(p-1)*num) / den^2
    dnum_ds = q > 0 ? q * s^(q-1) : zero(T)
    dden_ds = p > 0 ? p * s^(p-1) : zero(T)
    dy_ds = (dnum_ds * den - dden_ds * num) / (den * den)
    dy_dr = dy_ds * ds_dr

    # x = -1 + 2*(y - yin)/(ycut - yin)
    scale = 2 / (ycut - yin)
    x = -one(T) + scale * (y - yin)
    dx_dr = scale * dy_dr

    # Clamp: if outside [-1, 1], derivative is zero
    if x <= -one(T) || x >= one(T)
        return clamp(x, -one(T), one(T)), zero(T)
    end
    return x, dx_dr
end
""")
end

function _write_envelope(io, env::ACEpotentials.Models.PolyEnvelope2sX, iz, jz)
    println(io, """
# Envelope $(iz)-$(jz): PolyEnvelope2sX
const ENV_$(iz)_$(jz)_X1 = $(env.x1)
const ENV_$(iz)_$(jz)_X2 = $(env.x2)
const ENV_$(iz)_$(jz)_P1 = $(env.p1)
const ENV_$(iz)_$(jz)_P2 = $(env.p2)
const ENV_$(iz)_$(jz)_S = $(env.s)

@inline function envelope_$(iz)_$(jz)(r::T, x::T) where {T}
    x1, x2 = ENV_$(iz)_$(jz)_X1, ENV_$(iz)_$(jz)_X2
    if !(x1 < x < x2)
        return zero(T)
    end
    p1, p2, s = ENV_$(iz)_$(jz)_P1, ENV_$(iz)_$(jz)_P2, ENV_$(iz)_$(jz)_S
    return s * (x - x1)^p1 * (x2 - x)^p2
end

# Envelope with derivative denv/dx
@inline function envelope_d_$(iz)_$(jz)(r::T, x::T) where {T}
    x1, x2 = ENV_$(iz)_$(jz)_X1, ENV_$(iz)_$(jz)_X2
    if !(x1 < x < x2)
        return zero(T), zero(T)
    end
    p1, p2, s = ENV_$(iz)_$(jz)_P1, ENV_$(iz)_$(jz)_P2, ENV_$(iz)_$(jz)_S

    # env = s * (x - x1)^p1 * (x2 - x)^p2
    left = (x - x1)^p1
    right = (x2 - x)^p2
    env = s * left * right

    # denv/dx = s * (p1*(x-x1)^(p1-1)*(x2-x)^p2 - p2*(x-x1)^p1*(x2-x)^(p2-1))
    dleft_dx = p1 > 0 ? p1 * (x - x1)^(p1-1) : zero(T)
    dright_dx = p2 > 0 ? -p2 * (x2 - x)^(p2-1) : zero(T)
    denv_dx = s * (dleft_dx * right + left * dright_dx)

    return env, denv_dx
end
""")
end

function _write_spline(io, spl, iz, jz)
    # Extract spline data
    # The spline is a ScaledInterpolation wrapping a BSplineInterpolation
    itp = spl.itp  # ScaledInterpolation
    inner_itp = itp.itp  # BSplineInterpolation

    # Get the knots and coefficients
    knots = itp.ranges[1]  # The x values (StepRangeLen)
    x_start = first(knots)
    x_step = step(knots)
    x_len = length(knots)

    # Get coefficients (these are SVector values)
    coeffs = inner_itp.coefs
    n_rnl = length(first(coeffs))

    # Write coefficients as a Tuple of SVectors for type stability
    println(io, """
# Spline $(iz)-$(jz)
const SPL_$(iz)_$(jz)_XSTART = $(x_start)
const SPL_$(iz)_$(jz)_XSTEP = $(x_step)
const SPL_$(iz)_$(jz)_XLEN = $(x_len)
""")

    # Write each coefficient as an SVector
    print(io, "const SPL_$(iz)_$(jz)_COEFFS = (\n")
    for (i, c) in enumerate(coeffs)
        vals = join(["$v" for v in c], ", ")
        print(io, "    SVector{$(n_rnl), Float64}($(vals))")
        if i < length(coeffs)
            println(io, ",")
        else
            println(io)
        end
    end
    println(io, ")")
    println(io)

    println(io, """
# Linear interpolation of spline coefficients
@inline function spline_$(iz)_$(jz)(x::T) where {T}
    # Find interval
    idx_f = (x - SPL_$(iz)_$(jz)_XSTART) / SPL_$(iz)_$(jz)_XSTEP + 1
    idx = floor(Int, idx_f)
    idx = clamp(idx, 1, SPL_$(iz)_$(jz)_XLEN - 1)
    t = T(idx_f - idx)

    # Linear interpolation between spline coefficients (type-stable via Tuple indexing)
    @inbounds c0 = SPL_$(iz)_$(jz)_COEFFS[idx]
    @inbounds c1 = SPL_$(iz)_$(jz)_COEFFS[idx+1]
    return (one(T) - t) .* c0 .+ t .* c1
end

# Spline with derivative dspl/dx
@inline function spline_d_$(iz)_$(jz)(x::T) where {T}
    # Find interval
    idx_f = (x - SPL_$(iz)_$(jz)_XSTART) / SPL_$(iz)_$(jz)_XSTEP + 1
    idx = floor(Int, idx_f)
    idx = clamp(idx, 1, SPL_$(iz)_$(jz)_XLEN - 1)
    t = T(idx_f - idx)

    @inbounds c0 = SPL_$(iz)_$(jz)_COEFFS[idx]
    @inbounds c1 = SPL_$(iz)_$(jz)_COEFFS[idx+1]

    # spl = (1-t)*c0 + t*c1
    spl = (one(T) - t) .* c0 .+ t .* c1

    # dspl/dx = dspl/dt * dt/dx = (c1 - c0) / step
    dspl_dx = (c1 .- c0) ./ T(SPL_$(iz)_$(jz)_XSTEP)

    return spl, dspl_dx
end
""")
end

function _write_spherical_harmonics(io, maxl)
    # Generate inline solid harmonics code using SpheriCart's code generators
    # This produces trim-safe code that doesn't require SpheriCart at runtime
    ylm_code = generate_solid_harmonics_code(maxl)

    # Write the generated code
    print(io, ylm_code)
    println(io)
end

# Fallback for non-spline basis
function _write_radial_basis(io, rbasis, NZ)
    error("Only SplineRnlrzzBasis is currently supported for export. Use splinify_first=true or call splinify() on your model first.")
end

# Additional envelope types
function _write_envelope(io, env, iz, jz)
    error("Envelope type $(typeof(env)) not yet supported for export")
end
