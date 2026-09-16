# Code generation utilities for trim-safe exports
#
# These functions generate inline code derived from existing package implementations,
# avoiding Val-based dispatch and other patterns that break trim=safe.
#
# The key insight: we use the upstream packages' code generators at EXPORT TIME,
# then emit specialized code that doesn't require those packages at RUNTIME.
#
# Usage:
#   ylm_code = generate_solid_harmonics_code(maxl)
#   hermite_code = generate_hermite_spline_code(hermite_data, NZ, rcut)
#   # Write to separate files, then include() in main export

using SpheriCart
using StaticArrays
using Pkg
using Printf

"""
    generate_solid_harmonics_code(maxl::Int; T=Float64, normalisation=:L2)

Generate trim-safe inline solid harmonics code for a specific maxl value.
Uses SpheriCart's internal code generators but produces output that doesn't
require Val-based dispatch at runtime.

Returns a String containing Julia code.
"""
function generate_solid_harmonics_code(maxl::Int; T=Float64, normalisation=:L2)
    n_ylm = (maxl + 1)^2

    io = IOBuffer()

    # Get SpheriCart version for documentation
    sc_version = try
        deps = Pkg.dependencies()
        sc_uuid = Base.PkgId(SpheriCart).uuid
        string(deps[sc_uuid].version)
    catch
        "unknown"
    end

    println(io, """
# ============================================================================
# SOLID HARMONICS (inline, trim-safe)
# ============================================================================
# Generated from SpheriCart v$sc_version internals
# maxl = $maxl, normalisation = $normalisation
#
# This code is derived from SpheriCart's recurrence relations but avoids
# Val-based dispatch for trim=safe compatibility.

const MAXL = $maxl
const N_YLM = $n_ylm
""")

    # Generate value-only code using SpheriCart's internal generator
    value_code = SpheriCart._codegen_Zlm(maxl, T, normalisation)

    println(io, """
# Solid harmonics evaluation (values only)
@inline function eval_ylm(R::SVector{3, TT}) where {TT}
    x, y, z = R[1], R[2], R[3]
""")

    # Emit the generated code (skip the return statement, we'll add our own)
    for expr in value_code[1:end-1]
        println(io, "    ", expr)
    end

    # Add the return with proper type annotation
    z_vars = join(["Z_$i" for i in 1:n_ylm], ", ")
    println(io, "    return SVector{$n_ylm, TT}($z_vars)")
    println(io, "end")
    println(io)

    # Handle L=0 specially (SpheriCart._codegen_Zlm_grads has a bug for L=0)
    if maxl == 0
        # For L=0, Y_0^0 is constant (0.28209479...), so gradient is zero
        println(io, """
# Solid harmonics with gradients (L=0 special case)
# Y_0^0 is a constant, so its gradient is zero
@inline function eval_ylm_ed(R::SVector{3, TT}) where {TT}
    x, y, z = R[1], R[2], R[3]
    # Y_0^0 = 1/(2*sqrt(pi)) = 0.28209479177387814
    Z_1 = TT(0.28209479177387814)
    Z = SVector{1, TT}(Z_1)
    dZ = SVector{1, SVector{3, TT}}(zero(SVector{3, TT}))
    return Z, dZ
end
""")
    else
        # Generate gradient code using SpheriCart's internal generator
        grad_code = SpheriCart._codegen_Zlm_grads(maxl, T, normalisation)

        println(io, """
# Solid harmonics with gradients
@inline function eval_ylm_ed(R::SVector{3, TT}) where {TT}
    x, y, z = R[1], R[2], R[3]
""")

        # Emit the generated code (skip the return statement)
        for expr in grad_code[1:end-1]
            println(io, "    ", expr)
        end

        # Add the return with proper types
        dz_vars = join(["dZ_$i" for i in 1:n_ylm], ", ")
        println(io, "    Z = SVector{$n_ylm, TT}($z_vars)")
        println(io, "    dZ = SVector{$n_ylm, SVector{3, TT}}($dz_vars)")
        println(io, "    return Z, dZ")
        println(io, "end")
    end

    return String(take!(io))
end

"""
    hermite_pair_rows(hermite_data, rnl_used) -> Dict{Int, Vector{Int}}

For each ORDERED species pair, the `Rnl` rows whose knot tables are actually emitted: the rows
the `A` basis reads (`rnl_used`, see `_rnl_used` in write_radial.jl) that this pair does not
represent identically.

This is the single definition of that set.  `generate_hermite_spline_code` writes the knot
tables and the cubic at exactly this width, and `_write_evaluation_functions` builds its
per-pair A-accumulation blocks from the same mapping -- local slot `i` of the narrow radial
vector is global row `rows[k][i]`.  Computing it twice, in two files, from two readings of
the same data is exactly the kind of coupling that fails silently, so it is computed once.

`rnl_used = nothing` keeps every row (what the three-argument diagnostic callers get).
"""
function hermite_pair_rows(hermite_data::Dict, rnl_used = nothing)
    first_data = first(values(hermite_data))
    n_rnl = first_data.n_rnl
    used = rnl_used === nothing ? collect(1:n_rnl) : sort(collect(Int, rnl_used))
    @assert !isempty(used) && minimum(used) >= 1 && maximum(used) <= n_rnl """
        rnl_used = $used is not a subset of 1:$n_rnl (the spline tables' row count)"""
    rows = Dict{Int, Vector{Int}}()
    for (k, data) in hermite_data
        rows[k] = [t for t in used
                   if any(!=(0.0), @view data.F[:, t]) || any(!=(0.0), @view data.G[:, t])]
    end
    return rows
end

"""
    generate_hermite_spline_code(hermite_data::Dict, NZ::Int, rcut::Float64;
                                 rnl_used = nothing)

Generate trim-safe inline Hermite cubic spline evaluation code.
The hermite_data comes from `extract_hermite_spline_data()`.

This generates exact code that replicates EquivariantTensors' TransSelSplines
evaluation without any Interpolations.jl or P4ML dependencies.

`rnl_used` (Task 5 / B1) is the set of `Rnl` rows the `A` basis actually reads -- pass
`_rnl_used(tensor)`.  Each pair's knot tables are then written for only those of its rows
that are additionally not identically zero, and the cubic is evaluated at that reduced width
and scattered back into the full `SVector{N_RNL}` the dispatchers return.  On the fitted
Cantor model that is 9 columns instead of 74, i.e. an 8x smaller `F`/`G` table AND 8x less
Hermite arithmetic per edge; the dropped rows are exactly zero in every result (the argument
is in `_rnl_used`'s docstring in write_radial.jl), so this changes no number by one ulp.

`rnl_used = nothing` keeps every row, which is what the standalone diagnostic scripts under
`export/scripts/` that call the three-argument form get.

Returns a String containing Julia code.
"""
function generate_hermite_spline_code(hermite_data::Dict, NZ::Int, rcut::Float64;
                                      rnl_used = nothing)
    io = IOBuffer()

    first_data = first(values(hermite_data))
    n_rnl = first_data.n_rnl
    n_pairs = length(hermite_data)
    used = rnl_used === nothing ? collect(1:n_rnl) : sort(collect(Int, rnl_used))
    @assert !isempty(used) && minimum(used) >= 1 && maximum(used) <= n_rnl """
        rnl_used = $used is not a subset of 1:$n_rnl (the spline tables' row count)"""
    pair_rows = hermite_pair_rows(hermite_data, used)
    m_max = maximum(length(r) for r in values(pair_rows))
    @assert m_max >= 1 """
        no ordered species pair populates a single (n,l) row -- the exported radial basis
        would be identically zero"""

    # `extract_hermite_spline_data` keys the dictionary by the ORDERED pair index
    # k = pair_idx(iz, jz) = (iz-1)*NZ + jz, the same convention `RBASIS_W`, `PAIR_C` and the
    # generated `pair_idx` helper use.  Emitting a table for every k in 1:NZ^2 is what makes
    # the dispatchers at the bottom of this function correct for NZ >= 2; keying or
    # dispatching symmetrically (as this generator did before Task 2) silently swaps the
    # radial basis of the transposed species pairs.
    @assert n_pairs == NZ^2 """
        Hermite spline data has $n_pairs tables, expected NZ^2 = $(NZ^2) (one per ORDERED
        species pair).  extract_hermite_spline_data must key on pair_idx(iz, jz)."""
    @assert sort(collect(keys(hermite_data))) == collect(1:NZ^2) """
        Hermite spline table keys are $(sort(collect(keys(hermite_data)))), expected 1:$(NZ^2)."""
    for (k, data) in hermite_data
        @assert data.pair_idx == k && k == (data.iz - 1) * NZ + data.jz """
            Hermite table $k claims to be pair ($(data.iz), $(data.jz)) / index
            $(data.pair_idx); pair_idx would be $((data.iz - 1) * NZ + data.jz)."""
    end

    println(io, """
# ============================================================================
# HERMITE CUBIC SPLINE RADIAL BASIS (trim-safe)
# ============================================================================
# Exact Hermite cubic spline evaluation matching EquivariantTensors.
# Generated at export time from P4ML splinified model.
#
# Algorithm: Hermite cubic interpolation
#   - Transform r to y-space via Agnesi transform
#   - Find segment in uniform knot grid
#   - Evaluate cubic using knot function values (F) and gradients (G)
#   - Apply envelope function
#
# Reference: EquivariantTensors/src/embed/transsplines.jl

const N_RNL = $n_rnl
const RCUT_GLOBAL = $rcut

# The (n,l) rows any A basis function reads (Task 5 / B1).  Each pair's knot tables below are
# written for the subset of these that the pair actually populates, named PAIR_k_ROWS; every
# other row of the returned SVector{N_RNL} is exactly zero, and is exactly zero in the model
# too (a row outside RNL_USED can reach neither the energy nor the forces -- see _rnl_used in
# export/src/write_radial.jl).
const RNL_USED = $(repr(Tuple(used)))
const N_RNL_USED = $(length(used))  # of N_RNL = $n_rnl

# The widest per-pair row set.  Since Task 6 / B2 the per-pair evaluators return an
# SVector{M_RNL} carrying ONLY the rows that pair populates (PAIR_k_ROWS order), rather than
# scattering them into an SVector{N_RNL} of mostly zeros: the scatter was immediately undone
# by the old N_RNL-wide copy into the work arrays, so the pruning bought nothing until the
# width travelled with it.  `evaluate_Rnl` / `evaluate_Rnl_d` below are full-width wrappers
# for the tests and diagnostic scripts; the evaluation kernel does not call them.
const M_RNL = $(m_max)
""")

    # Generate Agnesi transform and envelope functions for each pair
    for pair_idx in sort(collect(keys(hermite_data)))
        data = hermite_data[pair_idx]
        p = data.agnesi_params

        # Task 5 / B1: keep only the rows the A basis reads AND that this pair populates.
        # One definition, shared with the evaluation writer -- see `hermite_pair_rows`.
        rows = pair_rows[pair_idx]
        m = length(rows)

        println(io, "# === Pair $pair_idx: Species ($(data.iz), $(data.jz)) ===")
        println(io, "# $m of $n_rnl Rnl rows are read by the A basis and nonzero for this pair;")
        println(io, "# the knot tables and the cubic below are written at that reduced width.")
        println(io, "const PAIR_$(pair_idx)_ROWS = SVector{$m, Int}($(repr(rows)))")
        println(io)

        # Agnesi transform parameters
        println(io, "const PAIR_$(pair_idx)_PIN = $(p.pin)")
        println(io, "const PAIR_$(pair_idx)_PCUT = $(p.pcut)")
        println(io, "const PAIR_$(pair_idx)_A = $(p.a)")
        println(io, "const PAIR_$(pair_idx)_B0 = $(p.b0)")
        println(io, "const PAIR_$(pair_idx)_B1 = $(p.b1)")
        println(io, "const PAIR_$(pair_idx)_RIN = $(p.rin)")
        println(io, "const PAIR_$(pair_idx)_REQ = $(p.req)")
        println(io)

        # Spline grid parameters
        println(io, "const PAIR_$(pair_idx)_Y_MIN = $(data.y_min)")
        println(io, "const PAIR_$(pair_idx)_Y_MAX = $(data.y_max)")
        println(io, "const PAIR_$(pair_idx)_N_KNOTS = $(data.n_knots)")
        h = (data.y_max - data.y_min) / (data.n_knots - 1)
        println(io, "const PAIR_$(pair_idx)_H = $h  # Knot spacing")
        println(io)

        # Write F matrix (function values at knots), pruned to PAIR_k_ROWS
        println(io, "# Function values at knots [n_knots × $m] (rows = PAIR_$(pair_idx)_ROWS)")
        println(io, "const PAIR_$(pair_idx)_F = (")
        for i in 1:data.n_knots
            vals = join([@sprintf("%.16e", data.F[i, t]) for t in rows], ", ")
            print(io, "    SVector{$m, Float64}($vals)")
            println(io, i < data.n_knots ? "," : "")
        end
        println(io, ")")
        println(io)

        # Write G matrix (gradients at knots), pruned to PAIR_k_ROWS
        println(io, "# Gradients at knots [n_knots × $m] (rows = PAIR_$(pair_idx)_ROWS)")
        println(io, "const PAIR_$(pair_idx)_G = (")
        for i in 1:data.n_knots
            vals = join([@sprintf("%.16e", data.G[i, t]) for t in rows], ", ")
            print(io, "    SVector{$m, Float64}($vals)")
            println(io, i < data.n_knots ? "," : "")
        end
        println(io, ")")
        println(io)

        # Agnesi transform function - correct generalized Agnesi formula
        # from EquivariantTensors: s = (r-rin)/(req-rin), x = 1/(1+a*s^pin/(1+s^(pin-pcut))), y = b1*x + b0
        println(io, """
# Generalized Agnesi transform: r → y ∈ [-1, 1]
# Formula: s = (r-rin)/(req-rin), x = 1/(1+a*s^pin/(1+s^(pin-pcut))), y = b1*x + b0
@inline function agnesi_transform_$pair_idx(r::T) where {T}
    rin = T(PAIR_$(pair_idx)_RIN)
    req = T(PAIR_$(pair_idx)_REQ)
    a = T(PAIR_$(pair_idx)_A)
    b0 = T(PAIR_$(pair_idx)_B0)
    b1 = T(PAIR_$(pair_idx)_B1)
    pin = PAIR_$(pair_idx)_PIN
    pcut = PAIR_$(pair_idx)_PCUT

    # This function and agnesi_transform_d_$pair_idx below MUST be structurally identical in
    # everything that produces `y`: the same guard, the same division to form `s` (as
    # ET.eval_agnesi does -- a reciprocal-multiply in one of them differs by 1 ulp), and the
    # same clamp.  site_energy takes this route while site_energy_forces* take the other one,
    # so any asymmetry here shows up as a disagreement between the exported entry points.
    s = (r - rin) / (req - rin)
    if s <= zero(T)
        return -one(T)           # r <= rin maps to y = -1 (ET.agnesi_params: xin -> -1)
    end

    # Generalized Agnesi: x = 1 / (1 + a * s^pin / (1 + s^(pin-pcut)))
    s_pin = s^pin
    s_diff = s^(pin - pcut)
    x = one(T) / (one(T) + a * s_pin / (one(T) + s_diff))

    # Linear rescaling to [-1, 1]
    y = b1 * x + b0
    return clamp(y, -one(T), one(T))
end

# Agnesi transform with analytical derivative: returns (y, dy/dr)
@inline function agnesi_transform_d_$pair_idx(r::T) where {T}
    rin = T(PAIR_$(pair_idx)_RIN)
    req = T(PAIR_$(pair_idx)_REQ)
    a = T(PAIR_$(pair_idx)_A)
    b0 = T(PAIR_$(pair_idx)_B0)
    b1 = T(PAIR_$(pair_idx)_B1)
    pin = PAIR_$(pair_idx)_PIN
    pcut = PAIR_$(pair_idx)_PCUT

    # s formed exactly as in agnesi_transform_$pair_idx (division, not reciprocal-multiply);
    # ds_dr is built separately and used only for the chain rule.
    s = (r - rin) / (req - rin)
    ds_dr = one(T) / (req - rin)

    # Same guard, same value, as agnesi_transform_$pair_idx; the derivative is zero because
    # y is constant at the clamp.
    if s <= zero(T)
        return -one(T), zero(T)
    end

    s_pin = s^pin
    s_diff = s^(pin - pcut)
    denom = one(T) + s_diff

    x = one(T) / (one(T) + a * s_pin / denom)
    y = b1 * x + b0

    # Derivative: dx/ds using quotient rule on x = 1/(1 + a*s^pin/(1+s^(pin-pcut)))
    # Let g = a * s^pin / (1 + s^(pin-pcut))
    # x = 1/(1+g), so dx/dg = -1/(1+g)^2 = -x^2
    # dg/ds = a * (pin*s^(pin-1) * (1+s^(pin-pcut)) - s^pin * (pin-pcut)*s^(pin-pcut-1)) / (1+s^(pin-pcut))^2
    #       = a * s^(pin-1) * (pin*(1+s^(pin-pcut)) - (pin-pcut)*s^(pin-pcut)) / (1+s^(pin-pcut))^2
    #       = a * s^(pin-1) * (pin + pin*s^(pin-pcut) - pin*s^(pin-pcut) + pcut*s^(pin-pcut)) / (1+s^(pin-pcut))^2
    #       = a * s^(pin-1) * (pin + pcut*s^(pin-pcut)) / (1+s^(pin-pcut))^2
    g = a * s_pin / denom
    dg_ds = a * s^(pin-1) * (pin + pcut * s_diff) / (denom^2)
    dx_ds = -x^2 * dg_ds

    dy_ds = b1 * dx_ds
    dy_dr = dy_ds * ds_dr

    # y is clamped exactly as in agnesi_transform_$pair_idx, and WHERE it clamps the
    # derivative must be zero: y is constant there, so returning the unclamped dy_dr (as this
    # generator did until Task 2) gives a nonzero force from a flat transform.
    y_clamped = clamp(y, -one(T), one(T))
    y_clamped != y && return y_clamped, zero(T)
    return y, dy_dr
end

# Envelope function: (1 - y²)² in Y-SPACE
# This is the standard ACE quartic envelope
@inline function envelope_$pair_idx(y::T) where {T}
    # Y-space envelope: (1 - y²)²
    y2 = y * y
    one_minus_y2 = max(zero(T), one(T) - y2)
    return one_minus_y2^2
end

# Envelope with derivative w.r.t. y
# d/dy[(1-y²)²] = 2(1-y²)(-2y) = -4y(1-y²)
@inline function envelope_d_$pair_idx(y::T) where {T}
    y2 = y * y
    one_minus_y2 = max(zero(T), one(T) - y2)
    env = one_minus_y2^2
    denv_dy = -4 * y * one_minus_y2
    return env, denv_dy
end
""")

        # Hermite cubic evaluation
        println(io, """
# Hermite cubic spline evaluation (values only), at the pair's own width
@inline function evaluate_Rnl_$pair_idx(r::T)::SVector{M_RNL, T} where {T}
    # Transform to y-space
    y = agnesi_transform_$pair_idx(r)

    # Find segment in uniform grid
    y_clamped = clamp(y, PAIR_$(pair_idx)_Y_MIN, PAIR_$(pair_idx)_Y_MAX)
    t_raw = (y_clamped - PAIR_$(pair_idx)_Y_MIN) / PAIR_$(pair_idx)_H
    t_frac, t_floor = modf(t_raw)
    il = unsafe_trunc(Int, t_floor) + 1  # 1-indexed

    # Clamp segment index
    il = clamp(il, 1, PAIR_$(pair_idx)_N_KNOTS - 1)

    # Get knot data (left and right endpoints of segment)
    @inbounds fl = PAIR_$(pair_idx)_F[il]
    @inbounds fr = PAIR_$(pair_idx)_F[il + 1]
    @inbounds gl = PAIR_$(pair_idx)_H .* PAIR_$(pair_idx)_G[il]   # Pre-scale by h
    @inbounds gr = PAIR_$(pair_idx)_H .* PAIR_$(pair_idx)_G[il + 1]

    # Hermite cubic polynomial (Horner's form)
    # f(t) = ((a3*t + a2)*t + a1)*t + a0
    # where:
    #   a0 = fl
    #   a1 = gl
    #   a2 = -3fl + 3fr - 2gl - gr
    #   a3 = 2fl - 2fr + gl + gr
    a0 = fl
    a1 = gl
    a2 = @. -3fl + 3fr - 2gl - gr
    a3 = @. 2fl - 2fr + gl + gr
    s = @. ((a3 * t_frac + a2) * t_frac + a1) * t_frac + a0

    # Apply envelope in Y-SPACE: (1 - y²)²
    env = envelope_$pair_idx(y)
    v = env .* s
    # Pad the $m computed rows to M_RNL (the widest pair), so every pair's evaluator has one
    # return type.  Local slot i is global (n,l) row PAIR_$(pair_idx)_ROWS[i]; slots beyond
    # $m are exactly zero in every quantity this model produces.
    @inbounds return SVector{M_RNL, T}(
$(_scatter_expr(collect(1:m), ["v[$i]" for i = 1:m], m_max))
    )
end

# Hermite cubic spline evaluation (with derivatives), at the pair's own width
@inline function evaluate_Rnl_d_$pair_idx(r::T)::Tuple{SVector{M_RNL, T}, SVector{M_RNL, T}} where {T}
    # Transform to y-space (with analytical derivative)
    y, dy_dr = agnesi_transform_d_$pair_idx(r)

    # Find segment
    y_clamped = clamp(y, PAIR_$(pair_idx)_Y_MIN, PAIR_$(pair_idx)_Y_MAX)
    t_raw = (y_clamped - PAIR_$(pair_idx)_Y_MIN) / PAIR_$(pair_idx)_H
    t_frac, t_floor = modf(t_raw)
    il = unsafe_trunc(Int, t_floor) + 1

    il = clamp(il, 1, PAIR_$(pair_idx)_N_KNOTS - 1)

    # Get knot data
    @inbounds fl = PAIR_$(pair_idx)_F[il]
    @inbounds fr = PAIR_$(pair_idx)_F[il + 1]
    @inbounds gl = PAIR_$(pair_idx)_H .* PAIR_$(pair_idx)_G[il]
    @inbounds gr = PAIR_$(pair_idx)_H .* PAIR_$(pair_idx)_G[il + 1]

    # Hermite cubic (value)
    a0 = fl
    a1 = gl
    a2 = @. -3fl + 3fr - 2gl - gr
    a3 = @. 2fl - 2fr + gl + gr
    s = @. ((a3 * t_frac + a2) * t_frac + a1) * t_frac + a0

    # Hermite cubic derivative w.r.t. t
    # ds/dt = (3a3*t + 2a2)*t + a1
    ds_dt = @. (3a3 * t_frac + 2a2) * t_frac + a1

    # Chain rule: ds/dy = (ds/dt) / h
    ds_dy = ds_dt ./ PAIR_$(pair_idx)_H

    # Envelope in Y-SPACE and its derivative w.r.t. y
    env, denv_dy = envelope_d_$pair_idx(y)

    # Product rule with chain rule: d/dr[env(y) * s(y)]
    #   = (denv/dy * s + env * ds/dy) * dy/dr
    v = env .* s
    dv = (denv_dy .* s .+ env .* ds_dy) .* dy_dr

    # Padded to M_RNL (see above).
    @inbounds Rnl = SVector{M_RNL, T}(
$(_scatter_expr(collect(1:m), ["v[$i]" for i = 1:m], m_max))
    )
    @inbounds dRnl_dr = SVector{M_RNL, T}(
$(_scatter_expr(collect(1:m), ["dv[$i]" for i = 1:m], m_max))
    )
    return Rnl, dRnl_dr
end

# Narrow (M_RNL, pair-local) -> full width (N_RNL, global (n,l) index).  Compatibility
# wrappers only; nothing on the hot path scatters.
@inline function _scatter_full_$pair_idx(v::SVector{M_RNL, T}) where {T}
    @inbounds return SVector{N_RNL, T}(
$(_scatter_expr(rows, ["v[$i]" for i = 1:m], n_rnl))
    )
end
""")
    end

    # Write dispatcher functions
    println(io, """
# ============================================================================
# DISPATCH FUNCTIONS
# ============================================================================

# Narrow radial basis dispatch (values only) -- what the per-neighbour kernel calls
@inline function _evaluate_Rnl_pair_m(r::T, k::Int)::SVector{M_RNL, T} where {T}""")

    for k in 1:n_pairs
        cond = k == 1 ? "if" : "elseif"
        println(io, "    $cond k == $k; return evaluate_Rnl_$k(r)")
    end
    println(io, "    end")
    # Cold branch: the chain above is exhaustive over 1:NZ^2, so reaching here means the
    # caller computed a species-pair index out of range.  RAISE rather than return zeros --
    # a silently zero radial basis is plausible-looking wrong energies, not a crash.
    println(io, "    error(\"_evaluate_Rnl_pair_m: species-pair index \$k is outside 1:\$(NZ*NZ)\")")
    println(io, "end")
    println(io)

    println(io, """
# Narrow radial basis dispatch (with derivatives)
@inline function _evaluate_Rnl_d_pair_m(r::T, k::Int)::Tuple{SVector{M_RNL, T}, SVector{M_RNL, T}} where {T}""")

    for k in 1:n_pairs
        cond = k == 1 ? "if" : "elseif"
        println(io, "    $cond k == $k; return evaluate_Rnl_d_$k(r)")
    end
    println(io, "    end")
    println(io, "    error(\"_evaluate_Rnl_d_pair_m: species-pair index \$k is outside 1:\$(NZ*NZ)\")")
    println(io, "end")
    println(io)

    println(io, """
# Narrow -> full width dispatch (compatibility wrappers only)
@inline function _scatter_full(v::SVector{M_RNL, T}, k::Int)::SVector{N_RNL, T} where {T}""")
    for k in 1:n_pairs
        cond = k == 1 ? "if" : "elseif"
        println(io, "    $cond k == $k; return _scatter_full_$k(v)")
    end
    println(io, "    end")
    println(io, "    error(\"_scatter_full: species-pair index \$k is outside 1:\$(NZ*NZ)\")")
    println(io, "end")
    println(io)

    println(io, """
# Public API, FULL WIDTH.  These are what the test suite and the diagnostic scripts compare
# against the model's own `Rnl`, so they keep the N_RNL layout.  The evaluation kernel uses
# the narrow dispatchers above instead.
@inline function evaluate_Rnl(r::T, iz::Int, jz::Int)::SVector{N_RNL, T} where {T}
    k = pair_idx(iz, jz)   # the ONE per-pair index; PAIR_k_* tables are written in that order
    return _scatter_full(_evaluate_Rnl_pair_m(r, k), k)
end

@inline function evaluate_Rnl_d(r::T, iz::Int, jz::Int)::Tuple{SVector{N_RNL, T}, SVector{N_RNL, T}} where {T}
    k = pair_idx(iz, jz)
    v, dv = _evaluate_Rnl_d_pair_m(r, k)
    return _scatter_full(v, k), _scatter_full(dv, k)
end
""")

    return String(take!(io))
end
