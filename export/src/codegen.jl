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
    generate_hermite_spline_code(hermite_data::Dict, NZ::Int)

Generate trim-safe inline Hermite cubic spline evaluation code.
The hermite_data comes from `extract_hermite_spline_data()`.

This generates exact code that replicates EquivariantTensors' TransSelSplines
evaluation without any Interpolations.jl or P4ML dependencies.

Returns a String containing Julia code.
"""
function generate_hermite_spline_code(hermite_data::Dict, NZ::Int, rcut::Float64)
    io = IOBuffer()

    first_data = first(values(hermite_data))
    n_rnl = first_data.n_rnl
    n_pairs = length(hermite_data)

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
""")

    # Generate Agnesi transform and envelope functions for each pair
    for pair_idx in sort(collect(keys(hermite_data)))
        data = hermite_data[pair_idx]
        p = data.agnesi_params

        println(io, "# === Pair $pair_idx: Species ($(data.iz), $(data.jz)) ===")
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

        # Write F matrix (function values at knots)
        println(io, "# Function values at knots [n_knots × n_rnl]")
        println(io, "const PAIR_$(pair_idx)_F = (")
        for i in 1:data.n_knots
            vals = join([@sprintf("%.16e", v) for v in data.F[i, :]], ", ")
            print(io, "    SVector{$n_rnl, Float64}($vals)")
            println(io, i < data.n_knots ? "," : "")
        end
        println(io, ")")
        println(io)

        # Write G matrix (gradients at knots)
        println(io, "# Gradients at knots [n_knots × n_rnl]")
        println(io, "const PAIR_$(pair_idx)_G = (")
        for i in 1:data.n_knots
            vals = join([@sprintf("%.16e", v) for v in data.G[i, :]], ", ")
            print(io, "    SVector{$n_rnl, Float64}($vals)")
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

    # Normalized distance from rin to req
    s = (r - rin) / (req - rin)

    # Generalized Agnesi: x = 1 / (1 + a * s^pin / (1 + s^(pin-pcut)))
    s_pin = s^pin
    s_diff = s^(pin - pcut)
    x = one(T) / (one(T) + a * s_pin / (one(T) + s_diff))

    # Linear rescaling to [-1, 1]
    y = b1 * x + b0
    y = clamp(y, -one(T), one(T))

    return y
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

    ds_dr = one(T) / (req - rin)
    s = (r - rin) * ds_dr

    # Avoid numerical issues at s=0
    if s <= zero(T)
        return -one(T), zero(T)
    end

    s_pin = s^pin
    s_diff = s^(pin - pcut)
    denom = one(T) + s_diff

    x = one(T) / (one(T) + a * s_pin / denom)
    y = b1 * x + b0
    y = clamp(y, -one(T), one(T))

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
# Hermite cubic spline evaluation (values only)
@inline function evaluate_Rnl_$pair_idx(r::T)::SVector{N_RNL, T} where {T}
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
    return env .* s
end

# Hermite cubic spline evaluation (with derivatives)
@inline function evaluate_Rnl_d_$pair_idx(r::T)::Tuple{SVector{N_RNL, T}, SVector{N_RNL, T}} where {T}
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
    Rnl = env .* s
    dRnl_dr = (denv_dy .* s .+ env .* ds_dy) .* dy_dr

    return Rnl, dRnl_dr
end
""")
    end

    # Write dispatcher functions
    println(io, """
# ============================================================================
# DISPATCH FUNCTIONS
# ============================================================================

# Radial basis dispatch (values only)
@inline function evaluate_Rnl(r::T, iz::Int, jz::Int)::SVector{N_RNL, T} where {T}
    pair_idx = zz2pair_sym(iz, jz)""")

    for pair_idx in 1:n_pairs
        cond = pair_idx == 1 ? "if" : "elseif"
        println(io, "    $cond pair_idx == $pair_idx; return evaluate_Rnl_$pair_idx(r)")
    end
    println(io, "    end")
    println(io, "    return zero(SVector{N_RNL, T})")
    println(io, "end")
    println(io)

    # Derivative dispatch
    println(io, """
# Radial basis dispatch (with derivatives)
@inline function evaluate_Rnl_d(r::T, iz::Int, jz::Int)::Tuple{SVector{N_RNL, T}, SVector{N_RNL, T}} where {T}
    pair_idx = zz2pair_sym(iz, jz)""")

    for pair_idx in 1:n_pairs
        cond = pair_idx == 1 ? "if" : "elseif"
        println(io, "    $cond pair_idx == $pair_idx; return evaluate_Rnl_d_$pair_idx(r)")
    end
    println(io, "    end")
    println(io, "    return zero(SVector{N_RNL, T}), zero(SVector{N_RNL, T})")
    println(io, "    end")

    return String(take!(io))
end
