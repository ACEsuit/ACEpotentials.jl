# Generic distance transform kernels for trim-safe exports
#
# These functions implement radial transforms (Agnesi, normalized, etc.)
# with parameterized constants instead of per-pair code generation.
#
# Usage in exported models:
#   Include this file, define TRANSFORM_PARAMS tuple, use dispatcher.

"""
Generalized Agnesi transform: r → y ∈ [-1, 1]

Formula from EquivariantTensors:
  s = (r - rin) / (req - rin)
  x = 1 / (1 + a * s^pin / (1 + s^(pin - pcut)))
  y = b1 * x + b0

Parameters (named tuple):
- `rin`: Inner cutoff
- `req`: Equilibrium distance (normalization point)
- `pin`: Inner exponent
- `pcut`: Cutoff smoothness exponent
- `a`: Agnesi steepness parameter
- `b0, b1`: Linear scaling to [-1, 1]
"""
@inline function agnesi_transform(r::T, p) where {T}
    if r <= p.rin
        return one(T)
    end

    s = (r - p.rin) / (p.req - p.rin)
    s_pin = s^p.pin
    s_diff = s^(p.pin - p.pcut)
    denom = one(T) + s_diff
    x = one(T) / (one(T) + p.a * s_pin / denom)
    y = p.b1 * x + p.b0
    return clamp(y, -one(T), one(T))
end

"""
Agnesi transform with analytical derivative dy/dr.

Uses quotient rule on the Agnesi formula.
"""
@inline function agnesi_transform_d(r::T, p) where {T}
    if r <= p.rin
        return one(T), zero(T)
    end

    ds_dr = one(T) / (p.req - p.rin)
    s = (r - p.rin) * ds_dr

    # Avoid numerical issues near s=0
    if s <= T(1e-12)
        return one(T), zero(T)
    end

    s_pin = s^p.pin
    s_diff = s^(p.pin - p.pcut)
    denom = one(T) + s_diff

    x = one(T) / (one(T) + p.a * s_pin / denom)
    y = p.b1 * x + p.b0

    # Derivative via chain rule
    # g = a * s^pin / (1 + s^(pin-pcut))
    # x = 1/(1+g), dx/dg = -x^2
    # dg/ds = a * s^(pin-1) * (pin + pcut*s^(pin-pcut)) / (1+s^(pin-pcut))^2
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

"""
Normalized transform (ACEpotentials.Models.NormalizedTransform pattern).

Formula:
  s = (r - rin) / (r0 - rin)
  y_inner = (a + s^q) / (a + s^p)
  y = -1 + 2 * (y_inner - yin) / (ycut - yin)

Parameters (named tuple):
- `rin, r0`: Distance parameters
- `p, q, a`: Transform shape parameters
- `yin, ycut`: Normalization endpoints
"""
@inline function normalized_transform(r::T, p) where {T}
    if r <= p.rin
        return one(T)
    end

    s = (r - p.rin) / (p.r0 - p.rin)
    num = p.a + s^p.q
    den = p.a + s^p.p
    y_inner = num / den

    y = -one(T) + 2 * (y_inner - p.yin) / (p.ycut - p.yin)
    return clamp(y, -one(T), one(T))
end

"""
Normalized transform with analytical derivative dy/dr.
"""
@inline function normalized_transform_d(r::T, params) where {T}
    if r <= params.rin
        return one(T), zero(T)
    end

    s = (r - params.rin) / (params.r0 - params.rin)
    ds_dr = one(T) / (params.r0 - params.rin)

    # y_inner = (a + s^q) / (a + s^p)
    num = params.a + s^params.q
    den = params.a + s^params.p
    y_inner = num / den

    # dy_inner/ds via quotient rule
    dnum_ds = params.q > 0 ? params.q * s^(params.q - 1) : zero(T)
    dden_ds = params.p > 0 ? params.p * s^(params.p - 1) : zero(T)
    dy_inner_ds = (dnum_ds * den - dden_ds * num) / (den * den)

    # y = -1 + 2*(y_inner - yin)/(ycut - yin)
    scale = 2 / (params.ycut - params.yin)
    y = -one(T) + scale * (y_inner - params.yin)
    dy_dr = scale * dy_inner_ds * ds_dr

    y_clamped = clamp(y, -one(T), one(T))
    if y_clamped != y
        return y_clamped, zero(T)
    end
    return y, dy_dr
end
