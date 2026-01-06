# Generic envelope function kernels for trim-safe exports
#
# Envelope functions smoothly decay to zero at the cutoff.
# These parameterized versions avoid per-pair code generation.

"""
Quartic envelope: (1 - y²)²

Standard ACE envelope for y ∈ [-1, 1].
Returns 0 at y = ±1, maximum at y = 0.
"""
@inline function envelope_quartic(y::T) where {T}
    y2 = y * y
    one_minus_y2 = max(zero(T), one(T) - y2)
    return one_minus_y2^2
end

"""
Quartic envelope with derivative.

Returns (env, denv/dy) where:
  env = (1 - y²)²
  denv/dy = -4y(1 - y²)
"""
@inline function envelope_quartic_d(y::T) where {T}
    y2 = y * y
    one_minus_y2 = max(zero(T), one(T) - y2)
    env = one_minus_y2^2
    denv_dy = -4 * y * one_minus_y2
    return env, denv_dy
end

"""
Polynomial envelope: (1 - (r/rcut)^p)^p

Alternative envelope in r-space (not y-space).
Parameters: p (exponent), rcut (cutoff distance)
"""
@inline function envelope_poly(r::T, p, rcut) where {T}
    if r >= rcut
        return zero(T)
    end
    u = r / rcut
    return (one(T) - u^p)^p
end

"""
Polynomial envelope with derivative.

Returns (env, denv/dr).
"""
@inline function envelope_poly_d(r::T, p, rcut) where {T}
    if r >= rcut
        return zero(T), zero(T)
    end
    u = r / rcut
    du_dr = one(T) / rcut
    u_p = u^p
    inner = one(T) - u_p
    env = inner^p

    # d/dr[(1-u^p)^p] = p*(1-u^p)^(p-1) * (-p*u^(p-1)) * du/dr
    denv_dr = -p^2 * inner^(p-1) * u^(p-1) * du_dr
    return env, denv_dr
end

"""
Two-parameter polynomial envelope: (1 - (r/rcut)^p0)^p1

ACEpotentials.Models.PolyEnvelope2 pattern.
Parameters: p0 (inner exponent), p1 (outer exponent), rcut
"""
@inline function envelope_poly2(r::T, p0, p1, rcut) where {T}
    if r >= rcut
        return zero(T)
    end
    u = r / rcut
    return (one(T) - u^p0)^p1
end

"""
Two-parameter polynomial envelope with derivative.
"""
@inline function envelope_poly2_d(r::T, p0, p1, rcut) where {T}
    if r >= rcut
        return zero(T), zero(T)
    end
    u = r / rcut
    du_dr = one(T) / rcut
    u_p0 = u^p0
    inner = one(T) - u_p0
    env = inner^p1

    # d/dr[(1-u^p0)^p1] = p1*(1-u^p0)^(p1-1) * (-p0*u^(p0-1)) * du/dr
    denv_dr = -p0 * p1 * inner^(p1-1) * u^(p0-1) * du_dr
    return env, denv_dr
end

"""
PolyEnvelope2sX envelope (x-space variant from ACEpotentials.Models).

Formula: s * (x - x1)^p1 * (x2 - x)^p2

Parameters (named tuple): x1, x2, p1, p2, s
"""
@inline function envelope_poly2sx(x::T, p) where {T}
    if !(p.x1 < x < p.x2)
        return zero(T)
    end
    return p.s * (x - p.x1)^p.p1 * (p.x2 - x)^p.p2
end

"""
PolyEnvelope2sX with derivative.
"""
@inline function envelope_poly2sx_d(x::T, p) where {T}
    if !(p.x1 < x < p.x2)
        return zero(T), zero(T)
    end

    left = (x - p.x1)^p.p1
    right = (p.x2 - x)^p.p2
    env = p.s * left * right

    # Product rule for derivative
    dleft_dx = p.p1 > 0 ? p.p1 * (x - p.x1)^(p.p1 - 1) : zero(T)
    dright_dx = p.p2 > 0 ? -p.p2 * (p.x2 - x)^(p.p2 - 1) : zero(T)
    denv_dx = p.s * (dleft_dx * right + left * dright_dx)

    return env, denv_dx
end
