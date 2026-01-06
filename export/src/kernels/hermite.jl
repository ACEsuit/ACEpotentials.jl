# Generic Hermite cubic spline evaluation kernels for trim-safe exports
#
# Hermite cubic interpolation provides C1-continuous interpolation
# with exact values and derivatives at knot points.
#
# Usage in exported models:
#   Include this file, define knot data as tuples, use generic evaluators.

"""
Hermite cubic interpolation on uniform grid.

Given a value `y` in the range [y_min, y_max] with `n_knots` uniformly spaced knots,
evaluates the Hermite cubic spline defined by function values `F` and slopes `G` at knots.

The spline uses the standard Hermite basis:
  H00(t) = 2t³ - 3t² + 1
  H10(t) = t³ - 2t² + t
  H01(t) = -2t³ + 3t²
  H11(t) = t³ - t²

In Horner's form for efficiency:
  f(t) = ((a3*t + a2)*t + a1)*t + a0
  where:
    a0 = f_left
    a1 = h * g_left
    a2 = -3*f_left + 3*f_right - 2*h*g_left - h*g_right
    a3 = 2*f_left - 2*f_right + h*g_left + h*g_right

Arguments:
- `y`: Evaluation point
- `y_min, y_max`: Grid bounds
- `h`: Knot spacing (y_max - y_min) / (n_knots - 1)
- `F`: Tuple of function value SVectors at each knot
- `G`: Tuple of slope SVectors at each knot

Returns: Interpolated SVector value
"""
@inline function hermite_eval(y::T, y_min, y_max, h, F, G) where {T}
    # Find segment in uniform grid
    y_clamped = clamp(y, y_min, y_max)
    t_raw = (y_clamped - y_min) / h
    t_frac, t_floor = modf(t_raw)
    il = unsafe_trunc(Int, t_floor) + 1  # 1-indexed

    # Clamp segment index to valid range
    n_knots = length(F)
    il = clamp(il, 1, n_knots - 1)

    # Get knot data (left and right endpoints)
    @inbounds fl = F[il]
    @inbounds fr = F[il + 1]
    @inbounds gl = h .* G[il]    # Pre-scale by h for Hermite formula
    @inbounds gr = h .* G[il + 1]

    # Hermite cubic (Horner's form)
    a0 = fl
    a1 = gl
    a2 = @. -3fl + 3fr - 2gl - gr
    a3 = @. 2fl - 2fr + gl + gr

    return @. ((a3 * t_frac + a2) * t_frac + a1) * t_frac + a0
end

"""
Hermite cubic interpolation with derivative.

Returns (value, d_value/dy) tuple.
Derivative formula (from differentiating Horner's form):
  df/dt = (3a3*t + 2a2)*t + a1
  df/dy = df/dt / h
"""
@inline function hermite_eval_d(y::T, y_min, y_max, h, F, G) where {T}
    # Find segment
    y_clamped = clamp(y, y_min, y_max)
    t_raw = (y_clamped - y_min) / h
    t_frac, t_floor = modf(t_raw)
    il = unsafe_trunc(Int, t_floor) + 1

    n_knots = length(F)
    il = clamp(il, 1, n_knots - 1)

    # Get knot data
    @inbounds fl = F[il]
    @inbounds fr = F[il + 1]
    @inbounds gl = h .* G[il]
    @inbounds gr = h .* G[il + 1]

    # Hermite cubic value
    a0 = fl
    a1 = gl
    a2 = @. -3fl + 3fr - 2gl - gr
    a3 = @. 2fl - 2fr + gl + gr
    val = @. ((a3 * t_frac + a2) * t_frac + a1) * t_frac + a0

    # Derivative w.r.t. t, then chain rule to y
    ds_dt = @. (3a3 * t_frac + 2a2) * t_frac + a1
    ds_dy = ds_dt ./ h

    return val, ds_dy
end

"""
Simple linear interpolation on uniform grid (fallback for less accuracy).

Faster than Hermite cubic but only C0-continuous.
"""
@inline function linear_interp(y::T, y_min, h, F) where {T}
    t_raw = (y - y_min) / h
    t_frac, t_floor = modf(t_raw)
    il = unsafe_trunc(Int, t_floor) + 1

    n_knots = length(F)
    il = clamp(il, 1, n_knots - 1)

    @inbounds f0 = F[il]
    @inbounds f1 = F[il + 1]

    return @. (one(T) - t_frac) * f0 + t_frac * f1
end

"""
Linear interpolation with derivative.
"""
@inline function linear_interp_d(y::T, y_min, h, F) where {T}
    t_raw = (y - y_min) / h
    t_frac, t_floor = modf(t_raw)
    il = unsafe_trunc(Int, t_floor) + 1

    n_knots = length(F)
    il = clamp(il, 1, n_knots - 1)

    @inbounds f0 = F[il]
    @inbounds f1 = F[il + 1]

    val = @. (one(T) - t_frac) * f0 + t_frac * f1
    dval_dy = (f1 .- f0) ./ h

    return val, dval_dy
end
