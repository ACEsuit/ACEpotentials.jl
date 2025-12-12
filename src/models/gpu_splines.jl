# GPU-compatible cubic spline implementation using KernelAbstractions
#
# This module provides cross-platform GPU spline evaluation that can be used
# with CUDA, Metal, ROCm, or CPU backends.

using KernelAbstractions
using StaticArrays

# =========================================================================
#  GPUCubicSpline - GPU-compatible cubic spline representation
# =========================================================================

"""
    GPUCubicSpline{T, N, A}

A GPU-compatible cubic spline representation storing coefficients for
efficient parallel evaluation.

The spline is defined on a uniform grid from `x_min` to `x_max` with
`n_segments` segments. Each segment stores 4 coefficients (c0, c1, c2, c3)
for the polynomial: `p(dx) = c0 + c1*dx + c2*dx^2 + c3*dx^3`
where `dx = x - x_seg_start`.

For multi-output splines (output dimension N), coefficients has shape
`(4, n_segments, N)`.

# Fields
- `coeffs`: Array of spline coefficients, shape (4, n_segments, N)
- `x_min`: Minimum x value
- `x_max`: Maximum x value
- `h`: Segment width (x_max - x_min) / n_segments
- `n_segments`: Number of spline segments
"""
struct GPUCubicSpline{T, N, A <: AbstractArray{T, 3}}
    coeffs::A           # (4, n_segments, N) - coefficients per segment
    x_min::T
    x_max::T
    h::T
    n_segments::Int
end

function Base.show(io::IO, spl::GPUCubicSpline{T, N}) where {T, N}
    print(io, "GPUCubicSpline{$T}($(spl.n_segments) segments, $N outputs, x ∈ [$(spl.x_min), $(spl.x_max)])")
end

output_dim(spl::GPUCubicSpline) = size(spl.coeffs, 3)

# =========================================================================
#  Conversion from Interpolations.jl splines
# =========================================================================

"""
    GPUCubicSpline(itp_spline)

Convert an Interpolations.jl cubic B-spline interpolation to a GPUCubicSpline.

Supports both scalar and vector-valued (SVector) splines.
"""
function GPUCubicSpline(itp)
    # Extract grid information from the scaled interpolation
    # itp is Extrapolation -> itp.itp is ScaledInterpolation -> itp.itp.ranges has the grid
    x_range = itp.itp.ranges[1]
    x_min = first(x_range)
    x_max = last(x_range)
    n_nodes = length(x_range)
    n_segments = n_nodes - 1
    h = step(x_range)  # Use step() for StepRangeLen

    # Determine output dimension from a test evaluation
    y_test = itp(x_min)
    if y_test isa Number
        N = 1
        T = typeof(y_test)
    elseif y_test isa SVector
        N = length(y_test)
        T = eltype(y_test)
    else
        error("Unsupported spline output type: $(typeof(y_test))")
    end

    # Compute polynomial coefficients for each segment
    # Using Taylor expansion at each segment start
    coeffs = zeros(T, 4, n_segments, N)

    for seg in 1:n_segments
        x0 = x_min + (seg - 1) * h
        x1 = x0 + h

        # Evaluate at segment endpoints and midpoint for fitting
        # For cubic spline, we use the function values and derivatives
        y0 = itp(x0)
        y1 = itp(x1)

        # Use finite differences for derivatives
        # Be careful at boundaries - use one-sided differences if needed
        eps_fd = h * 1e-6
        if seg == 1
            # Forward difference at left boundary
            dy0 = (itp(x0 + eps_fd) - itp(x0)) / eps_fd
        else
            dy0 = (itp(x0 + eps_fd) - itp(x0 - eps_fd)) / (2 * eps_fd)
        end

        if seg == n_segments
            # Backward difference at right boundary
            dy1 = (itp(x1) - itp(x1 - eps_fd)) / eps_fd
        else
            dy1 = (itp(x1 + eps_fd) - itp(x1 - eps_fd)) / (2 * eps_fd)
        end

        # Hermite interpolation coefficients
        # p(dx) = c0 + c1*dx + c2*dx^2 + c3*dx^3
        # p(0) = y0, p(h) = y1, p'(0) = dy0, p'(h) = dy1
        for j in 1:N
            y0_j = N == 1 ? y0 : y0[j]
            y1_j = N == 1 ? y1 : y1[j]
            dy0_j = N == 1 ? dy0 : dy0[j]
            dy1_j = N == 1 ? dy1 : dy1[j]

            # Hermite basis coefficients
            c0 = y0_j
            c1 = dy0_j
            c2 = (3 * (y1_j - y0_j) / h - 2 * dy0_j - dy1_j) / h
            c3 = (2 * (y0_j - y1_j) / h + dy0_j + dy1_j) / h^2

            coeffs[1, seg, j] = c0
            coeffs[2, seg, j] = c1
            coeffs[3, seg, j] = c2
            coeffs[4, seg, j] = c3
        end
    end

    return GPUCubicSpline{T, N, typeof(coeffs)}(coeffs, T(x_min), T(x_max), T(h), n_segments)
end

# =========================================================================
#  KernelAbstractions kernels for spline evaluation
# =========================================================================

"""
    @kernel function spline_eval_kernel!(out, x, coeffs, x_min, h, n_segments)

KernelAbstractions kernel for parallel spline evaluation.

Evaluates the spline at multiple x values simultaneously.
Each thread handles one x value and computes all N output components.
"""
@kernel function spline_eval_kernel!(out, @Const(x), @Const(coeffs),
                                     x_min, h, n_segments)
    i = @index(Global)

    # Clamp x to valid range
    xi = x[i]
    xi_clamped = clamp(xi, x_min, x_min + h * n_segments)

    # Find segment index (1-based)
    seg_float = (xi_clamped - x_min) / h
    seg = min(floor(Int, seg_float) + 1, n_segments)

    # Local coordinate within segment
    dx = xi_clamped - (x_min + (seg - 1) * h)

    # Evaluate polynomial using Horner's method for each output
    N = size(coeffs, 3)
    for j in 1:N
        c0 = coeffs[1, seg, j]
        c1 = coeffs[2, seg, j]
        c2 = coeffs[3, seg, j]
        c3 = coeffs[4, seg, j]

        # Horner's method: c0 + dx*(c1 + dx*(c2 + dx*c3))
        out[i, j] = c0 + dx * (c1 + dx * (c2 + dx * c3))
    end
end

"""
    @kernel function spline_eval_deriv_kernel!(out, dout, x, coeffs, x_min, h, n_segments)

KernelAbstractions kernel for parallel spline evaluation with derivatives.

Evaluates both the spline and its derivative at multiple x values.
"""
@kernel function spline_eval_deriv_kernel!(out, dout, @Const(x), @Const(coeffs),
                                           x_min, h, n_segments)
    i = @index(Global)

    # Clamp x to valid range
    xi = x[i]
    xi_clamped = clamp(xi, x_min, x_min + h * n_segments)

    # Find segment index (1-based)
    seg_float = (xi_clamped - x_min) / h
    seg = min(floor(Int, seg_float) + 1, n_segments)

    # Local coordinate within segment
    dx = xi_clamped - (x_min + (seg - 1) * h)

    # Evaluate polynomial and derivative for each output
    N = size(coeffs, 3)
    for j in 1:N
        c0 = coeffs[1, seg, j]
        c1 = coeffs[2, seg, j]
        c2 = coeffs[3, seg, j]
        c3 = coeffs[4, seg, j]

        # Value: c0 + dx*(c1 + dx*(c2 + dx*c3))
        out[i, j] = c0 + dx * (c1 + dx * (c2 + dx * c3))

        # Derivative: c1 + 2*c2*dx + 3*c3*dx^2
        dout[i, j] = c1 + dx * (2 * c2 + 3 * c3 * dx)
    end
end

# =========================================================================
#  High-level evaluation interface
# =========================================================================

"""
    evaluate(spl::GPUCubicSpline, x::AbstractVector)

Evaluate the GPU spline at multiple x values.

Returns an array of shape (length(x), N) where N is the output dimension.
"""
function evaluate(spl::GPUCubicSpline{T, N, A}, x::AbstractVector) where {T, N, A}
    n = length(x)
    backend = get_backend(spl.coeffs)

    # Allocate output on same device as coefficients
    out = KernelAbstractions.zeros(backend, T, n, N)

    # Move x to same device if needed
    x_dev = adapt(backend, x)

    # Launch kernel
    kernel! = spline_eval_kernel!(backend)
    kernel!(out, x_dev, spl.coeffs, spl.x_min, spl.h, spl.n_segments,
            ndrange=n)

    KernelAbstractions.synchronize(backend)
    return out
end

"""
    evaluate_ed(spl::GPUCubicSpline, x::AbstractVector)

Evaluate the GPU spline and its derivative at multiple x values.

Returns (out, dout) where both are arrays of shape (length(x), N).
"""
function evaluate_ed(spl::GPUCubicSpline{T, N, A}, x::AbstractVector) where {T, N, A}
    n = length(x)
    backend = get_backend(spl.coeffs)

    # Allocate outputs on same device as coefficients
    out = KernelAbstractions.zeros(backend, T, n, N)
    dout = KernelAbstractions.zeros(backend, T, n, N)

    # Move x to same device if needed
    x_dev = adapt(backend, x)

    # Launch kernel
    kernel! = spline_eval_deriv_kernel!(backend)
    kernel!(out, dout, x_dev, spl.coeffs, spl.x_min, spl.h, spl.n_segments,
            ndrange=n)

    KernelAbstractions.synchronize(backend)
    return out, dout
end

"""
    (spl::GPUCubicSpline)(x)

Callable interface for spline evaluation.

For a single x value, returns an SVector of length N.
For a vector of x values, returns a matrix of shape (length(x), N).
"""
function (spl::GPUCubicSpline{T, N, A})(x::Real) where {T, N, A}
    # Single-value evaluation (CPU fallback)
    xi = clamp(T(x), spl.x_min, spl.x_max)
    seg_float = (xi - spl.x_min) / spl.h
    seg = min(floor(Int, seg_float) + 1, spl.n_segments)
    dx = xi - (spl.x_min + (seg - 1) * spl.h)

    # Get coefficients (may need to copy from GPU)
    coeffs_cpu = spl.coeffs isa Array ? spl.coeffs : Array(spl.coeffs)

    result = zeros(T, N)
    for j in 1:N
        c0, c1, c2, c3 = coeffs_cpu[1, seg, j], coeffs_cpu[2, seg, j],
                         coeffs_cpu[3, seg, j], coeffs_cpu[4, seg, j]
        result[j] = c0 + dx * (c1 + dx * (c2 + dx * c3))
    end

    return N == 1 ? result[1] : SVector{N, T}(result)
end

function (spl::GPUCubicSpline)(x::AbstractVector)
    return evaluate(spl, x)
end

# =========================================================================
#  Utility functions
# =========================================================================

"""
    to_gpu(spl::GPUCubicSpline, device)

Move a GPUCubicSpline to a specific device (e.g., CUDA.cu, Metal.mtl).
"""
function to_gpu(spl::GPUCubicSpline{T, N, A}, device) where {T, N, A}
    coeffs_dev = device(spl.coeffs)
    return GPUCubicSpline{T, N, typeof(coeffs_dev)}(
        coeffs_dev, spl.x_min, spl.x_max, spl.h, spl.n_segments
    )
end

"""
    to_cpu(spl::GPUCubicSpline)

Move a GPUCubicSpline to CPU.
"""
function to_cpu(spl::GPUCubicSpline{T, N, A}) where {T, N, A}
    coeffs_cpu = Array(spl.coeffs)
    return GPUCubicSpline{T, N, typeof(coeffs_cpu)}(
        coeffs_cpu, spl.x_min, spl.x_max, spl.h, spl.n_segments
    )
end

# Adapt.jl support for automatic device transfer
using Adapt

Adapt.adapt_structure(to, spl::GPUCubicSpline{T, N}) where {T, N} =
    GPUCubicSpline{T, N, typeof(adapt(to, spl.coeffs))}(
        adapt(to, spl.coeffs),
        spl.x_min,
        spl.x_max,
        spl.h,
        spl.n_segments
    )
