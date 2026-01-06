# Generic polynomial evaluation kernels for trim-safe exports
#
# These functions implement the 3-term recurrence for orthogonal polynomials
# matching P4ML's OrthPolyBasis1D3T exactly.
#
# Usage in exported models:
#   Include this file, then call with model-specific coefficient arrays.

"""
Generic 3-term recurrence polynomial evaluation.

Matches P4ML's OrthPolyBasis1D3T recurrence:
  P[1] = A[1]
  P[2] = A[2] * y + B[2]
  P[n] = (A[n] * y + B[n]) * P[n-1] + C[n] * P[n-2]  for n >= 3

Arguments:
- `y`: Input value (typically transformed distance in [-1, 1])
- `A, B, C`: Coefficient tuples/arrays from orthonormal basis
- `P`: Output buffer (MVector or similar)
"""
@inline function eval_polys_generic!(P, A, B, C, y::T) where {T}
    n = length(P)
    @inbounds begin
        P[1] = T(A[1])
        if n >= 2
            P[2] = A[2] * y + B[2]
        end
        for i = 3:n
            P[i] = (A[i] * y + B[i]) * P[i-1] + C[i] * P[i-2]
        end
    end
    return P
end

"""
Generic 3-term recurrence with derivatives.

Returns both polynomial values P[i] and derivatives dP[i]/dy.
Derivative formula from differentiating recurrence:
  dP[1]/dy = 0
  dP[2]/dy = A[2]
  dP[n]/dy = A[n] * P[n-1] + (A[n] * y + B[n]) * dP[n-1] + C[n] * dP[n-2]
"""
@inline function eval_polys_ed_generic!(P, dP, A, B, C, y::T) where {T}
    n = length(P)
    @inbounds begin
        P[1] = T(A[1])
        dP[1] = zero(T)
        if n >= 2
            P[2] = A[2] * y + B[2]
            dP[2] = T(A[2])
        end
        for i = 3:n
            P[i] = (A[i] * y + B[i]) * P[i-1] + C[i] * P[i-2]
            dP[i] = A[i] * P[i-1] + (A[i] * y + B[i]) * dP[i-1] + C[i] * dP[i-2]
        end
    end
    return P, dP
end
