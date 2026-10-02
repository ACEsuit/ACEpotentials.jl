using StaticArrays
using LinearAlgebra
using SpheriCart

# Test: compare exported Ylm with SpheriCart reference

# Exported model's eval_ylm (copied from generated code)
const N_YLM = 9

function eval_ylm_exported(R::SVector{3, T}) where {T}
    Y = zeros(MVector{N_YLM, T})
    x, y, z = R
    r = norm(R)

    if r < 1e-12
        Y[1] = T(0.28209479177387814)
        return SVector{N_YLM, T}(Y)
    end

    # Solid harmonics: scaled by r^l
    # Using L2 normalization (SpheriCart convention)

    # l=0 (1 term)
    Y[1] = T(0.28209479177387814)  # 1/(2*sqrt(pi))

    # l=1 (3 terms): Y_1^{-1}, Y_1^0, Y_1^1
    c1 = T(0.4886025119029199)  # sqrt(3/(4*pi))
    Y[2] = c1 * y    # Y_1^{-1}
    Y[3] = c1 * z    # Y_1^0
    Y[4] = c1 * x    # Y_1^1

    # l=2 (5 terms): Y_2^{-2}, Y_2^{-1}, Y_2^0, Y_2^1, Y_2^2
    c2a = T(1.0925484305920792)   # sqrt(15/(4*pi))
    c2b = T(0.31539156525252005)  # sqrt(5/(16*pi))
    c2c = T(0.5462742152960396)   # sqrt(15/(16*pi))

    Y[5] = c2a * x * y                           # Y_2^{-2}
    Y[6] = c2a * y * z                           # Y_2^{-1}
    Y[7] = c2b * (2*z*z - x*x - y*y)            # Y_2^0
    Y[8] = c2a * z * x                           # Y_2^1
    Y[9] = c2c * (x*x - y*y)                     # Y_2^2

    return SVector{N_YLM, T}(Y)
end

# Create SpheriCart basis for comparison
ylm_basis = SolidHarmonics(2)

# Test vectors
test_Rs = [
    SVector(1.0, 0.0, 0.0),      # Along x
    SVector(0.0, 1.0, 0.0),      # Along y
    SVector(0.0, 0.0, 1.0),      # Along z
    SVector(1.0, 1.0, 1.0),      # Diagonal
    SVector(2.35, 0.0, 0.0),     # Scaled along x
    SVector(1.3575, 1.3575, 1.3575),  # Actual neighbor position
]

println("Comparing exported eval_ylm with SpheriCart:")
println("=" ^ 60)

for R in test_Rs
    println("\nR = $R (|R| = $(norm(R)))")

    # Exported implementation
    Y_exp = eval_ylm_exported(R)

    # SpheriCart reference
    Y_ref = ylm_basis(R)

    println("  Exported Y = $Y_exp")
    println("  SpheriCart Y = $Y_ref")

    # Compare
    diff = norm(Y_exp - Y_ref)
    println("  Max diff = $(maximum(abs.(Y_exp - Y_ref)))")
    println("  Match? $(diff < 1e-10)")
end

# Check what SpheriCart returns for the neighbor positions in our test
println("\n" * "=" ^ 60)
println("Testing with actual neighbor positions from debug:")

# Neighbors from atom 1
Rs_atom1 = [
    SVector(0.0, 0.0, -5.43),
    SVector(0.0, -5.43, 0.0),
    SVector(-5.43, 0.0, 0.0),
    SVector(5.43, 0.0, 0.0),
    SVector(0.0, 5.43, 0.0),
    SVector(0.0, 0.0, 5.43),
    SVector(1.3575, 1.3575, -4.0725),
    SVector(1.3575, -4.0725, 1.3575),
    SVector(-4.0725, 1.3575, 1.3575),
    SVector(1.3575, 1.3575, 1.3575),
]

println("\nFirst 10 neighbors of atom 1:")
for (i, R) in enumerate(Rs_atom1)
    Y_exp = eval_ylm_exported(R)
    Y_ref = ylm_basis(R)
    diff = maximum(abs.(Y_exp - Y_ref))
    println("  R[$i]: diff = $diff, match = $(diff < 1e-10)")
end
