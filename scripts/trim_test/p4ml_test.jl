using StaticArrays
using LinearAlgebra: norm
using Polynomials4ML
const P4ML = Polynomials4ML

# Create spherical harmonics basis for L=2
const YBASIS = P4ML.real_sphericalharmonics(2)
const NBASIS = length(YBASIS)

function evaluate_ylm(r::SVector{3, Float64})
    # Evaluate spherical harmonics at direction r (assumed normalized or will be internally)
    Y = zeros(Float64, NBASIS)
    P4ML.evaluate!(Y, YBASIS, r)
    return Y
end

function (@main)(ARGS)
    # Test position - avoid printing SVector directly as it causes dynamic dispatch
    r = SVector(1.0, 1.0, 1.0)
    r_normalized = r / norm(r)

    println(Core.stdout, "Evaluating spherical harmonics (L=2) at direction: (", r_normalized[1], ", ", r_normalized[2], ", ", r_normalized[3], ")")

    Y = evaluate_ylm(r_normalized)

    println(Core.stdout, "Number of basis functions: ", NBASIS)
    # Print Y[1] value as a simple check
    println(Core.stdout, "Y[1] = ", Y[1])
    # Sum all values as a checksum
    s = sum(Y)
    println(Core.stdout, "Sum of all Y values: ", s)

    return 0
end
