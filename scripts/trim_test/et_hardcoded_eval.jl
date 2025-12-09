# Hardcoded evaluation-only test for --trim=safe compilation
# This test avoids ALL construction code by using hardcoded pre-built structures.
#
# The goal: verify that RUNTIME evaluation paths are type-stable.
# Construction code has type instabilities that we accept - it runs once offline.

using StaticArrays
using LinearAlgebra: norm
using SparseArrays: sparse
using Polynomials4ML
const P4ML = Polynomials4ML
using EquivariantTensors
const ET = EquivariantTensors

# ============================================================================
# HARDCODED TENSOR STRUCTURES
# These were pre-computed from a simple basis and serialized here as literals.
# No construction code is called - we directly create the structs.
# ============================================================================

# From running _build_test_tensor() and inspecting the results:
#   abasis.spec: [(1, 1), (4, 1)]
#   aabasis.specs: ([(1,), (2,)],)
#   aabasis.ranges: (1:2,)
#   aabasis.hasconst: false
#   A2Bmaps[1]: sparse([1, 2], [1, 2], [1.0, 1.0], 2, 2)
#   𝔸spec: [[(n=1, l=0, m=0)], [(n=2, l=0, m=0)]]

# Create PooledSparseProduct directly with known spec
const ABASIS_SPEC = [(1, 1), (4, 1)]::Vector{Tuple{Int,Int}}
const ABASIS = ET.PooledSparseProduct(ABASIS_SPEC)

# Create SparseSymmProd directly with known spec
# spec[1] = [(1,), (2,)] corresponds to 1-body terms
const AABASIS_SPEC = [[1], [2]]::Vector{Vector{Int}}
const AABASIS = ET.SparseSymmProd(AABASIS_SPEC)

# The coupling map A2B (2x2 identity in this simple case)
const A2BMAP = (sparse([1, 2], [1, 2], [1.0, 1.0], 2, 2),)

# The readable spec for debugging (stored in meta dict for original API)
const AA_SPEC = [[(n=1, l=0, m=0)], [(n=2, l=0, m=0)]]::Vector{Vector{@NamedTuple{n::Int, l::Int, m::Int}}}

# Assemble the full SparseACEbasis using original Dict-based constructor
const META = Dict{String, Any}("𝔸spec" => AA_SPEC)
const TENSOR = ET.SparseACEbasis(ABASIS, AABASIS, A2BMAP, META)

# R specification and spherical harmonics basis
const R_SPEC = [(n=1, l=0), (n=1, l=1), (n=1, l=2), (n=2, l=0), (n=2, l=1)]
const MAXL = 2
const YBASIS = P4ML.real_sphericalharmonics(MAXL)

# ============================================================================
# EVALUATION PHASE - This is what needs to be --trim compatible
# ============================================================================

function compute_rnl_ylm(Rs::Vector{SVector{3, Float64}}, r_spec, ybasis)
    nneigh = length(Rs)
    nbasis_r = length(r_spec)
    nbasis_y = length(ybasis)

    # Compute radii
    rs = [norm(r) for r in Rs]

    # Radial basis: simple exp(-r) * r^l / (n+l)
    Rnl = zeros(Float64, nneigh, nbasis_r)
    for j in 1:nneigh
        for (i, spec) in enumerate(r_spec)
            n, l = spec.n, spec.l
            Rnl[j, i] = exp(-rs[j]) * rs[j]^l / (n + l)
        end
    end

    # Spherical harmonics
    Ylm = zeros(Float64, nneigh, nbasis_y)
    for j in 1:nneigh
        P4ML.evaluate!(view(Ylm, j, :), ybasis, Rs[j] / norm(Rs[j]))
    end

    return Rnl, Ylm
end

function evaluate_tensor(tensor, Rnl::Matrix{Float64}, Ylm::Matrix{Float64})
    # This is the core runtime evaluation - must be type-stable
    BB = ET.evaluate(tensor, Rnl, Ylm, NamedTuple(), NamedTuple())
    return BB[1]  # L=0 channel
end

# ============================================================================
# MAIN - Only calls evaluation functions
# ============================================================================

function (@main)(ARGS)
    # Test configuration: 3 neighbors
    Rs = [
        SVector(1.0, 0.0, 0.0),
        SVector(0.0, 1.0, 0.0),
        SVector(0.0, 0.0, 1.0)
    ]

    println(Core.stdout, "=== Hardcoded Evaluation-Only Trim Test ===")
    println(Core.stdout, "Tensor length: ", length(TENSOR))
    println(Core.stdout, "Number of neighbors: ", length(Rs))

    # Compute Rnl and Ylm (polynomial evaluations)
    Rnl, Ylm = compute_rnl_ylm(Rs, R_SPEC, YBASIS)

    # Evaluate tensor (the critical runtime path)
    B = evaluate_tensor(TENSOR, Rnl, Ylm)

    println(Core.stdout, "Evaluation successful!")
    println(Core.stdout, "B[1] = ", B[1])
    println(Core.stdout, "Sum of basis values: ", sum(B))

    return 0
end
