using StaticArrays
using LinearAlgebra: norm, dot

function compute_distance(v1::SVector{3, Float64}, v2::SVector{3, Float64})
    return norm(v1 - v2)
end

function compute_dot(v1::SVector{3, Float64}, v2::SVector{3, Float64})
    return dot(v1, v2)
end

function (@main)(ARGS)
    v1 = SVector(1.0, 0.0, 0.0)
    v2 = SVector(0.0, 1.0, 0.0)
    v3 = SVector(1.0, 1.0, 1.0)

    d = compute_distance(v1, v3)
    dp = compute_dot(v1, v2)

    println(Core.stdout, "Distance from (1,0,0) to (1,1,1): ", d)
    println(Core.stdout, "Dot product of (1,0,0) and (0,1,0): ", dp)

    return 0
end
