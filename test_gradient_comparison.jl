#!/usr/bin/env julia
# Debug script to compare gradients from different evaluators

using ACEpotentials
using ACEpotentials.ACE1compat
using LazyArtifacts, ExtXYZ
using LinearAlgebra

M = ACEpotentials.Models

println("Loading and fitting model...")
data = ExtXYZ.load(artifact"Si_tiny_dataset" * "/Si_tiny.xyz")
model = ace1_model(; elements = [:Si],
                     Eref = [:Si => -158.54496821],
                     rcut = 5.0,
                     order = 3,
                     totaldegree = 8)

data_keys = [:energy_key => "dft_energy",
             :force_key => "dft_force",
             :virial_key => "dft_virial"]
weights = Dict("default" => Dict("E"=>30.0, "F"=>1.0, "V"=>1.0))
acefit!(data, model; data_keys..., weights = weights, solver = ACEfit.BLR())

# Create fast evaluators
fpot = M.fast_evaluator(model; aa_static = true)
fpot_d = M.fast_evaluator(model; aa_static = false)

println("\nTesting gradient consistency...")

# Test on one random environment
Rs, Zs, z0 = M.rand_atenv(model.model, 10)

println("\nComputing gradients from three evaluators...")
E1 = M.eval_site(fpot, Rs, Zs, z0)
E2 = M.eval_site(model, Rs, Zs, z0)
v1, ∇v1 = M.eval_grad_site(fpot, Rs, Zs, z0)
v2, ∇v2 = M.eval_grad_site(model, Rs, Zs, z0)
v3, ∇v3 = M.eval_grad_site(fpot_d, Rs, Zs, z0)

println("\nEnergies:")
println("  fpot (static):  E1 = ", E1)
println("  model:          E2 = ", E2)
println("  fpot_d (dynamic): v3 = ", v3)
println("  All match? ", E1 ≈ E2 ≈ v1 ≈ v2 ≈ v3)

println("\nGradient comparison:")
println("  Length: ", length(∇v1), " atoms")
println("  ∇v1 (fpot static) first element: ", ∇v1[1])
println("  ∇v2 (model) first element:       ", ∇v2[1])
println("  ∇v3 (fpot_d dynamic) first element: ", ∇v3[1])

println("\nElement-wise comparison (first 3 atoms):")
for i = 1:min(3, length(∇v1))
   println("  Atom $i:")
   println("    ∇v1: ", ∇v1[i])
   println("    ∇v2: ", ∇v2[i])
   println("    ∇v3: ", ∇v3[i])
   println("    |∇v1 - ∇v2|: ", norm(∇v1[i] - ∇v2[i]))
   println("    |∇v1 - ∇v3|: ", norm(∇v1[i] - ∇v3[i]))
   println("    |∇v2 - ∇v3|: ", norm(∇v2[i] - ∇v3[i]))
end

println("\nMax differences:")
println("  max |∇v1 - ∇v2|: ", maximum(norm.(∇v1 .- ∇v2)))
println("  max |∇v1 - ∇v3|: ", maximum(norm.(∇v1 .- ∇v3)))
println("  max |∇v2 - ∇v3|: ", maximum(norm.(∇v2 .- ∇v3)))

println("\nRelative differences (as fraction of norm):")
n1 = norm(∇v1)
n2 = norm(∇v2)
n3 = norm(∇v3)
println("  ||∇v1||: ", n1)
println("  ||∇v2||: ", n2)
println("  ||∇v3||: ", n3)
println("  ||∇v1 - ∇v2|| / ||∇v2||: ", norm(∇v1 .- ∇v2) / n2)
println("  ||∇v1 - ∇v3|| / ||∇v3||: ", norm(∇v1 .- ∇v3) / n3)
println("  ||∇v2 - ∇v3|| / ||∇v3||: ", norm(∇v2 .- ∇v3) / n3)

println("\nChecking approximate equality (default tolerance):")
println("  ∇v1 ≈ ∇v2? ", all(∇v1 .≈ ∇v2))
println("  ∇v1 ≈ ∇v3? ", all(∇v1 .≈ ∇v3))
println("  ∇v2 ≈ ∇v3? ", all(∇v2 .≈ ∇v3))
