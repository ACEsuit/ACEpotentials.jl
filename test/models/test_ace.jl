
using Pkg; Pkg.activate(joinpath(@__DIR__(), "..", ".."))
# using TestEnv; TestEnv.activate();

using Test, ACEbase
using ACEbase.Testing: print_tf, println_slim

using ACEpotentials
M = ACEpotentials.Models

using Optimisers

using Random, LuxCore, StaticArrays, LinearAlgebra
rng = Random.MersenneTwister(1234)

##

elements = (:Si, :O)
level = M.TotalDegree()
max_level = 15
order = 3 

model = M.ace_model(; elements = elements, order = order, Ytype = :solid, 
                      level = level, max_level = max_level, maxl = 8, 
                      init_WB = :glorot_normal)

ps, st = LuxCore.setup(rng, model)

# TODO: the number of parameters is completely off, so something is 
#       likely wrong here. 


##

@info("Test Rotation-Invariance of the Model")

for ntest = 1:50 
   Nat = rand(8:16)
   Rs, Zs, Z0 = M.rand_atenv(model, Nat)
   val, st = M.evaluate(model, Rs, Zs, Z0, ps, st)

   p = shuffle(1:Nat)
   Rs1 = Ref(M.rand_iso()) .* Rs[p]
   Zs1 = Zs[p]
   val1, st = M.evaluate(model, Rs1, Zs1, Z0, ps, st)

   print_tf(@test abs(val - val1) < 1e-10)
end
println()

##

@info("Test derivatives w.r.t. positions")
Rs, Zs, z0 = M.rand_atenv(model, 16)
Ei, st = M.evaluate(model, Rs, Zs, z0, ps, st)
Ei1, ∇Ei, st = M.evaluate_ed(model, Rs, Zs, z0, ps, st)
println_slim(@test Ei ≈ Ei1)

for ntest = 1:20 
   Nat = rand(8:16)
   Rs, Zs, z0 = M.rand_atenv(model, Nat)
   Us = randn(SVector{3, Float64}, Nat)
   F(t) = M.evaluate(model, Rs + t * Us, Zs, z0, ps, st)[1] 
   dF(t) = dot(M.evaluate_ed(model, Rs + t * Us, Zs, z0, ps, st)[2], Us)
   print_tf(@test ACEbase.Testing.fdtest(F, dF, 0.0; verbose=false))
end
println() 

##

@info("Test derivatives w.r.t. parameters")
Nat = 15
Rs, Zs, z0 = M.rand_atenv(model, Nat)
Ei, st = M.evaluate(model, Rs, Zs, z0, ps, st)
Ei1, ∇Ei, st = M.grad_params(model, Rs, Zs, z0, ps, st)
println_slim(@test Ei ≈ Ei1)

for ntest = 1:20
   Nat = rand(8:16)
   Rs, Zs, z0 = M.rand_atenv(model, Nat)
   pvec, _restruct = destructure(ps)
   uvec = randn(length(pvec)) / sqrt(length(pvec))
   F(t) = M.evaluate(model, Rs, Zs, z0, _restruct(pvec + t * uvec), st)[1]
   dF0 = dot( destructure( M.grad_params(model, Rs, Zs, z0, ps, st)[2] )[1], uvec )
   print_tf(@test ACEbase.Testing.fdtest(F, t -> dF0, 0.0; verbose = false))
end

##


# first test shows the performance is not at all awful even without any 
# optimizations and reductions in memory allocations. 
using BenchmarkTools
Rs, Zs, z0 = M.rand_atenv(model, 16)
@btime M.evaluate($model, $Rs, $Zs, $z0, $ps, $st)
@btime M.evaluate_ed($model, $Rs, $Zs, $z0, $ps, $st)
@btime M.grad_params($model, $Rs, $Zs, $z0, $ps, $st)


##

@info("Test second mixed derivatives reverse-over-reverse")
for ntest = 1:20 
   Nat = rand(8:16)
   Rs, Zs, z0 = M.rand_atenv(model, Nat)
   Us = randn(SVector{3, Float64}, Nat)
   Ei = M.evaluate(model, Rs, Zs, z0, ps, st)
   Ei, ∂Ei, _ = M.grad_params(model, Rs, Zs, z0, ps, st)

   # test partial derivative w.r.t. the Ei component 
   ∂2_Ei = M.pullback_2_mixed(1.0, 0*Us, model, Rs, Zs, z0, ps, st)
   print_tf(@test destructure(∂2_Ei)[1] ≈ destructure(∂Ei)[1])

   # test partial derivative w.r.t. the ∇Ei component 
   ∂2_∇Ei = M.pullback_2_mixed(0.0, Us, model, Rs, Zs, z0, ps, st)
   ∂2_∇Ei_vec = destructure(∂2_∇Ei)[1]

   ps_vec, _restruct = destructure(ps)
   vs_vec = randn(length(ps_vec)) / sqrt(length(ps_vec))
   F(t) = dot(Us, M.evaluate_ed(model, Rs, Zs, z0, _restruct(ps_vec + t * vs_vec), st)[2])
   dF0 = dot(∂2_∇Ei_vec, vs_vec)
   print_tf(@test ACEbase.Testing.fdtest(F, t -> dF0, 0.0; verbose=false))
end

