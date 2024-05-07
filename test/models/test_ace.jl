
using Pkg; Pkg.activate(joinpath(@__DIR__(), "..", ".."))
# using TestEnv; TestEnv.activate();

using Test, ACEbase
using ACEbase.Testing: print_tf

using ACEpotentials
M = ACEpotentials.Models

using Random, LuxCore, StaticArrays
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

Rs, Zs, z0 = M.rand_atenv(model, 16)
Ei, st = M.evaluate(model, Rs, Zs, z0, ps, st)

Ei1, ∇Ei, st = M.evaluate_ed(model, Rs, Zs, z0, ps, st)

Ei ≈ Ei1

Us = randn(SVector{3, Float64}, length(Rs))
F(t) = M.evaluate(model, Rs + t * Us, Zs, z0, ps, st)[1] 
dF(t) = dot(M.evaluate_ed(model, Rs + t * Us, Zs, z0, ps, st)[2], Us)
ACEbase.Testing.fdtest(F, dF, 0.0)

##

# first test shows the performance is not at all awful even without any 
# optimizations and reductions in memory allocations. 
using BenchmarkTools
Rs, Zs, z0 = M.rand_atenv(model, 16)
@btime M.evaluate($model, $Rs, $Zs, $z0, $ps, $st)
@btime M.evaluate_ed($model, $Rs, $Zs, $z0, $ps, $st)

##

