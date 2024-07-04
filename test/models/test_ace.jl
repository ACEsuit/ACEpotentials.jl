
using Pkg; Pkg.activate(joinpath(@__DIR__(), "..", ".."))
# using TestEnv; TestEnv.activate();

##

using Test, ACEbase
using ACEbase.Testing: print_tf, println_slim

using ACEpotentials
M = ACEpotentials.Models

using Optimisers, ForwardDiff

using Random, LuxCore, StaticArrays, LinearAlgebra
rng = Random.MersenneTwister(1234)
Random.seed!(11)

##

elements = (:Si, :O)
level = M.TotalDegree()
max_level = 15
order = 3 

##

@info("Test ybasis of the Model is used correctly")
msolid = M.ace_model(; elements = elements, order = order, Ytype = :solid, 
level = level, max_level = max_level, maxl = 8, pair_maxn = 15, init_WB = :glorot_normal, init_Wpair = :glorot_normal)
mspherical = M.ace_model(; elements = elements, order = order, Ytype = :spherical, 
level = level, max_level = max_level, maxl = 8, pair_maxn = 15, init_WB = :glorot_normal, init_Wpair = :zero)
ps, st = LuxCore.setup(rng, msolid)

for ntest = 1:30 
   𝐫 = randn(SVector{3, Float64})
   Ysolid = msolid.ybasis(𝐫)
   Yspher = mspherical.ybasis(𝐫)
   ll = [ M.P4ML.SpheriCart.idx2lm(i)[1] for i in 1:length(Ysolid) ]
   print_tf(@test (Yspher .* (norm(𝐫)).^ll) ≈ Ysolid)
end
println() 

##

# for ybasis in [:spherical, :solid]
   ybasis = :spherical
   @info("=== Testing ybasis = $ybasis === ")
   local ps, st, Nat
   model = M.ace_model(; elements = elements, order = order, Ytype = ybasis, 
                        level = level, max_level = max_level, maxl = 8, 
                        pair_maxn = 15, 
                        init_WB = :glorot_normal, init_Wpair = :glorot_normal)

   ps, st = LuxCore.setup(rng, model)

##

   @info("Test Rotation-Invariance of the Model")

   for ntest = 1:50 
      local st1, Nat, Rs, Zs, Z0, val 

      Nat = rand(8:16)
      Rs, Zs, Z0 = M.rand_atenv(model, Nat)
      val, st1 = M.evaluate(model, Rs, Zs, Z0, ps, st)

      p = shuffle(1:Nat)
      Rs1 = Ref(M.rand_iso()) .* Rs[p]
      Zs1 = Zs[p]
      val1, st1 = M.evaluate(model, Rs1, Zs1, Z0, ps, st)

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
      local Nat, Rs, Zs, z0, Us, F, dF
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
      local Nat, Rs, Zs, z0, pvec, uvec, F, dF0, _restruct 

      Nat = rand(8:16)
      Rs, Zs, z0 = M.rand_atenv(model, Nat)
      pvec, _restruct = destructure(ps)
      uvec = randn(length(pvec)) / sqrt(length(pvec))
      F(t) = M.evaluate(model, Rs, Zs, z0, _restruct(pvec + t * uvec), st)[1]
      dF0 = dot( destructure( M.grad_params(model, Rs, Zs, z0, ps, st)[2] )[1], uvec )
      print_tf(@test ACEbase.Testing.fdtest(F, t -> dF0, 0.0; verbose = false))
   end
   println() 

   ##

   @info("Test second mixed derivatives reverse-over-reverse")
   for ntest = 1:20 
      local Nat, Rs, Zs, Us, Ei, ∂Ei, ∂2_Ei, 
            ps_vec, vs_vec, F, dF0, z0, _restruct 

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
   println() 

   ##

   @info("Test basis implementation")

   for ntest = 1:30 
      local Nat, Rs, Zs, z0, Ei, B, θ, st1 , ∇Ei

      Nat = 15
      Rs, Zs, z0 = M.rand_atenv(model, Nat)
      i_z0 = M._z2i(model, z0)
      Ei, st1 = M.evaluate(model, Rs, Zs, z0, ps, st)
      B, st1 = M.evaluate_basis(model, Rs, Zs, z0, ps, st)
      θ = M.get_basis_params(model, ps)
      print_tf(@test Ei ≈ dot(B, θ))

      Ei, ∇Ei, st1 = M.evaluate_ed(model, Rs, Zs, z0, ps, st)
      B, ∇B, st1 = M.evaluate_basis_ed(model, Rs, Zs, z0, ps, st)
      print_tf(@test Ei ≈ dot(B, θ))
      print_tf(@test ∇Ei ≈ sum(θ .* ∇B, dims=1)[:])   
   end
   println() 

   ##

   @info("Test the full mixed jacobian")

   for ntest = 1:30 
      local Nat, Rs, Zs, z0, Ei, ∇Ei, ∂∂Ei, Us, F, dF0

      Nat = 15
      Rs, Zs, z0 = M.rand_atenv(model, Nat)
      Us = randn(SVector{3, Float64}, Nat) / sqrt(Nat)
      F(t) = destructure( M.grad_params(model, Rs + t * Us, Zs, z0, ps, st)[2] )[1]
      dF0 = ForwardDiff.derivative(F, 0.0)
      ∂∂Ei = M.jacobian_grad_params(model, Rs, Zs, z0, ps, st)[3]
      print_tf(@test dF0 ≈ transpose.(∂∂Ei) * Us)
   end 
   println() 


   ##

   @info("check splinification")
   lin_ace = M.splinify(model, ps)
   ps_lin, st_lin = LuxCore.setup(rng, lin_ace)
   ps_lin.WB[:] .= ps.WB[:] 
   ps_lin.Wpair[:] .= ps.Wpair[:]

   for ntest = 1:10
      local len, Nat, Rs, Zs, z0, Ei 
      len = 10 
      mae = sum(1:len) do _
         Nat = rand(8:16)
         Rs, Zs, z0 = M.rand_atenv(model, Nat)
         Ei = M.evaluate(model, Rs, Zs, z0, ps, st)[1]
         Ei_lin = M.evaluate(lin_ace, Rs, Zs, z0, ps_lin, st_lin)[1]
         abs(Ei - Ei_lin)
      end
      mae /= len 
      print_tf(@test mae < 0.02)
   end
   println() 

end
##

#=
@info("Basic performance benchmarks")
# first test shows the performance is not at all awful even without any 
# optimizations and reductions in memory allocations. 
using BenchmarkTools
Nat = 15
Rs, Zs, z0 = M.rand_atenv(model, Nat)
Us = randn(SVector{3, Float64}, Nat)

@info("Evaluation and adjoints")
print("   evaluate : "); @btime M.evaluate($model, $Rs, $Zs, $z0, $ps, $st)
print("evaluate_ed : "); @btime M.evaluate_ed($model, $Rs, $Zs, $z0, $ps, $st)
print("grad_params : "); @btime M.grad_params($model, $Rs, $Zs, $z0, $ps, $st)
print("  reverse^2 : "); @btime M.pullback_2_mixed(rand(), $Us, $model, $Rs, $Zs, $z0, $ps, $st)

@info("Basis evaluation ")
@info("  NB: this is currently implemented using ForwardDiff and likely inefficient")
print("      evaluate_basis : "); @btime M.evaluate_basis($model, $Rs, $Zs, $z0, $ps, $st)
print("   evaluate_basis_ed : "); @btime M.evaluate_basis_ed($model, $Rs, $Zs, $z0, $ps, $st)
print("jacobian_grad_params : "); @btime M.jacobian_grad_params($model, $Rs, $Zs, $z0, $ps, $st)
=#
