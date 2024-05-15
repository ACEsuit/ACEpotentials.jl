


# using Pkg; Pkg.activate(joinpath(@__DIR__(), "..", ".."))
# using TestEnv; TestEnv.activate();

using ACEpotentials
M = ACEpotentials.Models

using Random, LuxCore, Test, ACEbase, LinearAlgebra
using ACEbase.Testing: print_tf
rng = Random.MersenneTwister(1234)

##

max_level = 8 
level = M.TotalDegree()
maxl = 3; maxn = max_level; 
elements = (:Si, :O)
basis = M.ace_learnable_Rnlrzz(; level=level, max_level=max_level, 
                                 maxl = maxl, maxn = maxn, elements = elements)

ps, st = LuxCore.setup(rng, basis)

r = 3.0 
Zi = basis._i2z[1]
Zj = basis._i2z[2]
Rnl, st1 = basis(r, Zi, Zj, ps, st)
Rnl, Rnl_d, st1 = M.evaluate_ed(basis, r, Zi, Zj, ps, st)

@info("Test derivatives of LearnableRnlrzzBasis")

for ntest = 1:20 
   local r, Zi, Zj, U, F, dF
   r = 2.0 + rand() 
   Zi = rand(basis._i2z)
   Zj = rand(basis._i2z)
   U = randn(eltype(Rnl), length(Rnl))
   F(t) = dot(U, basis(r + t, Zi, Zj, ps, st)[1])
   dF(t) = dot(U, M.evaluate_ed(basis, r + t, Zi, Zj, ps, st)[2])
   print_tf(@test ACEbase.Testing.fdtest(F, dF, 0.0; verbose=false))
end
println() 

##

@info("LearnableRnlrzz : Consistency of single and batched evaluation")

for ntest = 1:20 
   local Rs, Rnl, Zs, Z0, Nat, st1, ∇Rnl, rs 

   Nat = rand(8:16)
   Rs, Zs, Z0 = M.rand_atenv(basis, Nat)
   rs = norm.(Rs)

   Rnl = [ M.evaluate(basis, r, Z0, z, ps, st)[1] for (r, z) in zip(rs, Zs) ]
   Rnl_b, st1 = M.evaluate_batched(basis, rs, Z0, Zs, ps, st)
   print_tf(@test all([Rnl_b[j, :] for j = 1:Nat] .≈ Rnl))

   Rnl_b2, ∇Rnl_b, _ = M.evaluate_ed_batched(basis, rs, Z0, Zs, ps, st)
   ∇Rnl = [ M.evaluate_ed(basis, r, Z0, z, ps, st)[2]
            for (r, z) in zip(rs, Zs) ] 
                   
   print_tf(@test Rnl_b ≈ Rnl_b2)
   print_tf(@test all(∇Rnl .≈ [∇Rnl_b[j, :] for j = 1:Nat ]))
end
println() 

## 

basis_p = M.set_params(basis, ps)


@info("Testing SplineRnlrzzBasis consistency via splinify")

for ntest = 1:30 
   local Nat, Rs, Zs, Zi, r, Zj, Rnl

   Nat = 1
   Rs, Zs, Zi = M.rand_atenv(basis, Nat)
   r = norm(Rs[1]) 
   Zj = Zs[1] 

   Rnl, _ = basis(r, Zi, Zj, ps, st)

   for (nnodes, tol) in [(30, 1e-3), (100, 1e-5), (1000, 1e-8)]
      local basis_spl, ps_spl, st_spl, Rnl_spl 

      basis_spl = M.splinify(basis_p; nnodes = nnodes)
      ps_spl, st_spl = LuxCore.setup(rng, basis_spl)
      Rnl_spl, _ = basis_spl(r, Zi, Zj, ps_spl, st_spl)
      rel_err = (Rnl - Rnl_spl) ./ (1 .+ abs.(Rnl))
      # use 1-norm here to not stress about small outliers
      print_tf(@test norm(rel_err, 1)/length(Rnl) < tol)
      # @show norm(rel_err, 1) / length(Rnl)
   end
end
println() 

##

@info("Test derivatives of SplineRnlrzzBasis")

basis_p = M.set_params(basis, ps)
basis_spl = M.splinify(basis_p; nnodes = 100)

for ntest = 1:20 
   local Rs, Zs, Zi, Zj, r, Rnl, U, F, dF

   Rs, Zs, Zi = M.rand_atenv(basis_spl, 1)
   r = norm(Rs[1]); Zj = Zs[1] 
   Rnl = basis_spl(r, Zi, Zj, ps, st)[1]
   U = randn(eltype(Rnl), length(Rnl))
   F(t) = dot(U, basis(r + t, Zi, Zj, ps, st)[1])
   dF(t) = dot(U, M.evaluate_ed(basis, r + t, Zi, Zj, ps, st)[2])
   print_tf(@test ACEbase.Testing.fdtest(F, dF, 0.0; verbose=false))
end
println() 

##

@info("SplineRnlrzz : Consistency of single and batched evaluation")

basis_p = M.set_params(basis, ps)
basis_spl = M.splinify(basis_p; nnodes = 100)

for ntest = 1:20 
   local Rnl, Rs, Zs, Z0, Nat, st1, ∇Rnl, rs 

   Nat = rand(8:16)
   Rs, Zs, Z0 = M.rand_atenv(basis_spl, Nat)
   rs = norm.(Rs)

   Rnl = [ M.evaluate(basis_spl, r, Z0, z, ps, st)[1] for (r, z) in zip(rs, Zs) ]
   Rnl_b, st1 = M.evaluate_batched(basis_spl, rs, Z0, Zs, ps, st)
   print_tf(@test all([Rnl_b[j, :] for j = 1:Nat] .≈ Rnl))

   Rnl_b2, ∇Rnl_b, _ = M.evaluate_ed_batched(basis_spl, rs, Z0, Zs, ps, st)
   ∇Rnl = [ M.evaluate_ed(basis_spl, r, Z0, z, ps, st)[2]
            for (r, z) in zip(rs, Zs) ] 
                   
   print_tf(@test Rnl_b ≈ Rnl_b2)
   print_tf(@test all(∇Rnl .≈ [∇Rnl_b[j, :] for j = 1:Nat ]))
end
println() 