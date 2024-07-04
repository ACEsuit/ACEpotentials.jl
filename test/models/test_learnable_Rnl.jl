


# using Pkg; Pkg.activate(joinpath(@__DIR__(), "..", ".."))
# using TestEnv; TestEnv.activate();

using ACEpotentials
M = ACEpotentials.Models

using Random, LuxCore, Test, ACEbase, LinearAlgebra
using ACEbase.Testing: print_tf
rng = Random.MersenneTwister(1234)

Random.seed!(1234)

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
Rnl, Rnl_d = M.evaluate_ed(basis, r, Zi, Zj, ps, st)

@info("Test derivatives of Rnlrzz basis")

for ntest = 1:20 
   global ps, st
   r = 2.0 + rand() 
   Zi = rand(basis._i2z)
   Zj = rand(basis._i2z)
   U = randn(eltype(Rnl), length(Rnl))
   F(t) = dot(U, basis(r + t, Zi, Zj, ps, st))
   dF(t) = dot(U, M.evaluate_ed(basis, r + t, Zi, Zj, ps, st)[2])
   print_tf(@test ACEbase.Testing.fdtest(F, dF, 0.0; verbose=false))
end
println() 

##

@info("LearnableRnlrzz : Consistency of single and batched evaluation")

for ntest = 1:20 
   global ps, st
   Nat = rand(8:16)
   Rs, Zs, Z0 = M.rand_atenv(basis, Nat)
   rs = norm.(Rs)

   Rnl = [ M.evaluate(basis, r, Z0, z, ps, st) for (r, z) in zip(rs, Zs) ]
   Rnl_b = M.evaluate_batched(basis, rs, Z0, Zs, ps, st)
   print_tf(@test all([Rnl_b[j, :] for j = 1:Nat] .≈ Rnl))

   Rnl_b2, ∇Rnl_b = M.evaluate_ed_batched(basis, rs, Z0, Zs, ps, st)
   ∇Rnl = [ M.evaluate_ed(basis, r, Z0, z, ps, st)[2]
            for (r, z) in zip(rs, Zs) ] 
                   
   print_tf(@test Rnl_b ≈ Rnl_b2)
   print_tf(@test all(∇Rnl .≈ [∇Rnl_b[j, :] for j = 1:Nat ]))
end

## 

@info("quick splinification check")
basis_spl = M.splinify(basis, ps; nnodes = 100)
ps_spl, st_spl = LuxCore.setup(rng, basis_spl)

Rnl = basis(r, Zi, Zj, ps, st)
Rnl_spl = basis_spl(r, Zi, Zj, ps_spl, st_spl)

println_slim(@test norm(Rnl - Rnl_spl, Inf) < 1e-4)

Rnl, ∇Rnl = M.evaluate_ed(basis, r, Zi, Zj, ps, st)
Rnl_spl, ∇Rnl_spl = M.evaluate_ed(basis_spl, r, Zi, Zj, ps_spl, st_spl)

println_slim(@test norm(Rnl - Rnl_spl, Inf) < 1e-4 )
println_slim(@test norm(∇Rnl - ∇Rnl_spl, Inf) < 1e-2 )

