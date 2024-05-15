
# using Pkg; Pkg.activate(joinpath(@__DIR__(), "..", ".."))
# using TestEnv; TestEnv.activate();

using ACEpotentials
M = ACEpotentials.Models

using Random, LuxCore, Test, ACEbase, LinearAlgebra
using ACEbase.Testing: print_tf
rng = Random.MersenneTwister(1234)

##

max_level = 16
level = M.TotalDegree()
maxl = 0; maxn = max_level; 
elements = (:Si, :O)
basis = M.ace_learnable_Rnlrzz(; level=level, max_level=max_level, 
                                 maxl = maxl, maxn = maxn, 
                                 elements = elements, 
                                 transforms = (:agnesi, 1, 4), 
                                 envelopes = :poly1sr )

ps, st = LuxCore.setup(rng, basis)

r = 3.0 
Zi = basis._i2z[1]
Zj = basis._i2z[2]
Rnl1, st1 = basis(r, Zi, Zj, ps, st)
Rnl, Rnl_d, st1 = M.evaluate_ed(basis, r, Zi, Zj, ps, st)

basis_p = M.set_params(basis, ps)
basis_spl = M.splinify(basis_p)
ps_spl, st_spl = LuxCore.setup(rng, basis_spl)

Rnl2, _ = M.evaluate(basis_spl, r, Zi, Zj, ps_spl, st_spl)
Rnl2, Rnl_d2, _ = M.evaluate_ed(basis_spl, r, Zi, Zj, ps_spl, st_spl)

## 
# inspect the basis visually 

# using Plots

# rr = range(0.1, 6.0, length=300)
# Zs = fill(14, length(rr))
# z0 = 14
# Rnl, _ = M.evaluate_batched(basis_spl, rr, z0, Zs, ps, st)
# env_rr = M.evaluate.(Ref(basis_spl.envelopes[1,1]), rr, 0.0)

# plt1 = plot(; ylims = (-2.0, 5.0), )
# plt2 = plot(; ylims = (-3.0, 3.0), )
# for n = 1:5
#    plot!(plt1, rr, Rnl[:, n], label = "n=$n")
#    plot!(plt2, rr, Rnl[:, n] ./ env_rr, label ="")
# end
# vline!(plt1, [basis_spl.rin0cuts[1,1].r0], label = "r0")
# vline!(plt2, [basis_spl.rin0cuts[1,1].r0], label = "")

# plot(plt1, plt2, layout = (2,1))


##


@info("Test derivatives of Spline Rnl Basis for Pairpot")

for ntest = 1:20 
   local r, Zi, Zj, U, F, dF
   Zi = rand(basis_spl._i2z)
   Zj = rand(basis_spl._i2z)
   r = 2.0 + rand() 
   U = randn(length(basis_spl))
   F(t) = dot(U, basis_spl(r + t, Zi, Zj, ps, st)[1])
   dF(t) = dot(U, M.evaluate_ed(basis_spl, r + t, Zi, Zj, ps, st)[2])
   print_tf(@test ACEbase.Testing.fdtest(F, dF, 0.0; verbose=false))
end
println() 

##

