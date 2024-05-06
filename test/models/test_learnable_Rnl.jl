


using Pkg; Pkg.activate(joinpath(@__DIR__(), "..", ".."))
# using TestEnv; TestEnv.activate();

using ACEpotentials
M = ACEpotentials.Models

using Random, LuxCore
rng = Random.MersenneTwister(1234)

##

Dtot = 5
lmax = 3 
elements = (:Si, :O)
basis = M.ace_learnable_Rnlrzz(Dtot = Dtot, lmax = lmax, elements = elements)

ps, st = LuxCore.setup(rng, basis)

r = 3.0 
Zi = basis._i2z[1]
Zj = basis._i2z[2]
Rnl, st1 = basis(r, Zi, Zj, ps, st)

