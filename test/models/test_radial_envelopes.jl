

# using Pkg; Pkg.activate(".");
# using TestEnv; TestEnv.activate();

using ACEpotentials

# there are no real tests for envelopes yet. The only thing we have is 
# a plot of the envelopes to inspect manually.

##

#=
using Plots

rcut = 2.0 
envpair = ACEpotentials.Models.PolyEnvelope1sR(rcut, 1)
rr = range(0.0001, rcut+0.5, length=200)
y2 = ACEpotentials.Models.evaluate.(Ref(envpair), rr)

envmb = ACEpotentials.Models.PolyEnvelope2sX(0.0, 1.0, 2, 2) 
ymb = ACEpotentials.Models.evaluate.(Ref(envmb), rr)

plot(rr, y2, label="pair envelope", lw=2, legend=:topleft, ylims = (-1.0, 3.0))
plot!(rr, ymb, label="mb envelope", lw=2, )
=#

##


