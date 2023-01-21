
using ACE1pack
data = read_extxyz(joinpath(ACE1pack.artifact("TiAl_tutorial"), "TiAl_tutorial.xyz"))
model = acemodel(species = [:Ti, :Al], N = 3, maxdeg = 15, rcut = 5.5, 
                 maxdeg2 = 10, Eref = [:Ti => -1586.0195, :Al => -105.5954])
acefit!(model, data[1:5:end], ACEfit.RRQR(rtol = 1e-4))
# export_ace("TiAl.yace", model)