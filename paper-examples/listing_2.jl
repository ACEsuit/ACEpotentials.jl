using ACE1pack
model = acemodel(; elements = [:Ti, :Al],
            order = 3,
#            totaldegree = 15,
            totaldegree = 5,
            rcut = 5.5,
            Eref = [:Ti => -1586.0195, :Al => -105.5954]) # 1-body parameters

@info "BEGIN LISTING 2"

#model = ... # cf. Listing 1
P = smoothness_prior(model)
pathtodata = joinpath(ACE1pack.artifact("TiAl_tutorial"), "TiAl_tutorial.xyz")
data = read_extxyz(pathtodata)
acefit!(model, data; solver = ACEfit.RRQR(rtol = 1e-4, P = P))
# TODO
#export2lammps("TiAl.yace", model)
