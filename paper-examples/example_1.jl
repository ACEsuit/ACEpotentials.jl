@info "BEGIN EXAMPLE 1"

using ACE1pack
pathtodata = joinpath(ACE1pack.artifact("TiAl_tutorial"), "TiAl_tutorial.xyz")
data = read_extxyz(pathtodata)
#model = acemodel(elements = [:Ti, :Al], order = 3, totaldegree = 15,
model = acemodel(elements = [:Ti, :Al], order = 3, totaldegree = 5,
                 Eref = [:Ti => -1586.0195, :Al => -105.5954])
acefit!(model, data)
# TODO
# export2lammps("TiAl.yace", model)
#ACE1pack.ExportMulti.export_ACE("potential.yace", model.potential, export_pairpot_as_table=true)
