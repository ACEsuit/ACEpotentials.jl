using ACE1pack

@info "BEGIN LISTING 6"

pathtodata = joinpath(ACE1pack.artifact("TiAl_tutorial"), "TiAl_tutorial.xyz")
data = read_extxyz(pathtodata)
