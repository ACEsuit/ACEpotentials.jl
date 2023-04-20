using ACE1pack

@info "BEGIN LISTING 1"

model = acemodel(; elements = [:Ti, :Al],
            order = 3,
            totaldegree = 15,
            rcut = 5.5,
            Eref = [:Ti => -1586.0195, :Al => -105.5954])
