using ACE1pack

@info "BEGIN LISTING 4"

model = acemodel(
            elements = [:Ti, :Al],
            rcut = 5.5,
            order = 4,
            totaldegree = [25, 23, 20, 10],
            wL = 2.0)
