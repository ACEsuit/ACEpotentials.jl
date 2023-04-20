using ACE1pack
model = acemodel(species = [:Ti, :Al], N = 4, rcut = 5.5,
            wL = 2.0,
            maxdeg = [25, 23, 20, 10])

@info "BEGIN LISTING 5"

Γ = smoothness_prior(model; p = 2, wL = 1)
