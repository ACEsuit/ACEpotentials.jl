# Setup
using ACE1pack
model = acemodel(
            elements = [:Ti, :Al],
            rcut = 5.5,
            order = 4,
            totaldegree = [25, 23, 20, 10],
            wL = 2.0)
@info "BEGIN LISTING 5"
Γ = smoothness_prior(model; p = 2, wL = 1.0)
