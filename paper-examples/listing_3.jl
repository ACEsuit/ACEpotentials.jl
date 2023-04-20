using ACE1pack

@info "BEGIN LISTING 3"

species = [:Ti, :Al]
r0 = (rnn(:Ti) + rnn(:Al)) / 2
rin = 0.5 * r0
rcut = 2 * r0  # TODO: just added this
trans = AgnesiTransform(; r0=r0, rin=rin, p = 2)
fenv = ACE1.PolyEnvelope(1, r0, rcut)
radbasis = transformed_jacobi_env(12, trans, fenv, rcut)
model = model(species = species,
              N = 3, maxdeg = 15,
              radbasis = radbasis)
