# # Committee Potentials

using Plots, ACEpotentials

# ### Perform the fit

# load data
train, _, _ = ACEpotentials.example_dataset("Si_tiny")
data_keys = (energy_key = "dft_energy", force_key = "dft_force")

# create model
model = acemodel(elements = [:Si,], 
                 order = 3,   
                 totaldegree = 10,           
                 rcut = 5.0)

# create solver, setting a nonzero committee size
# at present, the SVD factorization is required for committees
solver = ACEfit.BLR(committee_size=10, factorization=:svd)

# perform the fit
acefit!(model, train; solver=solver, data_keys...);

# ### Inspect the committee energies

atoms = rattle!(bulk(:Si, cubic=true) * 2, 0.2)
energy, energies = ACE1.co_energy(model.potential, atoms)
plot(1:10, energy*ones(10), label="mean energy")
plot!(1:10, energies, seriestype=:scatter, label="committee energies")

# ### Committee forces

@show ACE1.co_forces(model.potential, atoms)

# ### Committee virial

@show ACE1.co_virial(model.potential, atoms)
