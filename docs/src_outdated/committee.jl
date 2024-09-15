# # Committee Potentials

using Plots, ACEpotentials, Statistics 

# ### Perform the fit

# load some example training data

train, _, _ = ACEpotentials.example_dataset("Si_tiny")
data_keys = (energy_key = "dft_energy", force_key = "dft_force");

# create model

model = acemodel(elements = [:Si,], order = 3, totaldegree = 8);

# create solver, setting a nonzero committee size at present, the SVD factorization is required for committees

solver = ACEfit.BLR(committee_size=10, factorization=:svd);

# perform the fit

acefit!(model, train; solver=solver, data_keys...);

# Inspect the total energies vs committee energies and error bars for a few perturbed structures. Note the training set is so small that we don't expect these committees to be particularly useful; this is only to illustrate how they might be used. Also note that the energy `E` is *not* in general the mean of `co_E` but it is the mean of the exact posterior distribution. 

atoms = bulk(:Si, cubic=true) * 2; rattle = [0.03, 0.1, 0.3]
plot(; size = (300, 300), xlabel = "rattle", ylabel = "energy [eV]", ylims = (-10650, -10250), 
      xlims = (0.015, 0.6), xticks = (rattle, string.(rattle)), xscale = :log10)
for (i, rt) in enumerate(rattle)
   rattle!(atoms, rt)
   E, co_E = ACE1.co_energy(model.potential, atoms)   
   scatter!(rt*ones(10), co_E, c = 1, label=(i==1 ? "committee" : ""))
   scatter!([rt,], [E,], yerror = [std(co_E),], c = 2, ms=6, label=(i==1 ? "mean" : ""))
end
plot!()

# Committee forces are computed analogously. `F` is a vector of mean forces (i.e. a vector of 3-vectors), while `co_F` is a list of vectors of committe forces (i.e. a vector of vectors of 3-vectors). 

F, co_F = ACE1.co_forces(model.potential, atoms)
@show typeof(F)
@show typeof(co_F);

# The situation is analogous for committee virials

V, co_V = ACE1.co_virial(model.potential, atoms)
@show typeof(V)
@show typeof(co_V);
