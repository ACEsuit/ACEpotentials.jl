# # ACE Descriptors
#
# This tutorial demonstrates a simple use of ACE descriptors.

using ACEpotentials, MultivariateStats, Plots

# Load a (tiny) silicon dataset, which has the isolated atom, 25 diamond-like
# configurations (dia), 25 beta-tin-like configurations and 2 liquid (liq)
# configurations.

dataset, _, _ = ACEpotentials.example_dataset("Si_tiny")

# Define a basis.

# TODO: requires an update to the pair potential implementation 
#basis = ACE1x.ace_basis(
#      elements = [:Si],
#      order = 3,         # correlation order = body-order - 1
#      totaldegree = 12,  # polynomial degree
#      r0 = 2.3,          # estimate for NN distance
#      rcut = 5.5,)

basis = ACE1.ace_basis(
    species = [:Si],
    N = 3,        # correlation order = body-order - 1
    maxdeg = 12,  # polynomial degree
    r0 = 2.3,     # estimate for NN distance
    rin = 0.1,
    rcut = 5.5,
    pin = 2)

# Compute an averaged structural descriptor for each configuration.

descriptors = []
config_types = []
for atoms in dataset
    descriptor = zeros(length(basis))
    for i in 1:length(atoms)
        descriptor += site_energy(basis, atoms, i)
    end
    descriptor /= length(atoms)
    push!(descriptors, descriptor)
    push!(config_types, atoms.data["config_type"].data)
end

# Finally, extract the descriptor principal components and plot. Note the
# segregation by configuration type.

descriptors = hcat(descriptors...)
M = fit(PCA, descriptors; maxoutdim=3, pratio=1)
descriptors_trans = transform(M, descriptors)
p = scatter(
    descriptors_trans[1,:], descriptors_trans[2,:], descriptors_trans[3,:],
    marker=:circle, linewidth=0, group=config_types, legend=:right)
plot!(p, xlabel="PC1", ylabel="PC2", zlabel="PC3", camera=(40,10))
