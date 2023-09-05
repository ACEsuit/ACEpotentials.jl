# # ACE Descriptors
#
# This tutorial demonstrates a simple use of ACE descriptors.

using ACEpotentials, MultivariateStats, Plots

# Load a (tiny) silicon dataset, which has the isolated atom, 25 diamond-like
# configurations (dia), 25 beta-tin-like configurations and 2 liquid (liq)
# configurations.

dataset, _, _ = ACEpotentials.example_dataset("Si_tiny");

# An ACE basis specifies a vector of invariant features of atomic environments and can therefore be used as a general descriptor.

basis = ACE1x.ace_basis(elements = [:Si],
                        rcut = 5.5,
                        order = 3,        # body-order - 1
                        totaldegree = 8);

# Compute an averaged structural descriptor for each configuration.

descriptors = []
config_types = []
for atoms in dataset
    struct_descriptor = sum(site_descriptors(basis, atoms)) / length(atoms)
    push!(descriptors, struct_descriptor)
    push!(config_types, atoms.data["config_type"].data)
end

# Finally, extract the descriptor principal components and plot. Note the segregation by configuration type.

descriptors1 = hcat(descriptors...)
M = fit(PCA, descriptors1; maxoutdim=3, pratio=1)
descriptors_trans = transform(M, descriptors1)
p = scatter(
    descriptors_trans[1,:], descriptors_trans[2,:], descriptors_trans[3,:],
    marker=:circle, linewidth=0, group=config_types, legend=:right)
plot!(p, xlabel="PC1", ylabel="PC2", zlabel="PC3", camera=(20,10))
