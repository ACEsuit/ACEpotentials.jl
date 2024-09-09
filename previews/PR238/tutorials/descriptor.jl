# # ACE Descriptors
#
# This tutorial demonstrates a simple use of ACE descriptors.

using ACEpotentials, MultivariateStats, Plots

# Load a tiny silicon dataset, which has the isolated atom, 25 diamond-like (dia)
# configurations, 25 beta-tin-like (bt) configurations, and 2 liquid (liq)
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

descriptors = hcat(descriptors...)
M = fit(PCA, descriptors; maxoutdim=3, pratio=1)
descriptors_trans = transform(M, descriptors)
p = scatter(
    descriptors_trans[1,:], descriptors_trans[2,:], descriptors_trans[3,:],
    marker=:circle, linewidth=0, group=config_types, legend=:right)
plot!(p, xlabel="PC1", ylabel="PC2", zlabel="PC3", camera=(20,10))


# The basis used above uses defaults that are suitable for regression of a potential energy surface, but other defaults might be better when using the ACE descriptor for other tasks such as classification. The following short script shows how to make some changes of this kind: 

model = acemodel(elements = [:Si,], order = 3, totaldegree = 10,
       pair_transform = (:agnesi, 1, 4, 0.0),
       pair_envelope = (:x, 0, 2),
       transform = (:agnesi, 1, 4, 0.0),
       envelope = (:x, 0, 2),
       r0 = :bondlen, # default, could specify explicitly
       )
basis = model.basis       

# - `[pair_]transfor = (:agnesi, 1, 4, 0.0)` : this generates a transform that behaves as t' ~ r^3 near zero and t' ~ r^-2 near the cutoff, but then actually enforces t' = 0 at the cutoff. 
# - `[pair_]envelope = (:x, 0, 2)` : this generates an envelope that is ~ (x - xcut)^2 at the cutoff and just ~ 1 for r -> 0. 