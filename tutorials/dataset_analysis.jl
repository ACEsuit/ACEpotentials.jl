
# # Elementary Dataset Analysis

# In this tutorial we show some basic tools in ACE1pack to analyze a dataset and how this connects to a model construction. 
# As usual we start by importing the relevant packages. For plotting we will use `Plots.jl` and `LaTexStrings` for nice labels.

using ACE1pack, Plots, LaTeXStrings

# Let's generate a naive dataset, just some random bulk Si structures that are rattled a bit.

rand_Si() = rattle!(bulk(:Si, cubic=true) * rand([2,3,4]), 0.25)
Si_data = [ rand_Si() for _=1:50 ]

# Two basic distributions we can look at to see how well the data fills in space are the radial and angular distribution functions. For the radial distribution function we use the cutoff of the model (see below). For the angular distribution we use a cutoff just above the nearest neighbour distance to we can clearly see the equilibrium bond-angles. 

r_cut = 6.0  #
rdf = ACE1pack.get_rdf(Si_data, r_cut; rescale = true)

r_cut_adf = 1.25 * rnn(:Si)
adf = ACE1pack.get_adf(Si_data, 1.25 * rnn(:Si))

# We can plot these distributions using the `histogram` function in `Plots.jl`. For the RDF we add some vertical lines to indicate the distances and first, second neighbours and so forth to confirm that the peaks are in the right place. For the ADF we add a vertical line to indicate the equilibrium bond angle.

plt_rdf = histogram(rdf[(:Si, :Si)], bins=150, label = "rdf", 
                     xlabel = L"r [\AA]", ylabel = "RDF", yticks = [])
vline!(rnn(:Si) * [1.0, 1.633, 1.915, 2.3, 2.5], label = "r1, r2, ...", lw=3)

eq_angle = 1.91 # radians 
plt_adf = histogram(adf, bins=25, label = "adf", yticks = [], 
                  xlabel = L"\theta", ylabel = "ADF", xlims = (0, π))
vline!([ eq_angle,], label = "109.5˚", lw=3)

plot(plt_rdf, plt_adf, layout = (2,1), size = (800, 400))


# One way we can use these distribution functions is to look at fitted potentials relative to where data is given. But even before a potential is fitted we can illustrate some properties of the basis functions used in ACE1pack. E.g. we can illustrate why we have chosen the distance transforms.
# First, we generate a default Si model. 

model = acemodel(elements = [:Si,], order = 3,   
                 totaldegree = 10, rcut = r_cut )

# We have a utility function `get_transforms` that extracts the transforms from the model. We can then plot the transform gradients. In regions of r space with high gradient we have higher resolution. We see that the transforms concentrate resolution near the nearest neighbour distance and ensure there is no resolution at all near ``r = 0``.

using ACE1.Transforms: transform_d 
trans, trans2 = ACE1pack.get_transforms(model)
t = trans[(:Si, :Si)]
rp = range(0.0, r_cut, length = 200)

plt_t = plot(rp, abs.(transform_d.(Ref(t), rp)), lw=3, 
             xlabel = L"r [\AA]", label = L"|t'(r)|",
             yticks = [], ylabel = "", xlims = (0, r_cut))
vline!([rnn(:Si),], lw=2, label = L"r_{\rm nn}")

plt_rdf = histogram(rdf[(:Si, :Si)], bins=100, label = "rdf", 
                     xlabel = L"r [\AA]", ylabel = "RDF",
                     yticks = [], xlims = (0, r_cut))
vline!([rnn(:Si),], label = L"r_{\rm nn}", lw=3)

plot(plt_t, plt_rdf, layout=grid(2, 1, heights=[0.7, 0.3]), size = (800, 400))

# To finish this tutorial, we quickly demonstrate what happens when there is more than one chemical species present in a dataset. 

using LazyArtifacts
data_file = joinpath(artifact"TiAl_tutorial", "TiAl_tutorial.xyz")
tial_data = read_extxyz(data_file)

rdf = ACE1pack.get_rdf(data, r_cut)
plt_TiTi = histogram(rdf[(:Ti, :Ti)], bins=100, xlabel = "", 
         ylabel = "RDF - TiTi", label = "", yticks = [], xlims = (0, r_cut) )
plt_TiAl = histogram(rdf[(:Ti, :Ti)], bins=100, xlabel = "", 
         ylabel = "RDF - TiAl", label = "", yticks = [], xlims = (0, r_cut) )
plt_AlAl = histogram(rdf[(:Al, :Al)], bins=100, xlabel = L"r [\AA]", 
         ylabel = "RDF - AlAl", label = "", yticks = [], xlims = (0, r_cut), )
plot(plt_TiTi, plt_TiAl, plt_AlAl, layout = (3,1), size = (700, 700))
         


