# # Smoothness Priors

using ACEpotentials, LinearAlgebra, Plots, LaTeXStrings

# ACEpotentials models make heavy use of smoothness priors, i.e., prior parameter distributions that impose smoothness on the fitted potential. This tutorial demonstrates how to use the smoothness priors implemented in ACEpotentials.
# We start by reading in a tiny testing dataset, and bring the data into a format
# that ACEfit likes. Note that using a very limited dataset makes the use of priors particularlty important. In general, the larger and more diverse the dataset, the less important the prior becomes.

rawdata, _, _ = ACEpotentials.example_dataset("Si_tiny")
datakeys = (energy_key = "dft_energy", force_key = "dft_force", virial_key = "dft_virial")

rcut = 6.0     # cut off distance 
r_nn = 2.3     # typical nearest neighbour distance

model = ACE1x.acemodel(elements = [:Si],
                       order = 3, totaldegree = 12,
                       rcut = rcut, r0 = r_nn, 
                       Eref = Dict("Si" => -158.54496821))

data = [ AtomsData(at; datakeys..., v_ref = model.Vref) for at in rawdata ]
A, Y, W = ACEfit.assemble(data, model.basis);

# A positive definite matrix P specifies a normal prior distribution in the Bayesian framework, but for the purpose of this tutorial it is maybe more intuitive to simply think of it as a regularisation operator. The regularised linear least squares problem is 
# ```math
#   \| A c - y \|^2 + \lambda \| P c \|^2
# ```
# where `A` is the design matrix, ``y`` is the vector of observations, ``c`` is the vector of parameters, and ``\lambda`` is a regularisation parameter. The prior matrix ``P`` is specified by the user. At present we support diagonal operators ``P``. The diagonal elements of ``P`` are the prior variances. The larger the prior variance, the smoother the fitted potential.
# Although not *strictly* true, we can think of each basis function as specified by a the parameters ``(n_t, l_t)_{t = 1}^N``, where ``N`` is the correlation-order. 
# The corresponding prior matrix element must be a function of those ``n_t, l_t`` values. We currently support three classes: algebraic, exponential and gaussian. 

# **TODO:** write down the precise definitions.

# In the following we demonstrate the usage of algebraic and gaussian priors. The choices for `σl, σn` made here may seem "magical", but there is a good justification and we plan to automate this in future releases. 

Pa2 = ACE1x.algebraic_smoothness_prior(model.basis; p=2)
Pa4 = ACE1x.algebraic_smoothness_prior(model.basis; p=4)
Pg  = ACE1x.gaussian_smoothness_prior( model.basis, σl = (2/r_nn)^2, σn = (0.5/r_nn)^2);

# Each of these object `Pa2, Pa4, Pg` are diagonal matrices. For each prior constructed above we now solve the regularised least squares problem. Note how design matrix need only be assembled once if we want to play with many different priors. Most of the time we would just use defaults however and then these steps are all taken care of behind the scenes. 

priors = Dict("Id" => I, "Algebraic(2)" => Pa2, "Algebraic(4)" => Pa4,"Gaussian" => Pg)
rmse = Dict() 
pots = Dict() 

for (prior_name, P) in priors
    print("Solving with ", prior_name, " prior ... ")

    ## solve the regularized least squares problem 
    Ã = Diagonal(W) * (A / P)
    ỹ = Diagonal(W) * Y
    c̃ = ACEfit.solve(ACEfit.BLR(; verbose=false), Ã, ỹ)["C"]
    ACE1x._set_params!(model, P \ c̃)

    ## compute errors and store them for later use (don't print them here)
    errs = ACEpotentials.linear_errors(rawdata, model; verbose=false, datakeys...)
    rmse[prior_name] = errs["rmse"]["set"]["F"]
    pots[prior_name] = model.potential
    println(" force=rmse = ", rmse[prior_name])
end

# The force RMSE errors are comparable for the three priors, though slightly better for the weaker smoothness priors `Algebraic(2)` and `Id`. This is unsurprising, since those priors are less restrictive. 

# On the other hand, we expect the stronger priors to generalize better. A typical intuition is that smooth potentials with similar accuracy will be more transferable than rougher potentials. We will show three one-dimensional slices through the fitted potentials: dimer curves, trimer curves and a decohesion curve. 

# First, the dimer curves: the utility function `ACEpotentials.dimers` can be used to generate the data for those curves, which are then plotted using `Plots.jl`. We also add a vertical line to indicate the nearest neighbour distance. The standard identity prior gives a completely unrealistic dimer curve. The `Algebraic(2)` regularised potential is missing a repulsive core behaviour. The two remaining smoothness priors give physically sensible dimer curve shapes. 

labels = sort(collect(keys(priors)))[[4,1,2,3]]
plt_dim = plot(legend = :topright, xlabel = L"r [\AA]", ylabel = "E [eV]", 
               xlims = (0, rcut), ylims = (-2, 5))
for l in labels
    D = ACEpotentials.dimers(pots[l], [:Si,])
    plot!(plt_dim, D[(:Si, :Si)]..., label = l, lw=2)
end
vline!([r_nn,], lw=2, ls=:dash, label = L"r_{\rm nn}")
plt_dim

# Next, we look at a trimer curve. This is generated using `ACEpotentials.trimers`. Both the `Id` and `Algebraic(2)` regularised models contain fairly significant oscillations while the `Algebraic(4)` and `Gaussian` models are much smoother. In addition, it appears that the `Gaussian` regularised model is somewhat more physically realistic on this slice with high energy at small bond-angles (thought the low energy at angle π seems somewhat strange).

plt_trim = plot(legend = :topright, xlabel = L"\theta", ylabel = "E [eV]", 
               xlims = (0, pi), ylims = (-0.35, 0.8))
for l in labels
    D = ACEpotentials.trimers(pots[l], [:Si,], r_nn,  r_nn)
    plot!(plt_trim, D[(:Si, :Si, :Si)]..., label = l, lw=2)
end
vline!(plt_trim, [1.90241,], lw=2, label = "equilibrium angle")
plt_trim

# Finally, we plot a decohesion curve, which contains more significant many-body effects. Arguably, none of our potentials perform very well on this test. Usually larger datasets, and longer cutoffs help in this case. 

at0 = bulk(:Si, cubic=true)
plt_dec = plot(legend = :topright, xlabel = L"r [\AA]", ylabel = "strain [eV/Å]", 
                xlim = (0.0, 5.0))
for l in labels
    aa, E, dE = ACEpotentials.decohesion_curve(at0, pots[l])
    plot!(plt_dec, aa, dE, label = l, lw=2)
end
plt_dec

