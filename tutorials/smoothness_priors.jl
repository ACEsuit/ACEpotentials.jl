using ACE1pack, ACE1x, LinearAlgebra

# read some tiny testing dataset, and bring the data into a format
# that ACEfit likes.
datapath = ACE1pack.artifact("Si_tiny_dataset")
rawdata = read_extxyz(datapath * "/Si_tiny.xyz")
datakeys = Dict("E" => "dft_energy", "F" => "dft_force", "V" => "dft_virial")

model = ACE1x.acemodel(elements = [:Si],
                       order = 3,
                       totaldegree = 10,
                       rcut = 6.0,
                       Eref = Dict("Si" => -158.54496821)
                       #transform=IdTransform(),
                       #transform_pair=IdTransform()
)

data = [ AtomsData(at, energy_key=datakeys["E"], force_key=datakeys["F"], virial_key=datakeys["V"], weights=Dict("default" => Dict("E" => 30.0, "F" => 1.0, "V" => 1.0)), v_ref=model.Vref) for at in rawdata ]


#Construct dictionary with regularisation corresponding to various priors
prior_dict = Dict()

#gaussian broadening
sigma = 2.0    #similar to atom_sigma in GAP, good values seem to be significantly larger than
rcut = 6.0       #cutoff distance
r_nn = 2.3     #typical nearest neighbour distance
sigma_n = (sigma/rcut)^2   #radial broadening factor
sigma_l = 0.2*(sigma/r_nn)^2 #angular braodening: the 0.2 here is empirical, based on python tests.
println(sigma_n)
println(sigma_l)

prior_dict["gaussian"] = gaussian_smoothness_prior(model.basis, σl = sigma_l, σn = sigma_n)

#exponential
prior_dict["exponential"] = exp_smoothness_prior(model.basis, al=0.1, an=0.1)

#algebraic
prior_dict["algebraic"] = algebraic_smoothness_prior(model.basis; p=2)

# assemble the unregularised least squares system
A, Y, W = ACEfit.linear_assemble(data, model.basis)

#Loop over prior options and print error table.
#Note how design matrix need only be assembled once.
for (prior_name, Γ) in prior_dict
    println("Solving with ", prior_name, " prior")
    #apply prior
    Ã = Diagonal(W) * (A / Γ)
    ỹ = Diagonal(W) * Y

    # solve the regularised least squares problem
    solver = ACEfit.SKLEARN_BRR(tol=1e-6, n_iter=500)
    c̃ = ACEfit.linear_solve(solver, Ã, ỹ)["C"]

    # convert parameters back to original, unregularized setting.
    c = Γ^-1 * c̃
    ACE1x._set_params!(model, c)

    # if you have a training and test set you can show errors for
    # both of course.
    ACE1pack.linear_errors(rawdata, model;
                energy_key = datakeys["E"],
                force_key = datakeys["F"],
                virial_key = datakeys["V"])
end