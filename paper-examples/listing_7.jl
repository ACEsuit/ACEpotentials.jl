model = ... # specify a model; see e.g. Listing 1
data = ... # load training data; see e.g. Listing 6
P = smoothness_prior(model) # regularisation operator; see § II C
weights = Dict( "default" => Dict("E" => 30.0, "F" => 1.0, "V" => 1.0),
                "liquid" => Dict("E" => 5.0, "F" => 0.5, "V" => 0.25) )
solver = BLR(tol = 1e-3, P = P) # specify the solver, see Table I for options
acefit!(model, data, solver) # assemble and solve lsq problem, update model parameters

# model accuracy on a test set
testdata = ... # load test data
errors(testdata, model)

# export the fitted potential
export2json("model.json", model)
export2lammps("model.yace", model)
