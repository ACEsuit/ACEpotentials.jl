# # Some Experimental Features
#
# This tutorial will go through a few experimental features 
# that are made available but may not be fully tested and should 
# note be relied upon. 

using ACEpotentials

# We do a quick fit of a TiAl potential following the same steps as in the
#  TiAl model tutorial.

model = acemodel(elements = [:Ti, :Al], order = 3, totaldegree = 6, 
					  rcut = 5.5, Eref = [:Ti => -1586.0195, :Al => -105.5954])
weights = Dict("FLD_TiAl" => Dict("E" => 60.0, "F" => 1.0 , "V" => 1.0 ),
               "TiAl_T5000" => Dict("E" => 5.0, "F" => 1.0 , "V" => 1.0 ))
solver = ACEfit.LSQR(damp = 1e-2, atol = 1e-6);
P = smoothness_prior(model; p = 4)
data, _, meta = ACEpotentials.example_dataset("TiAl_tutorial")
data_train = data[1:5:end]
acefit!(model, data_train; solver=solver, prior = P);

# Next we convert the model to a new experimental evaluator that should be 
# a lot faster - at least for small models. 

fpot = ACEpotentials.Experimental.fast_evaluator(model);

# The predictions should be correct to within 10-13 digits. 

for ntest = 1:10
   at = rattle!(rand(data), 0.01) 
   E = JuLIP.energy(model.potential, at) 
   E_fast = JuLIP.energy(fpot, at)
   @show abs(E - E_fast) 
end;

# Now let's look at timings, they should be significantly faster for the 
# new evaluator. Note that the speedup will be different depending on 
# the size of the model and the architecture of the computer.

JuLIP.forces(model.potential, data[1]);
JuLIP.forces(fpot, data[1]);
print("Energy, old evaluator: ")
@time for d in data; JuLIP.energy(model.potential, d); end
print("Energy, new evaluator: ")
@time for d in data; JuLIP.energy(fpot, d); end
print("Forces, old evaluator: ")
@time for d in data; JuLIP.forces(model.potential, d); end
print("Forces, new evaluator: ")
@time for d in data; JuLIP.forces(fpot, d); end
