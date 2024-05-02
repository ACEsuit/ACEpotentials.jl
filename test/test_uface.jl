
using ACEpotentials
using LazyArtifacts
using Test

## ----- setup -----

@info("Test UF_ACE evaluator")

@info("construct a Si model and fit parameters using RRQR")
model = acemodel(elements = [:Si],
                 Eref = [:Si => -158.54496821],
                 rcut = 5.5,
                 order = 3,
                 totaldegree = 10)
data = read_extxyz(artifact"Si_tiny_dataset" * "/Si_tiny.xyz")
data_keys = [:energy_key => "dft_energy",
             :force_key => "dft_force",
             :virial_key => "dft_virial"]
weights = Dict("default" => Dict("E"=>30.0, "F"=>1.0, "V"=>1.0),
               "liq" => Dict("E"=>10.0, "F"=>0.66, "V"=>0.25))

acefit!(model, data;
      data_keys...,
      weights = weights,
      solver = ACEfit.RRQR(rtol = 1e-6))


## 
@info("convert to UF_ACE format")      
fpot = ACEpotentials.Experimental.fast_evaluator(model)

##

@info("confirm that predictions are identical")

tolerance = 1e-8 
rattle = 0.1 

for ntest = 1:30
   at = bulk(:Si, cubic=true) * 2 
   rattle!(at, rattle)
   E1 = energy(model.potential, at) 
   E2 = energy(fpot, at)
   # @show abs(E1 - E2) < tolerance
   @test abs(E1 - E2) < tolerance
end

##

@info("construct a TiAl model and fit parameters using RRQR")


model = acemodel(elements = [:Ti, :Al],
					  order = 3,
					  totaldegree = 6, 
					  rcut = 5.5, 
					  Eref = [:Ti => -1586.0195, :Al => -105.5954])
@show length(model.basis);	

weights = Dict(
        "FLD_TiAl" => Dict("E" => 60.0, "F" => 1.0 , "V" => 1.0 ),
        "TiAl_T5000" => Dict("E" => 5.0, "F" => 1.0 , "V" => 1.0 ));

solver = ACEfit.LSQR(damp = 1e-2, atol = 1e-6);
P = smoothness_prior(model; p = 4)    #  (p = 4 is in fact the default)

data, _, meta = ACEpotentials.example_dataset("TiAl_tutorial")
data_train = data[1:5:end]
acefit!(model, data_train; solver=solver, prior = P);

##
@info("convert to UF_ACE format")      
fpot = ACEpotentials.Experimental.fast_evaluator(model)

##
@info("confirm that predictions are identical")

tolerance = 1e-8 
rattle = 0.01 

for ntest = 1:30
   at = rand(data) 
   rattle!(at, rattle)
   E1 = energy(model.potential, at) 
   E2 = energy(fpot, at)
   # @show abs(E1 - E2)  < tolerance
   @test abs(E1 - E2) < tolerance
end

