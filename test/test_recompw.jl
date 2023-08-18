using ACE1pack, Test 

@info("Checking recompute_weights")

# generate data
rawdata, _, _ = ACE1pack.example_dataset("Si_tiny")
datakeys = (energy_key="dft_energy", force_key="dft_force", virial_key="dft_virial")
weights = Dict("default" => Dict("E"=>30.0, "F"=>1.0, "V"=>1.0))
data = [AtomsData(at; datakeys..., weights = weights) for at in rawdata]

# test recompute weights with identical weights
W1 = ACEfit.assemble_weights(data)
W2 = ACE1pack.recompute_weights(rawdata; weights=weights, datakeys...)
println(@test W1 ≈ W2)

# test recompute_weights with different weights
weights2 = Dict("default" => Dict("E"=>30.0, "F"=>0.1, "V"=>10.0))
W3 = ACE1pack.recompute_weights(rawdata; weights=weights2, datakeys...)
println(@test !(W2 ≈ W3))
