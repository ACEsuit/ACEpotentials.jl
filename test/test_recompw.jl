

using ACE1pack, Test 
using ACE1pack: linear_assemble

##

@info("Checking recompute_weights")

fname=joinpath(ACE1pack.artifact("Si_tiny_dataset"), "Si_tiny.xyz")
datakeys = (energy_key="dft_energy", force_key="dft_force", virial_key="dft_virial")
rawdata = read_extxyz(fname)

model = acemodel(elements = [:Si,], order = 3, totaldegree = 6);

# standard assembly 
weights1 = Dict("default" => Dict("E"=>30.0, "F"=>1.0, "V"=>1.0))
A, Y, W1 = linear_assemble(rawdata, model; weights=weights1, datakeys...)

W2 = ACE1pack.recompute_weights(model, rawdata; weights=weights1, datakeys...)

weights2 = Dict("default" => Dict("E"=>30.0, "F"=>0.1, "V"=>10.0))
W3 = ACE1pack.recompute_weights(model, rawdata; weights=weights2, datakeys...)

println(@test W1 ≈ W2)
println(@test !(W2 ≈ W3))
