# tests scripts/ace_fit.jl

using ACE1pack, JuLIP, LazyArtifacts, Test

### create the param file
test_train_set = joinpath(artifact"TiAl_tiny_dataset", "TiAl_tiny.xyz")
species = [:Ti, :Al]
r0 = 2.88 
data = data_params(fname = test_train_set,
    energy_key = "energy",
    force_key = "force",
    virial_key = "virial")
ace_basis = basis_params(
    type = "ace",
    species = species, 
    N = 3, 
    maxdeg = 6, 
    r0 = r0, 
    radial = basis_params(
        type = "radial", 
        rcut = 5.0, 
        rin = 1.44,
        pin = 2))
pair_basis = basis_params(
    type = "pair", 
    species = species, 
    maxdeg = 6,
    r0 = r0,
    rcut = 5.0,
    rin = 0.0,
    pcut = 2, # TODO: check if it should be 1 or 2?
    pin = 0)
basis = Dict(
    "ace" => ace_basis,
    "pair" => pair_basis
)
solver = solver_params(type = :lsqr)
e0 = Dict("Ti" => -1586.0195, "Al" => -105.5954)
weights = Dict(
    "default" => Dict("E" => 5.0, "F" => 1.0, "V" => 1.0),
    "FLD_TiAl" => Dict("E" => 5.0, "F" => 1.0, "V" => 1.0),
    "TiAl_T5000" => Dict("E" => 30.0, "F" => 1.0, "V" => 1.0))
P = regularizer_params(type = "laplacian", rlap_scal = 3.0)
params = fit_params(
    data = data,
    basis = basis,
    solver = solver,
    e0 = e0,
    weights = weights,
    P = P,
    ACE_fname = "TiAl.json")
save_dict("TiAl.yaml", params)

### run the fit
isfile("TiAl.json") && rm("TiAl.json")
run(pipeline(`julia --project=.. ../scripts/ace_fit.jl --params TiAl.yaml`,
             stdout="TiAl.log",
             stderr="TiAl.log"))

### perform the tests
for line in readlines("TiAl.log")
   if occursin("set", line)
       ene_err = parse(Float64, split(line)[4]) * 1e-3
       for_err = parse(Float64, split(line)[6])
       vir_err = parse(Float64, split(line)[8]) * 1e-3
       @test isapprox(ene_err, 0.005, atol=1e-3)
       @test isapprox(for_err, 0.199, atol=1e-3)
       @test isapprox(vir_err, 0.066, atol=1e-3)
       break
   end
end

### tidy up
rm("TiAl.json")
rm("TiAl.log")
rm("TiAl.yace")
rm("TiAl.yaml")
rm("TiAl_pairpot.table")
