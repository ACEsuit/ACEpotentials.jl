
@testset "Read data" begin

using ACE1pack

fit_filename = @__DIR__()*"/files/TiAl_tutorial_DB_tenth.xyz"

@info("Test constructing `data_params` and reading data")
params = data_params(xyz_filename = fit_filename, energy_key = "energy", force_key = "force", virial_key = "virial")
data = ACE1pack.read_data(params)

end 
