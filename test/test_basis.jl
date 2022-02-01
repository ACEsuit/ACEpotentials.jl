
@testset "Basis" begin

using ACE1pack

@info("Test constructing rpi basis and parameters")
rpi_basis = rpi_basis_params(species = :Si, N = 3, maxdeg = 10)
ACE1pack.generate_rpi_basis(rpi_basis)

@info("Test constructing pair basis and parameters")
pair_basis = pair_basis_params(species = [:Ti, :Al], maxdeg = 6, r0 = 1.2)
ACE1pack.generate_pair_basis(pair_basis)

@info("Test constructing degree and degree params")
degree = degree_params()
D = ACE1pack.generate_degree(degree)

@info("Test constructing tranform and transform parameters")
# currently only "polynomial" transform implemented, default
trans_params = transform_params(r0 = 1.1)
transform = ACE1pack.generate_transform(trans_params)

@info("Test constructing radial basis parameters and rpi radial basis")
rad_basis = radbasis_params(rin = 0.0, pin = 0)
ACE1pack.generate_rpi_radbasis(rad_basis, D, 6, :Si, transform)

end
