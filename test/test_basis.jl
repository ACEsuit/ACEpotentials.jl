
@testset "Basis" begin

using ACE1pack

@info("Test constructing rpi basis and parameters")
rpi_basis = basis_params(type="rpi", species = :Si, N = 3, maxdeg = 10)
ACE1pack.generate_basis(rpi_basis)

@info("Test constructing pair basis and parameters")
pair_basis = basis_params(type="pair", species = [:Ti, :Al], maxdeg = 6, r0 = 1.2)
ACE1pack.generate_basis(pair_basis)

@info("Test constructing degree and degree params")
degree = degree_params()
D = ACE1pack.generate_degree(degree)

@info("Test constructing tranform and transform parameters")
trans_params = transform_params(r0 = 1.1)
transform = ACE1pack.generate_transform(trans_params)

@info("Test constructing radial basis parameters and rpi radial basis")
rad_basis = basis_params(type="rad", rin = 0.0, pin = 0)
ACE1pack.generate_rad_basis(rad_basis, D, 6, :Si, transform)


end
