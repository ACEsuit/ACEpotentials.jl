


using ACE1pack, Test 

@info("Test constructing rpi basis and parameters")
rpi_basis = basis_params(type="rpi", species = :Si, N = 3, maxdeg = 10)
ACE1pack.generate_basis(rpi_basis)


@info("Test constructing degree and degree params")
degree = degree_params()
D = ACE1pack.generate_degree(degree)
degreeM = degree_params(type = "sparseM", Dd = Dict("default" => 10, 1 => 7, (2, "H") => 3))
D = ACE1pack.generate_degree(degreeM)


@info("Test constructing tranforms and transform parameters")

# polytransform
polytransform_params = transform_params(r0 = 1.1)
transform_poly = ACE1pack.generate_transform(polytransform_params)

#multitransform 
cutoffs = Dict(
    (:C, :C) => (1.5, 2.3),
    (:H, :H) => (0.4, 2.1), 
    (:C, :H) => (1.2, 3.4))

transforms = Dict(
    (:C, :C) => transform_params(r0 = 2),
    (:H, :H) => transform_params(r0 = 1.1),
    (:C, :H) => transform_params(r0 = 2))

multitransform_params = transform_params(
    type = "multitransform", 
    transforms = transforms, 
    cutoffs = cutoffs)

transform_mult = ACE1pack.generate_transform(multitransform_params)


@info("Test constructing pair basis and parameters")
# with polytransform transform
pair_basis = basis_params(type="pair", species = [:Ti, :Al], maxdeg = 6, r0 = 1.2)
ACE1pack.generate_basis(pair_basis)

# with multitransform
pair_basis = basis_params(type="pair", species = [:Ti, :Al], maxdeg = 6, r0 = 1.2, transform = multitransform_params)
ACE1pack.generate_basis(pair_basis)


@info("Test constructing radial basis parameters and rpi radial basis")
rad_basis = basis_params(type="rad", rin = 0.0, pin = 0)
ACE1pack.generate_rad_basis(rad_basis, D, 6, :Si, transform_poly)
ACE1pack.generate_rad_basis(rad_basis, D, 6, :Si, transform_mult)
