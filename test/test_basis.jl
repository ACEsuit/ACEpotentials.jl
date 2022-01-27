
using ACE1pack

pair_basis = rpi_basis_params(species = :Si, N = 3, maxdeg = 10)

ACE1pack.generate_rpi_basis(pair_basis)
