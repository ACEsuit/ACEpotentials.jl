
@testset "Precon" begin

using ACE1pack

@info("Test generating laplacian precon")
# construct basis
ACE_basis = ACE1pack.generate_ace_basis(ace_basis_params(species = :Si, N = 2, maxdeg = 10))
pair_basis = ACE1pack.generate_pair_basis(pair_basis_params(species = :Si, maxdeg = 4))
basis = JuLIP.MLIPs.IPSuperBasis([pair_basis, ACE_basis])

rlap_scal = 2.0
precon = precon_params(type = "laplacian", rlap_scal = rlap_scal)
@test precon["rlap_scal"] == rlap_scal

expected_size = sum(length(b) for b in basis.BB)
P = generate_precon(basis, precon)
@test size(P) == (expected_size, expected_size)


end