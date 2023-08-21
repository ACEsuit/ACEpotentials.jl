
@testset "Regularizer" begin

using ACEpotentials

@info("Test generating laplacian regularizer")
# construct basis
ACE_basis = ACEpotentials.generate_ace_basis(ace_basis_params(species = :Si, N = 2, maxdeg = 10))
pair_basis = ACEpotentials.generate_pair_basis(pair_basis_params(species = :Si, maxdeg = 4))
basis = JuLIP.MLIPs.IPSuperBasis([pair_basis, ACE_basis])

rlap_scal = 2.0
regularizer = regularizer_params(type = "laplacian", rlap_scal = rlap_scal)
@test regularizer["rlap_scal"] == rlap_scal

expected_size = sum(length(b) for b in basis.BB)
P = generate_regularizer(basis, regularizer)
@test size(P) == (expected_size, expected_size)


end
