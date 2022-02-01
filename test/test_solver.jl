
@testset "Solver" begin

using ACE1pack, JuLIP

@info("Test generating solver params")
solver = solver_params(solver = :lsqr, rlap_scal = 3.0)
solver = ACE1pack.generate_solver(solver)
@test solver["solver"] == :lsqr

@info("Test applying preconditioning")
ACE_basis = ACE1pack.generate_rpi_basis(rpi_basis_params(species = :Si, N = 2, maxdeg = 10))
pair_basis = ACE1pack.generate_pair_basis(pair_basis_params(species = :Si, maxdeg = 4))
basis = JuLIP.MLIPs.IPSuperBasis([pair_basis, ACE_basis])

apply_preconditioning!(solver, basis=basis)
@test !haskey(solver, "rlap_scal")
@test haskey(solver, "P")
expected_size = sum(length(b) for b in basis.BB)
@test size(solver["P"]) == (expected_size, expected_size)

end
