


using ACE1pack, JuLIP, Test 

@info("Test generating solver params")
lsqr_damp = 1e-3
solver = solver_params(solver = :lsqr, lsqr_damp = lsqr_damp)
solver = ACE1pack.generate_solver(solver)
@test solver["solver"] == :lsqr
@test solver["lsqr_damp"] == lsqr_damp
