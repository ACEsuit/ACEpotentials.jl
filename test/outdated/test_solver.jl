


using ACEpotentials, JuLIP, Test 

@info("Test generating solver params")
damp = 1e-3
solver = ACEpotentials.solver_params(type = :lsqr, damp = damp)
solver = ACEpotentials.generate_solver(solver)
@test solver["solver"] == :lsqr
@test solver["damp"] == damp
