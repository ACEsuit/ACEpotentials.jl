


using ACE1pack, JuLIP, Test 

@info("Test generating solver params")
damp = 1e-3
solver = solver_params(type = :lsqr, damp = damp)
solver = ACE1pack.generate_solver(solver)
@test solver["solver"] == :lsqr
@test solver["damp"] == damp
