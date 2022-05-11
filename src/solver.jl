
# ------------------------------------------
#   solver

using ACEfit, ACE1, LinearAlgebra, JuLIP

export solver_params, generate_solver  

"""TODO documentation"""
function solver_params(; solver = nothing, kwargs...)
    # TODO error message
    solver = _solver_to_params(solver)
    @assert solver in keys(_solvers_params)
    return _solvers_params[solver](; kwargs...)
end


# TODO: add asserts & error messages for solvers' parameters
lsqr_params(; lsqr_damp = 5e-3, lsqr_atol = 1e-6, lsqr_conlim = 1e8, lsqr_maxiter = 100000, lsqr_verbose = false) =
    Dict("solver" => "lsqr", "lsqr_damp" => lsqr_damp, "lsqr_atol" => lsqr_atol,
         "lsqr_conlim" => lsqr_conlim, "lsqr_maxiter" => lsqr_maxiter, "lsqr_verbose" => lsqr_verbose)

rrqr_params(; rrqr_tol = 1e-5) =
    Dict("solver" => "rrqr", "rrqr_tol" => rrqr_tol)

brr_params(; brr_tol = 1e-3) = 
    Dict("solver" => "brr", "brr_tol" => brr_tol)

ard_params(; ard_tol = 1e-3, ard_threshold_lambda = 10000) = 
    Dict("solver" => "ard", "ard_tol" => ard_tol, "ard_threshold_lambda" => ard_threshold_lambda)


_solver_to_params(solver::Union{Symbol, AbstractString}) = 
    string(solver)


_solvers_params = Dict("lsqr" => lsqr_params, "rrqr" => rrqr_params, 
                        "brr" => brr_params, "ard" => ard_params)


_params_to_solver(solver::AbstractString) = 
    Symbol.(solver)


"""TODO documentation"""
function generate_solver(params::Dict)
    params = copy(params)
    params["solver"] = _params_to_solver(pop!(params, "solver"))
    return params
end

