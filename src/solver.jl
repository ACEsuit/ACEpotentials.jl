
# ------------------------------------------
#   solver

using IPFitting, ACE1, LinearAlgebra, JuLIP

export solver_params   


"""
`solver_params(; kwargs...)` : returns a dictionary containing the
complete set of parameters required to construct one of the solvers.
All parameters are passed as keyword argument and the kind of 
parameters required depend on "type".

### QR Parameters
* 'type = "qr"`

### LSQR Parameters 
* `type = "lsqr"`
* `lsqr_damp = 5e-3`
* `lsqr_atol = 1e-6`
* `lsqr_colim = 1e8`
* `lsqr_maxiter = 1e5`
* `lsqr_verbose = false`

### RRQR Parameters
* `type = "rrqr"`
* `rrqr_tol = 1e-5`

### BBR 
* `type = "bbr"`
* `brr_n_iter = 300`
* `brr_tol = 1e-3`

### ARD
* `type = "ard"`
* `ard_n_iter = 300`
* `ard_tol = 1e-3`
* `ard_threshold_lambda = 1e4` 

"""
function solver_params(; type = nothing, kwargs...)
    # TODO error message
    solver_type = _solver_to_params(type)
    @assert solver_type in keys(_solvers_params)
    return _solvers_params[solver_type](; kwargs...)
end

"""
`qr_params(; kwargs...)` : returns a dictionary containing the
complete set of parameters required to construct a qr solver.
All parameters are passed as keyword argument.

"""
qr_params() = Dict{Any,Any}("type" => "qr")

"""
`lsqr_params(; kwargs...)` : returns a dictionary containing the 
complete set of parameters required to construct a lsqr solver. 
All parameters are passed as keyword argument. 

### Parameters
* `lsqr_damp = 5e-3`
* `lsqr_atol = 1e-6`
* `lsqr_colim = 1e8`
* `lsqr_maxiter = 1e5`
* `lsqr_verbose = false`
"""
lsqr_params(; lsqr_damp = 5e-3, lsqr_atol = 1e-6, lsqr_conlim = 1e8, lsqr_maxiter = Integer(1e5), lsqr_verbose = false) =
    Dict("type" => "lsqr", "lsqr_damp" => lsqr_damp, "lsqr_atol" => lsqr_atol,
         "lsqr_conlim" => lsqr_conlim, "lsqr_maxiter" => lsqr_maxiter, "lsqr_verbose" => lsqr_verbose)

"""
`rrqr_params(; kwargs...)` : returns a dictionary containing the 
complete set of parameters required to construct a rrqr solver. 
All parameters are passed as keyword argument. 

### Parameters
* `rrqr_tol = 1e-5`
"""
rrqr_params(; rrqr_tol = 1e-5) =
    Dict("type" => "rrqr", "rrqr_tol" => rrqr_tol)

"""
`brr_params(; kwargs...)` : returns a dictionary containing the 
complete set of parameters required to construct a scikit-learn 
Bayesian ridge regression solver. All parameters are passed as 
keyword argument. 

### Parameters
* `brr_n_iter = 300`
* `brr_tol = 1e-3`
"""
brr_params(; brr_n_iter = 300, brr_tol = 1e-3) =
    Dict("type" => "brr", "brr_n_iter" => brr_n_iter, "brr_tol" => brr_tol)


"""
`ard_params(; kwargs...)` : returns a dictionary containing the 
complete set of parameters required to construct a scikit-learn 
Automatic Relevance Detemrination solver. All parameters are 
passed as keyword argument. 

### Parameters
* `ard_n_iter = 300`
* `ard_tol = 1e-3`
* `ard_threshold_lambda = 1e4`
"""
ard_params(; ard_n_iter = 300, ard_tol = 1e-3, ard_threshold_lambda = 1e4) =
    Dict("type" => "ard", "ard_n_iter" => ard_n_iter, "ard_tol" => ard_tol, "ard_threshold_lambda" => ard_threshold_lambda)


_solver_to_params(solver_type::Union{Symbol, AbstractString}) = 
    string(solver_type)


_solvers_params = Dict("qr" => qr_params,
                       "lsqr" => lsqr_params, "rrqr" => rrqr_params,
                       "brr" => brr_params, "ard" => ard_params)


_params_to_solver(solver_type::AbstractString) = 
    Symbol.(solver_type)


function generate_solver(params::Dict)
    params = copy(params)
    params["solver"] = _params_to_solver(pop!(params, "type"))
    return params
end

