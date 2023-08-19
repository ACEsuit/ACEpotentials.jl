
# ------------------------------------------
#   solver

using ACEfit, ACE1, LinearAlgebra, JuLIP

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
* `damp = 5e-3`
* `atol = 1e-6`
* `colim = 1e8`
* `maxiter = 1e5`
* `verbose = false`

### RRQR Parameters
* `type = "rrqr"`
* `rtol = 1e-5`

### SKLEARN_BRR
* `type = "sklearn_brr"`
* `n_iter = 300`
* `tol = 1e-3`

### SKLEARN_ARD
* `type = "sklearn_ard"`
* `n_iter = 300`
* `tol = 1e-3`
* `threshold_lambda = 1e4` 

### BLR
* `type` = "blr"
* `verbose = false`

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
qr_params(; lambda=0.0) = Dict{Any,Any}("type" => "qr", "lambda"=>lambda)

"""
`lsqr_params(; kwargs...)` : returns a dictionary containing the 
complete set of parameters required to construct a lsqr solver. 
All parameters are passed as keyword argument. 

### Parameters
* `damp = 5e-3`
* `atol = 1e-6`
* `colim = 1e8`
* `maxiter = 1e5`
* `verbose = false`
"""
lsqr_params(; damp = 5e-3, atol = 1e-6, conlim = 1e8, maxiter = Integer(1e5), verbose = false) =
    Dict("type" => "lsqr", "damp" => damp, "atol" => atol,
         "conlim" => conlim, "maxiter" => maxiter, "verbose" => verbose)

"""
`rrqr_params(; kwargs...)` : returns a dictionary containing the 
complete set of parameters required to construct a rrqr solver. 
All parameters are passed as keyword argument. 

### Parameters
* `rtol = 1e-5`
"""
rrqr_params(; rtol = 1e-5) =
    Dict("type" => "rrqr", "rtol" => rtol)

"""
`sklearn_brr_params(; kwargs...)` : returns a dictionary containing the 
complete set of parameters required to construct a scikit-learn 
Bayesian ridge regression solver. All parameters are passed as 
keyword argument. 

### Parameters
* `n_iter = 300`
* `tol = 1e-3`
"""
sklearn_brr_params(; n_iter = 300, tol = 1e-3) =
    Dict("type" => "sklearn_brr", "n_iter" => n_iter, "tol" => tol)


"""
`sklearn_ard_params(; kwargs...)` : returns a dictionary containing the 
complete set of parameters required to construct a scikit-learn 
Automatic Relevance Detemrination solver. All parameters are 
passed as keyword argument. 

### Parameters
* `n_iter = 300`
* `tol = 1e-3`
* `threshold_lambda = 1e4`
"""
sklearn_ard_params(; n_iter = 300, tol = 1e-3, threshold_lambda = 1e4) =
    Dict("type" => "sklearn_ard", "n_iter" => n_iter, "tol" => tol, "threshold_lambda" => threshold_lambda)

"""
`blr_params(; kwargs...)` : returns a dictionary containing the 
complete set of parameters required to construct a Bayesian 
linear regression solver. All parameters are passed as 
keyword argument. 

### Parameters
* `verbose = false`
"""
blr_params(; verbose = false, committee_size = 10, factorization=:svd) =
    Dict("type" => "blr", "verbose" => verbose, "committee_size" => committee_size, "factorization" => factorization)


_solver_to_params(solver_type::Union{Symbol, AbstractString}) = 
    string(solver_type)


_solvers_params = Dict("qr" => qr_params,
                       "lsqr" => lsqr_params,
                       "rrqr" => rrqr_params,
                       "sklearn_brr" => sklearn_brr_params,
                       "sklearn_ard" => sklearn_ard_params,
                       "blr" => blr_params)

_params_to_solver(solver_type::AbstractString) = 
    Symbol.(solver_type)


function generate_solver(params::Dict)
    params = copy(params)
    params["solver"] = _params_to_solver(pop!(params, "type"))
    return params
end

