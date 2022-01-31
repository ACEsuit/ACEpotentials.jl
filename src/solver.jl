
# ------------------------------------------
#   solver

using IPFitting, ACE1, LinearAlgebra, JuLIP

export solver_params, precondition_laplacian

"""TODO documentation"""
function solver_params(;
    solver = nothing,
    rlap_scal = nothing,
    lsqr_damp = 5e-3,
    lsqr_atol = 1e-6,
    rrqr_tol = 1e-5,
    ard_tol = 1e-4,
    ard_threshold_lambda = 1e-2 #ard
)
    #ENH: maybe use `kwargs...`, similarly to `transform_params`, instead of defining all of the parameters?

    # TODO: explain 
    @assert !isnothing(solver)

    solver_params = Dict("solver" => _solver_to_params(solver), "rlap_scal" => rlap_scal)
    if solver == :lsqr
        solver_params["lsqr_damp"] = lsqr_damp
        solver_params["lsqr_atol"] = lsqr_atol
    elseif solver == :rrqr
        solver_params["rrqr_tol"] = rrqr_tol
    elseif solver == :brr
    elseif solver == :ard
        solver_params["ard_tol"] = ard_tol
        solver_params["ard_threshold_lambda"] = ard_threshold_lambda
    else
        throw(error("Unrecognised solver \"$(solver)\". Available options: $(:lsqr), $(:rrqr), $(:brr), $(:ard)"))
    end

    return solver_params
end

_solver_to_params(solver::Union{Symbol, AbstractString}) = 
    string(solver)

_params_to_solver(solver::AbstractString) = 
    Symbol.(solver)


"""TODO documentation"""
function generate_solver(params::Dict)
    params = copy(params)
    params["solver"] = _params_to_solver(pop!(params, "solver"))
    return params
end


"""TODO documentation"""
function apply_preconditioning!(params::Dict; basis = nothing)
    rlap_scal = pop!(params, "rlap_scal", nothing)
    if !isnothing(rlap_scal)
        @info("Applying preconditioning")
        @assert !isnothing(basis)
        P = precondition_laplacian(basis, rlap_scal)
        params["P"] = P
    end
end


""" TODO documentation"""
function precondition_laplacian(basis::JuLIP.MLIPs.IPSuperBasis, rlap_scal)
    return Diagonal(vcat(ACE1.scaling.(basis.BB, rlap_scal)...))
end