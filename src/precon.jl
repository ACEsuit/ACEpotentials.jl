
# ------------------------------------------
#   preconditioning 

using ACE1

export precon_params

"""
`precon_params(; type = "laplacian", kwargs...)` : returns a dictionary containing the
complete set of parameters required to construct one of the solvers.
All parameters are passed as keyword argument and the kind of 
parameters required depend on "type".

### LSQR Parameters (default)
* `type = "laplacian"`
* `rlap_scal = 3.0`
"""
function precon_params(; type = "laplacian", kwargs...)
    @assert haskey(_preconditioners, type)
    return _preconditioners[type][2](; kwargs...)
end


"""
`laplacian_precon_params(; kwargs...)` : returns a dictionary containing the 
complete set of parameters required to construct a laplacian preconditioner. 
All parameters are passed as keyword argument. 

### Parameters
* `rlap_scal = 3.0`
"""
function laplacian_precon_params(; rlap_scal = 3.0)
    # TODO: check that value is reasonable
    return Dict(
        "type" => "laplacian", 
        "rlap_scal" => rlap_scal)
end


function laplacian_precon(basis::JuLIP.MLIPs.IPSuperBasis; rlap_scal)
    return Diagonal(vcat(ACE1.scaling.(basis.BB, rlap_scal)...))
end

function generate_precon(basis::JuLIP.MLIPs.IPSuperBasis, params::Dict)
    precon = _preconditioners[params["type"]][1]
    kwargs = Dict([Symbol(key) => val for (key, val) in params]...)
    delete!(kwargs, :type)
    return precon(basis; kwargs...)
end

_preconditioners = Dict("laplacian" => (laplacian_precon, laplacian_precon_params))