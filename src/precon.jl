
# ------------------------------------------
#   preconditioning 

using ACE1

# TODO remove generate export
export precon_params, generate_precon


function precon_params(; type = "laplacian", kwargs...)
    @assert haskey(_preconditioners, type)
    return _preconditioners[type][2](; kwargs...)
end

function laplacian_precon_params(; rlap_scal = 3.0)
    # TODO: check that value is reasonable
    return Dict(
        "type" => "laplacian", 
        "rlap_scal" => rlap_scal)
end


""" TODO documentation"""
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