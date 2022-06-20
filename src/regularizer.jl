
# ------------------------------------------
#   regularizers  

using ACE1

export regularizer_params

"""
`regularizer_params(; type = "laplacian", kwargs...)` : returns a dictionary containing the
complete set of parameters required to construct one of the solvers.
All parameters are passed as keyword argument and the kind of 
parameters required depend on "type".

### LSQR Parameters (default)
* `type = "laplacian"`
* `rlap_scal = 3.0`
"""
function regularizer_params(; type = "laplacian", kwargs...)
    @assert haskey(regularizers, type)
    return regularizers[type][2](; kwargs...)
end


"""
`laplacian_regularizer_params(; kwargs...)` : returns a dictionary containing the 
complete set of parameters required to construct a laplacian regularizer. 
All parameters are passed as keyword argument. 

### Parameters
* `rlap_scal = 3.0`
"""
function laplacian_regularizer_params(; rlap_scal = 3.0)
    # TODO: check that value is reasonable
    return Dict(
        "type" => "laplacian", 
        "rlap_scal" => rlap_scal)
end


function laplacian_regularizer(basis::JuLIP.MLIPs.IPSuperBasis; rlap_scal)
    return Diagonal(vcat(ACE1.scaling.(basis.BB, rlap_scal)...))
end

function generate_regularizer(basis::JuLIP.MLIPs.IPSuperBasis, params::Dict)
    regularizer = regularizers[params["type"]][1]
    kwargs = Dict([Symbol(key) => val for (key, val) in params]...)
    delete!(kwargs, :type)
    return regularizer(basis; kwargs...)
end

regularizers = Dict("laplacian" => (laplacian_regularizer, laplacian_regularizer_params))