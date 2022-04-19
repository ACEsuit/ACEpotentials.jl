
# ------------------------------------------
#    ACE Fitting

import IPFitting, ACE1pack, ACE1

export fit_params, fit_ace 

"""
TODO: documentation:
"""
function fit_ace(params::Dict)

    # ENH: don't generate redundant stuff if db is read from disk. 
    # ENH: make species non-mandatory and read from data
    data =  ACE1pack.read_data(params["data"])

    basis = [ACE1pack.generate_basis(basis_params) for (basis_name, basis_params) in params["basis"]]
    basis = JuLIP.MLIPs.IPSuperBasis(basis);

    if params["fit_from_LSQ_DB"]
        db = LsqDB(params["LSQ_DB_fname_stem"])
    else
        db = LsqDB(params["LSQ_DB_fname_stem"], basis, data)
    end


    solver = ACE1pack.generate_solver(params["solver"])

    if !isnothing(params["P"]) 
        solver["P"] = ACE1pack.generate_precon(basis, params["P"])
    end

    if typeof(params["e0"]) == Dict{Any, Any}
        # sometimes gets read in (from yaml?) as Dict{Any, Any} 
        # which gives StackOverflowError somewhere in OneBody
        params["e0"] = convert(Dict{String, Any}, params["e0"])
    end

    Vref = OneBody(params["e0"])

    weights = params["weights"]

    IP, lsqinfo = lsqfit(db, solver=solver, Vref=Vref, weights=weights, error_table=true)

    @info("Fitting errors")
    rmse_table(lsqinfo["errors"])

    lsqinfo["fit_params"] = params 

    _save_fit(params["ACE_fname"], IP, lsqinfo)

    return IP, lsqinfo
end

function fit_params(;
    data = nothing,
    basis = nothing,
    solver = nothing, 
    e0 = nothing, 
    weights = nothing, 
    P = nothing,
    ACE_fname = "ACE_fit.json", 
    LSQ_DB_fname_stem = "",
    fit_from_LSQ_DB = false)

    # TODO - friendlify
    @assert !isnothing(data)
    @assert !isnothing(basis)
    @assert !isnothing(solver)
    @assert !isnothing(e0)

    return Dict(
            "data" => data,
            "basis" => basis,
            "solver" => solver,
            "e0" => e0,
            "weights" => weights,
            "P" => P,
            "ACE_fname" => ACE_fname, 
            "LSQ_DB_fname_stem" => LSQ_DB_fname_stem,
            "fit_from_LSQ_DB" => fit_from_LSQ_DB)
end


function _save_fit(fname, IP, lsqinfo)
    # ENH: save to yace option
    if fname == ""
        return
    end
    if isfile(fname)
        stem = replace(fname, ".json" => "")
        fnew =  stem * "." * String(rand('a':'z', 5)) * ".json"
        @warn("The file $fname already exists. It will be renamed to $fnew to avoid overwriting.")
        mv(fname, fnew)
    end
    @info("Saving ace fit to $(fname)")
    save_dict(fname, Dict("IP" => write_dict(IP), "info" => lsqinfo))
end

