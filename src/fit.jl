
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

    ACE_basis = ACE1pack.generate_rpi_basis(params["rpi_basis"])
    pair_basis = ACE1pack.generate_pair_basis(params["pair_basis"])
    basis = JuLIP.MLIPs.IPSuperBasis([pair_basis, ACE_basis]);

    if params["fit_from_db"]
        db = LsqDB(params["ACE_fname_stem"] * "_kron.h5")
    else
        db = LsqDB(params["ACE_fname_stem"], basis, data)
    end


    solver = ACE1pack.generate_solver(params["solver"])

    if !isnothing(params["P"]) 
        solver["P"] = ACE1pack.generate_precon(basis, params["P"])
    end

    Vref = OneBody(params["e0"])

    weights = params["weights"]

    IP, lsqinfo = lsqfit(db, solver=solver, Vref=Vref, weights=weights, error_table=true)

    @info("Fitting errors")
    rmse_table(lsqinfo["errors"])

    lsqinfo["fit_dict"] = params 

    _save_fit(params["ACE_fname_stem"], IP, lsqinfo)

    return IP, lsqinfo
end

function fit_params(;
    data = nothing,
    rpi_basis = nothing,
    pair_basis = nothing,
    solver = nothing, 
    e0 = nothing, 
    weights = nothing, 
    P = nothing,
    ACE_fname_stem = "ace_fit", 
    fit_from_db = false)

    # TODO - friendlify
    @assert !isnothing(data)
    @assert !isnothing(rpi_basis)
    @assert !isnothing(pair_basis)
    @assert !isnothing(solver)
    @assert !isnothing(e0)

    return Dict(
            "data" => data,
            "rpi_basis" => rpi_basis,
            "pair_basis" => pair_basis,
            "solver" => solver,
            "e0" => e0,
            "weights" => weights,
            "P" => P,
            "ACE_fname_stem" => ACE_fname_stem, 
            "fit_from_db" => fit_from_db)
end


function _save_fit(stem, IP, lsqinfo)
    # ENH: save to yace option
    if stem == ""
        return
    end
    fname = stem * ".json"
    if isfile(fname)
        fnew =  stem * "." * String(rand('a':'z', 5)) * ".json"
        @warn("The file $fname already exists. It will be renamed to $fnew to avoid overwriting.")
        mv(fname, fnew)
        fname = fnew
    end
    @info("Saving ace fit to fname")
    save_dict(fname, Dict("IP" => write_dict(IP), "info" => lsqinfo))
end

