
# ------------------------------------------
#    ACE Fitting

import IPFitting, ACE1pack, ACE1
using Dates, Base 

export fit_params, fit_ace, make_ace_db, db_params, fit_ace_db, save_fit

"""
TODO: documentation:
"""
function fit_ace(params::Dict)

    db = _decide_how_to_get_db(params)
    IP, lsqinfo = fit_ace_db(db, params)

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


function save_fit(fname, IP, lsqinfo)
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


function fit_ace_db(params::Dict)
    @assert params["fit_from_LSQ_DB"]
    db = LsqDB(params["LSQ_DB_fname_stem"])
    IP, lsqinfo = fit_ace_db(db, params)
    return IP, lsqinfo    
end


function fit_ace_db(db::IPFitting.LsqDB, params::Dict)
    solver = ACE1pack.generate_solver(params["solver"])

    if !isnothing(params["P"]) 
        solver["P"] = ACE1pack.generate_precon(db.basis, params["P"])
    end

    if typeof(params["e0"]) == Dict{Any, Any}
        # sometimes gets read in (from yaml?) as Dict{Any, Any} 
        # which gives StackOverflowError somewhere in OneBody
        params["e0"] = convert(Dict{String, Any}, params["e0"])
    end

    Vref = OneBody(params["e0"])
    weights = params["weights"]

    vals, solve_time, bytes, _, _ = @timed IPFitting.Lsq.lsqfit(
        db,
        Vref=Vref,  
        solver=solver,
        weights=weights,
        error_table=true
    );

    IP = vals[1]
    lsqinfo = vals[2]
    solve_time = canonicalize(Dates.CompoundPeriod(Second(trunc(Int, solve_time))))
    bytes = Base.format_bytes(bytes)
    lsqinfo["solve_time"] = solve_time
    lsqinfo["solve_mem"] = bytes
    lsqinfo["fit_params"] = params

    save_fit(params["ACE_fname"], IP, lsqinfo)

    @info("Fitting errors")
    rmse_table(lsqinfo["errors"])
    @info("LsqDB solve time: $(solve_time)")
    @info("LsqDB memory: $(bytes)")

    return IP, lsqinfo    
end


function make_ace_db(params::Dict)
    data =  ACE1pack.read_data(params["data"])
    basis = [ACE1pack.generate_basis(basis_params) for (basis_name, basis_params) in params["basis"]]
    basis = JuLIP.MLIPs.IPSuperBasis(basis);
    db = LsqDB(params["LSQ_DB_fname_stem"], basis, data)
    return db
end

function db_params(;
    data = nothing,
    basis = nothing,
    LSQ_DB_fname_stem = "")

    # TODO - friendlify
    @assert !isnothing(data)
    @assert !isnothing(basis)

    return Dict(
            "data" => data,
            "basis" => basis,
            "LSQ_DB_fname_stem" => LSQ_DB_fname_stem)
end

function _decide_how_to_get_db(params::Dict)
    if params["LSQ_DB_fname_stem"] == ""
        return make_ace_db(params) 
    else
        db_fname = params["LSQ_DB_fname_stem"] * "_kron.h5"
        if ~isfile(db_fname)
            return make_ace_db(params) 
        else
            if params["fit_from_LSQ_DB"]
                return LsqDB(params["LSQ_DB_fname_stem"]) 
            else
                @warn("Found $(db_fname), but wasn't asked to fit it. Making a new one and saving to $db_fname).")
                return make_ace_db(params) 
            end
        end
    end
end
