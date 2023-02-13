
# ------------------------------------------
#    ACE Fitting

import ACEfit, ACE1pack, ACE1, ExtXYZ
using Dates, Base 

export fit_params, fit_ace, make_ace_db, db_params, fit_ace_db, save_fit

"""
`fit_ace(params::Dict) -> IP, lsqinfo` 

Function to set up and fit the least-squares 
problem of "atoms' positions" -> "energy, forces, virials". Takes in a 
dictionary with all the parameters. See `?fit_params` for details. 
"""
function fit_ace(params::Dict, mode=:serial)

    basis = [ACE1pack.generate_basis(basis_params) for (basis_name, basis_params) in params["basis"]]
    basis = JuLIP.MLIPs.IPSuperBasis(basis);

    Vref = OneBody(Dict(Symbol(k) => v for (k,v) in params["e0"]))
    energy_key = params["data"]["energy_key"]
    force_key = params["data"]["force_key"]
    virial_key = params["data"]["virial_key"]
    weight_key = params["data"]["weight_key"]
    weights = params["weights"]
    dataset = JuLIP.read_extxyz(params["data"]["fname"])

    data = AtomsData[]
    for atoms in dataset
        d = AtomsData(atoms; energy_key=energy_key, force_key=force_key, virial_key=virial_key, weight_key=weight_key, weights=weights, v_ref=Vref)
        push!(data, d)
    end
    assess_dataset(data)

    solver = ACEfit.create_solver(params["solver"])
    if isnothing(params["P"])
        P = nothing
    else
        P = ACE1pack.generate_regularizer(basis, params["P"])
    end
    fit = ACEfit.linear_fit(data, basis, solver, mode, P)
    C = fit["C"]
    IP = JuLIP.MLIPs.combine(basis, C)
    (Vref != nothing) && (IP = JuLIP.MLIPs.SumIP(Vref, IP))

    errors = linear_errors(data, IP)
    results = Dict{String,Any}("IP" => IP, "errors" => errors)

    save_fit(params["ACE_fname"], IP, Dict("errors" => errors, "params" => params))

    if haskey(fit, "committee")
        IP_com = committee_potential(basis, C, fit["committee"])
        (Vref != nothing) && (IP_com = JuLIP.MLIPs.SumIP(Vref, IP_com))
        results["IP_com"] = IP_com
        if !(params["ACE_fname"] == "") && !isnothing(params["ACE_fname"])
            save_fit("committee_"*params["ACE_fname"], IP_com, Dict("errors" => errors, "params" => params))
        end
    end

    return results
end

"""
`fit_params(; kwargs...)` 

Returns a dictionary containing all of the
parameters needed to make an ACE potential. All parameters are passed 
as keyword argumts. 

### Parameters
* `data` : data parameters, see `?data_params` for details (mandatory)
* `basis` : dictionary containing dictionaries that specify the basis used in fitting.  
    For example 

```julia
basis = Dict(
    "pair_short" => Dict( "type" => "pair", ...), 
    "pair_long" => Dict("type" => "pair", ...), 
    "manybody" => Dict("type" => "ace", ...), 
    "nospecies" => Dict("type" => "ace", species = ["X",], ...)
```

keys of `basis` are ignored, so that multiple basis with different specifications 
(e.g. smaller and larger cutoffs) can be combined. See `?basis_params` for more detail.  
* `solver` : dictionary containing parameters that specify the solver for 
    least squares problem (mandatory). See `?solver_params`.
* `e0` : `Dict{String, Float}` containing reference values for isolated atoms'
    energies (mandatory). 
* `weights` : dictionary of `Dict("config_type" => Dict("E" => Float, "F => Float))``
    entries specifying fitting weights. "default" is set to `1.0` for all of "E", "F",
    and "V" weights. 
* `P` : regularizer parameters (optional), see `?regularizer_params`.
* `ACE_fname = "ACE_fit.json"` : filename to save ACE to. Potential & info
    do not get saved if `ACE_fname` isnothing() or is set to `""`. Files already _parse_entry
    are renamed and not overwritten. 
* `LSQ_DB_fname_stem = ""` : stem to save LsqDB to. Doesn't get saved if set to an empty 
    string (""). If the file is already present, but `fit_from_LSQ_DB` is set to false,
    the old database is renamed, a new one constructed and saved under the given name. 
* `fit_from_LSQ_DB = false`: whether to fit from a least squares database specified with
    `LSQ_DB_fname_stem`. If `LSQ_DB_fname_stem * "_kron.h5"` file is not present, LsqDB is 
    constructed from scratch and saved.  
"""
function fit_params(;
    data = nothing,
    basis = nothing,
    solver = nothing, 
    e0 = nothing, 
    weights = Dict("default"=>Dict("E"=>1.0, "F"=>1.0, "V"=>1.0)),
    P = nothing,
    ACE_fname = "ACE_fit.json", 
    LSQ_DB_fname_stem = "",
    fit_from_LSQ_DB = false)

    @assert !isnothing(data) "`data` is mandatory"
    @assert !isnothing(basis) "`basis` is mandatory"
    @assert !isnothing(solver) "`solver` is mandatory"
    @assert !isnothing(e0) "`e0` is mandatory"

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


"""
`save_fit(fname, IP, lsqinfo)` 

Saves Dict("IP" => IP, "info" => lsqinfo) to fname.
If `fname` is already present, it is renamed and dictionary saved to `fname`. 
"""
function save_fit(fname, IP, lsqinfo)
    # ENH: save to yace option
    if fname == "" || isnothing(fname)
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

#"""
#`fit_ace_db(params::Dict)` : fits LsqDB with `params["LSQ_DB_fname_stem"], which must be already present. 
#`params["fit_from_LSQ_DB"]` must be set to true. See `?fit_params` for `params` specification, 
#of which `data` and `basis` aren't needed (are ignored).
#"""
#function fit_ace_db(params::Dict)
#    @assert params["fit_from_LSQ_DB"]
#    db = LsqDB(params["LSQ_DB_fname_stem"])
#    IP, lsqinfo = fit_ace_db(db, params)
#    return IP, lsqinfo    
#end

#"""
#`fit_ace_db(db::IPFitting.LsqDB, params::Dict)` : fits the given LsqDB. See `?fit_params` for 
#`params` specification, of which `data` and `basis` aren't needed (are ignored).
#"""
#function fit_ace_db(db::IPFitting.LsqDB, params::Dict)
#    solver = ACE1pack.generate_solver(params["solver"])
#
#    if !isnothing(params["P"]) 
#        solver["P"] = ACE1pack.generate_regularizer(db.basis, params["P"])
#    end
#
#    if typeof(params["e0"]) == Dict{Any, Any}
#        # sometimes gets read in (from yaml?) as Dict{Any, Any} 
#        # which gives StackOverflowError somewhere in OneBody
#        params["e0"] = convert(Dict{String, Any}, params["e0"])
#    end
#
#    Vref = OneBody(params["e0"])
#    weights = params["weights"]
#
#    vals, solve_time, bytes, _, _ = @timed IPFitting.Lsq.lsqfit(
#        db,
#        Vref=Vref,  
#        solver=solver,
#        weights=weights,
#        error_table=true
#    );
#
#    IP = vals[1]
#    lsqinfo = vals[2]
#    solve_time = canonicalize(Dates.CompoundPeriod(Second(trunc(Int, solve_time))))
#    bytes = Base.format_bytes(bytes)
#    lsqinfo["solve_time"] = solve_time
#    lsqinfo["solve_mem"] = bytes
#    lsqinfo["fit_params"] = params
#
#    save_fit(params["ACE_fname"], IP, lsqinfo)
#
#    @info("Fitting errors")
#    rmse_table(lsqinfo["errors"])
#    @info("LsqDB solve time: $(solve_time)")
#    @info("LsqDB memory: $(bytes)")
#
#    return IP, lsqinfo    
#end

"""
`make_ace_db(params::Dict)` -> LsqDB

Makes a LsqDB from given parameters' dictionary. For `params` see 
`?db_params`; parameters from `fit_params` also work, except 
unnecessary entries will be ignored. Returns `IPFitting.LsqDB`
"""
function make_ace_db(params::Dict)
    data =  ACE1pack.read_data(params["data"])
    basis = [ACE1pack.generate_basis(basis_params) for (basis_name, basis_params) in params["basis"]]
    basis = JuLIP.MLIPs.IPSuperBasis(basis);
    db = LsqDB(params["LSQ_DB_fname_stem"], basis, data)
    return db
end

"""
db_params(; kwargs...)` : returns a dictionary containing all of the
parameters needed for making a LsqDB. All parameters are passed 
as keyword argumts. 

### Parameters
* `data` : data parameters, see `?data_params` for details (mandatory)
* `basis` : dictionary containing dictionaries that specify the basis used in fitting.  
    For example 

    ```julia
    basis = Dict(
        "pair_short" => Dict( "type" => "pair", ...), 
        "pair_long" => Dict("type" => "pair", ...), 
        "manybody" => Dict("type" => "ace", ...), 
        "nospecies" => Dict("type" => "ace", species = ["X",], ...)
    ```

keys of `basis` are ignored, so that multiple basis with different specifications 
(e.g. smaller and larger cutoffs) can be combined. See `?basis_params` for more detail.  
* `LSQ_DB_fname_stem = ""` : stem to save LsqDB to. Doesn't get saved if set to an empty 
    string (""). If `LSQ_DB_fname_stem * "_kron.h5"` file is not present it gets renamed, 
    a new LsqDB is constructed and saved.  
"""
function db_params(;
    data = nothing,
    basis = nothing,
    LSQ_DB_fname_stem = "")

    @assert !isnothing(data) "`data` is mandatory"
    @assert !isnothing(basis) "`basis` is mandatory"

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
