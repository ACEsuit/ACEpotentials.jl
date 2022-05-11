
# ------------------------------------------
#    ACE Fitting

import ACEfit, ACE1pack, ACE1

export fit_params, fit_ace 

"""
TODO: documentation:
"""
function create_the_dataset(params::Dict)

    # ENH: don't generate redundant stuff if db is read from disk. 
    # ENH: make species non-mandatory and read from data
    data =  ACE1pack.read_data(params["data"])

    basis = [ACE1pack.generate_basis(basis_params) for (basis_name, basis_params) in params["basis"]]
    basis = JuLIP.MLIPs.IPSuperBasis(basis);

###    # wcw remove this old stuff eventually
###    if params["fit_from_LSQ_DB"]
###        db = LsqDB(params["LSQ_DB_fname_stem"])
###    else
###        db = LsqDB(params["LSQ_DB_fname_stem"], basis, data)
###    end
###
###
###    solver = ACE1pack.generate_solver(params["solver"])
###
###    if !isnothing(params["P"]) 
###        solver["P"] = ACE1pack.generate_precon(basis, params["P"])
###    end
###
###    if typeof(params["e0"]) == Dict{Any, Any}
###        # sometimes gets read in (from yaml?) as Dict{Any, Any} 
###        # which gives StackOverflowError somewhere in OneBody
###        params["e0"] = convert(Dict{String, Any}, params["e0"])
###    end
###
###    Vref = OneBody(params["e0"])
###
###    @info("Fitting errors")
###    rmse_table(lsqinfo["errors"])
###
###    lsqinfo["fit_params"] = params 
###
###    _save_fit(params["ACE_fname"], IP, lsqinfo)
###
###    return IP, lsqinfo

    # create one body potential
    if typeof(params["e0"]) == Dict{Any, Any}
        # sometimes gets read in (from yaml?) as Dict{Any, Any} 
        # which gives StackOverflowError somewhere in OneBody
        params["e0"] = convert(Dict{String, Any}, params["e0"])
    end
    Vref = OneBody(params["e0"])

    weights = params["weights"]

    energy_key = params["data"]["energy_key"]
    force_key = params["data"]["force_key"]
    virial_key = params["data"]["virial_key"]

    function create_dataset(julip_dataset)
        data = ACEfit.Dat[]
        for at in julip_dataset
            energy = nothing
            forces = nothing
            virial = nothing
            config_type = "default"
            for key in keys(at.data)
                if lowercase(key)=="config_type"; config_type=at.data[key].data; end
            end
            for key in keys(at.data)
                if lowercase(key) == lowercase(energy_key)
                    w = (config_type in keys(weights)) ? weights[config_type]["E"] : weights["default"]["E"]
                    energy_ref = JuLIP.energy(Vref, at)
                    energy = at.data[key].data - energy_ref
                    energy = ObsPotentialEnergy(energy, w, energy_ref)
                end
                if lowercase(key) == lowercase(force_key)
                    w = (config_type in keys(weights)) ? weights[config_type]["F"] : weights["default"]["F"]
                    forces = ObsForces(at.data[key].data[:], w)
                end
                if lowercase(key) == lowercase(virial_key)
                    w = (config_type in keys(weights)) ? weights[config_type]["V"] : weights["default"]["V"]
                    m = SMatrix{3,3}(at.data[key].data)
                    virial = ObsVirial(m, w)
                end
            end
            obs = Any[energy]
            if !isnothing(forces)
                push!(obs, forces)
            end
            if !isnothing(virial)
                insert!(obs, 1, virial)
            end
            dat = ACEfit.Dat(at, config_type, obs)
            push!(data, dat)
        end
        return data
    end

    julip_dataset = JuLIP.read_extxyz(params["data"]["fname"])

    return create_dataset(julip_dataset)

end

function fit_ace(params::Dict; parallelism="serial")

    basis = [ACE1pack.generate_basis(basis_params) for (basis_name, basis_params) in params["basis"]]
    basis = JuLIP.MLIPs.IPSuperBasis(basis);

    data = create_the_dataset(params)

    if parallelism == "serial"
        return ACEfit.llsq!(
            basis,
            data,
            :serial,
            solver=ACEfit.LSQR(damp=params["solver"]["lsqr_damp"],
                               atol=params["solver"]["lsqr_atol"],
                               conlim=params["solver"]["lsqr_conlim"],
                               maxiter=params["solver"]["lsqr_maxiter"],
                               verbose=params["solver"]["lsqr_verbose"])
            )
    elseif parallelism == "distributed"
        return ACEfit.llsq!(basis, data, :dist, solver=ACEfit.LSQR(atol=1e-12))
    else
        println("bad parallelism input")
    end

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

