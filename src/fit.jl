
# ------------------------------------------
#    ACE Fitting

import ACEfit, ACE1pack, ACE1, ExtXYZ

export fit_params, fit_ace 


function fit_ace(params::Dict; parallelism="serial")

    basis = [ACE1pack.generate_basis(basis_params) for (basis_name, basis_params) in params["basis"]]
    basis = JuLIP.MLIPs.IPSuperBasis(basis);

    Vref = OneBody(convert(Dict{String,Any},params["e0"]))
    energy_key = params["data"]["energy_key"]
    force_key = params["data"]["force_key"]
    virial_key = params["data"]["virial_key"]
    weights = params["weights"]
    julip_dataset = read_data(params["data"])
    data = ACEfit.Dat[]
    for atoms in julip_dataset
        dat = ACEfit._atoms_to_data(atoms, Vref, weights, energy_key, force_key, virial_key)
        push!(data, dat)
    end

    if parallelism == "serial"
        return ACEfit.llsq!(
            basis,
            data,
            Vref,
            :serial,
            solver=ACEfit.create_solver(params["solver"])
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

