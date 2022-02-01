
# ------------------------------------------
#    ACE Fitting

import IPFitting, ACE1pack, ACE1

export ace_params, fit_ace 

"""
TODO: documentation:
"""
function fit_ace(params::Dict)

    # ENH: make species non-mandatory and read from data
    data =  ACE1pack.read_data(params["data"])

    ACE_basis = ACE1pack.generate_rpi_basis(params["rpi_basis"])
    pair_basis = ACE1pack.generate_pair_basis(params["pair_basis"])
    basis = JuLIP.MLIPs.IPSuperBasis([pair_basis, ACE_basis]);

    #ENH option to read-in LsqDB from file
    db = LsqDB(params["db_filename"], basis, data)

    solver = ACE1pack.generate_solver(params["solver"])
    apply_preconditioning!(solver, basis=basis)

    # ENH: altrnative/default to get isolated atom energies from the dataset (like gap)
    Vref = OneBody(params["e0"])

    weights = params["weights"]

    IP, lsqinfo = lsqfit(db, solver=solver, Vref=Vref, weights=weights, error_table=true)

    @info("Fitting errors")
    rmse_table(lsqinfo["errors"])

    lsqinfo["fit_dict"] = params 

    if !isnothing(params["ACE_fname_stem"])
        # ENH: save to yace option
        @info("Saving fit to $(params["ACE_fname_stem"] * ".json")")
        save_dict(params["ACE_fname_stem"] * ".json", Dict("IP" => write_dict(IP), "info" => lsqinfo))
    end
    return IP, lsqinfo

end

function ace_params(;
    data = nothing,
    rpi_basis = nothing,
    pair_basis = nothing,
    solver = nothing, 
    e0 = nothing, 
    weights = nothing, 
    ACE_fname_stem = nothing, 
    db_filename = "", 
    )

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
            "ACE_fname_stem" => ACE_fname_stem,
            "db_filename" => db_filename,
            )
end




