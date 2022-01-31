
# ------------------------------------------
#    ACE Fitting

import IPFitting, ACE1pack, ACE1

export ace_params, fit_ace 

"""
TODO: documentation:
"""
function fit_ace(params::Dict)

    # ENH: remove need for specifying species, because they can be read from data (any caveats??)
    data =  ACE1pack.read_data(params["data_params"])

    # ENH: think if anything is ever specified for both rpi_basis and pair_basis and maybe have just one "basis_params" dict
    # ENH: alternative/default to get r0 from data
    ACE_basis = ACE1pack.generate_rpi_basis(params["rpi_basis_params"])
    pair_basis = ACE1pack.generate_pair_basis(params["pair_basis_params"])
    basis = JuLIP.MLIPs.IPSuperBasis([pair_basis, ACE_basis]);

    #ENH option to read-in LsqDB from file
    db = LsqDB(params["db_filename"], basis, data)

    solver = ACE1pack.generate_solver(params["solver_params"])
    apply_preconditioning!(solver, basis=basis)

    # ENH: altrnative/default to get isolated atom energies from the dataset (like gap)
    Vref = OneBody(params["e0_dict"])

    weights = params["weights"]
    error_table = params["error_table"]

    IP, lsqinfo = lsqfit(db, solver=solver, Vref=Vref, weights=weights, error_table=error_table)

    @info("Fitting errors")
    rmse_table(lsqinfo["errors"])

    lsqinfo["fit_dict"] = params 

    if !isnothing(params["ACE_filename"])
        # ENH: save to yace option
        @info("Saving fit to $(params["ACE_filename"])")
        save_dict(params["ACE_filename"], Dict("IP" => write_dict(IP), "info" => lsqinfo))
    end
    return IP, lsqinfo

end

function ace_params(;
    data_params = nothing,
    rpi_basis_params = nothing,
    pair_basis_params = nothing,
    solver_params = nothing, 
    e0_dict = nothing, 
    weights = nothing, 
    ACE_filename = nothing, 
    db_filename = "", 
    error_table = true, 
    )

    # TODO - friendlify
    @assert !isnothing(data_params)
    @assert !isnothing(rpi_basis_params)
    @assert !isnothing(pair_basis_params)
    @assert !isnothing(solver_params)
    @assert !isnothing(e0_dict)

    return Dict(
            "data_params" => data_params,
            "rpi_basis_params" => rpi_basis_params,
            "pair_basis_params" => pair_basis_params,
            "solver_params" => solver_params,
            "e0_dict" => e0_dict,
            "weights" => weights,
            "ACE_filename" => ACE_filename,
            "db_filename" => db_filename,
            "error_table" => error_table
            )
end




