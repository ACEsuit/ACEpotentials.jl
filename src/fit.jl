
# ------------------------------------------
#    ACE Fitting

import ACEfit, ACE1pack, ACE1, ExtXYZ
using Dates, Base 

export fit_params, fit_ace, make_ace_db, db_params, fit_ace_db, save_fit

"""
`fit_ace(params::Dict)` : function to set up and fit the least-squares 
problem of "atoms' positions" -> "energy, forces, (virials)". Takes in a 
dictionary with all the parameters. see `?fit_params` for details. 

Returns `IP`, `lsqinfo`
"""
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

#function fit_elemental(params::Dict)
#
#    to = TimerOutput()
#
#    print("Counting observations ... "); flush(stdout)
#    Nobs = ACEfit.count_observations(
#        params["data"]["fname"],
#        params["data"]["energy_key"],
#        params["data"]["force_key"],
#        params["data"]["virial_key"])
#    print("the design matrix will have ", Nobs, " rows.\n"); flush(stdout)
#
#    print("Creating basis ... "); flush(stdout)
#    @everywhere workers() params = $params
#    @everywhere basis = [ACE1pack.generate_basis(basis_params) for (basis_name, basis_params) in params["basis"]]
#    @everywhere basis = JuLIP.MLIPs.IPSuperBasis(basis);
#    print("the design matrix will have ", length(basis), " columns.\n"); flush(stdout)
#
#    print("Attempting to allocate arrays ... "); flush(stdout)
#    A = dzeros((Nobs, length(basis)), workers(), [nworkers(),1])
#    Y = dzeros((Nobs,1), workers(), (nworkers(),1))
#    W = dzeros((Nobs,1), workers(), (nworkers(),1))
#    @everywhere A = $A
#    @everywhere Y = $Y
#    @everywhere W = $W
#    print("successful.\n"); flush(stdout)
#
#    print("Assembling system with matrix size (", Nobs, ", ", length(basis), ") ... "); flush(stdout)
#    @timeit to "assembling system" begin
#        #@everywhere workers() ACEfit.assemble_dist!(A, Y, W, data, basis)
#        @sync for worker in workers()
#            # TODO: there is some redundancy here, in that basis is sent, but it's already been constructed everywhere
#            #       what would be the fastest?
#            @spawnat worker ACEfit.assemble_dist_new!(A, Y, W, params, basis)
#        end
#    end
#    print("successful.\n"); flush(stdout)
#
#    print("Converting to an Elemental system ... "); flush(stdout)
#    @timeit to "convert to elemental system" begin
#        @everywhere function convert(::Type{Elemental.DistMatrix{Float64}}, DA::DistributedArrays.DArray{Float64})
#            A = Elemental.DistMatrix(Float64)
#            Elemental.zeros!(A, size(DA)...)
#            function localcopyto!(A, DA)
#                lA = localpart(DA)
#                li = first(localindices(DA)[1])
#                lj = first(localindices(DA)[2])
#                Elemental.reserve(A, length(lA))
#                for j = 1:size(lA,2), i = 1:size(lA,1)
#                    Elemental.queueUpdate(A, li+i-1, lj+j-1, lA[i,j])
#                end
#                Elemental.processQueues(A)
#            end
#            localcopyto!(A, DA)
#            return A
#        end
#        @mpi_do manager EA = convert(Elemental.DistMatrix{Float64}, A)
#        @mpi_do manager EB = convert(Elemental.DistMatrix{Float64}, Y)
#    end
#    print("completed.\n"); flush(stdout)
#    
#    print("Solving the Elemental system ... "); flush(stdout)
#    @timeit to "solve elemental system" begin
#        #@mpi_do manager EX = Elemental.ridge(EA, EB, 0.01)
#        @mpi_do manager EX = Elemental.leastSquares(EA, EB)
#    end    
#    print("completed.\n"); flush(stdout)
#
#    print("Converting solution to an ordinary array ... "); flush(stdout)
#    @timeit to "convert to regular array" begin
#        @everywhere X = Array{Float64}(undef,Elemental.height(EX),Elemental.width(EX))
#        @everywhere Elemental.copyto!(X, EX)
#    end
#    print("completed.\n"); flush(stdout)
#
#    print("Computing errors ... "); flush(stdout)
#    @timeit to "Computing errors" begin
#        config_errors = ACEfit.error_llsq_new(params, convert(Array,A*distribute(X))./convert(Array,W), convert(Array,Y)./convert(Array,W))
#    end
#    print("completed.\n"); flush(stdout)
#
#    println(to)
#
#    return X, config_errors
#end

"""
`fit_params(; kwargs...)` : returns a dictionary containing all of the
parameters needed for making an ACE potential. All parameters are passed 
as keyword argumts. 

### Parameters
* `data` : data parameters, see `?data_params` for details (mandatory)
* `basis` : dictionary containing dictionaries that specify the basis used
in fitting. Usually just `ace_params` and `pair_params` (mandatory). 
For example
```
basis = Dict(
    "ace" => ace_params(; kwargs...),
    "pair" => pair_params(; kwargs...))
````
keys of `basis` must correspond to one of the "type" of `basis_params`, see 
`?basis_params` and values are corresponding parameters' dictionaries.
* `solver` : dictionary containing parameters that specify the solver for 
least squares problem (mandatory). See `?solver_params`.
* `e0` : Dict{String, Float} containing reference values for isolated atoms'
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
    weights = nothing, 
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
`save_fit(fname, IP, lsqinfo)` : saves Dict("IP" => IP, "info" => lsqinfo) to fname.
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
`make_ace_db(params::Dict)` : makes a LsqDB from given parameters' dictionary. 
For `params` see `?db_params`; parameters from `fit_params` also work, except 
unnecessary entries will be ignored. 


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
* `basis` : dictionary containing dictionaries that specify the basis used
in fitting. Usually just `ace_params` and `pair_params` (mandatory). 
For example
```
basis = Dict{
    "ace" => ace_params(; kwargs...),
    "pair" => pair_params(; kwargs...)}
````
keys of `basis` must correspond to one of the "type" of `basis_params`, see 
`?basis_params` and values are corresponding parameters' dictionaries.
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
