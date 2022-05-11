using TimerOutputs

using Distributed
@everywhere using Distributed

using DistributedArrays, JuLIP, ACEfit, ACE1pack, Revise
@everywhere using DistributedArrays, JuLIP, ACEfit, ACE1pack, Revise

function fit_distributed(params::Dict)

    to = TimerOutput()

    #println("preparing data"); flush(stdout)
    #@timeit to "preparing data" begin
    #    @everywhere workers() params = $params
    #    @everywhere data = ACE1pack.create_the_dataset(params)
    #end

    print("Counting observations ... "); flush(stdout)
    Nobs = ACE1pack.count_observations(
        params["data"]["fname"],
        params["data"]["energy_key"],
        params["data"]["force_key"],
        params["data"]["virial_key"])
    print("the design matrix will have ", Nobs, " rows.\n"); flush(stdout)

    print("Creating basis ... "); flush(stdout)
    @everywhere workers() params = $params
    @everywhere basis = [ACE1pack.generate_basis(basis_params) for (basis_name, basis_params) in params["basis"]]
    @everywhere basis = JuLIP.MLIPs.IPSuperBasis(basis);
    print("the design matrix will have ", length(basis), " columns.\n"); flush(stdout)

    print("Attempting to allocate arrays ... "); flush(stdout)
    A = dzeros((Nobs, length(basis)), workers(), [nworkers(),1])
    Y = dzeros(tuple(Nobs), workers(), nworkers())
    W = dzeros(tuple(Nobs), workers(), nworkers())
    print("successful.\n"); flush(stdout)

    print("Assembling system with matrix size (", Nobs, ", ", length(basis), ") ... "); flush(stdout)
    @timeit to "assembling system" begin
        #@everywhere workers() ACEfit.assemble_dist!(A, Y, W, data, basis)
        @sync for worker in workers()
            # TODO: there is some redundancy here, in that basis is sent, but it's already been constructed everywhere
            #       what would be the fastest?
            @spawnat worker ACEfit.assemble_dist_new!(A, Y, W, params, basis)
        end
    end
    print("successful.\n"); flush(stdout)


    println("solving system"); flush(stdout)
    @timeit to "solving system" begin
        coef = ACEfit.solve_llsq(
            ACEfit.LSQR(damp=params["solver"]["lsqr_damp"],
                        atol=params["solver"]["lsqr_atol"],
                        conlim=params["solver"]["lsqr_conlim"],
                        maxiter=params["solver"]["lsqr_maxiter"],
                        verbose=params["solver"]["lsqr_verbose"]),
            A,
            Y)
    end

    @timeit to "computing errors" begin
        config_errors = ACEfit.error_llsq_new(params, convert(Vector,A*coef)./convert(Vector,W), convert(Vector,Y)./convert(Vector,W))
    end

    println(to)

    return coef, config_errors
end
