using TimerOutputs, ExtXYZ
@everywhere using LinearAlgebra, DistributedArrays, JuLIP, ACEfit, ACE1pack
@everywhere import Base.convert

function fit_distributed(params::Dict)

    to = TimerOutput()

    print("Counting observations ... "); flush(stdout)
    Nobs, obs_per_config, frames = ACEfit.count_observations_new(
        params["data"]["fname"],
        params["data"]["energy_key"],
        params["data"]["force_key"],
        params["data"]["virial_key"])
    @everywhere Nobs = $Nobs
    print("the design matrix will have ", Nobs, " rows.\n"); flush(stdout)

    print("Creating basis ... "); flush(stdout)
    @everywhere workers() params = $params
    @everywhere basis = [ACE1pack.generate_basis(basis_params) for (basis_name, basis_params) in params["basis"]]
    @everywhere basis = JuLIP.MLIPs.IPSuperBasis(basis);
    print("the design matrix will have ", length(basis), " columns.\n"); flush(stdout)

    print("Attempting to allocate arrays ... "); flush(stdout)
    @mpi_do manager A = Elemental.DistMatrix(Float64)
    @mpi_do manager Y = Elemental.DistMatrix(Float64)
    @mpi_do manager W = Elemental.DistMatrix(Float64)
    @mpi_do manager Elemental.zeros!(A, Nobs, length(basis))
    @mpi_do manager Elemental.zeros!(Y, Nobs, 1)
    @mpi_do manager Elemental.zeros!(A, Nobs, 1)
    print("successful.\n"); flush(stdout)

    print("Assembling system with matrix size (", Nobs, ", ", length(basis), ") ... "); flush(stdout)
    @everywhere v_ref = OneBody(convert(Dict{String,Any},params["e0"]))
    @everywhere energy_key = params["data"]["energy_key"]
    @everywhere force_key = params["data"]["force_key"]
    @everywhere virial_key = params["data"]["virial_key"]
    @everywhere weights = params["weights"]
    @everywhere first_row = 1
    @everywhere f = x -> ACEfit.assemble_pmap!(
                            A, Y, W,
                            first_row, x,
                            v_ref, energy_key, force_key, virial_key, weights)
    frames = ExtXYZ.read_frames(params["data"]["fname"]) 
    @timeit to "assembling system" begin
        pmap(f, frames)
    end
    print("successful.\n"); flush(stdout)
    
    print("Solving the Elemental system ... "); flush(stdout)
    @timeit to "solve elemental system" begin
        #@mpi_do manager EX = Elemental.ridge(EA, EB, 0.01)
        @mpi_do manager EX = Elemental.leastSquares(EA, EB)
    end    
    print("completed.\n"); flush(stdout)

    print("Converting solution to an ordinary array ... "); flush(stdout)
    @timeit to "convert to regular array" begin
        @everywhere X = Array{Float64}(undef,Elemental.height(EX),Elemental.width(EX))
        @everywhere Elemental.copyto!(X, EX)
    end
    print("completed.\n"); flush(stdout)

    print("Computing errors ... "); flush(stdout)
    @timeit to "Computing errors" begin
        config_errors = ACEfit.error_llsq_new(params, convert(Array,A*distribute(X))./convert(Array,W), convert(Array,Y)./convert(Array,W))
    end
    print("completed.\n"); flush(stdout)

    println(to)

    return X, config_errors
end
