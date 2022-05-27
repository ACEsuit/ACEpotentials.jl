using TimerOutputs
@everywhere using LinearAlgebra, DistributedArrays, JuLIP, ACEfit, ACE1pack
@everywhere import Base.convert

function fit_distributed(params::Dict)

    to = TimerOutput()

    print("Counting observations ... "); flush(stdout)
    Nobs = ACEfit.count_observations(
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
    Y = dzeros((Nobs,1), workers(), (nworkers(),1))
    W = dzeros((Nobs,1), workers(), (nworkers(),1))
    @everywhere A = $A
    @everywhere Y = $Y
    @everywhere W = $W
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

    print("Converting to an Elemental system ... "); flush(stdout)
    @timeit to "convert to elemental system" begin
        @everywhere function convert(::Type{Elemental.DistMatrix{Float64}}, DA::DistributedArrays.DArray{Float64})
            A = Elemental.DistMatrix(Float64)
            Elemental.zeros!(A, size(DA)...)
            function localcopyto!(A, DA)
                lA = localpart(DA)
                li = first(localindices(DA)[1])
                lj = first(localindices(DA)[2])
                Elemental.reserve(A, length(lA))
                for j = 1:size(lA,2), i = 1:size(lA,1)
                    Elemental.queueUpdate(A, li+i-1, lj+j-1, lA[i,j])
                end
                Elemental.processQueues(A)
            end
            localcopyto!(A, DA)
            return A
        end
        @mpi_do manager EA = convert(Elemental.DistMatrix{Float64}, A)
        @mpi_do manager EB = convert(Elemental.DistMatrix{Float64}, Y)
    end
    print("completed.\n"); flush(stdout)
    
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
