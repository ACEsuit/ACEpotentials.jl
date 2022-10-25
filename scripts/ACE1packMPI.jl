using Distributed
using MPIClusterManagers
using ProgressMeter

manager = MPIClusterManagers.start_main_loop(MPI_TRANSPORT_ALL)

@everywhere using ACE1pack
@everywhere using ACEfit
@everywhere using Suppressor
@mpi_do manager @suppress using Elemental

function ace_fit_mpi(params::Dict)

    energy_key = params["data"]["energy_key"]
    force_key = params["data"]["force_key"]
    virial_key = params["data"]["virial_key"]
    weights = params["weights"]
    v_ref = OneBody(convert(Dict{String,Any},params["e0"]))

    data = ACE1pack.AtomsData[]
    for atoms in JuLIP.read_extxyz(params["data"]["fname"])
        d = ACE1pack.AtomsData(atoms, energy_key, force_key, virial_key, weights, v_ref)
        push!(data, d)
    end

    basis = [ACE1pack.generate_basis(basis_params) for (basis_name, basis_params) in params["basis"]]
    basis = JuLIP.MLIPs.IPSuperBasis(basis);

    solver = ACEfit.create_solver(params["solver"])

    @info "Assembling linear problem."
    row_start, row_count = ACEfit.row_info(data)

    @info "  - Creating feature matrix with size ($(sum(row_count)), $(length(basis)))."
    @everywhere n_rows = sum($row_count)
    @everywhere n_cols = length($basis)
    @mpi_do manager A = Elemental.DistMatrix(Float64)
    @mpi_do manager Y = Elemental.DistMatrix(Float64)
    @mpi_do manager W = Elemental.DistMatrix(Float64)
    @mpi_do manager Elemental.zeros!(A, n_rows, n_cols)
    @mpi_do manager Elemental.zeros!(Y, n_rows, 1)
    @mpi_do manager Elemental.zeros!(W, n_rows, 1)

    @everywhere basis = $basis
    @everywhere function linear_fill(dat, row_start=1)
        i1 = row_start
        i2 = row_start + ACEfit.count_observations(dat) - 1
        a = ACEfit.feature_matrix(dat, basis)
        y = ACEfit.target_vector(dat)
        w = ACEfit.weight_vector(dat)

        Elemental.reserve(A, length(a))
        for j = 1:size(a,2), i = 1:size(a,1)
            Elemental.queueUpdate(A, row_start+i-1, j, w[i]*a[i,j])
        end
        Elemental.reserve(Y, length(y))
        for i = 1:length(y)
            Elemental.queueUpdate(Y, row_start+i-1, 1, w[i]*y[i])
        end
        Elemental.reserve(W, length(w))
        for i = 1:length(w)
            Elemental.queueUpdate(W, row_start+i-1, 1, w[i])
        end

        return nothing
    end
   
    @info "  - Beginning assembly in MPI mode with $(nworkers()) workers."
    @everywhere f = x -> linear_fill(x[1], x[2])
    @showprogress pmap(f, zip(data, row_start))
    @info "  - Processing queues to scatter data."
    @mpi_do manager Elemental.processQueues(A)
    @mpi_do manager Elemental.processQueues(Y)
    @mpi_do manager Elemental.processQueues(W)
    @info "  - Assembly completed."

    @info "Solving the linear problem."
    @info "  - Finding least squares solution."
    @mpi_do manager EC = Elemental.leastSquares(A, Y)
    @info "  - Converting result to local array."
    @everywhere C = Array{Float64}(undef,Elemental.height(EC),Elemental.width(EC))
    @everywhere Elemental.copyto!(C, EC)
    @info "  - Creating the final potential."
    IP = JuLIP.MLIPs.combine(basis, vec(C))
    (v_ref != nothing) && (IP = JuLIP.MLIPs.SumIP(v_ref, IP))

    @info "Computing fit error."
    errors = ACE1pack.linear_errors(data, IP)
    return IP, errors
end

function ACE1packMPI_finalize()
    @mpi_do manager Elemental.Finalize()
    MPIClusterManagers.stop_main_loop(manager)
end
