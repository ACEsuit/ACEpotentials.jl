#!/usr/bin/env julia
#=
ACE potential driver for i-PI socket protocol.

This script loads an ACE potential model and connects as an i-PI driver
to serve energy/force/virial calculations via socket communication.

Usage:
    julia --project=. ace_driver.jl --model path/to/model.json --port 31415
    julia --project=. ace_driver.jl --model path/to/model.json --unixsocket ace_socket

The driver connects to ASE's SocketIOCalculator and responds to calculation
requests using the specified ACE model.
=#

using ArgParse
using ACEpotentials
using IPICalculator
using AtomsBase
using Unitful
using UnitfulAtomic

function parse_commandline()
    s = ArgParseSettings(
        description = "ACE potential i-PI driver for ASE integration"
    )

    @add_arg_table! s begin
        "--model"
            help = "Path to ACE model JSON file"
            required = true
        "--port"
            help = "TCP port to connect to (default: 31415)"
            arg_type = Int
            default = 31415
        "--unixsocket"
            help = "Unix socket name (mutually exclusive with --port)"
            default = nothing
        "--host"
            help = "Host address to connect to (default: localhost)"
            default = "localhost"
        "--verbose", "-v"
            help = "Enable verbose logging"
            action = :store_true
    end

    return parse_args(s)
end

function create_template_system(elements::Vector{Symbol})
    #=
    Create a minimal template AtomsBase system.

    IPICalculator needs a template system to know the atom types.
    The actual positions come from the socket client.
    =#
    n = length(elements)
    positions = [zeros(3)u"Å" for _ in 1:n]
    cell = [10.0u"Å" 0.0u"Å" 0.0u"Å";
            0.0u"Å" 10.0u"Å" 0.0u"Å";
            0.0u"Å" 0.0u"Å" 10.0u"Å"]
    atoms = [AtomsBase.Atom(el, pos) for (el, pos) in zip(elements, positions)]
    return periodic_system(atoms, cell)
end

function main()
    args = parse_commandline()

    if args["verbose"]
        ENV["JULIA_DEBUG"] = "IPICalculator"
    end

    # Load the ACE model
    model_path = args["model"]
    @info "Loading ACE model" path=model_path

    potential, meta = ACEpotentials.load_model(model_path)

    @info "Model loaded successfully" elements=meta.elements cutoff=meta.rcut
    @info "Julia configuration" threads=Threads.nthreads()

    # Create template system with the model's elements
    elements = meta.elements
    template = create_template_system(elements)

    # Connect as i-PI driver
    if args["unixsocket"] !== nothing
        socket_name = args["unixsocket"]
        @info "Connecting to Unix socket" socket=socket_name
        IPICalculator.run_driver(template, potential;
            unixsocket=socket_name,
        )
    else
        host = args["host"]
        port = args["port"]
        @info "Connecting to TCP socket" host=host port=port
        IPICalculator.run_driver(template, potential;
            address=host,
            port=port,
        )
    end
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
