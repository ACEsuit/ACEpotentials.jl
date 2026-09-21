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
            # 127.0.0.1, NOT "localhost".  The Python side is ASE's SocketServer, which
            # is socket.socket(AF_INET) bound to ('', port) -- IPv4 only.  On any host
            # whose resolver returns ::1 before 127.0.0.1 for "localhost" (the default on
            # most modern Linux), Julia's connect("localhost", port) dials IPv6, finds
            # nothing listening, and fails with ECONNREFUSED.  The driver then dies and
            # ASE blocks forever waiting for a client that is already gone.
            help = "Host address to connect to (default: 127.0.0.1)"
            default = "127.0.0.1"
        "--species"
            # The i-PI protocol never transmits chemical species: POSDATA carries the cell
            # and the positions and nothing else.  IPICalculator therefore takes the species
            # (and masses) from the template passed to run_driver and asserts on every
            # POSDATA that the incoming position count matches:
            #     @assert length(atom_species) == length(positions)
            # A template built from the model's element LIST is one atom per element type,
            # so a single-species model gave a 1-atom template and every real system failed
            # that assert -- 2-atom Si diamond included.  The caller knows the composition,
            # so it passes it here.
            help = "Comma-separated chemical symbols of the system, e.g. Si,Si. " *
                   "Defaults to one atom per element in the model, which is only correct " *
                   "for a system with exactly one atom of each element."
            default = nothing
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
    # AtomsBase.periodic_system takes the cell as a vector of lattice VECTORS, not as a
    # 3x3 matrix; passing a Matrix is a MethodError, which is as far as this driver ever
    # got.  See AtomsBase/src/implementation/utils.jl: the accepted forms are
    # NTuple{D,AbstractVector} and AbstractVector{<:AbstractVector}.
    cell = [[10.0, 0.0, 0.0]u"Å",
            [0.0, 10.0, 0.0]u"Å",
            [0.0, 0.0, 10.0]u"Å"]
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

    # `load_model` returns (model, parsed_json) -- the SECOND value is the whole parsed
    # file, not a metadata struct.  This used to be `potential, meta = ...` followed by
    # `meta.elements`, which cannot work: a Dict has no field `elements`, so the driver
    # died with a MethodError before it ever reached the socket.  Nor is `D["meta"]` the
    # place to look: that key holds the `meta` kwarg of `save_model`, which is an empty
    # Dict unless the caller passed one.
    #
    # Species and cutoff come from `model_spec`, and it is always present here: reaching
    # this line means `load_model` already did `make_model(D["model_spec"])` and would
    # have thrown otherwise.  JSON gives the elements back as strings, and
    # `create_template_system` takes a Vector{Symbol}.
    potential, D = ACEpotentials.load_model(model_path)

    spec = D["model_spec"]
    rcut = spec["rcut"]

    # The template must have the SAME atom count and species as the system the client will
    # send -- see --species above.  Falling back to the model's element list preserves the
    # old behaviour for a caller that does not pass one.
    elements = if args["species"] !== nothing && !isempty(args["species"])
        Symbol.(strip.(split(args["species"], ",")))
    else
        Symbol.(spec["elements"])
    end

    @info "Model loaded successfully" elements cutoff=rcut
    @info "Julia configuration" threads=Threads.nthreads()

    # Create template system with the model's elements
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
