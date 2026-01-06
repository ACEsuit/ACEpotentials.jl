#!/usr/bin/env julia
"""
Minimal C interface for ACE models using exported modules.

This interface loads exported ACE models and provides C-callable functions
for energy, force, and basis evaluation.

Design:
- Loads exported minimal modules (created by export_ace_model_minimal)
- Provides C-compatible API for LAMMPS, Python, etc.
- Zero code generation - calls actual ACEpotentials code
- Thread-safe model storage with unique IDs

API Functions:
- ace_load_model: Load an exported model from directory
- ace_site_energy: Compute site energy
- ace_site_energy_forces: Compute site energy and forces
- ace_site_basis: Compute ACE basis vector
- ace_get_cutoff: Get model cutoff radius
- ace_get_species: Get list of atomic species
- ace_unload_model: Unload a model
"""

module ACE_C_Interface_Minimal

using ACEpotentials
using StaticArrays
using Serialization

# Thread-safe model storage
const LOADED_MODELS = Dict{Int32, Module}()
const LOADED_PATHS = Dict{Int32, String}()
const MODEL_LOCK = ReentrantLock()
const NEXT_MODEL_ID = Ref{Int32}(1)

"""
    generate_model_id() -> Int32

Generate unique model ID.
"""
function generate_model_id()::Int32
    lock(MODEL_LOCK) do
        id = NEXT_MODEL_ID[]
        NEXT_MODEL_ID[] += 1
        return id
    end
end

"""
    ace_load_model(model_path::Cstring) -> Cint

Load an exported ACE model from directory.

Parameters:
    model_path: C string pointer to model directory path

Returns:
    model_id: Positive integer on success, -1 on failure

Example:
    model_id = ace_load_model(pointer("path/to/model"))
"""
function ace_load_model(model_path::Cstring)::Cint
    try
        path = unsafe_string(model_path)

        # Check if model directory exists
        if !isdir(path)
            @error "Model directory not found: $path"
            return Cint(-1)
        end

        # Find the wrapper file (look for ace_model.jl or similar)
        wrapper_files = filter(f -> endswith(f, ".jl") && !endswith(f, "_data.jl"),
                              readdir(path))

        if isempty(wrapper_files)
            @error "No Julia wrapper file found in $path"
            return Cint(-1)
        end

        wrapper_file = joinpath(path, wrapper_files[1])

        if !isfile(wrapper_file)
            @error "Model file not found: $wrapper_file"
            return Cint(-1)
        end

        # Load the exported module into Main
        @info "Loading ACE model from $wrapper_file..."
        Base.include(Main, wrapper_file)

        # Get the module (need to determine module name from file)
        module_name = split(basename(wrapper_file), ".")[1]
        module_name_symbol = Symbol(uppercasefirst(module_name))

        if !isdefined(Main, module_name_symbol)
            @error "Module $module_name_symbol not defined after loading $wrapper_file"
            @error "Available modules in Main: $(names(Main))"
            return Cint(-1)
        end

        model_module = getfield(Main, module_name_symbol)

        # Verify required functions exist
        required_funcs = [:site_energy, :site_energy_forces, :site_basis]
        for func in required_funcs
            if !isdefined(model_module, func)
                @error "Module $module_name_symbol missing required function: $func"
                return Cint(-1)
            end
        end

        # Generate unique model ID and store
        model_id = generate_model_id()
        lock(MODEL_LOCK) do
            LOADED_MODELS[model_id] = model_module
            LOADED_PATHS[model_id] = path
        end

        @info "✓ Loaded model: $path (ID: $model_id)"
        @info "  Species: $(model_module.I2Z)"
        @info "  Cutoff: $(model_module.RCUT) Å"

        return model_id

    catch e
        @error "Failed to load model" exception=(e, catch_backtrace())
        return Cint(-1)
    end
end

"""
    ace_unload_model(model_id::Cint) -> Cint

Unload a previously loaded model.

Returns: 0 on success, -1 on failure
"""
function ace_unload_model(model_id::Cint)::Cint
    try
        lock(MODEL_LOCK) do
            if !haskey(LOADED_MODELS, model_id)
                @warn "Attempt to unload non-existent model: $model_id"
                return Cint(-1)
            end

            delete!(LOADED_MODELS, model_id)
            delete!(LOADED_PATHS, model_id)
            @info "Unloaded model ID: $model_id"
            return Cint(0)
        end
    catch e
        @error "Failed to unload model" exception=(e, catch_backtrace())
        return Cint(-1)
    end
end

"""
    ace_get_cutoff(model_id::Cint, cutoff_ptr::Ptr{Float64}) -> Cint

Get the cutoff radius for a loaded model.

Returns: 0 on success, -1 on failure
"""
function ace_get_cutoff(model_id::Cint, cutoff_ptr::Ptr{Float64})::Cint
    try
        model_module = lock(MODEL_LOCK) do
            get(LOADED_MODELS, model_id, nothing)
        end

        if model_module === nothing
            @error "Model not loaded: $model_id"
            return Cint(-1)
        end

        unsafe_store!(cutoff_ptr, Float64(model_module.RCUT))
        return Cint(0)

    catch e
        @error "Error getting cutoff" exception=(e, catch_backtrace())
        return Cint(-1)
    end
end

"""
    ace_get_species(model_id::Cint, species_ptr::Ptr{Cint}, n_species_ptr::Ptr{Cint}) -> Cint

Get the list of atomic species for a loaded model.

Parameters:
    model_id: Model ID
    species_ptr: Pointer to array to store species (atomic numbers)
    n_species_ptr: Pointer to store number of species

Returns: 0 on success, -1 on failure
"""
function ace_get_species(model_id::Cint, species_ptr::Ptr{Cint}, n_species_ptr::Ptr{Cint})::Cint
    try
        model_module = lock(MODEL_LOCK) do
            get(LOADED_MODELS, model_id, nothing)
        end

        if model_module === nothing
            @error "Model not loaded: $model_id"
            return Cint(-1)
        end

        species = model_module.I2Z
        unsafe_store!(n_species_ptr, Cint(length(species)))

        for i in 1:length(species)
            unsafe_store!(species_ptr, Cint(species[i]), i)
        end

        return Cint(0)

    catch e
        @error "Error getting species" exception=(e, catch_backtrace())
        return Cint(-1)
    end
end

"""
    ace_site_energy(model_id::Cint, n_neigh::Cint, Rs_ptr::Ptr{Float64},
                    Zs_ptr::Ptr{Cint}, Z0::Cint) -> Float64

Compute site energy using minimal export.

Parameters:
    model_id: Model ID from ace_load_model
    n_neigh: Number of neighbors
    Rs_ptr: Pointer to neighbor positions (flat array: x1,y1,z1,x2,y2,z2,...)
    Zs_ptr: Pointer to neighbor atomic numbers
    Z0: Central atom atomic number

Returns:
    Site energy in eV, or NaN on error
"""
function ace_site_energy(
    model_id::Cint,
    n_neigh::Cint,
    Rs_ptr::Ptr{Float64},
    Zs_ptr::Ptr{Cint},
    Z0::Cint
)::Float64
    try
        # Get model module
        model_module = lock(MODEL_LOCK) do
            get(LOADED_MODELS, model_id, nothing)
        end

        if model_module === nothing
            @error "Model not loaded: $model_id"
            return NaN
        end

        # Convert C arrays to Julia types
        Rs = Vector{SVector{3, Float64}}(undef, n_neigh)
        Zs = Vector{Int}(undef, n_neigh)

        for i in 1:n_neigh
            Rs[i] = SVector{3, Float64}(
                unsafe_load(Rs_ptr, 3*(i-1) + 1),
                unsafe_load(Rs_ptr, 3*(i-1) + 2),
                unsafe_load(Rs_ptr, 3*(i-1) + 3)
            )
            Zs[i] = Int(unsafe_load(Zs_ptr, i))
        end

        # Call exported function
        E = model_module.site_energy(Rs, Zs, Int(Z0))

        return Float64(E)

    catch e
        @error "Error computing site energy" exception=(e, catch_backtrace())
        return NaN
    end
end

"""
    ace_site_energy_forces(model_id::Cint, n_neigh::Cint, Rs_ptr::Ptr{Float64},
                           Zs_ptr::Ptr{Cint}, Z0::Cint, energy_ptr::Ptr{Float64},
                           forces_ptr::Ptr{Float64}) -> Cint

Compute site energy and forces.

Parameters:
    model_id: Model ID
    n_neigh: Number of neighbors
    Rs_ptr: Pointer to neighbor positions (flat array)
    Zs_ptr: Pointer to neighbor atomic numbers
    Z0: Central atom atomic number
    energy_ptr: Pointer to store energy
    forces_ptr: Pointer to store forces (flat array: fx1,fy1,fz1,...)

Returns: 0 on success, -1 on failure
"""
function ace_site_energy_forces(
    model_id::Cint,
    n_neigh::Cint,
    Rs_ptr::Ptr{Float64},
    Zs_ptr::Ptr{Cint},
    Z0::Cint,
    energy_ptr::Ptr{Float64},
    forces_ptr::Ptr{Float64}
)::Cint
    try
        # Get model module
        model_module = lock(MODEL_LOCK) do
            get(LOADED_MODELS, model_id, nothing)
        end

        if model_module === nothing
            @error "Model not loaded: $model_id"
            return Cint(-1)
        end

        # Convert C arrays to Julia types
        Rs = Vector{SVector{3, Float64}}(undef, n_neigh)
        Zs = Vector{Int}(undef, n_neigh)

        for i in 1:n_neigh
            Rs[i] = SVector{3, Float64}(
                unsafe_load(Rs_ptr, 3*(i-1) + 1),
                unsafe_load(Rs_ptr, 3*(i-1) + 2),
                unsafe_load(Rs_ptr, 3*(i-1) + 3)
            )
            Zs[i] = Int(unsafe_load(Zs_ptr, i))
        end

        # Call exported function
        E, F = model_module.site_energy_forces(Rs, Zs, Int(Z0))

        # Write results to C arrays
        unsafe_store!(energy_ptr, Float64(E))
        for i in 1:n_neigh
            unsafe_store!(forces_ptr, Float64(F[i][1]), 3*(i-1) + 1)
            unsafe_store!(forces_ptr, Float64(F[i][2]), 3*(i-1) + 2)
            unsafe_store!(forces_ptr, Float64(F[i][3]), 3*(i-1) + 3)
        end

        return Cint(0)

    catch e
        @error "Error computing site energy and forces" exception=(e, catch_backtrace())
        return Cint(-1)
    end
end

"""
    ace_site_basis(model_id::Cint, n_neigh::Cint, Rs_ptr::Ptr{Float64},
                   Zs_ptr::Ptr{Cint}, Z0::Cint, basis_ptr::Ptr{Float64},
                   n_basis_ptr::Ptr{Cint}) -> Cint

Compute ACE basis vector.

Parameters:
    model_id: Model ID
    n_neigh: Number of neighbors
    Rs_ptr: Pointer to neighbor positions (flat array)
    Zs_ptr: Pointer to neighbor atomic numbers
    Z0: Central atom atomic number
    basis_ptr: Pointer to store basis (must be pre-allocated!)
    n_basis_ptr: Pointer to store basis size

Returns: 0 on success, -1 on failure
"""
function ace_site_basis(
    model_id::Cint,
    n_neigh::Cint,
    Rs_ptr::Ptr{Float64},
    Zs_ptr::Ptr{Cint},
    Z0::Cint,
    basis_ptr::Ptr{Float64},
    n_basis_ptr::Ptr{Cint}
)::Cint
    try
        # Get model module
        model_module = lock(MODEL_LOCK) do
            get(LOADED_MODELS, model_id, nothing)
        end

        if model_module === nothing
            @error "Model not loaded: $model_id"
            return Cint(-1)
        end

        # Convert C arrays to Julia types
        Rs = Vector{SVector{3, Float64}}(undef, n_neigh)
        Zs = Vector{Int}(undef, n_neigh)

        for i in 1:n_neigh
            Rs[i] = SVector{3, Float64}(
                unsafe_load(Rs_ptr, 3*(i-1) + 1),
                unsafe_load(Rs_ptr, 3*(i-1) + 2),
                unsafe_load(Rs_ptr, 3*(i-1) + 3)
            )
            Zs[i] = Int(unsafe_load(Zs_ptr, i))
        end

        # Call exported function
        B = model_module.site_basis(Rs, Zs, Int(Z0))

        # Write results to C arrays
        unsafe_store!(n_basis_ptr, Cint(length(B)))
        for i in 1:length(B)
            unsafe_store!(basis_ptr, Float64(B[i]), i)
        end

        return Cint(0)

    catch e
        @error "Error computing site basis" exception=(e, catch_backtrace())
        return Cint(-1)
    end
end

"""
    ace_get_n_basis(model_id::Cint, n_basis_ptr::Ptr{Cint}) -> Cint

Get the basis size for a loaded model (for pre-allocation).

Returns: 0 on success, -1 on failure
"""
function ace_get_n_basis(model_id::Cint, n_basis_ptr::Ptr{Cint})::Cint
    try
        model_module = lock(MODEL_LOCK) do
            get(LOADED_MODELS, model_id, nothing)
        end

        if model_module === nothing
            @error "Model not loaded: $model_id"
            return Cint(-1)
        end

        # Get basis size from model parameters
        n_basis = size(model_module.PS.readout.W, 2)
        unsafe_store!(n_basis_ptr, Cint(n_basis))

        return Cint(0)

    catch e
        @error "Error getting basis size" exception=(e, catch_backtrace())
        return Cint(-1)
    end
end

# Module initialization
function __init__()
    @info "ACE C Interface (Minimal Export) initialized"
    @info "  Thread-safe model storage enabled"
    @info "  Available functions:"
    @info "    - ace_load_model(path)"
    @info "    - ace_site_energy(model_id, n_neigh, Rs, Zs, Z0)"
    @info "    - ace_site_energy_forces(model_id, n_neigh, Rs, Zs, Z0, E, F)"
    @info "    - ace_site_basis(model_id, n_neigh, Rs, Zs, Z0, B, n_B)"
    @info "    - ace_get_cutoff(model_id, rcut)"
    @info "    - ace_get_species(model_id, species, n_species)"
    @info "    - ace_get_n_basis(model_id, n_basis)"
    @info "    - ace_unload_model(model_id)"
end

end # module
