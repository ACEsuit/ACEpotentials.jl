
# EquivariantTensors-based calculator for ACE potentials
# This provides an AtomsCalculators-compatible interface using
# the Lux-based ET model with Zygote AD for forces and virial.

import EquivariantTensors as ET
import Polynomials4ML as P4ML

import AtomsBase
import AtomsCalculators
import AtomsCalculators: energy_forces_virial, potential_energy,
                         forces, virial, energy_unit, length_unit

using Unitful: ustrip, @u_str
using StaticArrays
using Zygote
using ForwardDiff
using LinearAlgebra: norm
using Random
import Lux
import LuxCore

# =========================================================================
#  Shared utilities for graph construction and device handling
# =========================================================================

"""
    _build_graph(sys, rcut)

Build an ETGraph from an atomic system with the given cutoff radius.
"""
function _build_graph(sys, rcut)
    G = ET.Atoms.interaction_graph(sys, rcut)
    return G
end

"""
    _prepare_for_device(G, ps, st, device, precision)

Prepare the graph, parameters, and states for the target device.
"""
function _prepare_for_device(G, ps, st, device, precision)
    if precision == :Float32
        G = ET.ETGraph(G.ii, G.jj, G.first,
                       ET.float32.(G.node_data),
                       ET.float32.(G.edge_data),
                       G.maxneigs)
        ps = ET.float32(ps)
        st = ET.float32(st)
    end

    if device !== identity
        G = device(G)
        ps = device(ps)
        st = device(st)
    end

    return G, ps, st
end

"""
    _reconstruct_graph(𝐫_vec, G, node_data, s0_edges, s1_edges, dev; graph_data=nothing)

Reconstruct an ETGraph with new edge positions. Used for differentiating
through graph evaluation.

If graph_data is not provided, uses G.graph_data.
"""
function _reconstruct_graph(𝐫_vec, G, node_data, s0_edges, s1_edges, dev; graph_data=nothing)
    edge_data = [(𝐫 = r, s0 = s0, s1 = s1)
                 for (r, s0, s1) in zip(𝐫_vec, s0_edges, s1_edges)]
    # Use provided graph_data, or use G.graph_data (may be nothing)
    gd = graph_data !== nothing ? graph_data : G.graph_data
    G_new = ET.ETGraph(G.ii, G.jj, G.first, node_data, edge_data, gd, G.maxneigs)
    return dev === identity ? G_new : dev(G_new)
end

"""
    _scatter_forces_virial(ii, jj, ∇𝐫, edge_positions, n_atoms; offset=0)

Scatter edge gradients to atomic forces and compute virial tensor.

# Arguments
- `ii`, `jj`: Edge endpoint indices (global)
- `∇𝐫`: Gradients w.r.t. edge positions (Vector of SVector{3})
- `edge_positions`: Original edge position vectors (for virial)
- `n_atoms`: Number of atoms in the result arrays
- `offset`: Index offset for batched evaluation (default 0)

# Returns
(forces, virial) tuple where forces is Vector{SVector{3,Float64}} and
virial is SMatrix{3,3,Float64}.
"""
function _scatter_forces_virial(ii, jj, ∇𝐫, edge_positions, n_atoms; offset=0)
    T = Float64
    F = zeros(SVector{3, T}, n_atoms)
    virial = @SMatrix zeros(T, 3, 3)

    for (k, (i, j)) in enumerate(zip(ii, jj))
        i_local = i - offset
        j_local = j - offset
        ∂E_∂𝐫 = SVector{3, T}(∇𝐫[k])

        F[i_local] += ∂E_∂𝐫
        F[j_local] -= ∂E_∂𝐫

        𝐫ij = SVector{3, T}(edge_positions[k])
        virial -= ∂E_∂𝐫 * 𝐫ij'
    end

    return F, virial
end

# =========================================================================
#  ETCalculator - Unified Lux-based ACE calculator using EquivariantTensors
# =========================================================================

"""
    ETCalculator

An AtomsCalculators-compatible calculator that uses the EquivariantTensors
Lux model for energy, force, virial, and basis evaluation.

Supports both CPU and GPU evaluation. Forces and virial are computed via
Zygote automatic differentiation through the Lux model. Basis derivatives
use ForwardDiff for efficiency.

# Fields
- `model`: The Lux Chain model (et_model) for energy evaluation
- `basis_model`: The Lux Chain model for basis evaluation (without readout/sum)
- `ps`: Model parameters (NamedTuple)
- `st`: Model states (NamedTuple)
- `basis_ps`: Basis model parameters (NamedTuple)
- `basis_st`: Basis model states (NamedTuple)
- `rcut`: Cutoff radius (with units)
- `zlist`: List of chemical species for basis indexing
- `len_Bi`: Length of basis per species
- `device`: Device function for GPU (e.g., `Metal.mtl`, `CUDA.cu`, or `identity` for CPU)
- `precision`: `:Float64` or `:Float32` for GPU

# Example
```julia
calc = build_et_calculator(ace_model, ps, st)
E = potential_energy(system, calc)
efv = energy_forces_virial(system, calc)
B = evaluate_basis(calc, system)
B, dB = evaluate_basis_ed(calc, system)
```
"""
mutable struct ETCalculator{MOD, BMOD, PS, ST, BPS, BST, ZLIST, DEV}
    model::MOD
    basis_model::BMOD
    ps::PS
    st::ST
    basis_ps::BPS
    basis_st::BST
    rcut::typeof(1.0u"Å")
    zlist::ZLIST
    len_Bi::Int
    device::DEV
    precision::Symbol
end

energy_unit(::ETCalculator) = u"eV"
length_unit(::ETCalculator) = u"Å"

"""
    length_basis(calc::ETCalculator)

Return the total basis length: `len_Bi * NZ` where NZ is the number of species.
Note: This does not include pair basis (not yet implemented in ET backend).
"""
length_basis(calc::ETCalculator) = calc.len_Bi * length(calc.zlist)

"""
    get_basis_inds(calc::ETCalculator, Z)

Get the basis indices for species Z.
"""
function get_basis_inds(calc::ETCalculator, Z)
    i_z = ET.cat2idx(calc.zlist, Z)
    return (i_z - 1) * calc.len_Bi .+ (1:calc.len_Bi)
end

# =========================================================================
#  Core evaluation functions
# =========================================================================

"""
    _energy_from_edge_positions(𝐫_vec, G, s0_edges, s1_edges, model, ps, st, dev)

Compute energy as a function of edge position vectors.
This is the differentiable function for Zygote.
"""
function _energy_from_edge_positions(𝐫_vec, G, node_data, s0_edges, s1_edges,
                                      model, ps, st, dev)
    G_new = _reconstruct_graph(𝐫_vec, G, node_data, s0_edges, s1_edges, dev)
    return model(G_new, ps, st)[1]
end

# =========================================================================
#  AtomsCalculators interface
# =========================================================================

function AtomsCalculators.potential_energy(sys, calc::ETCalculator; kwargs...)
    G = _build_graph(sys, calc.rcut)
    G_dev, ps_dev, st_dev = _prepare_for_device(G, calc.ps, calc.st, calc.device, calc.precision)
    E = calc.model(G_dev, ps_dev, st_dev)[1]
    return Float64(E) * energy_unit(calc)
end

function AtomsCalculators.forces(sys, calc::ETCalculator; kwargs...)
    efv = energy_forces_virial(sys, calc; kwargs...)
    return efv.forces
end

function AtomsCalculators.virial(sys, calc::ETCalculator; kwargs...)
    efv = energy_forces_virial(sys, calc; kwargs...)
    return efv.virial
end

function AtomsCalculators.energy_forces_virial(sys, calc::ETCalculator; kwargs...)
    G = _build_graph(sys, calc.rcut)

    # Extract edge data components
    𝐫_edges = [ed.𝐫 for ed in G.edge_data]
    s0_edges = [ed.s0 for ed in G.edge_data]
    s1_edges = [ed.s1 for ed in G.edge_data]

    # Prepare parameters and states for device
    dev = calc.device
    ps_work = calc.ps
    st_work = calc.st
    node_data = G.node_data

    if calc.precision == :Float32
        𝐫_edges = ET.float32.(𝐫_edges)
        ps_work = ET.float32(ps_work)
        st_work = ET.float32(st_work)
        node_data = ET.float32.(node_data)
    end

    if dev !== identity
        ps_work = dev(ps_work)
        st_work = dev(st_work)
    end

    # Energy function for differentiation
    function _efunc(𝐫_vec)
        _energy_from_edge_positions(𝐫_vec, G, node_data, s0_edges, s1_edges,
                                    calc.model, ps_work, st_work, dev)
    end

    # Compute energy and gradient
    E = _efunc(𝐫_edges)
    ∇𝐫 = Zygote.gradient(_efunc, 𝐫_edges)[1]

    if ∇𝐫 === nothing
        error("Zygote gradient returned nothing - check model differentiability")
    end

    # Convert to CPU if needed
    if dev !== identity
        ∇𝐫 = collect(∇𝐫)
    end

    # Scatter edge gradients to forces and virial
    edge_positions = [ed.𝐫 for ed in G.edge_data]
    F, virial = _scatter_forces_virial(G.ii, G.jj, ∇𝐫, edge_positions, length(sys))

    return (
        energy = Float64(E) * energy_unit(calc),
        forces = F .* (energy_unit(calc) / length_unit(calc)),
        virial = virial * energy_unit(calc)
    )
end

# =========================================================================
#  Basis evaluation functions (integrated into ETCalculator)
# =========================================================================

"""
    evaluate_basis(calc::ETCalculator, sys)

Evaluate the ACE basis for all atoms in the system.

Returns a matrix B of shape (n_atoms, length_basis) where each row contains
the basis values for that atom, indexed by species.

# Arguments
- `calc`: ETCalculator instance
- `sys`: AtomsBase-compatible atomic system

# Returns
Matrix B where B[i, :] contains basis values for atom i.
"""
function evaluate_basis(calc::ETCalculator, sys)
    G = _build_graph(sys, calc.rcut)
    G_dev, ps_dev, st_dev = _prepare_for_device(G, calc.basis_ps, calc.basis_st,
                                                 calc.device, calc.precision)

    # Evaluate basis - returns per-atom basis vectors
    # Shape: (n_atoms, len_Bi)
    Bi_all, _ = calc.basis_model(G_dev, ps_dev, st_dev)

    # Convert to CPU if needed
    if calc.device !== identity
        Bi_all = collect(Bi_all)
    end

    # Assemble into full basis matrix indexed by species
    T = Float64
    n_atoms = length(sys)
    B = zeros(T, n_atoms, length_basis(calc))

    for i in 1:n_atoms
        Z = AtomsBase.atomic_symbol(sys, i)
        Zs = AtomsBase.ChemicalSpecies(Z)
        inds = get_basis_inds(calc, Zs)
        # Bi_all is (n_atoms, len_Bi), so Bi_all[i, :] is basis for atom i
        B[i, inds] .= T.(Bi_all[i, :])
    end

    return B
end

# Helper functions for ForwardDiff (matching ace.jl pattern)
# Convert Vector{SVector{3,T}} to flat Vector{T} and back
__vec_edges(𝐫s::AbstractVector{<:SVector{3}}) = reinterpret(eltype(eltype(𝐫s)), 𝐫s)
__svec_edges(v::AbstractVector{T}) where {T} = reinterpret(SVector{3, T}, v)

"""
    _basis_from_edge_positions(𝐫_vec, G, node_data, s0_edges, s1_edges, model, ps, st, dev)

Compute per-atom basis as a function of edge position vectors.
This is the differentiable function for ForwardDiff jacobian.
"""
function _basis_from_edge_positions(𝐫_vec, G, node_data, s0_edges, s1_edges,
                                     model, ps, st, dev)
    G_new = _reconstruct_graph(𝐫_vec, G, node_data, s0_edges, s1_edges, dev)
    Bi_all, _ = model(G_new, ps, st)
    return Bi_all  # Shape: (n_atoms, len_Bi)
end

"""
    evaluate_basis_ed(calc::ETCalculator, sys)

Evaluate the ACE basis and its gradients with respect to atomic positions.

Returns (B, dB) where:
- B: Matrix of shape (n_atoms, length_basis) with basis values
- dB: Array of shape (n_atoms, length_basis, n_atoms) of SVector{3} gradients

The gradient dB[i, b, j] = ∂B[i,b]/∂X[j] is a 3-vector giving the derivative
with respect to the 3D position of atom j.

Uses ForwardDiff.jacobian for efficient gradient computation. This is optimal
for the "few inputs → many outputs" structure of this problem (edge positions
→ basis values).

# Arguments
- `calc`: ETCalculator instance
- `sys`: AtomsBase-compatible atomic system

# Returns
Tuple (B, dB) with basis values and gradients.
"""
function evaluate_basis_ed(calc::ETCalculator, sys)
    G = _build_graph(sys, calc.rcut)

    # Extract edge data components
    𝐫_edges = [ed.𝐫 for ed in G.edge_data]
    s0_edges = [ed.s0 for ed in G.edge_data]
    s1_edges = [ed.s1 for ed in G.edge_data]

    # Prepare parameters and states (CPU only for ForwardDiff)
    ps_work = calc.basis_ps
    st_work = calc.basis_st
    node_data = G.node_data

    # ForwardDiff requires CPU - warn if GPU was requested
    if calc.device !== identity
        @warn "evaluate_basis_ed uses ForwardDiff (CPU only). GPU device ignored." maxlog=1
    end

    # Basis function taking flat vector of edge positions for ForwardDiff
    function _bfunc_flat(𝐫_flat)
        𝐫_vec = __svec_edges(𝐫_flat)
        Bi_all = _basis_from_edge_positions(𝐫_vec, G, node_data, s0_edges, s1_edges,
                                            calc.basis_model, ps_work, st_work, identity)
        return vec(Bi_all)  # Flatten to 1D for jacobian
    end

    # Flatten edge positions for ForwardDiff
    𝐫_flat = collect(__vec_edges(𝐫_edges))

    # Evaluate basis and Jacobian using ForwardDiff
    Bi_all_flat = _bfunc_flat(𝐫_flat)
    dB_flat = ForwardDiff.jacobian(_bfunc_flat, 𝐫_flat)

    # Reshape results
    len_Bi = calc.len_Bi
    n_atoms = length(sys)
    n_edges = length(𝐫_edges)
    T = Float64

    # dB_flat: (n_atoms * len_Bi) × (3 * n_edges)
    # Reshape to (n_atoms, len_Bi, 3, n_edges) then permute to (n_atoms, len_Bi, n_edges, 3)
    dB_reshaped = permutedims(reshape(dB_flat, n_atoms, len_Bi, 3, n_edges), (1, 2, 4, 3))

    # Convert edge gradients to atomic position gradients
    dB = zeros(SVector{3, T}, n_atoms, length_basis(calc), n_atoms)

    for (k, (i, j)) in enumerate(zip(G.ii, G.jj))
        for atom_idx in 1:n_atoms
            Z = AtomsBase.atomic_symbol(sys, atom_idx)
            Zs = AtomsBase.ChemicalSpecies(Z)
            inds = get_basis_inds(calc, Zs)

            for (local_b, global_b) in enumerate(inds)
                ∂B_∂𝐫 = SVector{3, T}(dB_reshaped[atom_idx, local_b, k, :])
                # ∂B/∂X_i = -∂B/∂𝐫_ij (since ∂𝐫_ij/∂X_i = -I)
                # ∂B/∂X_j = +∂B/∂𝐫_ij (since ∂𝐫_ij/∂X_j = +I)
                dB[atom_idx, global_b, i] -= ∂B_∂𝐫
                dB[atom_idx, global_b, j] += ∂B_∂𝐫
            end
        end
    end

    # Assemble B matrix
    Bi_all = reshape(Bi_all_flat, n_atoms, len_Bi)
    B = zeros(T, n_atoms, length_basis(calc))
    for i in 1:n_atoms
        Z = AtomsBase.atomic_symbol(sys, i)
        Zs = AtomsBase.ChemicalSpecies(Z)
        inds = get_basis_inds(calc, Zs)
        B[i, inds] .= T.(Bi_all[i, :])
    end

    return B, dB
end

# =========================================================================
#  Convenience constructors
# =========================================================================

"""
    build_et_calculator(ace_model, ps, st; device=identity, precision=:Float64)

Build an ETCalculator from an ACEModel by converting it to a Lux-based
EquivariantTensors model. This unified calculator supports both energy/forces/virial
evaluation and basis evaluation.

# Arguments
- `ace_model`: An ACEModel instance
- `ps`: Parameters from `Lux.setup(rng, ace_model)`
- `st`: States from `Lux.setup(rng, ace_model)`
- `device`: GPU device function (default: `identity` for CPU)
- `precision`: `:Float64` or `:Float32`

# Returns
An `ETCalculator` that can be used with AtomsCalculators interface
and for basis evaluation.
"""
function build_et_calculator(ace_model::ACEModel, ps, st;
                             device = identity,
                             precision = :Float64)

    # Get element list
    rbasis = ace_model.rbasis
    et_i2z = AtomsBase.ChemicalSpecies.(rbasis._i2z)

    # Build Rnl basis
    et_rbasis = _convert_Rnl_learnable(rbasis; zlist = et_i2z, rfun = x -> norm(x.𝐫))
    et_rspec = rbasis.spec

    # Build Ylm basis
    et_ybasis = Lux.Chain(𝐫ij = ET.NTtransform(x -> x.𝐫), Y = ace_model.ybasis)
    et_yspec = P4ML.natural_indices(et_ybasis.layers.Y)

    # Build embedding layer
    et_embed = ET.EdgeEmbed(Lux.BranchLayer(; Rnl = et_rbasis, Ylm = et_ybasis))

    # Build ACE tensor basis
    AA_spec = ace_model.tensor.meta["𝔸spec"]
    et_mb_spec = unique([[(n=b.n, l=b.l) for b in bb] for bb in AA_spec])

    et_mb_basis = ET.sparse_equivariant_tensor(
        L = 0,
        mb_spec = et_mb_spec,
        Rnl_spec = et_rspec,
        Ylm_spec = et_yspec,
        basis = real
    )

    # Build readout layer
    et_readout = let zlist = et_i2z
        __zi = x -> ET.cat2idx(zlist, x.s)
        ET.SelectLinL(
            et_mb_basis.lens[1],
            1,
            length(et_i2z),
            __zi
        )
    end

    # Build basis-only model (no readout or sum)
    et_basis = Lux.Chain(;
        embed = et_embed,
        ace = et_mb_basis,
        unwrp = Lux.WrappedFunction(x -> x[1])
    )

    # Build full energy model
    et_model = Lux.Chain(
        L1 = Lux.BranchLayer(;
            basis = et_basis,
            nodes = Lux.WrappedFunction(G -> G.node_data)
        ),
        Ei = et_readout,
        E = Lux.WrappedFunction(sum)
    )

    # Setup parameters and states for both models
    rng = Random.MersenneTwister(1234)
    et_ps, et_st = LuxCore.setup(rng, et_model)
    basis_ps, basis_st = LuxCore.setup(rng, et_basis)

    # Copy parameters from ACEModel
    NZ = length(et_i2z)
    for i in 1:NZ, j in 1:NZ
        idx = (i-1)*NZ + j
        et_ps.L1.basis.embed.Rnl.connection.W[:, :, idx] = ps.rbasis.Wnlq[:, :, i, j]
        basis_ps.embed.Rnl.connection.W[:, :, idx] = ps.rbasis.Wnlq[:, :, i, j]
    end

    for i in 1:NZ
        et_ps.Ei.W[1, :, i] .= ps.WB[:, i]
    end

    # Get cutoff and basis length
    rcut = maximum(a.rcut for a in ace_model.pairbasis.rin0cuts) * u"Å"
    len_Bi = length(ace_model.tensor)

    return ETCalculator(et_model, et_basis, et_ps, et_st, basis_ps, basis_st,
                        rcut, et_i2z, len_Bi, device, precision)
end

# Legacy alias for backwards compatibility
"""
    build_et_basis_calculator(ace_model, ps, st; device=identity, precision=:Float64)

DEPRECATED: Use `build_et_calculator` instead. The unified ETCalculator now supports
both energy/forces/virial evaluation and basis evaluation.

This function is provided for backwards compatibility and simply calls
`build_et_calculator`.
"""
function build_et_basis_calculator(ace_model::ACEModel, ps, st; kwargs...)
    @warn "build_et_basis_calculator is deprecated. Use build_et_calculator instead, which now supports both energy and basis evaluation."
    return build_et_calculator(ace_model, ps, st; kwargs...)
end
