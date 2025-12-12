
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
using LinearAlgebra: norm
using Random
import Lux
import LuxCore

# =========================================================================
#  ETCalculator - Lux-based ACE calculator using EquivariantTensors
# =========================================================================

"""
    ETCalculator

An AtomsCalculators-compatible calculator that uses the EquivariantTensors
Lux model for energy, force, and virial evaluation.

Supports both CPU and GPU evaluation. Forces and virial are computed via
Zygote automatic differentiation through the Lux model.

# Fields
- `model`: The Lux Chain model (et_model)
- `ps`: Model parameters (NamedTuple)
- `st`: Model states (NamedTuple)
- `rcut`: Cutoff radius (with units)
- `device`: Device function for GPU (e.g., `Metal.mtl`, `CUDA.cu`, or `identity` for CPU)
- `precision`: `:Float64` or `:Float32` for GPU

# Example
```julia
calc = ETCalculator(et_model, et_ps, et_st, 5.5u"Å")
E = potential_energy(system, calc)
efv = energy_forces_virial(system, calc)
```
"""
mutable struct ETCalculator{MOD, PS, ST, DEV}
    model::MOD
    ps::PS
    st::ST
    rcut::typeof(1.0u"Å")
    device::DEV
    precision::Symbol
end

function ETCalculator(model, ps, st, rcut;
                      device = identity,
                      precision = :Float64)
    ETCalculator(model, ps, st, rcut, device, precision)
end

energy_unit(::ETCalculator) = u"eV"
length_unit(::ETCalculator) = u"Å"

# =========================================================================
#  Core evaluation functions
# =========================================================================

"""
    _build_graph(sys, calc::ETCalculator)

Build an ETGraph from an atomic system for the given calculator.
"""
function _build_graph(sys, calc::ETCalculator)
    rcut = calc.rcut
    G = ET.Atoms.interaction_graph(sys, rcut)
    return G
end

"""
    _prepare_for_device(G, ps, st, calc::ETCalculator)

Prepare the graph, parameters, and states for the target device.
"""
function _prepare_for_device(G, ps, st, calc::ETCalculator)
    dev = calc.device
    if calc.precision == :Float32
        G = ET.ETGraph(G.ii, G.jj, G.first,
                       ET.float32.(G.node_data),
                       ET.float32.(G.edge_data),
                       G.maxneigs)
        ps = ET.float32(ps)
        st = ET.float32(st)
    end

    if dev !== identity
        G = dev(G)
        ps = dev(ps)
        st = dev(st)
    end

    return G, ps, st
end

"""
    _energy_from_edge_positions(𝐫_vec, G, s0_edges, s1_edges, model, ps, st, dev)

Compute energy as a function of edge position vectors.
This is the differentiable function for Zygote.
"""
function _energy_from_edge_positions(𝐫_vec, G, node_data, s0_edges, s1_edges,
                                      model, ps, st, dev)
    # Reconstruct edge_data from position vectors
    edge_data = [(𝐫 = r, s0 = s0, s1 = s1)
                 for (r, s0, s1) in zip(𝐫_vec, s0_edges, s1_edges)]
    G_new = ET.ETGraph(G.ii, G.jj, G.first, node_data, edge_data, G.maxneigs)

    if dev !== identity
        G_new = dev(G_new)
    end

    return model(G_new, ps, st)[1]
end

# =========================================================================
#  AtomsCalculators interface
# =========================================================================

function AtomsCalculators.potential_energy(sys, calc::ETCalculator; kwargs...)
    G = _build_graph(sys, calc)
    G_dev, ps_dev, st_dev = _prepare_for_device(G, calc.ps, calc.st, calc)
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
    G = _build_graph(sys, calc)

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

    # Compute forces via scatter
    # F[i] = +∂E/∂𝐫_ij (since ∂𝐫_ij/∂X_i = -I)
    # F[j] = -∂E/∂𝐫_ij (since ∂𝐫_ij/∂X_j = +I)
    T = Float64  # Always return Float64 forces
    F = zeros(SVector{3, T}, length(sys))
    for (k, (i, j)) in enumerate(zip(G.ii, G.jj))
        ∂E_∂𝐫 = SVector{3, T}(∇𝐫[k])
        F[i] += ∂E_∂𝐫
        F[j] -= ∂E_∂𝐫
    end

    # Compute virial: σ = -∑_edges (∂E/∂𝐫_ij) ⊗ 𝐫_ij
    virial = @SMatrix zeros(T, 3, 3)
    for (k, ed) in enumerate(G.edge_data)
        𝐫ij = SVector{3, T}(ed.𝐫)
        ∂E_∂𝐫 = SVector{3, T}(∇𝐫[k])
        virial -= ∂E_∂𝐫 * 𝐫ij'
    end

    return (
        energy = Float64(E) * energy_unit(calc),
        forces = F .* (energy_unit(calc) / length_unit(calc)),
        virial = virial * energy_unit(calc)
    )
end

# =========================================================================
#  Convenience constructors
# =========================================================================

"""
    build_et_calculator(ace_model, ps, st; device=identity, precision=:Float64)

Build an ETCalculator from an ACEModel by converting it to a Lux-based
EquivariantTensors model.

# Arguments
- `ace_model`: An ACEModel instance
- `ps`: Parameters from `Lux.setup(rng, ace_model)`
- `st`: States from `Lux.setup(rng, ace_model)`
- `device`: GPU device function (default: `identity` for CPU)
- `precision`: `:Float64` or `:Float32`

# Returns
An `ETCalculator` that can be used with AtomsCalculators interface.
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

    # Build full model
    et_basis = Lux.Chain(;
        embed = et_embed,
        ace = et_mb_basis,
        unwrp = Lux.WrappedFunction(x -> x[1])
    )

    et_model = Lux.Chain(
        L1 = Lux.BranchLayer(;
            basis = et_basis,
            nodes = Lux.WrappedFunction(G -> G.node_data)
        ),
        Ei = et_readout,
        E = Lux.WrappedFunction(sum)
    )

    # Setup parameters and states
    rng = Random.MersenneTwister(1234)
    et_ps, et_st = LuxCore.setup(rng, et_model)

    # Copy parameters from ACEModel
    NZ = length(et_i2z)
    for i in 1:NZ, j in 1:NZ
        idx = (i-1)*NZ + j
        et_ps.L1.basis.embed.Rnl.connection.W[:, :, idx] = ps.rbasis.Wnlq[:, :, i, j]
    end

    for i in 1:NZ
        et_ps.Ei.W[1, :, i] .= ps.WB[:, i]
    end

    # Get cutoff
    rcut = maximum(a.rcut for a in ace_model.pairbasis.rin0cuts) * u"Å"

    return ETCalculator(et_model, et_ps, et_st, rcut; device=device, precision=precision)
end

# =========================================================================
#  ETBasisCalculator - Basis evaluation (returns B, not E)
# =========================================================================

"""
    ETBasisCalculator

A calculator for evaluating the ACE basis functions using the EquivariantTensors
Lux model. Returns per-atom basis vectors B (for linear fitting) rather than
energy.

This is the ET equivalent of `evaluate_basis()` and `evaluate_basis_ed()`
from the CPU backend.

# Fields
- `basis_model`: The Lux Chain model returning per-atom basis (embed → ace → unwrap)
- `ps`: Model parameters (NamedTuple)
- `st`: Model states (NamedTuple)
- `rcut`: Cutoff radius (with units)
- `zlist`: List of chemical species (for basis indexing)
- `len_Bi`: Length of basis per species
- `device`: Device function for GPU
- `precision`: `:Float64` or `:Float32`
"""
mutable struct ETBasisCalculator{MOD, PS, ST, ZLIST, DEV}
    basis_model::MOD
    ps::PS
    st::ST
    rcut::typeof(1.0u"Å")
    zlist::ZLIST
    len_Bi::Int
    device::DEV
    precision::Symbol
end

function ETBasisCalculator(basis_model, ps, st, rcut, zlist, len_Bi;
                           device = identity,
                           precision = :Float64)
    ETBasisCalculator(basis_model, ps, st, rcut, zlist, len_Bi, device, precision)
end

"""
    length_basis(calc::ETBasisCalculator)

Return the total basis length: `len_Bi * NZ` where NZ is the number of species.
Note: This does not include pair basis (not yet implemented in ET backend).
"""
length_basis(calc::ETBasisCalculator) = calc.len_Bi * length(calc.zlist)

"""
    get_basis_inds(calc::ETBasisCalculator, Z)

Get the basis indices for species Z.
"""
function get_basis_inds(calc::ETBasisCalculator, Z)
    i_z = ET.cat2idx(calc.zlist, Z)
    return (i_z - 1) * calc.len_Bi .+ (1:calc.len_Bi)
end

"""
    build_et_basis_calculator(ace_model, ps, st; device=identity, precision=:Float64)

Build an ETBasisCalculator from an ACEModel. This creates a Lux model that
returns per-atom basis vectors B (not energy).

# Arguments
- `ace_model`: An ACEModel instance
- `ps`: Parameters from `Lux.setup(rng, ace_model)`
- `st`: States from `Lux.setup(rng, ace_model)`
- `device`: GPU device function (default: `identity` for CPU)
- `precision`: `:Float64` or `:Float32`

# Returns
An `ETBasisCalculator` for basis evaluation.
"""
function build_et_basis_calculator(ace_model::ACEModel, ps, st;
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

    # Build basis-only model (no readout or sum)
    # Returns per-atom basis vectors
    et_basis_model = Lux.Chain(;
        embed = et_embed,
        ace = et_mb_basis,
        unwrp = Lux.WrappedFunction(x -> x[1])  # Extract L=0 component
    )

    # Setup parameters and states
    rng = Random.MersenneTwister(1234)
    et_ps, et_st = LuxCore.setup(rng, et_basis_model)

    # Copy radial basis parameters from ACEModel
    NZ = length(et_i2z)
    for i in 1:NZ, j in 1:NZ
        idx = (i-1)*NZ + j
        et_ps.embed.Rnl.connection.W[:, :, idx] = ps.rbasis.Wnlq[:, :, i, j]
    end

    # Get cutoff and basis length
    rcut = maximum(a.rcut for a in ace_model.pairbasis.rin0cuts) * u"Å"
    len_Bi = length(ace_model.tensor)

    return ETBasisCalculator(et_basis_model, et_ps, et_st, rcut, et_i2z, len_Bi;
                             device=device, precision=precision)
end

"""
    _build_graph(sys, calc::ETBasisCalculator)

Build an ETGraph from an atomic system for the given calculator.
"""
function _build_graph(sys, calc::ETBasisCalculator)
    rcut = calc.rcut
    G = ET.Atoms.interaction_graph(sys, rcut)
    return G
end

"""
    _prepare_for_device(G, ps, st, calc::ETBasisCalculator)

Prepare the graph, parameters, and states for the target device.
"""
function _prepare_for_device(G, ps, st, calc::ETBasisCalculator)
    dev = calc.device
    if calc.precision == :Float32
        G = ET.ETGraph(G.ii, G.jj, G.first,
                       ET.float32.(G.node_data),
                       ET.float32.(G.edge_data),
                       G.maxneigs)
        ps = ET.float32(ps)
        st = ET.float32(st)
    end

    if dev !== identity
        G = dev(G)
        ps = dev(ps)
        st = dev(st)
    end

    return G, ps, st
end

"""
    evaluate_basis(calc::ETBasisCalculator, sys)

Evaluate the ACE basis for all atoms in the system.

Returns a matrix B of shape (n_atoms, length_basis) where each row contains
the basis values for that atom, indexed by species.

# Arguments
- `calc`: ETBasisCalculator instance
- `sys`: AtomsBase-compatible atomic system

# Returns
Matrix B where B[i, :] contains basis values for atom i.
"""
function evaluate_basis(calc::ETBasisCalculator, sys)
    G = _build_graph(sys, calc)
    G_dev, ps_dev, st_dev = _prepare_for_device(G, calc.ps, calc.st, calc)

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

"""
    _basis_from_edge_positions(𝐫_vec, G, node_data, s0_edges, s1_edges, model, ps, st, dev)

Compute per-atom basis as a function of edge position vectors.
This is the differentiable function for Zygote jacobian.
"""
function _basis_from_edge_positions(𝐫_vec, G, node_data, s0_edges, s1_edges,
                                     model, ps, st, dev)
    # Reconstruct edge_data from position vectors
    edge_data = [(𝐫 = r, s0 = s0, s1 = s1)
                 for (r, s0, s1) in zip(𝐫_vec, s0_edges, s1_edges)]
    G_new = ET.ETGraph(G.ii, G.jj, G.first, node_data, edge_data, G.maxneigs)

    if dev !== identity
        G_new = dev(G_new)
    end

    Bi_all, _ = model(G_new, ps, st)
    return Bi_all  # Shape: (n_atoms, len_Bi)
end

"""
    evaluate_basis_ed(calc::ETBasisCalculator, sys)

Evaluate the ACE basis and its gradients with respect to atomic positions.

Returns (B, dB) where:
- B: Matrix of shape (n_atoms, length_basis) with basis values
- dB: Array of shape (n_atoms, length_basis, n_atoms, 3) with gradients

The gradient dB[i, b, j, α] = ∂B[i,b]/∂X[j,α] where X[j,α] is the α-th
coordinate of atom j.

# Arguments
- `calc`: ETBasisCalculator instance
- `sys`: AtomsBase-compatible atomic system

# Returns
Tuple (B, dB) with basis values and gradients.
"""
function evaluate_basis_ed(calc::ETBasisCalculator, sys)
    G = _build_graph(sys, calc)

    # Extract edge data components
    𝐫_edges = [ed.𝐫 for ed in G.edge_data]
    s0_edges = [ed.s0 for ed in G.edge_data]
    s1_edges = [ed.s1 for ed in G.edge_data]

    # Prepare parameters and states
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

    # Basis function for differentiation
    function _bfunc(𝐫_vec)
        _basis_from_edge_positions(𝐫_vec, G, node_data, s0_edges, s1_edges,
                                   calc.basis_model, ps_work, st_work, dev)
    end

    # Evaluate basis
    Bi_all = _bfunc(𝐫_edges)

    # Compute Jacobian via Zygote
    # We need ∂Bi/∂𝐫_edges, then convert to ∂Bi/∂X using chain rule
    # For each basis function output, compute gradient w.r.t. all edge positions

    len_Bi = calc.len_Bi
    n_atoms = length(sys)
    n_edges = length(𝐫_edges)

    # Use Zygote pullback to get gradients
    # For each output component (atom i, basis b), get gradient w.r.t. edge positions
    _, pullback = Zygote.pullback(_bfunc, 𝐫_edges)

    # Initialize gradient storage
    # dB_edges[i, b, k] = ∂B[i,b]/∂𝐫_edges[k] (3-vector)
    T = Float64
    dB_edges = zeros(SVector{3, T}, n_atoms, len_Bi, n_edges)

    # Compute gradients for each output component
    # Bi_all has shape (n_atoms, len_Bi), so tangent should match
    for i in 1:n_atoms
        for b in 1:len_Bi
            # Create one-hot tangent vector matching Bi_all shape (n_atoms, len_Bi)
            tangent = zeros(T, n_atoms, len_Bi)
            tangent[i, b] = one(T)

            # Pullback to get gradient w.r.t. edges
            ∇𝐫 = pullback(tangent)[1]

            if ∇𝐫 !== nothing
                for k in 1:n_edges
                    dB_edges[i, b, k] = SVector{3, T}(∇𝐫[k])
                end
            end
        end
    end

    # Convert edge gradients to atomic position gradients
    # ∂B/∂X_i = ∑_{edges (i,j)} ∂B/∂𝐫_ij * ∂𝐫_ij/∂X_i
    # where ∂𝐫_ij/∂X_i = -I and ∂𝐫_ij/∂X_j = +I
    dB = zeros(SVector{3, T}, n_atoms, length_basis(calc), n_atoms)

    for (k, (i, j)) in enumerate(zip(G.ii, G.jj))
        for atom_idx in 1:n_atoms
            Z = AtomsBase.atomic_symbol(sys, atom_idx)
            Zs = AtomsBase.ChemicalSpecies(Z)
            inds = get_basis_inds(calc, Zs)

            for (local_b, global_b) in enumerate(inds)
                ∂B_∂𝐫 = dB_edges[atom_idx, local_b, k]
                # ∂B/∂X_i = -∂B/∂𝐫_ij (since ∂𝐫_ij/∂X_i = -I)
                # ∂B/∂X_j = +∂B/∂𝐫_ij (since ∂𝐫_ij/∂X_j = +I)
                dB[atom_idx, global_b, i] -= ∂B_∂𝐫
                dB[atom_idx, global_b, j] += ∂B_∂𝐫
            end
        end
    end

    # Assemble B matrix
    if calc.device !== identity
        Bi_all = collect(Bi_all)
    end

    B = zeros(T, n_atoms, length_basis(calc))
    for i in 1:n_atoms
        Z = AtomsBase.atomic_symbol(sys, i)
        Zs = AtomsBase.ChemicalSpecies(Z)
        inds = get_basis_inds(calc, Zs)
        # Bi_all is (n_atoms, len_Bi)
        B[i, inds] .= T.(Bi_all[i, :])
    end

    return B, dB
end
