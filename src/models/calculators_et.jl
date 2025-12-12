
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
