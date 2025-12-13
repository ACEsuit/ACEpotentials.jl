# Batched evaluation for multiple structures in a single GPU call
#
# This module provides utilities for efficiently evaluating the ACE model
# on multiple atomic structures simultaneously, which is essential for
# training workflows.

import EquivariantTensors as ET
import AtomsBase
using StaticArrays
using Unitful: ustrip, @u_str
using Zygote

# Import shared utilities from calculators_et.jl
# _build_graph, _prepare_for_device, _energy_from_edge_positions,
# _reconstruct_graph, and _scatter_forces_virial are defined there

# Type alias for EFV result to improve readability
const EFVResult = NamedTuple{(:energy, :forces, :virial),
                              Tuple{typeof(1.0u"eV"),
                                    Vector{SVector{3, typeof(1.0u"eV/Å")}},
                                    SMatrix{3, 3, typeof(1.0u"eV"), 9}}}

# =========================================================================
#  BatchedETGraph - Multiple structures in a single graph
# =========================================================================

"""
    BatchedETGraph{VECI, TN, TE}

A batched graph representation containing multiple atomic structures.
This allows evaluating multiple structures in a single GPU call.

# Fields
- `graph`: The concatenated ETGraph containing all structures
- `n_structures`: Number of structures in the batch
- `atom_offsets`: Cumulative atom counts, atom_offsets[i] = start index of structure i
- `edge_offsets`: Cumulative edge counts, edge_offsets[i] = start index of structure i edges
- `n_atoms_per_structure`: Number of atoms in each structure
- `n_edges_per_structure`: Number of edges in each structure
"""
struct BatchedETGraph{G <: ET.ETGraph}
    graph::G
    n_structures::Int
    atom_offsets::Vector{Int}      # length n_structures + 1
    edge_offsets::Vector{Int}      # length n_structures + 1
    n_atoms_per_structure::Vector{Int}
    n_edges_per_structure::Vector{Int}
end

function Base.show(io::IO, bg::BatchedETGraph)
    n_atoms = ET.nnodes(bg.graph)
    n_edges = ET.nedges(bg.graph)
    print(io, "BatchedETGraph($(bg.n_structures) structures, $n_atoms atoms, $n_edges edges)")
end

n_structures(bg::BatchedETGraph) = bg.n_structures
total_atoms(bg::BatchedETGraph) = ET.nnodes(bg.graph)
total_edges(bg::BatchedETGraph) = ET.nedges(bg.graph)

"""
    get_structure_atoms(bg::BatchedETGraph, i::Int)

Get the atom indices for structure i (1-based).
"""
function get_structure_atoms(bg::BatchedETGraph, i::Int)
    start_idx = bg.atom_offsets[i] + 1
    end_idx = bg.atom_offsets[i+1]
    return start_idx:end_idx
end

"""
    get_structure_edges(bg::BatchedETGraph, i::Int)

Get the edge indices for structure i (1-based).
"""
function get_structure_edges(bg::BatchedETGraph, i::Int)
    start_idx = bg.edge_offsets[i] + 1
    end_idx = bg.edge_offsets[i+1]
    return start_idx:end_idx
end

# =========================================================================
#  Batch construction
# =========================================================================

"""
    batch_graphs(systems, rcut)

Build a BatchedETGraph from multiple atomic systems.

# Arguments
- `systems`: Vector of AtomsBase-compatible atomic systems
- `rcut`: Cutoff radius (with or without units)

# Returns
A `BatchedETGraph` containing all structures concatenated.

# Example
```julia
systems = [bulk(:Si) * (2,2,2), bulk(:Si) * (3,3,3)]
bg = batch_graphs(systems, 5.5u"Å")
```
"""
function batch_graphs(systems, rcut)
    n_sys = length(systems)

    # Build individual graphs using shared utility
    graphs = [_build_graph(sys, rcut) for sys in systems]

    # Compute offsets
    n_atoms_per = [ET.nnodes(g) for g in graphs]
    n_edges_per = [ET.nedges(g) for g in graphs]

    atom_offsets = [0; cumsum(n_atoms_per)]
    edge_offsets = [0; cumsum(n_edges_per)]

    total_n_atoms = atom_offsets[end]
    total_n_edges = edge_offsets[end]

    # Concatenate graph data with offset adjustments
    # Edge indices need to be shifted by atom offset
    all_ii = Vector{Int}(undef, total_n_edges)
    all_jj = Vector{Int}(undef, total_n_edges)

    # First array needs special handling - it's per-atom
    # first[i] = index of first edge for atom i
    all_first = Vector{Int}(undef, total_n_atoms + 1)

    for (s, g) in enumerate(graphs)
        atom_off = atom_offsets[s]
        edge_off = edge_offsets[s]
        n_atoms = n_atoms_per[s]
        n_edges = n_edges_per[s]

        # Copy edge indices with offset
        edge_range = (edge_off + 1):(edge_off + n_edges)
        all_ii[edge_range] .= g.ii .+ atom_off
        all_jj[edge_range] .= g.jj .+ atom_off

        # Copy first array with offset
        atom_range = (atom_off + 1):(atom_off + n_atoms)
        all_first[atom_range] .= g.first[1:n_atoms] .+ edge_off
    end
    all_first[end] = total_n_edges + 1

    # Concatenate node and edge data
    all_node_data = vcat([g.node_data for g in graphs]...)
    all_edge_data = vcat([g.edge_data for g in graphs]...)

    # Compute max neighbors
    max_neigs = maximum(g.maxneigs for g in graphs)

    # Create the concatenated graph (with nothing for graph_data)
    batched_graph = ET.ETGraph(all_ii, all_jj, all_first,
                               all_node_data, all_edge_data, nothing, max_neigs)

    return BatchedETGraph(batched_graph, n_sys,
                          atom_offsets, edge_offsets,
                          n_atoms_per, n_edges_per)
end

# =========================================================================
#  Batched energy evaluation
# =========================================================================

"""
    evaluate_batched_energies(model, ps, st, batched_graph)

Evaluate per-structure energies for a batched graph.

# Arguments
- `model`: The Lux Chain model (et_model from build_et_calculator)
- `ps`: Model parameters
- `st`: Model states
- `batched_graph`: A BatchedETGraph

# Returns
Vector of energies, one per structure in the batch.
"""
function evaluate_batched_energies(model, ps, st, bg::BatchedETGraph)
    # Evaluate the model on the full batched graph
    # The model returns total energy, but we need per-structure energies

    # First, get the per-atom energies by modifying the model to not sum
    # We need to extract the per-atom contributions

    # Get the L1 layer output (before final sum)
    # The model structure is: L1 -> Ei -> E (sum)
    # We want to stop at Ei to get per-atom energies

    G = bg.graph

    # Run through L1 (basis + nodes)
    L1_out, st_L1 = model.layers.L1(G, ps.L1, st.L1)

    # Run through Ei (readout) to get per-atom energies
    Ei, st_Ei = model.layers.Ei(L1_out, ps.Ei, st.Ei)

    # Ei is now a vector of per-atom energies
    # Sum within each structure to get per-structure energies
    n_sys = bg.n_structures
    energies = zeros(eltype(Ei), n_sys)

    for s in 1:n_sys
        atom_range = get_structure_atoms(bg, s)
        energies[s] = sum(Ei[atom_range])
    end

    return energies
end

"""
    evaluate_batched_energies(calc::ETCalculator, systems)

Convenience function to evaluate energies for multiple systems using ETCalculator.

# Arguments
- `calc`: An ETCalculator instance
- `systems`: Vector of AtomsBase-compatible atomic systems

# Returns
Vector of energies with units.
"""
function evaluate_batched_energies(calc::ETCalculator, systems)
    bg = batch_graphs(systems, calc.rcut)

    # Prepare for device using shared utility
    G_dev, ps_dev, st_dev = _prepare_for_device(bg.graph, calc.ps, calc.st,
                                                 calc.device, calc.precision)

    # Create modified batched graph with device arrays
    bg_dev = BatchedETGraph(G_dev, bg.n_structures,
                            bg.atom_offsets, bg.edge_offsets,
                            bg.n_atoms_per_structure, bg.n_edges_per_structure)

    energies = evaluate_batched_energies(calc.model, ps_dev, st_dev, bg_dev)

    # Convert to CPU if needed
    if calc.device !== identity
        energies = collect(energies)
    end

    return Float64.(energies) .* energy_unit(calc)
end

# =========================================================================
#  Batched basis evaluation
# =========================================================================

"""
    evaluate_batched_basis(calc::ETCalculator, systems)

Evaluate basis for multiple systems in a single batched call.

# Arguments
- `calc`: An ETCalculator instance
- `systems`: Vector of AtomsBase-compatible atomic systems

# Returns
Vector of basis matrices, one per system. Each matrix has shape (n_atoms, length_basis).
"""
function evaluate_batched_basis(calc::ETCalculator, systems)
    bg = batch_graphs(systems, calc.rcut)

    # Prepare for device using shared utility
    G_dev, ps_dev, st_dev = _prepare_for_device(bg.graph, calc.basis_ps, calc.basis_st,
                                                 calc.device, calc.precision)

    # Evaluate basis on full batched graph
    Bi_all, _ = calc.basis_model(G_dev, ps_dev, st_dev)

    # Convert to CPU if needed
    if calc.device !== identity
        Bi_all = collect(Bi_all)
    end

    # Split into per-structure basis matrices
    T = Float64
    n_sys = bg.n_structures
    basis_list = Vector{Matrix{T}}(undef, n_sys)

    for s in 1:n_sys
        n_atoms = bg.n_atoms_per_structure[s]
        atom_range = get_structure_atoms(bg, s)

        B = zeros(T, n_atoms, length_basis(calc))

        for (local_i, global_i) in enumerate(atom_range)
            # Get species for this atom
            # We need the original system to get species info
            sys = systems[s]
            Z = AtomsBase.atomic_symbol(sys, local_i)
            Zs = AtomsBase.ChemicalSpecies(Z)
            inds = get_basis_inds(calc, Zs)

            # Bi_all is (total_atoms, len_Bi)
            B[local_i, inds] .= T.(Bi_all[global_i, :])
        end

        basis_list[s] = B
    end

    return basis_list
end

# =========================================================================
#  Batched force/virial evaluation
# =========================================================================

"""
    evaluate_batched_efv(calc::ETCalculator, systems)

Evaluate energies, forces, and virials for multiple systems.

# Arguments
- `calc`: An ETCalculator instance
- `systems`: Vector of AtomsBase-compatible atomic systems

# Returns
Vector of NamedTuples (energy, forces, virial), one per system.
"""
function evaluate_batched_efv(calc::ETCalculator, systems)
    bg = batch_graphs(systems, calc.rcut)
    G = bg.graph

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

    # Energy function for differentiation (uses shared utility)
    function _efunc(𝐫_vec)
        _energy_from_edge_positions(𝐫_vec, G, node_data, s0_edges, s1_edges,
                                    calc.model, ps_work, st_work, dev)
    end

    # Compute total energy and gradient
    E_total = _efunc(𝐫_edges)
    ∇𝐫 = Zygote.gradient(_efunc, 𝐫_edges)[1]

    if ∇𝐫 === nothing
        error("Zygote gradient returned nothing - check model differentiability")
    end

    # Convert to CPU if needed
    if dev !== identity
        ∇𝐫 = collect(∇𝐫)
    end

    # Split results per structure
    n_sys = bg.n_structures
    results = Vector{EFVResult}(undef, n_sys)

    for s in 1:n_sys
        sys = systems[s]
        n_atoms = bg.n_atoms_per_structure[s]
        edge_range = get_structure_edges(bg, s)
        atom_offset = bg.atom_offsets[s]

        # Extract edges for this structure
        ii_s = G.ii[edge_range]
        jj_s = G.jj[edge_range]
        ∇𝐫_s = ∇𝐫[edge_range]
        edge_positions_s = [G.edge_data[k].𝐫 for k in edge_range]

        # Use shared utility for force/virial scatter
        F, virial = _scatter_forces_virial(ii_s, jj_s, ∇𝐫_s, edge_positions_s, n_atoms;
                                           offset=atom_offset)

        # Compute per-structure energy by re-evaluating on just this structure
        # TODO: Could optimize by extracting per-atom energies from forward pass
        E_s = evaluate_batched_energies(calc, [sys])[1]

        results[s] = (
            energy = E_s,
            forces = F .* (energy_unit(calc) / length_unit(calc)),
            virial = virial * energy_unit(calc)
        )
    end

    return results
end
