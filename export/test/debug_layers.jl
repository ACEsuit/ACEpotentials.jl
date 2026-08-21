using ACEpotentials
using ACEpotentials.Models
using ACEpotentials.ETModels
using ACEpotentials.ETModels: splinify, ETACEPotential, ETACE
using StaticArrays
using LinearAlgebra
using Random
using Lux
using LuxCore
import AtomsBase
using Unitful
using AtomsCalculators
import EquivariantTensors as ET

const M = ACEpotentials.Models

# Create splinified model
elements = (:Si,)
rcut = 5.5

rin0cuts = M._default_rin0cuts(elements)
rin0cuts = (x -> (rin = x.rin, r0 = x.r0, rcut = rcut)).(rin0cuts)

rng = Random.MersenneTwister(1234)

ace_model = M.ace_model(;
    elements = elements,
    order = 2,
    Ytype = :solid,
    level = M.TotalDegree(),
    max_level = 8,
    maxl = 2,
    pair_maxn = 8,
    rin0cuts = rin0cuts,
    init_WB = :glorot_normal,
    init_Wpair = :glorot_normal
)

ps, st = Lux.setup(rng, ace_model)

et_model = ACEpotentials.ETModels.convert2et(ace_model)
et_ps, et_st = LuxCore.setup(MersenneTwister(1234), et_model)

n_species = length(elements)
for iz in 1:n_species
    for jz in 1:n_species
        et_ps.rembed.post.W[:, :, (iz-1)*n_species + jz] .= ps.rbasis.Wnlq[:, :, iz, jz]
    end
end
for iz in 1:n_species
    et_ps.readout.W[1, :, iz] .= ps.WB[:, iz]
end

# Splinify
et_model_splined = splinify(et_model, et_ps, et_st; Nspl=50)
et_ps_splined, et_st_splined = LuxCore.setup(MersenneTwister(1234), et_model_splined)

for iz in 1:n_species
    et_ps_splined.readout.W[1, :, iz] .= et_ps.readout.W[1, :, iz]
end

calc_splined = ETACEPotential(et_model_splined, et_ps_splined, et_st_splined, rcut)

# Create test system
a0 = 5.43
positions = [
    SVector(0.0, 0.0, 0.0),
    SVector(a0/4, a0/4, a0/4),
    SVector(a0/2, a0/2, 0.0),
    SVector(a0/2, 0.0, a0/2),
]
box = [SVector(a0, 0.0, 0.0), SVector(0.0, a0, 0.0), SVector(0.0, 0.0, a0)]

sys = AtomsBase.periodic_system(
    [:Si => pos * u"Å" for pos in positions],
    [b * u"Å" for b in box]
)

# Build graph
G = ET.Atoms.interaction_graph(sys, rcut * u"Å")

println("=== Graph structure ===")
println("Number of nodes: ", length(G.node_data))
println("Number of edges: ", length(G.edge_data))

# Evaluate each layer of the reference model
println("\n=== Reference model layer-by-layer evaluation ===")

# Get model components
rembed = et_model_splined.rembed
yembed = et_model_splined.yembed
basis = et_model_splined.basis
readout = et_model_splined.readout

# Step 1: rembed (radial embedding)
Rnl_ref, st_rembed_out = rembed(G, et_ps_splined.rembed, et_st_splined.rembed)
println("\nRnl (rembed output):")
println("  Type: ", typeof(Rnl_ref))
println("  Shape: ", size(Rnl_ref))
if ndims(Rnl_ref) == 2
    println("  Rnl_ref[1:5, 1:5] = ")
    for i in 1:min(5, size(Rnl_ref, 1))
        println("    ", Rnl_ref[i, 1:min(5, size(Rnl_ref, 2))])
    end
else
    println("  First element type: ", typeof(Rnl_ref[1]))
    println("  First 5 elements:")
    for i in 1:min(5, length(Rnl_ref))
        println("    Rnl[$i] = ", Rnl_ref[i])
    end
end

# Step 2: yembed (angular embedding)
Ylm_ref, st_yembed_out = yembed(G, et_ps_splined.yembed, et_st_splined.yembed)
println("\nYlm (yembed output):")
println("  Type: ", typeof(Ylm_ref))
println("  Shape: ", size(Ylm_ref))
if ndims(Ylm_ref) == 2
    println("  Ylm_ref[1:5, :] = ")
    for i in 1:min(5, size(Ylm_ref, 1))
        println("    ", Ylm_ref[i, :])
    end
end

# Step 3: basis (many-body pooling)
println("\n=== Calling basis layer ===")
basis_out, st_basis_out = basis((Rnl_ref, Ylm_ref), et_ps_splined.basis, et_st_splined.basis)
println("Basis output type: ", typeof(basis_out))
if isa(basis_out, Tuple)
    println("  Tuple of length: ", length(basis_out))
    𝔹_ref = basis_out[1]
else
    𝔹_ref = basis_out
end
println("  𝔹_ref type: ", typeof(𝔹_ref))
println("  𝔹_ref shape: ", size(𝔹_ref))
println("  𝔹_ref[1, 1:10] (node 1, first 10 basis functions):")
println("    ", 𝔹_ref[1, 1:min(10, size(𝔹_ref, 2))])
println("  𝔹_ref[2, 1:10] (node 2, first 10 basis functions):")
println("    ", 𝔹_ref[2, 1:min(10, size(𝔹_ref, 2))])

# Step 4: readout
println("\n=== Calling readout layer ===")
φ_ref, st_readout_out = readout((𝔹_ref, G.node_data), et_ps_splined.readout, et_st_splined.readout)
println("Readout output type: ", typeof(φ_ref))
println("  φ_ref (site energies): ", φ_ref)
println("  sum(φ_ref): ", sum(φ_ref))

# Now compare with exported model
println("\n" * "=" ^ 60)
println("=== Comparing with exported model ===")

# Include exported model
exp_mod = Module(:ExportedModel)
Base.include(exp_mod, joinpath(@__DIR__, "build", "hermite_accuracy_test.jl"))

# Build neighbor list for atom 1
Rs1 = SVector{3, Float64}[]
Zs1 = Int[]
for (edge_idx, edge) in enumerate(G.edge_data)
    if G.ii[edge_idx] == 1
        push!(Rs1, SVector{3, Float64}(edge.𝐫))
        push!(Zs1, Int(edge.z1.atomic_number))
    end
end
println("\nAtom 1 has $(length(Rs1)) neighbors")

# Compute exported embeddings
Rnl_exp, Ylm_exp = exp_mod.compute_embeddings(Rs1, Zs1, 14)
println("\nExported embeddings:")
println("  Rnl_exp shape: ", size(Rnl_exp))
println("  Ylm_exp shape: ", size(Ylm_exp))
println("  Rnl_exp[1, 1:5]: ", Rnl_exp[1, 1:min(5, size(Rnl_exp, 2))])
println("  Ylm_exp[1, :]: ", Ylm_exp[1, :])

# Compare Rnl from reference (edges for node 1) vs exported
println("\n=== Comparing Rnl values (node 1 edges) ===")

# Get edge indices for node 1
edge_indices_node1 = [idx for idx in 1:length(G.edge_data) if G.ii[idx] == 1]
println("Edges belonging to node 1: ", length(edge_indices_node1))

# Compare Rnl values
# Rnl_ref is (18, 4, 19) = (max_neighbors, nodes, n_rnl)
# So Rnl_ref[j, i, :] is Rnl for j-th neighbor of node i
println("\nComparing Rnl (reference vs exported) for node 1:")
println("Reference Rnl shape: ", size(Rnl_ref))  # Should be (18, 4, 19)
println("Exported Rnl shape: ", size(Rnl_exp))   # Should be (18, 19)

# For node 1 (index 1), compare Rnl
for j in 1:min(5, size(Rnl_ref, 1))
    ref_val = Rnl_ref[j, 1, :]  # j-th neighbor of node 1
    exp_val = Rnl_exp[j, :]      # j-th neighbor in my extracted list

    # Get distance for this neighbor
    exp_r = norm(Rs1[j])

    println("  Neighbor $j (r=$(round(exp_r, digits=3))):")
    println("    Ref[1:5]: ", ref_val[1:min(5, length(ref_val))])
    println("    Exp[1:5]: ", exp_val[1:min(5, length(exp_val))])
    println("    Max diff: ", maximum(abs.(ref_val - exp_val)))
    println("    Match? ", ref_val ≈ exp_val)
end

# Check if neighbors are in the same order by comparing distances
println("\n=== Checking neighbor ordering ===")
# Reference doesn't directly give distances, but we can check via edge data
# Actually, we need to match edges to the reshaped Rnl

# The reference model uses ET.Atoms.interaction_graph which creates edges
# The rembed layer then evaluates Rnl for each edge
# The reshape happens via ET.reshape_embedding

# Let me check the edge order in the graph
println("First 5 edges in graph:")
for k in 1:min(5, length(G.edge_data))
    e = G.edge_data[k]
    println("  Edge $k: ii=$(G.ii[k]), jj=$(G.jj[k]), |𝐫|=$(norm(e.𝐫))")
end

println("\nFirst 5 neighbors of node 1 (my extraction):")
for j in 1:min(5, length(Rs1))
    println("  Neighbor $j: |R|=$(norm(Rs1[j])), R=$(Rs1[j])")
end

# The issue is the reshaping! Let me check how ET reshapes the embedding
# EdgeEmbed evaluates on flat edge list, then reshapes to (maxneigs, nnodes, nfuncs)
# The neighbor ordering within each node depends on the edge ordering in the graph

# Let me extract neighbors in the SAME order as the graph does for node 1
println("\n=== Re-extracting neighbors in graph edge order ===")
Rs1_correct = SVector{3, Float64}[]
Zs1_correct = Int[]
neig_idx_in_graph = Int[]
for (edge_idx, edge) in enumerate(G.edge_data)
    if G.ii[edge_idx] == 1
        push!(Rs1_correct, SVector{3, Float64}(edge.𝐫))
        push!(Zs1_correct, Int(edge.z1.atomic_number))
        push!(neig_idx_in_graph, edge_idx)
    end
end

println("Node 1 neighbors (in graph edge order):")
for j in 1:min(5, length(Rs1_correct))
    edge_idx = neig_idx_in_graph[j]
    println("  Neighbor $j: edge_idx=$edge_idx, |R|=$(round(norm(Rs1_correct[j]), digits=3))")
end

# Now compute 𝔹 for exported model
B_exp, A_exp = exp_mod.tensor_evaluate(Rnl_exp, Ylm_exp)
println("\n=== Comparing B-basis (node 1) ===")
println("Reference 𝔹[1, :] (node 1 B-basis):")
println("  ", 𝔹_ref[1, :])
println("\nExported B (from tensor_evaluate):")
println("  ", B_exp)

println("\n=== B-basis detailed comparison ===")
for i in 1:min(10, length(B_exp))
    ref = 𝔹_ref[1, i]
    exp = B_exp[i]
    println("  B[$i]: ref = $ref, exp = $exp, ratio = $(ref/exp)")
end

println("\n=== Computing energies ===")
# Reference site energy for node 1
ref_site_energy_1 = dot(et_ps_splined.readout.W[1, :, 1], 𝔹_ref[1, :])
println("Reference site energy (node 1): ", ref_site_energy_1, " eV")

# Exported site energy
exp_site_energy_1 = exp_mod.site_energy(Rs1, Zs1, 14)
println("Exported site energy (node 1): ", exp_site_energy_1, " eV")

println("\nRatio ref/exp: ", ref_site_energy_1 / exp_site_energy_1)

# Total reference energy
E_ref_total = sum(φ_ref)
println("\nReference total energy: ", E_ref_total, " eV")
