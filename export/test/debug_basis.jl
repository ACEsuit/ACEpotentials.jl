using ACEpotentials
using ACEpotentials.Models
using ACEpotentials.ETModels
using ACEpotentials.ETModels: splinify, ETACEPotential
using StaticArrays
using LinearAlgebra
using Random
using Lux
using LuxCore
import AtomsBase
using Unitful
using AtomsCalculators
import EquivariantTensors as ET
using SparseArrays

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

# Get total energy from reference
E_ref = ustrip(u"eV", AtomsCalculators.potential_energy(sys, calc_splined))
println("Reference total energy: $E_ref eV")

# Include exported model
println("\n=== Including exported model ===")
exp_mod = Module(:ExportedModel)
Base.include(exp_mod, joinpath(@__DIR__, "build", "hermite_accuracy_test.jl"))

# Build neighbor lists
G = ET.Atoms.interaction_graph(sys, rcut * u"Å")

neighbor_data = Vector{Tuple{Vector{SVector{3,Float64}}, Vector{Int}}}()
for i in 1:4
    Rs_i = SVector{3, Float64}[]
    Zs_i = Int[]
    for (edge_idx, edge) in enumerate(G.edge_data)
        if G.ii[edge_idx] == i
            push!(Rs_i, SVector{3, Float64}(edge.𝐫))
            push!(Zs_i, Int(edge.z1.atomic_number))
        end
    end
    push!(neighbor_data, (Rs_i, Zs_i))
end

# Get the reference A2B matrix and compare
println("\n=== Comparing A2B matrices ===")
ref_A2B = et_model_splined.basis.A2Bmaps[1]
println("Reference A2B type: ", typeof(ref_A2B))
println("Reference A2B size: ", size(ref_A2B))
println("Reference A2B nnz: ", nnz(ref_A2B))

# Convert to vectors for comparison
ref_A2B_I, ref_A2B_J, ref_A2B_V = findnz(ref_A2B)
println("\nReference A2B sparse structure:")
println("  I (first 20): ", ref_A2B_I[1:min(20, length(ref_A2B_I))])
println("  J (first 20): ", ref_A2B_J[1:min(20, length(ref_A2B_J))])
println("  V (first 20): ", ref_A2B_V[1:min(20, length(ref_A2B_V))])

println("\nExported A2B sparse structure:")
println("  I (first 20): ", exp_mod.A2BMAP_1_I[1:min(20, length(exp_mod.A2BMAP_1_I))])
println("  J (first 20): ", exp_mod.A2BMAP_1_J[1:min(20, length(exp_mod.A2BMAP_1_J))])
println("  V (first 20): ", exp_mod.A2BMAP_1_V[1:min(20, length(exp_mod.A2BMAP_1_V))])

# Build dense matrices
ref_A2B_dense = Matrix(ref_A2B)
exp_A2B_dense = zeros(Float64, exp_mod.A2BMAP_1_SIZE...)
for k in 1:length(exp_mod.A2BMAP_1_I)
    exp_A2B_dense[exp_mod.A2BMAP_1_I[k], exp_mod.A2BMAP_1_J[k]] = exp_mod.A2BMAP_1_V[k]
end

println("\nA2B dense matrix difference (Frobenius norm): ", norm(ref_A2B_dense - exp_A2B_dense))
println("A2B matrices match? ", ref_A2B_dense ≈ exp_A2B_dense)

# Compare basis specs
println("\n=== Comparing basis specs ===")
ref_aspec = et_st_splined.basis.aspec
ref_aaspecs = et_st_splined.basis.aaspecs

println("Reference A-spec (first 10): ", ref_aspec[1:min(10, length(ref_aspec))])
println("Exported A-spec (first 10): ", exp_mod.ABASIS_SPEC[1:min(10, length(exp_mod.ABASIS_SPEC))])
println("A-spec match? ", ref_aspec == collect(exp_mod.ABASIS_SPEC))

println("\nReference AA-spec order 1: ", ref_aaspecs[1])
println("Exported AA-spec order 1: ", exp_mod.AABASIS_SPECS_1)
println("AA-spec order 1 match? ", ref_aaspecs[1] == collect(exp_mod.AABASIS_SPECS_1))

println("\nReference AA-spec order 2 (first 10): ", ref_aaspecs[2][1:min(10, length(ref_aaspecs[2]))])
println("Exported AA-spec order 2 (first 10): ", exp_mod.AABASIS_SPECS_2[1:min(10, length(exp_mod.AABASIS_SPECS_2))])
println("AA-spec order 2 match? ", ref_aaspecs[2] == collect(exp_mod.AABASIS_SPECS_2))

# Now let's try to directly evaluate the reference model's site energy
# by calling the ETACE model directly
println("\n=== Direct reference model evaluation ===")

# The ETACE model takes an ETGraph as input
# Let's look at what's inside the graph
println("\nGraph structure:")
println("  # nodes: ", length(G.node_data))
println("  # edges: ", length(G.edge_data))

# Let me try to manually trace through the reference model
# The reference model structure is:
# rembed (EdgeEmbed{TransSelSplines}) -> yembed (EdgeEmbed{EmbedDP}) -> basis (SparseACEbasis) -> readout (SelectLinL)

println("\n=== Trying to trace through reference model ===")

# Get model components
rembed = et_model_splined.rembed
yembed = et_model_splined.yembed
basis = et_model_splined.basis
readout = et_model_splined.readout

# Try to call each layer
# First, let's evaluate rembed for a single edge
R_test = SVector(1.3575, 1.3575, 1.3575)  # One of the actual neighbor positions
edge = (𝐫 = R_test, z0 = AtomsBase.ChemicalSpecies(:Si), z1 = AtomsBase.ChemicalSpecies(:Si))

println("\nEdge: ", edge)
println("r = ", norm(R_test))

# Try to call rembed.layer directly
# TransSelSplines is the layer inside EdgeEmbed
rembed_layer = rembed.layer

# Let's look at what TransSelSplines expects
println("\nrembed_layer type: ", typeof(rembed_layer))

# Check if we can access the evaluate function
# For TransSelSplines, the forward pass is: (x, ps, st) -> (y, st)
# where x is the input (edge), ps is parameters, st is state

# The state for rembed is in et_st_splined.rembed
rembed_st = et_st_splined.rembed
println("rembed state type: ", typeof(rembed_st))
println("rembed state: ", rembed_st)

# Check the full state for splined model
println("\n=== Full state structure ===")
println("et_st_splined.rembed: ", et_st_splined.rembed)
println("\net_st_splined.yembed: ", et_st_splined.yembed)

# The state should have params for the splines
if haskey(et_st_splined.rembed, :params)
    println("\net_st_splined.rembed.params type: ", typeof(et_st_splined.rembed.params))
end

# Actually let's check calc_splined.st which is the state for the calculator
println("\n=== Calculator state structure ===")
calc_st = calc_splined.st
println("calc_st keys: ", keys(calc_st))
println("\ncalc_st.rembed: ", typeof(calc_st.rembed))
if isa(calc_st.rembed, NamedTuple)
    println("  keys: ", keys(calc_st.rembed))
    if haskey(calc_st.rembed, :params)
        println("  params type: ", typeof(calc_st.rembed.params))
        if isa(calc_st.rembed.params, NamedTuple)
            println("  params keys: ", keys(calc_st.rembed.params))
        end
    end
end

# Check basis state
println("\ncalc_st.basis: ", typeof(calc_st.basis))
if isa(calc_st.basis, NamedTuple)
    println("  keys: ", keys(calc_st.basis))
end

# NOW let's look at something different:
# What if the issue is that the reference uses a DIFFERENT E0?
# Let me check if there's a V_REF or reference energy

println("\n=== Checking for reference energies ===")

# Check et_ps for any reference energies
println("et_ps keys: ", keys(et_ps))
println("et_ps_splined keys: ", keys(et_ps_splined))

# Check if there's a vref or e0 in the parameters
for (k, v) in pairs(et_ps_splined)
    if occursin("ref", lowercase(string(k))) || occursin("e0", lowercase(string(k)))
        println("Found $k: ", v)
    end
end

# Actually, let me look at the WB values more carefully
# Maybe there's a difference we missed
println("\n=== Detailed WB comparison ===")
WB_ref = et_ps_splined.readout.W[1, :, 1]
WB_exp = exp_mod.WB_1

println("Reference WB (all 31 values):")
for i in 1:length(WB_ref)
    println("  WB[$i] = $(WB_ref[i]) vs $(WB_exp[i]) | diff = $(abs(WB_ref[i] - WB_exp[i]))")
end

println("\nTotal WB diff: ", norm(WB_ref - WB_exp))
println("WB match? ", WB_ref ≈ WB_exp)

# Let me try yet another thing: evaluate the exported model and reference model
# for exactly the same neighbor list and compare intermediate values

println("\n=== Computing A-basis manually from embeddings ===")

Rs1, Zs1 = neighbor_data[1]
n_neighbors = length(Rs1)

# Get embeddings from exported model
Rnl_exp, Ylm_exp = exp_mod.compute_embeddings(Rs1, Zs1, 14)
println("Exported Rnl shape: ", size(Rnl_exp))
println("Exported Ylm shape: ", size(Ylm_exp))

# Compute A-basis from exported embeddings
N_A = length(exp_mod.ABASIS_SPEC)
A_exp = zeros(Float64, N_A)
for (iA, (n, l)) in enumerate(exp_mod.ABASIS_SPEC)
    for j in 1:n_neighbors
        A_exp[iA] += Rnl_exp[j, n] * Ylm_exp[j, l]
    end
end

println("\nExported A-basis (computed manually):")
println("  A[1:10] = ", A_exp[1:10])
println("  sum(A) = ", sum(A_exp))
println("  max(abs(A)) = ", maximum(abs.(A_exp)))

# Now compute AA-basis
N_AA = sum(length.(exp_mod.AABASIS_SPECS_1)) + sum(length.(exp_mod.AABASIS_SPECS_2))
println("\n  N_AA = ", N_AA, " (order 1: ", length(exp_mod.AABASIS_SPECS_1), ", order 2: ", length(exp_mod.AABASIS_SPECS_2), ")")

AA_exp = zeros(Float64, 47)  # Based on AABASIS_RANGES

# Order 1
for (i_local, ϕ) in enumerate(exp_mod.AABASIS_SPECS_1)
    i = i_local
    AA_exp[i] = A_exp[ϕ[1]]
end

# Order 2
for (i_local, ϕ) in enumerate(exp_mod.AABASIS_SPECS_2)
    i = 8 + i_local
    AA_exp[i] = A_exp[ϕ[1]] * A_exp[ϕ[2]]
end

println("\nExported AA-basis:")
println("  AA[1:8] (order 1) = ", AA_exp[1:8])
println("  AA[9:17] (order 2, first 9) = ", AA_exp[9:17])
println("  sum(AA) = ", sum(AA_exp))
println("  max(abs(AA)) = ", maximum(abs.(AA_exp)))

# Now compute B = A2B * AA
B_exp = exp_A2B_dense * AA_exp

println("\nExported B-basis:")
println("  B[1:10] = ", B_exp[1:10])
println("  sum(B) = ", sum(B_exp))
println("  max(abs(B)) = ", maximum(abs.(B_exp)))

# Compute site energy
E_site_manual = dot(WB_ref, B_exp)
println("\nSite energy (manual): ", E_site_manual, " eV")

# Compare with exported model's site_energy
E_site_exp = exp_mod.site_energy(Rs1, Zs1, 14)
println("Site energy (exported): ", E_site_exp, " eV")

# The BIG question: what is the reference site energy for atom 1?
println("\n=== Reference site energy ===")
println("Reference total for 4 atoms: ", E_ref, " eV")
println("Reference per atom (avg): ", E_ref / 4, " eV")

# The issue is clear: our exported gives -0.01 eV per atom
# but reference gives -4.89 eV per atom
# That's a 489x difference!

# Let me check if there's something we're missing in the A-basis computation
# Maybe the reference uses a DIFFERENT pooling?

println("\n=== Checking A-basis computation pattern ===")
println("ABASIS_SPEC[1] = ", exp_mod.ABASIS_SPEC[1], " means A[1] = sum_j(Rnl[j,$(exp_mod.ABASIS_SPEC[1][1])] * Ylm[j,$(exp_mod.ABASIS_SPEC[1][2])])")
println("  => A[1] = sum of (Rnl[:, 1] .* Ylm[:, 1])")
println("  => A[1] = ", sum(Rnl_exp[:, 1] .* Ylm_exp[:, 1]))

# This should match what we computed above
println("  vs A_exp[1] = ", A_exp[1])
